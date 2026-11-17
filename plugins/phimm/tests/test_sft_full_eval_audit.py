"""Strict source coverage and documented prompt-conditioning audit contracts."""

import copy
import hashlib
import json
import sys
from concurrent.futures import Future
from types import ModuleType, SimpleNamespace

import pytest
from recipe.phimm.sft import full_eval_audit as audit


def export_identity():
    root = "az://account/data/outputs/sft_continue1736"
    model_path = root + "/exports/step1736/qwen_hf"
    settings = {"candidate": True, "source_optimizer_step": 1736,
                "source_checkpoint": root + "/global_step_1736",
                "source_training_revision": "a" * 40, "model_path": model_path}
    producer = {"trainer": {"default_hdfs_dir": root, "experiment_name": "sft_continue1736"},
                "model": {"path": "az://baseline/qwen_hf", "lora_rank": 320, "lora_alpha": 640}}
    exported = {"checkpoint_step": 1736, "checkpoint": settings["source_checkpoint"],
                "source_training_revision": "a" * 40, "effective_producer_config": producer,
                "base_model": "az://baseline/qwen_hf", "lora_rank": 320, "lora_alpha": 640,
                "cpu_loader_verified": True, "all_tensors_finite": True,
                "files": [{"path": model_path + "/model.safetensors", "bytes": 123}]}
    return settings, exported


@pytest.mark.parametrize("layout", ["sft", "remax"])
def test_explicit1736_identity_verifies_actual_export_and_both_producer_layouts(monkeypatch, layout):
    settings, exported = export_identity()
    if layout == "remax":
        producer = exported["effective_producer_config"]
        producer["actor_rollout_ref"] = {"model": producer.pop("model")}
    requested = []
    monkeypatch.setattr(audit, "load_json", lambda uri: (requested.append(uri) or exported, "export-sha"))
    step, proof = audit.source_checkpoint_provenance(settings)
    assert step == proof["checkpoint_step"] == 1736
    assert requested == [settings["model_path"] + "/export_provenance.json"]
    assert proof["source_training_revision"] == settings["source_training_revision"]
    assert proof["sha256"] == "export-sha"


@pytest.mark.parametrize("field,value,message", [
    ("checkpoint_step", 868, "optimizer step differs"),
    ("checkpoint", "az://other/global_step_1736", "source checkpoint differs"),
    ("source_training_revision", "b" * 40, "producer revision differs"),
    ("cpu_loader_verified", False, "successful model verification"),
    ("all_tensors_finite", False, "successful model verification"),
    ("files", [{"path": "az://other/model.safetensors", "bytes": 123}], "this merged model"),
])
def test_explicit1736_rejects_mismatched_or_unverified_export(monkeypatch, field, value, message):
    settings, exported = export_identity()
    exported[field] = value
    monkeypatch.setattr(audit, "load_json", lambda uri: (exported, "sha"))
    with pytest.raises(ValueError, match=message):
        audit.source_checkpoint_provenance(settings)


def test_explicit_step_cannot_relabel_an868_checkpoint_before_reading_export(monkeypatch):
    settings, _ = export_identity()
    settings["source_checkpoint"] = settings["source_checkpoint"].replace("1736", "868")
    monkeypatch.setattr(audit, "load_json", lambda uri: pytest.fail("must reject checkpoint before IO"))
    with pytest.raises(ValueError, match="path and optimizer step differ"):
        audit.source_checkpoint_provenance(settings)


def test_historical868_inputs_keep_default_identity_without_new_export_requirement(monkeypatch):
    monkeypatch.setattr(audit, "load_json", lambda uri: pytest.fail("legacy inputs must not require new artifacts"))
    for candidate in (False, True):
        assert audit.source_checkpoint_provenance({"candidate": candidate}) == (868, None)


def test_complete_candidate_audit_publishes_verified1736_not_historical868(monkeypatch):
    settings, exported = export_identity()
    settings.update(rescore_baseline=False, candidate_root="az://evaluation", submission_id="eval1736",
                    experiment="candidate1736")
    names = [f"source{i}" for i in range(35)]
    records = [{"dataset": name, "candidate_status": "AUDITED",
                "baseline": {"segments": 1, "groups": 1}} for name in names]
    document = {"settings": settings, "reference": {"datasets": [{"source": name} for name in names]},
                "suites": {name: "test" for name in names}, "baseline_current_audit_uri": "baseline"}
    documents = {"input": document, settings["model_path"] + "/export_provenance.json": exported,
                 "az://evaluation/val_data_gen/0.metrics.json": {"step": 0},
                 "baseline": {"audit_status": "COMPLETE", "all35_baseline_artifacts_freshly_verified": True,
                              "scoring_module_sha256": {}, "scorer_config_sha256": audit.json_hash({}),
                              "execution_revision": "baseline-revision", "results": records}}
    saved = {}
    monkeypatch.setattr(audit, "load_json", lambda uri: (documents[uri], "sha-" + uri))
    monkeypatch.setattr(audit, "scoring_snapshot", lambda: {"execution_revision": "eval-revision", "scoring_module_sha256": {}})
    monkeypatch.setattr(audit, "scorer_config", lambda settings: {})
    monkeypatch.setattr(audit, "candidate_coverage_policy", lambda settings: {})
    monkeypatch.setattr(audit, "validate_record", lambda *args, **kwargs: None)
    monkeypatch.setattr(audit.bf, "exists", lambda uri: False)
    monkeypatch.setattr(audit, "save_json", lambda uri, data: saved.update({uri: data}))
    monkeypatch.setattr(audit, "as_completed", lambda futures: list(futures))
    job = SimpleNamespace(status="SUCCEEDED", entrypoint=f"eval_openfull_zip candidate1736 {settings['model_path']}",
                          metadata={}, message="done")
    ray = ModuleType("ray")
    submission = ModuleType("ray.job_submission")
    submission.JobSubmissionClient = lambda address: SimpleNamespace(get_job_info=lambda identifier: job)
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.job_submission", submission)

    class Pool:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, fn, reference, *args):
            future = Future()
            future.set_result(next(record for record in records if record["dataset"] == reference["source"]))
            return future

    monkeypatch.setattr(audit, "ProcessPoolExecutor", Pool)
    result = audit.run("input", "output")
    assert result["full35_accepted"] is True
    assert saved["output"]["source_optimizer_step"] == 1736
    assert saved["output"]["source_checkpoint_provenance"]["checkpoint"] == settings["source_checkpoint"]
    assert saved["output"]["baseline_identity"] == "original standard synchronous ReMax Base3a@868"


@pytest.fixture
def policy():
    return {"version": audit.COVERAGE_POLICY_VERSION,
            "language_rules": {"monsoon_en_in": {"baseline": "English In", "candidate": "English",
                "prompt_before": "The languages are English and In.", "prompt_after": "The language is English."}}}


def pair():
    old = {"id": "one", "gts": "reference", "data_source": "monsoon_en_in", "language": "English In",
           "input": "user\nTranscribe verbatim.\nThe languages are English and In.\nassistant\n"}
    new = {**old, "language": "English", "input": old["input"].replace("The languages are English and In.", "The language is English.")}
    return old, new


def test_exact_core8_pair_preserves_raw9_difference_and_records_every_prompt(policy):
    old, new = pair()
    proof = audit.approve_conditioning_change("monsoon_en_in", [old], [new], policy)
    assert audit.identity(old) != audit.identity(new)
    assert audit.core_identity(old) == audit.core_identity(new)
    assert proof["status"] == "DOCUMENTED_PROMPT_CONDITIONING_CHANGE"
    assert proof["changed_language_segments"] == proof["paired_source_segments"] == 1
    assert proof["language_pair_counts"] == [{"baseline":"English In", "candidate":"English", "count":1}]
    assert proof["prompt_pair_counts"][0]["baseline"] == old["input"]
    assert proof["prompt_pair_counts"][0]["candidate"] == new["input"]


@pytest.mark.parametrize("field,value", [("gts","other"),("id","other"),("audio_path","other"),
                                          ("seg_start",1),("seg_end",2),("parent_audio_path","other")])
def test_any_core_identity_drift_fails_closed(policy, field, value):
    old, new = pair()
    new[field] = value
    with pytest.raises(ValueError, match="core8 coverage mismatch"):
        audit.approve_conditioning_change("monsoon_en_in", [old], [new], policy)


@pytest.mark.parametrize("change", ["extra_text", "missing_sentence", "duplicate_sentence", "wrong_language", "equal_language_changed_prompt"])
def test_unapproved_prompt_or_language_change_fails(policy, change):
    old, new = pair()
    if change == "extra_text": new["input"] += "X"
    elif change == "missing_sentence": new["input"] = "no language hint"
    elif change == "duplicate_sentence": new["input"] += "The language is English."
    elif change == "wrong_language": new["language"] = "Hindi"
    else: new["language"] = old["language"]
    with pytest.raises(ValueError, match="Input differs|Unapproved|input change"):
        audit.approve_conditioning_change("monsoon_en_in", [old], [new], policy)


def test_hindi_and_other_sources_cannot_use_english_exception(policy):
    old, new = pair()
    for source in ("monsoon_hi_in", "ls_clean"):
        with pytest.raises(ValueError, match="No approved conditioning rule"):
            audit.approve_conditioning_change(source, [old], [new], policy)


def test_ambiguous_duplicate_core_is_not_deduplicated_into_acceptance(policy):
    old, new = pair()
    with pytest.raises(ValueError, match="Ambiguous duplicate"):
        audit.approve_conditioning_change("monsoon_en_in", [old, copy.deepcopy(old)], [new], policy)


def test_audit_one_preserves_observed9_and_independently_checks_scores(tmp_path, monkeypatch, policy):
    old, new = pair()
    old.update(n_err=1, n_ref=2, output="historical")
    baseline = tmp_path / "baseline.jsonl"
    baseline.write_text(json.dumps(old) + "\n")
    ref = {"source":"monsoon_en_in", "uri":str(baseline), "sha256":hashlib.sha256(baseline.read_bytes()).hexdigest(),
           "segments":1, "inactive_child_rows":0, "n_err":1, "n_ref":2}
    root = tmp_path / "candidate"
    target = root / "val_data_gen/monsoon_en_in/0.jsonl"
    target.parent.mkdir(parents=True)
    emitted = {"step":0, "asr_data_source":"monsoon_en_in", "gts":new["gts"], "input":new["input"], "n_err":0, "n_ref":2,
               "asr_detail":{"parent_info":new, "segment_outputs":["candidate"], "segment_rows":[{"extra_info":new}]}}
    target.write_text(json.dumps(emitted)+"\n")
    current = {"monsoon_en_in":{"source":str(baseline),"sha256":ref["sha256"],
               "coverage_sha256":audit.coverage_hash(audit.Counter([audit.identity(old)])),
               "current_scorer":{"n_err":1,"n_ref":2,"value":.5,"groups":1}}}
    monkeypatch.setattr(audit, "rescore", lambda *a,**kw:{"n_err":0,"n_ref":2})
    settings = {"candidate":True,"rescore_baseline":False,"candidate_root":str(root)}
    metrics = {"metrics":{"val-core/monsoon_en_in/p_err/mean@1":0.0}}
    result = audit.audit_one(ref,"OpenEN",settings,{},metrics,current,policy)
    b,c = result["baseline"],result["candidate"]
    assert result["candidate_status"] == "AUDITED"
    assert c["coverage_sha256"] != b["coverage_sha256"]
    assert c["core8_coverage_sha256"] == b["core8_coverage_sha256"]
    assert c["missing_source_segments"] == c["extra_source_segments"] == 1
    assert not c["strict9_coverage_matches"] and c["coverage_matches"]
    emitted["n_err"] = 1
    target.write_text(json.dumps(emitted)+"\n")
    with pytest.raises(ValueError,match="Independent candidate rescore differs"):
        audit.audit_one(ref,"OpenEN",settings,{},metrics,current,policy)


def test_real_spawn_parent_scoring_matches_serial_counts_and_coverage(tmp_path):
    config = audit.scorer_config({"model_path":"az://model", "experiment":"test-audit"})
    path = tmp_path / "parents.jsonl"
    rows = []
    for i, words in enumerate((["hello", "world"], ["hello brave world"], ["hello"], ["hello world"])):
        outputs = ["<src=English><tgt=English>\n<VERBATIM>\n" + text for text in words]
        info = {"id":f"parent{i}","language":"English"}
        score = audit.rescore(config,"earnings21",{"gts":"hello world"},outputs,info)
        rows.append({"step":0,"asr_data_source":"earnings21","gts":"hello world", "input":"same prompt",
                     "n_err":score["n_err"],"n_ref":score["n_ref"],
                     "asr_detail":{"parent_info":info,"segment_outputs":outputs,"segment_rows":[{"extra_info":info}]}})
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    results = []
    for workers in (1,2):
        digest = hashlib.sha256()
        scored = list(audit.candidate_scored_rows(str(path),"earnings21",config,workers,digest))
        assert digest.hexdigest() == hashlib.sha256(path.read_bytes()).hexdigest()
        totals = tuple(sum(score[key] for _,_,score in scored) for key in ("n_err","n_ref"))
        coverage = audit.Counter(audit.core_identity({**detail["parent_info"],"gts":row["gts"],"data_source":"earnings21"})
                                 for row,detail,_ in scored)
        results.append((totals,coverage))
    assert results[0] == results[1]
    assert results[0][0] == (2,8)


def test_failed_future_is_logged_and_other_successes_commit_before_final_failure(monkeypatch, capsys):
    names = [f"source{i}" for i in range(35)]
    document = {"settings":{"candidate":False}, "reference":{"datasets":[{"source":n} for n in names]},
                "suites":{n:"test" for n in names}}
    saved = []
    monkeypatch.setattr(audit,"load_json",lambda uri:(document,"input-sha"))
    monkeypatch.setattr(audit,"scoring_snapshot",lambda:{"execution_revision":"test","scoring_module_sha256":{}})
    monkeypatch.setattr(audit,"scorer_config",lambda settings:{})
    monkeypatch.setattr(audit,"validate_record",lambda *args,**kwargs:None)
    monkeypatch.setattr(audit.bf,"exists",lambda path:False)
    monkeypatch.setattr(audit,"save_json",lambda path,data:saved.append((path,data)))
    monkeypatch.setattr(audit,"as_completed",lambda futures:list(futures))

    class Pool:
        def __init__(self,**kwargs): pass
        def __enter__(self): return self
        def __exit__(self,*args):
            # The driver has committed all successful peers before pool cleanup.
            assert len(saved) == 34
        def submit(self,fn,ref,*args):
            future = Future()
            if ref["source"] == names[0]: future.set_exception(ValueError("known coverage failure"))
            else: future.set_result({"dataset":ref["source"]})
            return future

    monkeypatch.setattr(audit,"ProcessPoolExecutor",Pool)
    with pytest.raises(ValueError,match="Dataset audits failed; successful cohorts remain committed"):
        audit.run("input","output")
    output = capsys.readouterr().out
    assert output.index("AUDIT_FAILED source0") < output.index("COMMITTED source1")
    assert all(path != "output" for path,_ in saved)
