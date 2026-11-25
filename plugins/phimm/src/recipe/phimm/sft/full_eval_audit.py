"""Persist and resume corpus audits of the final OpenFull35 ASR evaluation.

This CPU-only reader never loads model weights or submits experiments. Run it as
an ordinary persistent Ray job after syncing this module. Inputs are JSON data:
``settings``, archived ``reference``, source-to-``suites`` mapping, optional
``seed_records`` and ``seed_scoring_snapshot``, and (for candidate audits)
``baseline_current_audit_uri``. Every completed dataset is committed separately
before the final audit, so interrupted runs can resume without repeating work.
Candidate ``settings.source_optimizer_step`` defaults to868 for historical
inputs. Explicit steps require ``source_checkpoint`` and
``source_training_revision`` matching the merged model's export provenance.
"""

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
import datetime
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import subprocess
import traceback

import blobfile as bf

SCHEMA_VERSION = 1
FIELDS = (
    "id", "gts", "data_source", "language", "audio_path", "parent_audio_path",
    "seg_index", "seg_start", "seg_end",
)
CORE_FIELDS = tuple(field for field in FIELDS if field != "language")
COVERAGE_POLICY_VERSION = "core8-with-paired-configured-english-prompt-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def identity(row):
    return json_hash({key: row.get(key) for key in FIELDS})


def core_identity(row):
    return json_hash({key: row.get(key) for key in CORE_FIELDS})


def candidate_coverage_policy(settings):
    """Authorize only the observed English change, backed by composed config."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    directory = Path(__file__).resolve().parents[1] / "config/v2611_dev/eval"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        config = compose(config_name="eval_openfull_zip", overrides=[
            "actor_rollout_ref.model.path=" + settings["model_path"],
            "trainer.experiment_name=" + settings["experiment"],
        ])
    data = OmegaConf.to_container(config.data.val_data, resolve=True)
    found = []

    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            source = value.get("post_process", {}).get("add_field", {}).get("fields", {}).get("data_source")
            if source == "monsoon_en_in":
                found.append(value)
            for item in value.values():
                if isinstance(item, (list, dict)):
                    visit(item)

    visit(data)
    require(len(found) == 1, "Expected one composed monsoon_en_in data definition")
    task = found[0].get("add_task_info", {})
    require(task.get("language") == "English" and task.get("lang_hint") is True,
            "Observed English conditioning change is unsupported by composed config")
    return {
        "version": COVERAGE_POLICY_VERSION, "core_fields": list(CORE_FIELDS), "strict_fields": list(FIELDS),
        "audit_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "configured_dataset_sha256": json_hash(found[0]), "configured_add_task_info": task,
        "language_rules": {"monsoon_en_in": {"baseline": "English In", "candidate": "English",
                           "prompt_before": "The languages are English and In.",
                           "prompt_after": "The language is English."}},
    }


def approve_conditioning_change(name, baseline_rows, candidate_rows, policy):
    """Pair exact core identities; permit only the recorded prompt sentence edit."""
    rule = policy["language_rules"].get(name)
    require(rule is not None, f"No approved conditioning rule for {name}")
    require(policy["version"] == COVERAGE_POLICY_VERSION, "Unknown coverage policy")
    baseline = {core_identity(row): row for row in baseline_rows}
    candidate = {core_identity(row): row for row in candidate_rows}
    require(len(baseline) == len(baseline_rows) and len(candidate) == len(candidate_rows),
            f"Ambiguous duplicate core identity in conditioning comparison: {name}")
    require(baseline.keys() == candidate.keys(), f"Conditioning comparison core8 coverage mismatch: {name}")
    language_pairs, prompt_pairs = Counter(), Counter()
    changed = 0
    for key, old in baseline.items():
        new = candidate[key]
        old_lang, new_lang = old.get("language"), new.get("language")
        old_prompt, new_prompt = old.get("input"), new.get("input")
        require(isinstance(old_prompt, str) and isinstance(new_prompt, str), f"Missing paired input: {name}")
        if old_lang == new_lang:
            require(old_prompt == new_prompt, f"Unapproved input change at equal language: {name}/{old.get('id')}")
        else:
            require((old_lang, new_lang) == (rule["baseline"], rule["candidate"]),
                    f"Unapproved language pair: {name}/{old.get('id')}: {old_lang!r}->{new_lang!r}")
            require(old_prompt.count(rule["prompt_before"]) == 1
                    and new_prompt.count(rule["prompt_after"]) == 1
                    and old_prompt.replace(rule["prompt_before"], rule["prompt_after"]) == new_prompt,
                    f"Input differs beyond approved language sentence: {name}/{old.get('id')}")
            changed += 1
        language_pairs[(old_lang, new_lang)] += 1
        prompt_pairs[(old_prompt, new_prompt)] += 1
    require(changed > 0, f"Conditioning approval requires an actual documented change: {name}")
    return {
        "status": "DOCUMENTED_PROMPT_CONDITIONING_CHANGE", "policy_version": policy["version"],
        "policy_sha256": json_hash(policy), "dataset": name, "configured_rule": rule,
        "paired_source_segments": len(baseline), "changed_language_segments": changed,
        "exact_core8_multiset_matches": True, "all_paired_inputs_checked": True,
        "language_pair_counts": [{"baseline": a, "candidate": b, "count": n}
                                 for (a, b), n in sorted(language_pairs.items())],
        "prompt_pair_counts": [{"baseline": a, "candidate": b, "baseline_sha256": hashlib.sha256(a.encode()).hexdigest(),
                                "candidate_sha256": hashlib.sha256(b.encode()).hexdigest(), "count": n}
                               for (a, b), n in sorted(prompt_pairs.items())],
    }


def coverage_hash(counter):
    return hashlib.sha256("\n".join(sorted(counter.elements())).encode()).hexdigest()


def load_json(uri):
    with bf.BlobFile(uri, "rb") as stream:
        raw = stream.read()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def save_json(uri, document):
    # BlobFile commits the completed object on close; the final audit is written
    # only after every per-dataset record has already been committed.
    with bf.BlobFile(uri, "w") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)


def source_checkpoint_provenance(settings):
    """Bind an explicit candidate step to its actual producer and HF export."""
    step = settings.get("source_optimizer_step", 868)
    require(type(step) is int and step > 0, "Source optimizer step must be a positive integer")
    if not settings["candidate"] or "source_optimizer_step" not in settings:
        require(step == 868, "Historical baseline source optimizer step must remain868")
        return step, None
    checkpoint = settings.get("source_checkpoint", "").rstrip("/")
    revision = settings.get("source_training_revision")
    require(checkpoint.endswith(f"/global_step_{step}"), "Source checkpoint path and optimizer step differ")
    require(isinstance(revision, str) and len(revision) == 40
            and all(char in "0123456789abcdef" for char in revision),
            "Expected source training revision must be a full commit SHA")
    model_path = settings["model_path"].rstrip("/")
    uri = model_path + "/export_provenance.json"
    exported, sha = load_json(uri)
    require(exported.get("checkpoint_step") == step, "Export provenance optimizer step differs")
    require(exported.get("checkpoint", "").rstrip("/") == checkpoint,
            "Export provenance source checkpoint differs")
    require(exported.get("source_training_revision") == revision, "Export provenance producer revision differs")
    producer = exported.get("effective_producer_config", {})
    trainer = producer.get("trainer", {})
    require(trainer.get("default_hdfs_dir", "").rstrip("/") == checkpoint.rsplit("/", 1)[0],
            "Export provenance producer output root differs")
    require(trainer.get("experiment_name") == checkpoint.rsplit("/", 2)[-2],
            "Export provenance producer experiment differs")
    model = producer.get("model") or producer.get("actor_rollout_ref", {}).get("model", {})
    require(bool(exported.get("base_model")) and isinstance(exported.get("lora_rank"), int)
            and exported["lora_rank"] > 0 and isinstance(exported.get("lora_alpha"), (int, float))
            and exported["lora_alpha"] > 0
            and model.get("path", "").rstrip("/") == exported["base_model"].rstrip("/")
            and model.get("lora_rank") == exported.get("lora_rank")
            and model.get("lora_alpha") == exported.get("lora_alpha"),
            "Export provenance model or adapter differs from producer config")
    require(exported.get("cpu_loader_verified") is True and exported.get("all_tensors_finite") is True,
            "Export provenance lacks successful model verification")
    require(any(item.get("path") == model_path + "/model.safetensors"
                and item.get("bytes", 0) > 0 for item in exported.get("files", [])),
            "Export provenance does not identify this merged model")
    return step, {"source": uri, "sha256": sha, "checkpoint": checkpoint,
                  "checkpoint_step": step, "source_training_revision": revision,
                  "producer_config_sha256": exported.get("producer_config_sha256")}


def source_stat_matches(document):
    stat = bf.stat(document["source"])
    return stat.size == document["bytes"] and stat.mtime == document["mtime"]


def scoring_snapshot():
    package = Path(__file__).resolve().parents[1]
    repository = next(parent for parent in package.parents if (parent / ".git").exists())
    paths = (
        set((package / "reward").glob("*.py"))
        | set((package / "utils/open_asr_normalizer").glob("*.py"))
        | {package / "agentic/scoring.py", package / "utils/tn.py", package / "utils/languages.py"}
    )
    return {
        "execution_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True,
        ).strip(),
        "scoring_module_sha256": {
            str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)
        },
    }


def scorer_config(settings):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    directory = Path(__file__).resolve().parents[1] / "config/v2611_dev/eval"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        config = compose(config_name="eval_openfull_zip", overrides=[
            "actor_rollout_ref.model.path=" + settings["model_path"],
            "trainer.experiment_name=" + settings["experiment"],
        ])
    return {
        "rewards": OmegaConf.to_container(config.rewards, resolve=True),
        "reward_map": OmegaConf.to_container(config.reward_map, resolve=True),
    }


def rescore(config, name, row, outputs, info=None):
    from recipe.phimm.agentic.scoring import score_response

    return score_response(config, {
        "data_source": name,
        "reward_model": {"ground_truth": row["gts"]},
        "extra_info": row if info is None else info,
    }, outputs)


def score_candidate_parent(config, name, ground_truth, outputs, parent_info, expected):
    """Spawn-safe scoring payload excludes the giant emitted detail/segment rows."""
    score = rescore(config, name, {"gts": ground_truth}, outputs, parent_info)
    for key in ("n_err", "n_ref"):
        require(math.isclose(score[key], expected[key], rel_tol=0, abs_tol=1e-8),
                f"Independent candidate rescore differs: {name} {key}")
    return {key: score[key] for key in ("n_err", "n_ref")}


def candidate_scored_rows(path, name, config, workers, content_sha):
    """Stream original bytes in order, bounding optional parent scoring to2W."""
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) if workers > 1 else None
    pending = {}

    def completed():
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            row, detail = pending.pop(future)
            yield row, detail, future.result()

    try:
        with bf.BlobFile(path, "rb") as stream:
            for line in stream:
                content_sha.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                require(
                    row["step"] == 0 and row.get("asr_data_source", row.get("data_source")) == name,
                    f"Wrong candidate step/source: {name}",
                )
                detail = row["asr_detail"]
                detail = json.loads(detail) if isinstance(detail, str) else detail
                payload = (
                    config,
                    name,
                    row["gts"],
                    detail["segment_outputs"],
                    detail.get("extra_info", detail.get("parent_info", {})),
                    {key: row[key] for key in ("n_err", "n_ref")},
                )
                if pool is None:
                    yield row, detail, score_candidate_parent(*payload)
                else:
                    pending[pool.submit(score_candidate_parent, *payload)] = (row, detail)
                    if len(pending) >= 2 * workers:
                        yield from completed()
            while pending:
                yield from completed()
    except Exception as error:
        print(f"PARENT_AUDIT_FAILED {name}: {error!r}", flush=True)
        traceback.print_exc()
        raise
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)


def audit_one(reference, suite, settings, config, metrics, current_baseline, coverage_policy=None, parent_workers=1):
    """Spawn-safe worker: read raw source/candidate and return one cohort proof."""
    name = reference["source"]
    uri = reference["uri"]
    stat = bf.stat(uri)
    sha = hashlib.sha256()
    coverage = Counter()
    core_coverage = Counter()
    conditioning_rows = []
    errors = refs = inactive = rows = 0
    rescored_errors = rescored_refs = rescored_groups = 0
    long_groups = {}
    with bf.BlobFile(uri, "rb") as stream:
        for line in stream:
            sha.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            coverage[identity(row)] += 1
            core_coverage[core_identity(row)] += 1
            if coverage_policy and name in coverage_policy["language_rules"]:
                conditioning_rows.append({**{field: row.get(field) for field in FIELDS}, "input": row.get("input")})
            require(row.get("data_source") == name, f"Wrong original baseline source: {name}")
            if row.get("n_err") is None or row.get("n_ref") is None:
                require(row.get("n_err") is None and row.get("n_ref") is None,
                        f"Only one original baseline count is null: {name}")
                inactive += 1
            else:
                errors += row["n_err"]
                refs += row["n_ref"]
            if settings["rescore_baseline"]:
                if row.get("parent_audio_path") and (row.get("n_segments") or 0) > 1:
                    entry = long_groups.setdefault(row["parent_audio_path"], {
                        "row": row, "outputs": [], "indices": [],
                    })
                    require(entry["row"]["gts"] == row["gts"], f"Grouped baseline references differ: {name}")
                    entry["outputs"].append(row["output"])
                    entry["indices"].append(row["seg_index"])
                else:
                    score = rescore(config, name, row, row["output"])
                    rescored_errors += score["n_err"]
                    rescored_refs += score["n_ref"]
                    rescored_groups += 1
    for entry in long_groups.values():
        row = entry["row"]
        order = sorted(range(len(entry["indices"])), key=lambda index: entry["indices"][index])
        require(len(order) == row["n_segments"] and len(set(entry["indices"])) == len(order),
                f"Incomplete original baseline parent: {name}")
        score = rescore(config, name, row, [entry["outputs"][index] for index in order])
        rescored_errors += score["n_err"]
        rescored_refs += score["n_ref"]
        rescored_groups += 1
    require(sha.hexdigest() == reference["sha256"], f"Original baseline artifact SHA drift: {name}")
    require((rows, inactive, errors, refs) == (
        reference["segments"], reference["inactive_child_rows"], reference["n_err"], reference["n_ref"],
    ), f"Original baseline corpus counts drift: {name}")
    baseline = {
        "dataset": name, "source": uri, "sha256": sha.hexdigest(), "bytes": stat.size, "mtime": stat.mtime,
        "segments": rows, "groups": rows - inactive, "unique_source_identities": len(coverage),
        "inactive_child_rows": inactive, "n_err": errors, "n_ref": refs, "value": errors / refs,
        "coverage_sha256": coverage_hash(coverage), "matches_archived_original": True,
    }
    if settings["rescore_baseline"]:
        require(rescored_refs > 0 and math.isfinite(rescored_errors), f"Invalid rescored baseline counts: {name}")
        require(rescored_groups == baseline["groups"], f"Rescored baseline grouping mismatch: {name}")
        baseline["current_scorer"] = {
            "n_err": rescored_errors, "n_ref": rescored_refs, "value": rescored_errors / rescored_refs,
            "groups": rescored_groups,
            "matches_original_counts": (rescored_errors, rescored_refs) == (errors, refs),
        }
    result = {"dataset": name, "baseline": baseline, "suite": suite}
    if not settings["candidate"]:
        return result
    checked = current_baseline[name]
    require((checked["source"], checked["sha256"], checked["coverage_sha256"]) == (
        baseline["source"], baseline["sha256"], baseline["coverage_sha256"],
    ), f"Candidate baseline differs from independently rescored original: {name}")
    baseline["current_scorer"] = checked["current_scorer"]
    baseline["core8_coverage_sha256"] = coverage_hash(core_coverage)
    path = settings["candidate_root"].rstrip("/") + f"/val_data_gen/{name}/0.jsonl"
    require(bf.exists(path), f"Missing candidate detail artifact: {name}")
    candidate_stat = bf.stat(path)
    candidate_sha = hashlib.sha256()
    identities = set()
    candidate_sources = {}
    groups = set()
    candidate_errors = candidate_refs = candidate_rows = acoustic_segments = 0
    parallelism = parent_workers if name in {"earnings21", "earnings22", "coraal"} else 1
    for row, detail, score in candidate_scored_rows(path, name, config, parallelism, candidate_sha):
        candidate_rows += 1
        require(row["step"] == 0 and row.get("asr_data_source", row.get("data_source")) == name,
                f"Wrong candidate step/source: {name}")
        shared_info = detail.get("extra_info", detail.get("parent_info", {}))
        group = identity({**shared_info, "gts": row["gts"], "data_source": name})
        require(group not in groups, f"Duplicate candidate scoring group: {name}")
        groups.add(group)
        candidate_errors += score["n_err"]
        candidate_refs += score["n_ref"]
        acoustic_segments += len(detail["segment_outputs"])
        for segment in detail.get("asr_segments", detail.get("segment_rows", [])):
            info = {**(detail.get("extra_info") or {}), **(segment.get("extra_info") or {})}
            source = {**info, "gts": row["gts"], "data_source": name}
            key = identity(source)
            identities.add(key)
            if key not in candidate_sources:
                candidate_sources[key] = {**{field: source.get(field) for field in FIELDS}, "input": row.get("input")}
            elif coverage_policy and name in coverage_policy["language_rules"]:
                require(candidate_sources[key]["input"] == row.get("input"), f"Repeated source has different input: {name}")
    candidate_coverage = Counter(identities)
    missing = sum((coverage - candidate_coverage).values())
    extra = sum((candidate_coverage - coverage).values())
    candidate_core = Counter(core_identity(row) for row in candidate_sources.values())
    core_missing = sum((core_coverage - candidate_core).values())
    core_extra = sum((candidate_core - core_coverage).values())
    require(not core_missing and not core_extra, f"Candidate core8 source/reference coverage mismatch: {name}")
    approval = None
    if missing or extra:
        require(coverage_policy is not None, "Missing candidate coverage policy")
        approval = approve_conditioning_change(name, conditioning_rows, list(candidate_sources.values()), coverage_policy)
    require(candidate_refs > 0 and math.isfinite(candidate_errors) and candidate_errors >= 0,
            f"Invalid candidate corpus counts: {name}")
    key = f"val-core/{name}/p_err/mean@1"
    logged = metrics["metrics"].get(key)
    matches_metric = logged is not None and math.isclose(
        candidate_errors / candidate_refs, logged, rel_tol=0, abs_tol=1e-10,
    )
    candidate = {
        "dataset": name, "source": path, "sha256": candidate_sha.hexdigest(),
        "bytes": candidate_stat.size, "mtime": candidate_stat.mtime,
        "groups": candidate_rows, "source_segments": len(identities), "decoded_acoustic_segments": acoustic_segments,
        "n_err": candidate_errors, "n_ref": candidate_refs, "value": candidate_errors / candidate_refs,
        "logged_value": logged, "metric_matches": matches_metric,
        # Preserve observed historical nine-field semantics even for an
        # explicitly approved prompt-conditioning difference.
        "coverage_sha256": coverage_hash(candidate_coverage), "strict9_coverage_matches": not missing and not extra,
        "coverage_matches": (not missing and not extra) or approval is not None,
        "core8_coverage_sha256": coverage_hash(candidate_core), "core8_missing": core_missing, "core8_extra": core_extra,
        "coverage_approval": approval,
        "missing_source_segments": missing, "extra_source_segments": extra,
        # Compound-aware WER denominators may depend on the hypothesis. Exact
        # source and reference identity, rather than n_ref equality, is the gate.
        "reference_count_matches": candidate_refs == baseline["current_scorer"]["n_ref"],
        "group_count_matches": candidate_rows == baseline["groups"],
        "all_emitted_counts_independently_rescored": True,
    }
    result["candidate"] = candidate
    result["candidate_status"] = "AUDITED" if (
        candidate["coverage_matches"] and candidate["metric_matches"] and candidate["group_count_matches"]
    ) else "MISMATCH"
    value = baseline["current_scorer"]["value"]
    result["delta_percentage_points"] = 100 * (candidate["value"] - value)
    result["werr"] = 1 - candidate["value"] / value if value > 0 else None
    return result


def validate_record(record, reference, settings, require_current=False):
    """Validate persisted/seeded data before allowing a completed cohort reuse."""
    baseline = record["baseline"]
    require(record["dataset"] == reference["source"], "Resume dataset identity mismatch")
    require((baseline["source"], baseline["sha256"]) == (reference["uri"], reference["sha256"]),
            "Resume original artifact identity mismatch")
    require((baseline["segments"], baseline["inactive_child_rows"], baseline["n_err"], baseline["n_ref"]) == (
        reference["segments"], reference["inactive_child_rows"], reference["n_err"], reference["n_ref"],
    ), "Resume original corpus counts mismatch")
    require(baseline["groups"] == baseline["segments"] - baseline["inactive_child_rows"], "Resume grouping mismatch")
    require(source_stat_matches(baseline), "Original source changed since its audit")
    if require_current:
        current = baseline["current_scorer"]
        require(current["groups"] == baseline["groups"] and current["n_ref"] > 0,
                "Invalid current-scorer resume counts/grouping")
        require(math.isclose(current["value"], current["n_err"] / current["n_ref"], rel_tol=0, abs_tol=1e-12),
                "Invalid current-scorer resume corpus ratio")
    if settings["candidate"]:
        candidate = record["candidate"]
        expected = settings["candidate_root"].rstrip("/") + f"/val_data_gen/{record['dataset']}/0.jsonl"
        require(candidate["source"] == expected and source_stat_matches(candidate), "Candidate source changed since audit")
        require(record["candidate_status"] == "AUDITED", "Cannot reuse a failed candidate audit")
        require(candidate["coverage_matches"] and candidate["metric_matches"] and candidate["group_count_matches"],
                "Invalid candidate acceptance flags")
        require(candidate["core8_coverage_sha256"] == baseline["core8_coverage_sha256"]
                and candidate["core8_missing"] == candidate["core8_extra"] == 0, "Invalid core8 coverage proof")
        if not candidate["strict9_coverage_matches"]:
            approval = candidate["coverage_approval"]
            require(approval and approval["status"] == "DOCUMENTED_PROMPT_CONDITIONING_CHANGE"
                    and approval["policy_version"] == COVERAGE_POLICY_VERSION
                    and approval["all_paired_inputs_checked"] and approval["exact_core8_multiset_matches"],
                    "Invalid prompt-conditioning approval")


def run(input_uri, output_uri, workers=8, parent_workers=1):
    document, input_sha = load_json(input_uri)
    settings = dict(document["settings"])
    settings.setdefault("candidate", False)
    settings.setdefault("rescore_baseline", not settings["candidate"])
    require(not (settings["candidate"] and settings["rescore_baseline"]),
            "Candidate audit consumes the separate completed baseline audit; do not rescore it again")
    source_step, source_provenance = source_checkpoint_provenance(settings)
    reference = document["reference"]
    suites = document["suites"]
    names = [row["source"] for row in reference["datasets"]]
    require(len(names) == len(set(names)) == 35 and set(names) == set(suites), "OpenFull requires exactly35 sources")
    snapshot = scoring_snapshot()
    config = scorer_config(settings)
    coverage_policy = candidate_coverage_policy(settings) if settings["candidate"] else None
    fingerprint_inputs = {
        "schema_version": SCHEMA_VERSION, "scoring_module_sha256": snapshot["scoring_module_sha256"],
        "scorer_config": config, "settings": settings, "reference": reference, "suites": suites,
    }
    if coverage_policy:
        fingerprint_inputs["candidate_coverage_policy"] = coverage_policy
    if source_provenance:
        fingerprint_inputs["source_checkpoint_provenance"] = source_provenance
    fingerprint = json_hash(fingerprint_inputs)
    job_info = metrics = metrics_sha = None
    current_baseline = {}
    baseline_dependency = None
    if settings["candidate"]:
        from ray.job_submission import JobSubmissionClient

        job = JobSubmissionClient(settings.get("ray_address", "http://127.0.0.1:9209")).get_job_info(settings["submission_id"])
        job_info = {"submission_id": settings["submission_id"], "status": str(job.status),
                    "entrypoint": job.entrypoint, "metadata": job.metadata, "message": job.message}
        require(str(job.status) == "SUCCEEDED", "Full evaluation Ray job is not SUCCEEDED")
        require(all(token in job.entrypoint for token in (
            settings["model_path"], settings["experiment"], "eval_openfull_zip",
        )), "Final job model/experiment/config mismatch")
        metrics_uri = settings["candidate_root"].rstrip("/") + "/val_data_gen/0.metrics.json"
        metrics, metrics_sha = load_json(metrics_uri)
        require(metrics["step"] == 0, "Standalone evaluation step must be0")
        baseline_uri = document["baseline_current_audit_uri"]
        baseline_audit, baseline_sha = load_json(baseline_uri)
        require(baseline_audit["audit_status"] == "COMPLETE" and baseline_audit["all35_baseline_artifacts_freshly_verified"],
                "Current-scorer baseline audit is incomplete")
        require(baseline_audit["scoring_module_sha256"] == snapshot["scoring_module_sha256"],
                "Baseline and candidate current scorer source differs")
        require(baseline_audit["scorer_config_sha256"] == json_hash(config), "Baseline and candidate scorer config differs")
        current_baseline = {row["dataset"]: row["baseline"] for row in baseline_audit["results"]}
        require(set(current_baseline) == set(names), "Current baseline does not cover35 sources")
        for ref in reference["datasets"]:
            validate_record({"dataset": ref["source"], "baseline": current_baseline[ref["source"]]},
                            ref, {"candidate": False}, require_current=True)
        baseline_dependency = {"source": baseline_uri, "sha256": baseline_sha,
                               "execution_revision": baseline_audit["execution_revision"]}
    seed = {row["dataset"]: row for row in document.get("seed_records", [])}
    if seed:
        require(not settings["candidate"] and settings["rescore_baseline"], "Seeds are restricted to baseline rescoring")
        require(set(seed).issubset(names), "Unexpected seed dataset")
        require(document["seed_scoring_snapshot"]["scoring_module_sha256"] == snapshot["scoring_module_sha256"],
                "Seed and current scorer code differs")
    completed = {}
    references = {row["source"]: row for row in reference["datasets"]}
    record_root = output_uri + ".datasets"
    for name in names:
        path = record_root + f"/{name}.json"
        record = None
        if bf.exists(path):
            envelope, _ = load_json(path)
            if envelope.get("fingerprint") == fingerprint:
                require(envelope["record_sha256"] == json_hash(envelope["record"]), "Resume record integrity mismatch")
                record = envelope["record"]
        if record is None and name in seed:
            record = seed[name]
        if record is not None:
            validate_record(record, references[name], settings, require_current=settings["rescore_baseline"] or settings["candidate"])
            require(record["suite"] == suites[name], "Resume suite mismatch")
            save_json(path, {"schema_version": SCHEMA_VERSION, "fingerprint": fingerprint,
                             "record_sha256": json_hash(record), "record": record,
                             "input_uri": input_uri, "input_sha256": input_sha, "seeded": name in seed})
            completed[name] = record
            print(f"REUSED {name} {len(completed)}/35", flush=True)
    pending = [row for row in reference["datasets"] if row["source"] not in completed]
    failures = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {
            pool.submit(audit_one, ref, suites[ref["source"]], settings, config, metrics, current_baseline, coverage_policy, parent_workers): ref["source"]
            for ref in pending
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                record = future.result()
                validate_record(record, references[name], settings, require_current=settings["rescore_baseline"] or settings["candidate"])
                save_json(record_root + f"/{name}.json", {
                    "schema_version": SCHEMA_VERSION, "fingerprint": fingerprint,
                    "record_sha256": json_hash(record), "record": record,
                    "input_uri": input_uri, "input_sha256": input_sha, "seeded": False,
                })
                completed[name] = record
                print(f"COMMITTED {name} {len(completed)}/35", flush=True)
            except Exception as error:
                failures.append({"dataset": name, "error": repr(error)})
                print(f"AUDIT_FAILED {name}: {error!r}", flush=True)
                traceback.print_exc()
    require(not failures, "Dataset audits failed; successful cohorts remain committed: " + json.dumps(failures))
    results = [completed[name] for name in names]
    accepted = bool(settings["candidate"] and job_info["status"] == "SUCCEEDED"
                    and all(row.get("candidate_status") == "AUDITED" for row in results))
    evidence = {
        "schema_version": SCHEMA_VERSION, "audit_status": "COMPLETE", "fingerprint": fingerprint,
        "collected_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **snapshot,
        "scorer_config_sha256": json_hash(config), "input_uri": input_uri, "input_sha256": input_sha,
        "source_optimizer_step": source_step, "evaluation_global_step": 0,
        "source_checkpoint_provenance": source_provenance,
        "baseline_identity": "original standard synchronous ReMax Base3a@868",
        "baseline_historical_ray_job_id": "raysubmit_wxL1nMu9RXPrAAWC",
        "all35_baseline_artifacts_freshly_verified": len(results) == 35,
        "baseline_total_source_segments": sum(row["baseline"]["segments"] for row in results),
        "baseline_total_groups": sum(row["baseline"]["groups"] for row in results),
        "all35_baseline_current_scorer_counts_match_original": all(
            row["baseline"].get("current_scorer", {}).get("matches_original_counts", False) for row in results
        ),
        "baseline_comparison_scoring": "same current scorer applied to original ReMax raw responses; comparison uses baseline.current_scorer; historical emitted counts remain baseline main fields",
        "baseline_current_scorer_audit": baseline_dependency, "candidate_job": job_info,
        "candidate_model_path": settings["model_path"] if settings["candidate"] else None,
        "full35_accepted": accepted, "results": results, "record_root": record_root,
        "metrics_artifact": settings.get("candidate_root", "").rstrip("/") + "/val_data_gen/0.metrics.json" if settings["candidate"] else None,
        "metrics_artifact_sha256": metrics_sha,
        "candidate_coverage_policy": coverage_policy,
        "parent_scoring_workers": parent_workers,
        "coverage_method": "exact core8 source/gts multiset; observed9 language identities retained; only configured English In to English paired prompt-sentence changes explicitly approved; repeated acoustic splits deduplicated by full observed9 identity",
    }
    save_json(output_uri, evidence)
    print(f"AUDIT_COMPLETE {output_uri} full35_accepted={accepted}", flush=True)
    if settings["candidate"]:
        require(accepted, "Full35 candidate acceptance failed; committed dataset records retained")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True, help="Blob URI containing audit settings/reference/suites data")
    parser.add_argument("--output-json", required=True, help="Blob URI for final evidence; dataset records use .datasets suffix")
    parser.add_argument("--workers", type=int, default=8, help="Independent spawn worker processes; reserve matching CPU/RAM")
    parser.add_argument("--parent-workers", type=int, default=1, help="Optional independent scoring processes per candidate OpenLong dataset (default: serial)")
    args = parser.parse_args()
    require(args.workers > 0, "workers must be positive")
    require(args.parent_workers > 0, "parent-workers must be positive")
    run(args.input_json, args.output_json, args.workers, args.parent_workers)


if __name__ == "__main__":
    main()
