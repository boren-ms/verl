"""Freeze only the five transferred OSS increments for 4x8 R2 mining."""
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path

import blobfile as bf
import yaml

ROOT = Path(__file__).resolve().parents[1]
CASE = "oss_gap_2611r2_errors_20261112"
BLOB = "az://orngwus2cresco/data/boren/data/" + CASE
MODEL = "az://orngwus2cresco/data/speech/projects/phi-fastllm-2611/amlt-results/fast-llm-2611-qwen9b-flashenc-mtp-s2-data-v3-8-2-1-shadow/38000/qwen_hf/"
# Single-language display names: the existing prompt formatter treats spaces
# as code-switch separators. ISO identity is retained independently.
LANGUAGE_NAMES = {"bn": "Bengali", "ckb": "Sorani", "luo": "Luo", "nso": "Sepedi", "nb": "Norwegian"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(out, upload=False):
    out.mkdir(parents=True, exist_ok=True)
    transfer = ROOT / "outputs/oss_missing_transfer_20261112"
    complete = json.loads((transfer / "completion.json").read_text())
    assert complete["complete"]
    sources = json.loads((transfer / "selected_sources.json").read_text())
    by_family_lang = {(s["family"], s["language"]): s for s in sources}
    raw = copy.deepcopy(yaml.safe_load((ROOT / "recipe/phimm/config/data/train_data/oss/missing_from_moe_20261112.yaml").read_text()))
    chunks, manifests = [], {}
    for c in raw:
        fields = c["post_process"]["add_field"]["fields"]
        family, lang = fields["source_family"], fields["source_language"]
        upstream = by_family_lang[family, lang]
        spec = c["specs"][0]
        manifest_path = transfer / "manifests" / Path(spec["manifest_file"]).name
        manifest_digest = sha(manifest_path)
        assert manifest_digest == complete["manifest_sha256"][manifest_path.name]
        manifest = json.loads(manifest_path.read_text())
        declarations = [str(upstream["source"]["manifest_file"]), str(upstream["source"]["chunk_path"])]
        assert any("train" in s.lower() for s in declarations), declarations
        spec["language"] = LANGUAGE_NAMES.get(lang, spec["language"])
        assert " " not in spec["language"], spec["language"]
        c["add_task_info"] = {"task": "lang_asr_verb", "prefix_prob": 0.0, "lang_hint": True}
        c["chunk_shuffle"], c["num_proc"] = False, 8
        fields["data_source"] = "openml"
        fields["selection_model"] = MODEL
        c["post_process"]["verl_format"]["extra_keys"].append("selection_model")
        manifests[spec["manifest_file"]] = manifest_digest
        for ch in manifest["fileInfo"]:
            audio = spec["chunk_path"].rstrip("/") + "/" + ch["name"] + "." + c["chunk_types"][0]
            chunks.append({"audio_key": audio, "count": ch["count"], "family": family, "language": lang,
                           "display_language": spec["language"], "split_status": "upstream_declared_train",
                           "source_split": "train", "split_evidence": declarations,
                           "manifest": spec["manifest_file"], "manifest_sha256": manifest_digest,
                           "upstream_manifest": upstream["source"]["manifest_file"],
                           "upstream_manifest_sha256": upstream["metadata"]["sha256"],
                           "selection_metric": "cer" if lang in {"zh", "yue"} else "wer"})
    assert len(chunks) == len({x["audio_key"] for x in chunks}) == 20538
    assert sum(x["count"] for x in chunks) == 36993182
    old_path = ROOT / "tmp/run_openasr_exp/oss_2611r2_errors_20261112/source_chunks.json"
    old = {x["audio_key"] for x in json.loads(old_path.read_text())}
    assert not old.intersection(x["audio_key"] for x in chunks), "Overlap with running older OSS case"
    frozen = out / "source_chunks.json"
    if frozen.exists():
        assert json.loads(frozen.read_text()) == chunks, "Source freeze changed"
    else:
        frozen.write_text(json.dumps(chunks, ensure_ascii=False, indent=2))
    data_name = "oss/" + CASE + "_raw"
    data_path = ROOT / f"recipe/phimm/config/data/train_data/{data_name}.yaml"
    data_path.write_text("# @package data.gen_data\n" + yaml.safe_dump(raw, allow_unicode=True, sort_keys=False))
    gen = {"hydra": {"searchpath": ["file://recipe/phimm/config", "file://verl/trainer/config"]},
           "defaults": ["gen_base", f"/data/train_data/{data_name}@data.gen_data", "_self_"],
           "custom_reward_function": {"path": "recipe/phimm/reward/oss_gap_error_mining.py", "name": "compute_score", "reward_kwargs": {"version": 2609}},
           "actor_rollout_ref": {"model": {"path": MODEL}}, "model": {"lora_rank": 0},
           "data": {"output_path": BLOB + "/generation", "chunk_load_mode": "indexed", "max_audio_dur": None,
                    "validation_shuffle": False, "output_split_size": 20000, "num_workers": 16, "prefetch_factor": 2,
                    "distributed_loading": {"enabled": True, "workers_per_node": 8, "prefetch_batches": 2, "timeout_s": 300}},
           "rollout": {"dtype": "bfloat16", "engine_kwargs": {"vllm": {"quantization": None}}},
           "ray_kwargs": {"ray_init": {"runtime_env": {"env_vars": {"VERL_AUDIO_CACHE_ROOT": "/root/data/" + CASE + "_cache"}}}},
           "trainer": {"experiment_name": "gen_" + CASE, "nnodes": 4, "n_gpus_per_node": 8}}
    gen_path = ROOT / f"recipe/phimm/config/v2611_dev/gen/gen_{CASE}.yaml"
    gen_path.write_text(yaml.safe_dump(gen, allow_unicode=True, sort_keys=False))
    summary = {"case_id": CASE, "expected_samples": 36993182, "unique_chunks": len(chunks),
               "families": dict(collections.Counter({f: sum(c["count"] for c in chunks if c["family"] == f) for f in sorted({c['family'] for c in chunks})})),
               "source_chunks_sha256": sha(frozen), "manifest_sha256": manifests,
               "source_languages": sorted({c["language"] for c in chunks}), "source_specs": len(raw),
               "raw_yaml": str(data_path.relative_to(ROOT)), "generation_config": str(gen_path.relative_to(ROOT)),
               "model": MODEL, "generation_topology": {"nnodes": 4, "n_gpus_per_node": 8},
               "older_generation_chunk_overlap": 0, "uploaded": False,
               "train_eligibility": "Train declarations frozen; final heldout/reference checks required before publishing filtered data"}
    if upload:
        for local in [frozen]:
            remote = BLOB + "/" + local.name
            if bf.exists(remote):
                with bf.BlobFile(remote, "rb") as f:
                    assert hashlib.sha256(f.read()).hexdigest() == sha(local)
            else:
                bf.copy(str(local), remote)
                with bf.BlobFile(remote, "rb") as f:
                    assert hashlib.sha256(f.read()).hexdigest() == sha(local)
        summary["uploaded"] = True
    (out / "source_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "manifest_sha256"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "tmp/run_openasr_exp" / CASE)
    p.add_argument("--upload", action="store_true")
    a = p.parse_args()
    prepare(a.out, a.upload)
