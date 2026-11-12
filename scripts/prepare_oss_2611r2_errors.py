"""Freeze raw OSS chunk identities for 2611 R2 generation (no row filters).

Run with the repo's blobfile environment. Existing inventory metadata supplies
main OSS sources; supplementary manifests are read directly. Audio paths are
deduplicated in source order; differing transcript aliases remain in the audit.
Training eligibility is deliberately separate from generation membership.
"""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

import blobfile as bf
import yaml

ROOT = Path(__file__).resolve().parents[1]
CASE = "oss_2611r2_errors_20261112"
BLOB = "az://orngwus2cresco/data/boren/data/oss_refine_2611r2_20261112"
MODEL = "az://orngwus2cresco/data/speech/projects/phi-fastllm-2611/amlt-results/fast-llm-2611-qwen9b-flashenc-mtp-s2-data-v3-8-2-1-shadow/38000/qwen_hf/"


def canonical(path):
    return re.sub(r"^/datablob1?/", "az://orngwus2cresco/data/speech/", path).rstrip("/")


def family(path):
    for match, name in [("open_mls_", "MLS"), ("open_zh_", "WenetSpeech"),
                        ("gigaspeech", "GigaSpeech"), ("cv15", "CommonVoice15"),
                        ("voxpopuli", "VoxPopuli"), ("fleurs", "FLEURS"),
                        ("openasr_ami", "AMI"), ("openasr_earnings", "Earnings22"),
                        ("wavllm", "WavLLM"), ("VOiCES", "VOiCES"),
                        ("VCTK", "VCTK"), ("LibriTTS", "LibriTTS")]:
        if match in path:
            return name
    raise ValueError(path)


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def prepare(out, upload=False):
    out.mkdir(parents=True, exist_ok=True)
    manifests = out / "manifests"
    manifests.mkdir(exist_ok=True)
    inventory = json.loads((ROOT / "outputs/oss_training_inventory_20261112/inventory.json").read_text())
    by_spec = {s["spec"]: s for s in inventory}
    index = ROOT / "recipe/phimm/config/data/train_data/oss/all.yaml"
    sources = []
    for spec in yaml.safe_load(index.read_text())["specs"]:
        for source in by_spec[spec]["sources"]:
            sources.append((spec, source))
    supplement = ROOT / "recipe/phimm/config/data/train_data/opensource_voices_vctk_libritts.yaml"
    for source in yaml.safe_load(supplement.read_text())["specs"]:
        with bf.BlobFile(source["manifest_file"], "r") as f:
            manifest = json.load(f)
        source = dict(source, chunks=manifest["fileInfo"], manifest_meta={k: v for k, v in manifest.items() if k != "fileInfo"})
        sources.append((str(supplement.relative_to(ROOT)), source))

    seen = {}
    chunks = []
    aliases = []
    groups = []
    uploads = []
    for spec, source in sources:
        audio_type = "audio" if "audio" in source["manifest_meta"]["fileType"] else "audios"
        types = ["audio", "transcription"] if audio_type == "audio" else ["audios", "sft.0.messages.1.content"]
        fam = family(spec + source["chunk_path"])
        manifest_chunks = []
        for chunk in source["chunks"]:
            key = canonical(source["chunk_path"]) + "/" + chunk["name"] + "." + audio_type
            text_type = "transcription" if audio_type == "audio" else "sft"
            text_path = canonical(source.get("trans_path", source["chunk_path"]) if audio_type == "audio" else source["chunk_path"])
            text_path += "/" + chunk["name"] + "." + text_type
            if key in seen:
                if seen[key]["count"] != chunk["count"]:
                    raise ValueError(f"Count disagreement for repeated audio: {key}")
                aliases.append({"audio_key": key, "kept_text_path": seen[key]["text_path"], "alias_text_path": text_path, "source_spec": spec})
                continue
            # Explicit train directories establish upstream declaration only;
            # record/parent heldout overlap checks are still required.
            declared_train = bool(re.search(r"/(train(?:/|_)|train-other-|asr_train|libriheavy_train)", key))
            row = {"audio_key": key, "count": int(chunk["count"]), "duration": chunk.get("duration"),
                   "text_path": text_path, "family": fam, "language": source["language"],
                   "source_spec": spec, "source_manifest": canonical(source["manifest_file"]),
                   "split_status": "upstream_train_declared" if declared_train else "unresolved_generation_only",
                   "training_eligible": False, "eligibility_reason": "Heldout ID/audio/parent audit pending"}
            seen[key] = row
            chunks.append(row)
            manifest_chunks.append(chunk)
        if not manifest_chunks:
            continue
        data = json.dumps({**source["manifest_meta"], "fileInfo": manifest_chunks}, ensure_ascii=False, sort_keys=True).encode()
        sha = digest_bytes(data)
        name = f"{len(groups):03d}_{fam}_{source['language']}_{sha[:12]}.json"
        local = manifests / name
        local.write_bytes(data)
        remote = f"{BLOB}/source_manifests/{name}"
        uploads.append((local, remote, sha))
        conf = {"dataset_name": "chunk", "specs": [{"manifest_file": remote,
                "chunk_path": canonical(source["chunk_path"]),
                "trans_path": canonical(source.get("trans_path", source["chunk_path"])),
                "language": source["language"]}], "chunk_types": types,
                "chunk_shuffle": False, "num_proc": 8,
                "add_task_info": {"task": "lang_asr_verb", "prefix_prob": 0.0, "lang_hint": True},
                "post_process": {"add_field": {"fields": {"data_source": "openml", "source_family": fam}},
                                 "verl_format": {"prompt_key": "prompt", "extra_keys": ["source_family"]}}}
        if audio_type == "audios":
            conf["pre_process"] = {"rename_fields": {"mappings": {"text": "sft.0.messages.1.content"}}}
        groups.append(conf)

    count = sum(x["count"] for x in chunks)
    if count != 15058014:
        raise ValueError(f"Universe disagrees with frozen main+supplement inventory: {count}")
    (out / "source_chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2))
    (out / "source_aliases.json").write_text(json.dumps(aliases, ensure_ascii=False, indent=2))
    data_config = ROOT / f"recipe/phimm/config/data/train_data/{CASE}_raw.yaml"
    data_config.write_text(yaml.safe_dump(groups, allow_unicode=True, sort_keys=False))
    gen_config = ROOT / f"recipe/phimm/config/v2611_dev/gen/gen_{CASE}.yaml"
    config = {"hydra": {"searchpath": ["file://recipe/phimm/config", "file://verl/trainer/config"]},
              "defaults": ["gen_base", {f"/data/train_data/{CASE}_raw@data.gen_data": None}, "_self_"]}
    # Hydra defaults for loading a named file uses string notation.
    config["defaults"][1] = f"/data/train_data/{CASE}_raw@data.gen_data"
    config.update(custom_reward_function={"path": "recipe/phimm/reward/oss_error_mining.py", "name": "compute_score", "reward_kwargs": {"version": 2609}},
                  actor_rollout_ref={"model": {"path": MODEL}}, model={"lora_rank": 0},
                  data={"output_path": f"{BLOB}/generation", "chunk_load_mode": "indexed", "max_audio_dur": None,
                        "num_workers": 16, "prefetch_factor": 2,
                        "validation_shuffle": False, "output_split_size": 20000},
                  rollout={"dtype": "bfloat16", "engine_kwargs": {"vllm": {"quantization": None}}},
                  trainer={"experiment_name": f"gen_{CASE}", "nnodes": 4, "n_gpus_per_node": 8})
    gen_config.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    summary = {"unique_chunks": len(chunks), "expected_samples": count, "alias_chunks_removed": len(aliases),
               "families": dict(collections.Counter({f: sum(c["count"] for c in chunks if c["family"] == f) for f in sorted({c['family'] for c in chunks})})),
               "language_codes": sorted({c["language"] for c in chunks}),
               "data_config": str(data_config.relative_to(ROOT)), "generation_config": str(gen_config.relative_to(ROOT)),
               "source_chunks_sha256": digest_bytes((out / 'source_chunks.json').read_bytes()),
               "selection_model": MODEL, "training_eligibility": "pending per-record heldout and split audit",
               "scoring": "zh: canonical expansion CER; other21: canonical openasr_eval WER; rank absolute counts, mixed units disclosed",
               "uploaded": False}
    if upload:
        for local, remote, sha in uploads:
            if bf.exists(remote):
                with bf.BlobFile(remote, "rb") as f:
                    if digest_bytes(f.read()) != sha:
                        raise ValueError(f"Frozen blob identity conflict: {remote}")
            else:
                bf.copy(str(local), remote)
                with bf.BlobFile(remote, "rb") as f:
                    if digest_bytes(f.read()) != sha:
                        raise ValueError(f"Uploaded blob hash mismatch: {remote}")
        for name in ["source_chunks.json", "source_aliases.json"]:
            bf.copy(str(out / name), f"{BLOB}/{name}", overwrite=True)
        summary["uploaded"] = True
    (out / "source_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / f"tmp/run_openasr_exp/{CASE}")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    prepare(args.out, args.upload)
