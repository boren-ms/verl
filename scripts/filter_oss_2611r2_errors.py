"""Coverage-gated, disk-backed absolute-error selection of OSS generation.

No training files are published without an independently completed split and
heldout audit. The SQLite primary key and per-chunk index bounds prove unique
coverage, avoiding an in-memory sort of millions of generation records.
"""
import argparse
import collections
import copy
import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path

import blobfile as bf
import yaml


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def run(args):
    scorer_id = getattr(args, "scorer_id", "oss_error_mining_v1")
    per_language_top_k = getattr(args, "per_language_top_k", None)
    if per_language_top_k is not None and per_language_top_k <= 0:
        raise ValueError("per_language_top_k must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    chunks = json.loads(args.sources.read_text())
    by_audio = {r["audio_key"]: r for r in chunks}
    if len(chunks) != len(by_audio):
        raise ValueError("Source universe has duplicate audio chunks")
    expected = sum(c["count"] for c in chunks)
    db_path = args.out / "scores.sqlite"
    if db_path.exists():
        raise FileExistsError(f"Use a fresh output directory to avoid mixing selection runs: {db_path}")
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE scores (id TEXT PRIMARY KEY, audio TEXT, n_err INTEGER, eligible INTEGER DEFAULT 0, record TEXT) WITHOUT ROWID")
    parts = sorted(bf.glob(args.generation.rstrip("/") + "/part-*.jsonl"), key=lambda p: int(re.search(r"part-(\d+)\.jsonl$", p).group(1)))
    if not parts:
        raise ValueError("No generation parts")
    coverage = collections.Counter()
    part_hashes = {}
    total = retained = at_limit = 0
    for part_idx, path in enumerate(parts):
        if int(re.search(r"part-(\d+)\.jsonl$", path).group(1)) != part_idx:
            raise ValueError(f"Non-contiguous generation prefix: {path}")
        h = hashlib.sha256()
        with bf.BlobFile(path, "rb") as f:
            for line in f:
                h.update(line)
                r = json.loads(line)
                audio, count, index = r["audio_chunk"].rsplit(":", 2)
                count, index = int(count), int(index)
                source = by_audio.get(audio)
                if source is None or count != source["count"] or not 0 <= index < count:
                    raise ValueError(f"Unexpected source identity: {r['audio_chunk']}")
                if per_language_top_k and r.get("source_language") != source["language"]:
                    raise ValueError("Per-language Top-K requires frozen ISO source_language identity")
                n_err, n_ref = r["n_err"], r["n_ref"]
                for value in (n_err, n_ref):
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or int(value) != value:
                        raise ValueError(f"Invalid error count in {path}: {value!r}")
                if not isinstance(r["text"], str) or r.get("selection_scorer") != scorer_id:
                    raise ValueError("Missing trusted source reference or wrong scorer")
                if r.get("selection_metric") != source.get("selection_metric", "cer" if source["language"] == "zh" else "wer"):
                    raise ValueError("Metric/source language mismatch")
                expected_model = getattr(args, "expected_selection_model", None)
                if expected_model and r.get("selection_model") != expected_model:
                    raise ValueError("Generation selection model identity mismatch")
                if all(k in r for k in ("n_ins", "n_del", "n_sub")):
                    edits = [r[k] for k in ("n_ins", "n_del", "n_sub")]
                    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or int(v) != v for v in edits) or sum(edits) != n_err:
                        raise ValueError("Invalid edit counts or n_err != n_ins+n_del+n_sub")
                # This is the deterministic tie ID, independent of UUIDs or output order.
                identity = f"{audio}:{count}:{index}"
                r["canonical_id"] = identity
                r["source_family"] = source["family"]
                r["source_split_status"] = source["split_status"]
                db.execute("INSERT INTO scores(id,audio,n_err,record) VALUES (?,?,?,?)",
                           (identity, audio, int(n_err), json.dumps(r, ensure_ascii=False)))
                coverage[audio] += 1
                total += 1
                retained += n_err > 0
                at_limit += bool(r.get("response_at_limit"))
                if total % 20000 == 0:
                    db.commit()
        part_hashes[path] = h.hexdigest()
        db.commit()
        print(f"Scored coverage {total}/{expected}; errors {retained}", flush=True)
    mismatches = {a: {"expected": s["count"], "observed": coverage[a]} for a, s in by_audio.items() if coverage[a] != s["count"]}
    if total != expected or mismatches:
        raise ValueError(f"Incomplete coverage: {total}/{expected}; {len(mismatches)} mismatched chunks. No final subset emitted.")
    proof = {"status": "COMPLETE_GENERATION_COVERAGE", "expected": expected, "observed_unique": total,
             "n_err_positive_before_split_audit": retained, "response_at_limit": at_limit,
             "source_chunks_sha256": sha(args.sources), "generation_parts_sha256": part_hashes,
             "training_published": False}
    (args.out / "coverage.json").write_text(json.dumps(proof, indent=2))
    if not args.eligibility:
        print("Coverage verified. Training publication waits for --eligibility split/heldout audit.")
        return
    audit = json.loads(args.eligibility.read_text())
    if audit.get("status") != "PASSED" or audit.get("source_chunks_sha256") != sha(args.sources) or not audit.get("audit_evidence"):
        raise ValueError("Eligibility audit is not passed, lacks evidence, or refers to another source universe")
    eligible = set(audit["eligible_audio_keys"])
    excluded = set(audit["excluded_audio_keys"])
    if eligible & excluded or eligible | excluded != set(by_audio):
        raise ValueError("Split audit must explicitly partition every source chunk")
    db.execute("CREATE INDEX audio_lookup ON scores(audio)")
    # Per-record heldout exclusions are applied as well as coarse split exclusions.
    for audio in eligible:
        db.execute("UPDATE scores SET eligible=1 WHERE audio=?", (audio,))
    for identity in audit.get("excluded_record_ids", []):
        db.execute("UPDATE scores SET eligible=0 WHERE id=?", (identity,))
    db.commit()
    db.execute("CREATE INDEX rank_errors ON scores(eligible,n_err DESC,id ASC)")
    files = []
    kept = 0
    selected_by_language = collections.Counter()
    previous = None
    writer = None
    membership = hashlib.sha256()
    audit_writer = (args.out / "selection_audit.jsonl").open("w")
    try:
        order = "json_extract(record,'$.source_language') ASC,n_err DESC,id ASC" if per_language_top_k else "n_err DESC,id ASC"
        for identity, n_err, raw in db.execute("SELECT id,n_err,record FROM scores WHERE eligible=1 AND n_err>0 ORDER BY " + order):
            r = json.loads(raw)
            lang = r.get("source_language")
            if per_language_top_k and selected_by_language[lang] >= per_language_top_k:
                continue
            key = (lang, -n_err, identity) if per_language_top_k else (-n_err, identity)
            if previous is not None and key < previous:
                raise ValueError("Non-monotonic global selection order")
            previous = key
            if kept % args.shard_rows == 0:
                if writer:
                    writer.close()
                path = args.out / f"train-{len(files):08d}.jsonl"
                writer = path.open("w")
                files.append(path)
            kept += 1
            selected_by_language[lang] += 1
            train = {"id": identity, "audio_chunk": r["audio_chunk"], "text": r["text"],
                     "language": r["language"], "source_family": r["source_family"],
                     "source_split_status": r["source_split_status"], "selection_rank": kept,
                     "selection_n_err": n_err, "selection_n_ref": r["n_ref"],
                     "selection_metric": r["selection_metric"], "selection_model": args.model}
            if "source_language" in r:
                train["source_language"] = r["source_language"]
            if per_language_top_k:
                train["selection_language_rank"] = selected_by_language[lang]
            writer.write(json.dumps(train, ensure_ascii=False, allow_nan=False) + "\n")
            audit_writer.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
            membership.update((identity + "\n").encode())
    finally:
        if writer:
            writer.close()
        audit_writer.close()
    if kept < 128:
        raise ValueError(f"Only {kept} eligible errors, fewer than the inherited batch size")
    data_config = [{"dataset_name": "jsonl", "jsonl_paths": [args.publish_root.rstrip("/") + "/" + p.name for p in files],
                    "num_proc": 8, "add_task_info": {"task": "lang_asr_verb", "prefix_prob": 0.0, "lang_hint": True},
                    "post_process": {"add_field": {"fields": {"data_source": "openml"}},
                                     "verl_format": {"prompt_key": "prompt", "extra_keys": ["source_family", "source_split_status", "selection_rank", "selection_n_err", "selection_n_ref", "selection_metric", "selection_model"]}}}]
    (args.out / "training_data.yaml").write_text(yaml.safe_dump(data_config, allow_unicode=True, sort_keys=False))
    if any("source_language" in json.loads(raw) for (raw,) in db.execute("SELECT record FROM scores LIMIT 1")):
        data_config[0]["post_process"]["verl_format"]["extra_keys"].append("source_language")
        (args.out / "training_data.yaml").write_text(yaml.safe_dump(data_config, allow_unicode=True, sort_keys=False))
    if per_language_top_k:
        data_config[0]["post_process"]["verl_format"]["extra_keys"].append("selection_language_rank")
        (args.out / "training_data.yaml").write_text(yaml.safe_dump(data_config, allow_unicode=True, sort_keys=False))
    parent = getattr(args, "training_parent", "remax_main_base2c_2611r2")
    data_config_name = getattr(args, "data_config_name", "oss_2611r2_errors_20261112")
    experiment_name = getattr(args, "experiment_name", "remax_main_base2c_2611r2_oss_errors_20261112")
    recipe = {"hydra": {"searchpath": ["file://recipe/phimm/config", "file://verl/trainer/config"]},
              "defaults": [parent, f"/data/train_data/{data_config_name}@_oss_errors_train", "_self_"],
              "data": {"train_data": ["${_oss_errors_train}"], "mix": [{"shuffle": True, "datasource": [{"name": "openml", "sample": kept, "shuffle": True}]}]},
              "trainer": {"experiment_name": experiment_name}}
    template = getattr(args, "training_template", None)
    if template:
        # A primary config's hydra.searchpath cannot be inherited as a defaults
        # child. Materialize the exact parent mapping, preserving every field
        # except the new dataset/mix and unique experiment name.
        recipe = copy.deepcopy(yaml.safe_load(Path(template).read_text()))
        defaults = [d for d in recipe.get("defaults", []) if d != "_self_"]
        recipe["defaults"] = defaults + [f"/data/train_data/{data_config_name}@_oss_errors_train", "_self_"]
        recipe.setdefault("hydra", {"searchpath": ["file://recipe/phimm/config", "file://verl/trainer/config"]})
        recipe["data"]["train_data"] = ["${_oss_errors_train}"]
        recipe["data"]["mix"] = [{"shuffle": True, "datasource": [{"name": "openml", "sample": kept, "shuffle": True}]}]
        recipe.setdefault("trainer", {})["experiment_name"] = experiment_name
    (args.out / "training_recipe.yaml").write_text(yaml.safe_dump(recipe, allow_unicode=True, sort_keys=False))
    proof.update(selected_train_rows=kept, ordered_membership_sha256=membership.hexdigest(),
                 output_files_sha256={p.name: sha(p) for p in files}, eligibility_sha256=sha(args.eligibility),
                 training_published=False, next_gate="Native loader/audio/reward checks, effective data-only diff and T verification before publication/launch")
    if per_language_top_k:
        proof.update(per_language_top_k=per_language_top_k, selected_by_language=dict(selected_by_language),
                     ranking="ISO language ASC; absolute n_err DESC; canonical audio_chunk ASC; eligibility applied before quota")
    (args.out / "selection_prepared.json").write_text(json.dumps(proof, indent=2))
    print(f"Prepared {kept} sorted training rows. Publication and training checks are still required.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--generation", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--eligibility", type=Path)
    p.add_argument("--shard-rows", type=int, default=100000)
    p.add_argument("--publish-root", default="az://orngwus2cresco/data/boren/data/oss_refine_2611r2_20261112/filtered")
    p.add_argument("--model", default="2611r2_shadow38000_qwen_hf")
    p.add_argument("--scorer-id", default="oss_error_mining_v1")
    p.add_argument("--training-parent", default="remax_main_base2c_2611r2")
    p.add_argument("--data-config-name", default="oss_2611r2_errors_20261112")
    p.add_argument("--experiment-name", default="remax_main_base2c_2611r2_oss_errors_20261112")
    p.add_argument("--expected-selection-model")
    p.add_argument("--training-template", type=Path)
    p.add_argument("--per-language-top-k", type=int, help="Optional quota after split/heldout exclusions; requires ISO source_language")
    a = p.parse_args()
    if a.shard_rows <= 0:
        p.error("--shard-rows must be positive")
    run(a)
