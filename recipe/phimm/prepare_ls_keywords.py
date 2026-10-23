"""Build reproducible, speaker-disjoint LibriSpeech joint-keyword manifests.

Existing source files are read-only. All labels use the frozen common5k policy;
legacy test ground_truth lists are checked, then mapped to keywords explicitly.
"""

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from recipe.phimm.data.keyword_output import ANNOTATION_POLICY, target_keywords, words

COMMON_SHA = "30eadb83c338df9376e4fabd31bd1349becc06b896b81dbc5fb7dac6fca72a8e"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def split_for(speaker):
    bucket = int(hashlib.sha256(f"ls-keywords-v1:{speaker}".encode()).hexdigest(), 16) % 100
    return "dev" if bucket < 5 else "internal_test" if bucket < 10 else "train"


def prepare(data_root, output_dir, audio_prefix=None, require_local_audio=False, pilot_size=10000, seed=42):
    data_root, output_dir = Path(data_root), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite manifest directory: {output_dir}")
    common_path = data_root / "librispeech_biasing/words/common_words_5k.txt"
    policy_sha = digest(common_path)
    if policy_sha != COMMON_SHA:
        raise ValueError("common5k SHA does not match frozen ls-common5k-v1 policy")
    common = {line.split()[0].lower() for line in common_path.read_text().splitlines() if line.strip()}
    sets = defaultdict(list)
    source_hashes, stats = {}, Counter()
    test_texts, test_ids, test_speakers = set(), set(), set()
    ids_seen = set()

    def record(uid, text, local, split, source):
        sid, chapter, _ = uid.split("-")
        local = Path(local)
        if uid in ids_seen:
            raise ValueError(f"duplicate source id: {uid}")
        ids_seen.add(uid)
        keys = target_keywords(text, common)
        # Prepared paths can target a verified remote data root without changing labels.
        audio = (audio_prefix.rstrip("/") + "/" + str(local.relative_to(data_root))) if audio_prefix else str(local)
        return {
            "id": uid,
            "text": text,
            "keywords": keys,
            "audio_path": audio,
            "split": split,
            "speaker_id": sid,
            "chapter_id": chapter,
            "keyword_annotation_valid": True,
            "annotation_policy": ANNOTATION_POLICY,
            "annotation_policy_sha": policy_sha,
            "source_manifest_sha": source,
        }

    for name in ("test-clean", "test-other"):
        path = data_root / f"librispeech_biasing/ref/{name}.biasing_100.jsonl"
        source_hashes[str(path)] = sha = digest(path)
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            uid, text = row["id"], row["text"]
            existing = row.get("ground_truth")
            if not isinstance(existing, list) or len(existing) != len(set(existing)):
                raise ValueError(f"{uid}: missing/invalid original keyword list")
            if set(target_keywords(text, common)) != set(existing):
                raise ValueError(f"{uid}: original test labels disagree with common5k")
            sid, chapter, _ = uid.split("-")
            local = data_root / f"LibriSpeech/{name}/{sid}/{chapter}/{uid}.flac"
            if not local.is_file():
                raise FileNotFoundError(local)
            test_ids.add(uid)
            test_speakers.add(sid)
            test_texts.add(" ".join(words(text)))
            sets[name.replace("-", "_")].append(record(uid, text, local, name, sha))

    train_ids_seen = set()
    train_path = data_root / "LibriSpeech/asr_train_transcribe.tsv"
    source_hashes[str(train_path)] = sha = digest(train_path)
    for line in train_path.open():
        if not line.strip():
            continue
        name, audio_list, conversations = line.rstrip("\n").split("\t", 2)
        uid = Path(name).stem
        if uid in train_ids_seen:
            raise ValueError(f"duplicate training id: {uid}")
        train_ids_seen.add(uid)
        sid, _, _ = uid.split("-")
        stats["source_train_rows"] += 1
        if uid in test_ids or sid in test_speakers:
            raise ValueError(f"training/test id or speaker overlap: {uid}")
        chats = ast.literal_eval(conversations)
        refs = [m["content"] for conv in chats for m in conv["messages"] if m["role"] == "assistant"]
        paths = ast.literal_eval(audio_list)
        if len(refs) != 1 or len(paths) != 1:
            raise ValueError(f"{uid}: expected one reference and one audio")
        text = refs[0]
        if " ".join(words(text)) in test_texts:
            stats["removed_official_duplicate_text"] += 1
            continue
        local = data_root / paths[0][len("/root/data/") :] if paths[0].startswith("/root/data/") else Path(paths[0])
        stats["missing_local_audio"] += not local.is_file()
        if require_local_audio and not local.is_file():
            continue
        sets[split_for(sid)].append(record(uid, text, local, split_for(sid), sha))

    # Remove repeated text across held-out folds as well, without moving speakers.
    internal_texts = {" ".join(words(x["text"])) for x in sets["internal_test"]}
    dev_before = len(sets["dev"])
    sets["dev"] = [x for x in sets["dev"] if " ".join(words(x["text"])) not in internal_texts]
    stats["removed_dev_internal_duplicate_text"] = dev_before - len(sets["dev"])
    heldout_texts = internal_texts | {" ".join(words(x["text"])) for x in sets["dev"]}
    train_before = len(sets["train"])
    sets["train"] = [x for x in sets["train"] if " ".join(words(x["text"])) not in heldout_texts]
    stats["removed_train_heldout_duplicate_text"] = train_before - len(sets["train"])
    for name in sets:
        sets[name].sort(key=lambda x: x["id"])

    rng = random.Random(seed)
    empty = [r for r in sets["train"] if not r["keywords"]]
    nonempty = [r for r in sets["train"] if r["keywords"]]
    size = min(pilot_size, len(sets["train"]))
    if not size:
        raise ValueError("no usable training data")
    ne = min(len(empty), round(size * len(empty) / len(sets["train"])))
    pilot = rng.sample(empty, ne) + rng.sample(nonempty, size - ne)
    rng.shuffle(pilot)
    sets["train_pilot10k_seed42"] = pilot
    sets["train_probe"] = pilot[:256]
    # Smoke-dev is train-disjoint and only for fast preflight, never official tuning.
    sets["dev_smoke512"] = random.Random(seed).sample(sets["dev"], min(512, len(sets["dev"])))
    output_dir.mkdir(parents=True)
    metadata = {
        "annotation_policy": ANNOTATION_POLICY,
        "annotation_policy_sha": policy_sha,
        "source_hashes": source_hashes,
        "stats": dict(stats),
        "seed": seed,
        "require_local_audio": require_local_audio,
        "audio_prefix": audio_prefix,
        "split_rule": "int(SHA256(ls-keywords-v1:<speaker>).hexdigest(),16)%100: dev<5, internal<10, train otherwise",
        "manifests": {},
    }
    for name, rows in sets.items():
        path = output_dir / f"{name}.jsonl"
        with path.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        metadata["manifests"][name] = {
            "rows": len(rows),
            "nonempty": sum(bool(x["keywords"]) for x in rows),
            "empty": sum(not x["keywords"] for x in rows),
            "sha256": digest(path),
        }
    (output_dir / "manifest_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/home/boren/data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audio-prefix")
    parser.add_argument("--require-local-audio", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_root, args.output_dir, args.audio_prefix, args.require_local_audio), indent=2))


if __name__ == "__main__":
    main()
