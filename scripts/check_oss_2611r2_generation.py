"""Preflight the frozen OSS generation recipe on its dedicated Ray node."""
import argparse
import hashlib
import json
from pathlib import Path

import blobfile as bf
import yaml


def main(out):
    from recipe.phimm.data.dataset import create_audio_dataset
    from recipe.phimm.data.chunk import load_chunk_sample
    from recipe.phimm.reward.oss_error_mining import compute_score
    from recipe.phimm.utils.languages import get_language_code
    from scripts.prepare_oss_2611r2_errors import MODEL

    root = Path(__file__).resolve().parents[1]
    conf = yaml.safe_load((root / "recipe/phimm/config/data/train_data/oss_2611r2_errors_20261112_raw.yaml").read_text())
    expected = 0
    selected = {}
    manifest_hashes = {}
    for c in conf:
        for s in c["specs"]:
            with bf.BlobFile(s["manifest_file"], "rb") as f:
                raw = f.read()
            manifest_hashes[s["manifest_file"]] = hashlib.sha256(raw).hexdigest()
            manifest = json.loads(raw)
            expected += sum(x["count"] for x in manifest["fileInfo"])
            for chunk in [manifest["fileInfo"][0], manifest["fileInfo"][-1]]:
                audio = s["chunk_path"].rstrip("/") + "/" + chunk["name"] + "." + c["chunk_types"][0]
                if not bf.exists(audio):
                    raise FileNotFoundError(audio)
            fam = c["post_process"]["add_field"]["fields"]["source_family"]
            if fam in {"MLS", "WenetSpeech", "CommonVoice15", "VOiCES", "AMI"}:
                selected.setdefault(fam, c)
    if expected != 15058014:
        raise ValueError(f"Manifest universe mismatch: {expected}")
    files = sorted(bf.listdir(MODEL))
    index = MODEL.rstrip("/") + "/model.safetensors.index.json"
    if bf.exists(index):
        with bf.BlobFile(index) as f:
            weights = json.load(f)
        for name in set(weights["weight_map"].values()):
            if not bf.exists(MODEL.rstrip("/") + "/" + name):
                raise FileNotFoundError(name)
    elif not bf.exists(MODEL.rstrip("/") + "/model.safetensors"):
        raise FileNotFoundError("No HF weight index or model.safetensors")
    checks = {}
    for fam, c in selected.items():
        c = dict(c, max_chunks=1, num_proc=1, version=2609)
        ds = create_audio_dataset(**c)
        with bf.BlobFile(c["specs"][0]["manifest_file"]) as f:
            first = json.load(f)["fileInfo"][0]
        if len(ds) != first["count"]:
            raise ValueError(f"{fam}: loader dropped rows: {len(ds)} != {first['count']}")
        e = ds[0]
        ref = e["reward_model"]["ground_truth"]
        extra = e["extra_info"]
        good = compute_score(ref, ref, version=2609, extra_info=extra)
        bad = compute_score("", ref, version=2609, extra_info=extra)
        if good["n_err"] != 0:
            raise ValueError(f"Identity scoring failed: {fam}")
        wave = load_chunk_sample(e["audio_chunk"])
        waves = wave if isinstance(wave, list) else [wave]
        seconds = sum(len(w[0]) / w[1] for w in waves)
        if seconds <= 0:
            raise ValueError(f"Empty audio: {fam}")
        checks[fam] = {"loader_rows": len(ds), "audio_chunk": e["audio_chunk"], "ref": ref,
                       "language": extra["language"], "audio_seconds": seconds,
                       "correct_score": good, "deletion_score": bad, "prompt": e["prompt"]}
    result = {"status": "PASSED", "expected_samples": expected,
              "model": MODEL, "model_files": files, "manifest_hashes": manifest_hashes,
              "native_loader_audio_scorer_checks": checks}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"status": "PASSED", "expected_samples": expected, "checked_families": list(checks)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/tmp/oss_generation_preflight.json"))
    main(parser.parse_args().out)
