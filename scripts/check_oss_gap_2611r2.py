"""CPU reference/scorer and optional native waveform checks for OSS gaps."""
import argparse
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path

import blobfile as bf
import yaml

from recipe.phimm.data.chunk import load_chunks, load_examples
from recipe.phimm.data.prompts import get_task_prompt
from recipe.phimm.reward.oss_gap_error_mining import compute_score
from scripts.prepare_oss_gap_2611r2 import CASE, MODEL, ROOT


def check_one(c):
    fields = c["post_process"]["add_field"]["fields"]
    language = fields["source_language"]
    chunks = load_chunks(c["specs"])
    chunk = chunks[0]
    e = load_examples(chunk, c["chunk_types"])
    ref_field = c["chunk_types"][1]
    assert len(e["audio_chunk"]) == len(e[ref_field]) == chunk["count"]
    # Prefer a nonempty example for the deletion check, preserve the original.
    i = next((i for i, t in enumerate(e[ref_field]) if t.strip()), 0)
    ref = e[ref_field][i]
    extra = {**fields, "language": chunk["language"]}
    good = compute_score(ref, ref, version=2609, extra_info=extra)
    deletion = compute_score("", ref, version=2609, extra_info=extra)
    insertion = compute_score(ref + " xyz", ref, version=2609, extra_info=extra)
    assert good["n_err"] == 0
    assert deletion["n_err"] == deletion["n_ref"]
    assert insertion["n_err"] > 0
    for r in [good, deletion, insertion]:
        assert r["n_err"] == r["n_ins"] + r["n_del"] + r["n_sub"]
        assert all(math.isfinite(r[k]) for k in ["score", "n_ref", "n_err"])
    prompt = get_task_prompt("lang_asr_verb", version=2609, lang=chunk["language"])
    assert "The language is " in prompt and "The languages are " not in prompt
    manifest = c["specs"][0]["manifest_file"]
    with bf.BlobFile(manifest, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return {"family": fields["source_family"], "source_language": language, "display_language": chunk["language"],
            "chunks": len(chunks), "samples": sum(x["count"] for x in chunks),
            "manifest": manifest, "manifest_sha256": digest, "audio_chunk": e["audio_chunk"][i],
            "reference": ref, "reference_field": ref_field, "good": good, "deletion": deletion,
            "insertion": insertion, "prompt": prompt, "passed": True}


def main(out, audio=False):
    conf = yaml.safe_load((ROOT / f"recipe/phimm/config/data/train_data/oss/{CASE}_raw.yaml").read_text())
    expected = json.loads((out / "source_summary.json").read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        checks = list(pool.map(check_one, conf))
    assert sum(c["samples"] for c in checks) == 36993182
    assert sum(c["chunks"] for c in checks) == 20538
    assert len(checks) == 93
    assert {c["manifest"]: c["manifest_sha256"] for c in checks} == expected["manifest_sha256"]
    waves = []
    if audio:
        from recipe.phimm.data.dataset import create_audio_dataset
        from recipe.phimm.data.chunk import load_chunk_sample
        seen = set()
        for c in conf:
            family = c["post_process"]["add_field"]["fields"]["source_family"]
            if family in seen:
                continue
            seen.add(family)
            ds = create_audio_dataset(**dict(c, max_chunks=1, num_proc=1, version=2609))
            e = ds[0]
            wave = load_chunk_sample(e["audio_chunk"])
            wave = wave if isinstance(wave, list) else [wave]
            seconds = sum(len(w[0]) / w[1] for w in wave)
            assert seconds > 0 and math.isfinite(seconds)
            waves.append({"family": family, "seconds": seconds, "audio_chunk": e["audio_chunk"], "prompt": e["prompt"]})
    result = {"status": "PASSED", "scope": "All93 source reference chunks and scorer cases; waveform/runtime checks separate",
              "model": MODEL, "unique_languages": len({c["source_language"] for c in checks}),
              "expected_samples": 36993182, "checks": checks, "waveforms": waves,
              "waveform_gate_passed": len(waves) == 5}
    (out / ("native_audio_checks.json" if audio else "reference_scorer_checks.json")).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "tmp/run_openasr_exp" / CASE)
    p.add_argument("--audio", action="store_true", help="Run on the audited dedicated GPU pool before generation")
    args = p.parse_args()
    main(args.out, args.audio)
