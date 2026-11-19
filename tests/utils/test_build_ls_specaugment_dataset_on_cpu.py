"""Verify row lineage, publication and randomized per-audio generation."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.build_ls_specaugment_dataset import sample_seed


def test_sample_seed_is_stable_and_distinct():
    assert sample_seed(42, "a", "/source.wav") == sample_seed(42, "a", "/source.wav")
    assert len({sample_seed(42, str(i), "/source.wav") for i in range(1000)}) == 1000


def test_batch_preserves_cohort_uses_dry_sources_and_publishes_last(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "source.wav"
    audio = 0.3 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)
    sf.write(source, audio, 16000)
    other_source = tmp_path / "other.wav"
    sf.write(other_source, audio, 16000)
    rows = [
        dict(
            id=f"sample-{i}",
            text=f"speech {i}",
            source_audio_path=str(path),
            audio_path="/nonexistent-old-noisy-audio.wav",
            speed_factor=1.5,
            response="stale",
        )
        for i, path in enumerate([source, other_source])
    ]
    manifest = tmp_path / "input.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    output = tmp_path / "output"
    command = [
        sys.executable,
        str(root / "scripts/build_ls_specaugment_dataset.py"),
        "--source-manifest",
        str(manifest),
        "--source-sha256",
        digest,
        "--config",
        str(root / "experiments/urgent_specaugment/config/augment_mild.yaml"),
        "--expected-rows",
        "2",
        "--output-dir",
        str(output),
        "--work-dir",
        str(tmp_path / "work"),
        "--workers",
        "2",
    ]
    subprocess.run([*command, "--max-examples", "1"], check=True, capture_output=True)
    assert not (output / "data.jsonl").exists()
    subprocess.run(command, check=True, capture_output=True)
    generated = [json.loads(line) for line in (output / "data.jsonl").read_text().splitlines()]
    assert [(r["id"], r["text"]) for r in generated] == [(r["id"], r["text"]) for r in rows]
    assert generated[0]["specaugment_seed"] != generated[1]["specaugment_seed"]
    assert generated[0]["specaugment_sampled"] != generated[1]["specaugment_sampled"]
    assert all(r["speed_factor"] == 1 and "response" not in r for r in generated)
    for row in generated:
        wav, rate = sf.read(row["audio_path"])
        assert wav.shape == audio.shape and rate == 16000
        assert hashlib.sha256(Path(row["audio_path"]).read_bytes()).hexdigest() == row["audio_sha256"]
    assert json.loads((output / "summary.json").read_text())["uploaded_wav_sha256_verified"]
    assert subprocess.run(command, capture_output=True).returncode != 0
