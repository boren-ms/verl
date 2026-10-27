import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pytest
import soundfile as sf
import yaml

from scripts.build_ls_rir_dataset import original_draws, run


@pytest.mark.parametrize("use_replacements", [False, True])
def test_regenerates_dry_sources_and_publishes_only_complete_manifest(tmp_path, use_replacements):
    sr = 16000
    speech = tmp_path / "dry.wav"
    noise = tmp_path / "noise.wav"
    rir = tmp_path / "rir.wav"
    sf.write(speech, np.sin(np.arange(sr) * 0.1) * 0.15, sr)
    sf.write(noise, np.random.default_rng(42).normal(0, 0.1, sr * 2), sr)
    sf.write(rir, np.array([0, 1, 0.7, 0.4, 0.2]), sr, subtype="FLOAT")
    noise_manifest = tmp_path / "noise.jsonl"
    noise_manifest.write_text(json.dumps({"audio_path": str(noise)}) + "\n")
    rir_manifest = tmp_path / "rir.jsonl"
    rir_manifest.write_text(
        json.dumps(
            {"audio_path": str(rir), "sha256": hashlib.sha256(rir.read_bytes()).hexdigest(), "license": "CC-BY-4.0"}
        )
        + "\n"
    )
    cfg = {
        "original_seed": 0,
        "rir_seed": 100,
        "rir_prob": 1,
        "speed_range": [1, 2],
        "snr_range": [10, 12],
        "workers": 2,
        "noise_manifest": str(noise_manifest),
        "rir_manifest": str(rir_manifest),
        "source_manifest": str(tmp_path / "source.jsonl"),
        "output_dir": str(tmp_path / "output"),
    }
    replacement = tmp_path / "replacement.wav"
    if use_replacements:
        sf.write(replacement, np.random.default_rng(99).normal(0, 0.1, sr * 2), sr)
        replacement_manifest = tmp_path / "replacement.jsonl"
        replacement_manifest.write_text(
            json.dumps(
                {
                    "audio_path": str(replacement),
                    "license": "CC-BY-4.0",
                    "sha256": hashlib.sha256(replacement.read_bytes()).hexdigest(),
                }
            )
            + "\n"
        )
        cfg.update(replacement_noise_manifest=str(replacement_manifest), resource_license="CC-BY-4.0")
    rows = []
    for index in range(2):
        rng = random.Random(index)
        rng.random()
        speed = rng.uniform(1, 2)
        rng.random()
        rng.choice([str(noise)])
        snr = rng.uniform(10, 12)
        rows.append(
            {
                "id": str(index),
                "source_audio_path": str(speech),
                "audio_path": f"/never_read_old_mixture/{index:08d}_old.wav",
                "noise_path": str(noise),
                "speed_factor": speed,
                "snr_db": snr,
                "text": "hello world",
                "wer": 0.2,
                "response": "old prediction",
            }
        )
    # Distinct path spellings to represent distinct original sources.
    second_speech = tmp_path / "second.wav"
    second_speech.write_bytes(speech.read_bytes())
    rows[1]["source_audio_path"] = str(second_speech)
    Path(cfg["source_manifest"]).write_text("".join(json.dumps(r) + "\n" for r in rows))
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    args = argparse.Namespace(
        config=str(config),
        work_dir=str(tmp_path / "work"),
        source_local=None,
        noise_local=None,
        rir_manifest_local=None,
        rir_local_dir=None,
        workers=2,
        max_examples=1,
    )
    run(args)
    final = Path(cfg["output_dir"]) / "data.jsonl"
    assert not final.exists()
    args.max_examples = None
    run(args)  # Resume the first row and generate the second.
    output = [json.loads(line) for line in final.read_text().splitlines()]
    assert [r["id"] for r in output] == ["0", "1"]
    for before, after in zip(rows, output, strict=True):
        data, sample_rate = sf.read(after["audio_path"])
        assert sample_rate == sr
        assert len(data) == round(sr / before["speed_factor"])
        assert after["text"] == before["text"]
        assert after["augmentation_order"] == ["speed", "rir", "noise"]
        assert abs(after["measured_snr_db_before_pcm16"] - before["snr_db"]) < 0.001
        assert "response" not in after and "wer" not in after
        assert after["source_generation"]["response"] == "old prediction"
        assert after["audio_sha256"] == hashlib.sha256(Path(after["audio_path"]).read_bytes()).hexdigest()
        if use_replacements:
            assert after["noise_path"] == str(replacement)
            assert after["parent_noise_path"] == str(noise)
            assert after["rir_license"] == after["noise_license"] == "CC-BY-4.0"
    with pytest.raises(ValueError, match="speed mismatch"):
        original_draws({**rows[0], "speed_factor": 1}, [str(noise)], cfg)
