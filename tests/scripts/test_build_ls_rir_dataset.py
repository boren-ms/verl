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
from scripts.validate_ls_rir_dataset import run as validate


@pytest.mark.parametrize("use_replacements", [False, True])
@pytest.mark.parametrize("target_snr_range", [None, [-5, 5]])
def test_regenerates_dry_sources_and_publishes_only_complete_manifest(tmp_path, use_replacements, target_snr_range):
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
    if target_snr_range is not None:
        cfg.update(original_snr_range=[10, 12], snr_range=target_snr_range)
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
        expected_snr = before["snr_db"] if target_snr_range is None else -5 + 5 * (before["snr_db"] - 10)
        assert after["snr_db"] == pytest.approx(expected_snr, abs=1e-10)
        assert after["parent_snr_db"] == before["snr_db"]
        assert abs(after["measured_snr_db_before_pcm16"] - expected_snr) < 0.001
        assert "response" not in after and "wer" not in after
        assert after["source_generation"]["response"] == "old prediction"
        assert after["audio_sha256"] == hashlib.sha256(Path(after["audio_path"]).read_bytes()).hexdigest()
        if use_replacements:
            assert after["noise_path"] == str(replacement)
            assert after["parent_noise_path"] == str(noise)
            assert after["rir_license"] == after["noise_license"] == "CC-BY-4.0"
    with pytest.raises(ValueError, match="speed mismatch"):
        original_draws({**rows[0], "speed_factor": 1}, [str(noise)], cfg)
    with pytest.raises(ValueError, match="noise/SNR mismatch"):
        original_draws({**rows[0], "snr_db": 0}, [str(noise)], cfg)
    if target_snr_range is not None:
        old_cfg = {**cfg, "snr_range": [10, 12]}
        _, old_speed, old_snr, old_rng = original_draws(rows[0], [str(noise)], old_cfg)
        _, new_speed, new_snr, new_rng = original_draws(rows[0], [str(noise)], cfg)
        assert old_speed == new_speed
        assert new_snr == pytest.approx(-5 + 5 * (old_snr - 10), abs=1e-10)
        assert old_rng.getstate() == new_rng.getstate()
    if use_replacements:
        validation = argparse.Namespace(
            config=str(config), expected_rows=2, samples=2, output=str(tmp_path / "validation.json")
        )
        validate(validation)
        assert json.loads(Path(validation.output).read_text())["all_remote_paths_and_sizes_verified"]
        # A published manifest must not hide a missing or truncated upload.
        uploaded = Path(output[0]["audio_path"])
        original_bytes = uploaded.read_bytes()
        uploaded.write_bytes(original_bytes[:44])
        with pytest.raises(ValueError, match="WAV size mismatch"):
            validate(validation)
        uploaded.unlink()
        with pytest.raises(ValueError, match="Missing 1 uploaded WAVs"):
            validate(validation)
