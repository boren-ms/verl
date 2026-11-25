"""Hard cuts cover full audio once, including tails missed by old segmentation."""

import io
import json
import zipfile
from copy import deepcopy

import numpy as np
import pytest
import soundfile as sf
from omegaconf import OmegaConf

from recipe.phimm.agentic.dataset import ASRDataset
from recipe.phimm.agentic.details import segment_row
from recipe.phimm.tools.openlong_hard_chunk import hard_chunk_parent, validation_source
from recipe.phimm.utils.audio import _load_time_chunk


def parent(member="audio.zip!100:160044", duration=5.0):
    return {
        "id": "parent",
        "text": "The exact full recording transcript.",
        "parent_audio_path": "original.wav",
        "duration": duration,
        "sampling_rate": 16000,
        # Old segments overlap and omit the tail; hard cuts must ignore these times.
        "audio_path": [f"{member}#0.0:2.7", f"{member}#2.4:4.8"],
        "n_segments": 2,
        "seg_start": 0.0,
        "seg_end": 4.8,
        "segment_reference_texts": ["Old reference", "history"],
        "source_metadata_json": ["obsolete alignment"],
    }


@pytest.mark.parametrize("frames", [80000, 64002])
def test_real_validation_loader_and_audio_preserve_every_sample(tmp_path, frames):
    waveform = np.linspace(-0.8, 0.8, frames, dtype=np.float32)
    encoded = io.BytesIO()
    sf.write(encoded, waveform, 16000, format="WAV", subtype="FLOAT")
    archive = tmp_path / "audio.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("audio.wav", encoded.getvalue())
        info = zf.getinfo("audio.wav")
        offset = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
        member = f"{archive}!{offset}:{info.file_size}"
    original = parent(member, duration=frames / 16000)
    before = deepcopy(original)
    row = hard_chunk_parent(original)
    assert original == before
    assert row["audio_path"] == [f"{member}#0.0:2.0", f"{member}#2.0:4.0", f"{member}#4.0:{frames / 16000}"]
    assert "segment_reference_texts" not in row and "source_metadata_json" not in row
    path = tmp_path / "parents.jsonl"
    path.write_text(json.dumps(row) + "\n")
    config = OmegaConf.create(
        {"version": 2609, "val_files": "val", "val_data": [[validation_source("earnings21", str(path))]]}
    )
    dataset = ASRDataset("val", None, config)
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["reward_model"]["ground_truth"] == original["text"]
    segments = [segment_row(sample, segment) for segment in sample["asr_segments"]]
    assert all(set(segment) == {"audio_path"} for segment in sample["asr_segments"])
    assert len(segments) == sample["extra_info"]["n_segments"] == 3
    assert all(segment["asr_full_reference"] for segment in segments)
    decoded = [_load_time_chunk(segment["audio_path"])[0] for segment in segments]
    assert [len(audio) for audio in decoded] == [32000, 32000, frames - 64000]
    np.testing.assert_array_equal(np.concatenate(decoded), waveform)


@pytest.mark.parametrize("duration,lengths", [(4.0, [2.0, 2.0]), (0.03125, [0.03125]), (2.0000625, [2.0, 0.0000625])])
def test_exact_multiple_and_short_tails(duration, lengths):
    row = hard_chunk_parent(parent(duration=duration))
    times = [[float(t) for t in path.rsplit("#", 1)[1].split(":")] for path in row["audio_path"]]
    assert [end - start for start, end in times] == pytest.approx(lengths)
    assert times[0][0] == 0 and times[-1][1] == duration
    assert all(times[i][1] == times[i + 1][0] for i in range(len(times) - 1))


@pytest.mark.parametrize("seconds", [0, -2, float("nan"), float("inf"), True, 0.000001])
def test_invalid_chunk_lengths(seconds):
    with pytest.raises(ValueError):
        hard_chunk_parent(parent(), seconds)


def test_different_members_and_fractional_samples_are_rejected():
    original = parent()
    original["audio_path"][1] = "different.zip!100:160044#2.4:4.8"
    with pytest.raises(ValueError, match="same complete ZIP member"):
        hard_chunk_parent(original)
    with pytest.raises(ValueError, match="whole number"):
        hard_chunk_parent(parent(duration=5.00001))
