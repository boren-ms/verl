import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from scripts.run_openlong_qwen_windows import DecodedAudio, align_window, parse_zip_member, read_completed, shard_for, split_retry_windows


class FakeAligner:
    aligner_processor = SimpleNamespace(tokenize_space_lang=lambda text: text.split())

    def align(self, audio, text, language):
        samples, rate = audio
        assert rate == 16000 and len(samples) == 16000
        assert text == "hello world" and language == "English"
        return [[SimpleNamespace(text="hello", start_time=0.1, end_time=0.3),
                 SimpleNamespace(text="world", start_time=0.4, end_time=0.8)]]


def test_align_window_uses_audio_range_and_parent_offsets(tmp_path):
    path = tmp_path / "audio.wav"
    sf.write(path, np.zeros(48000, dtype=np.float32), 16000)
    window = {"audio_start": 1.0, "audio_end": 2.0, "seed_text": "hello world"}
    with sf.SoundFile(path) as audio:
        result = align_window(FakeAligner(), window, audio)
    assert result["status"] == "ok"
    assert result["words"][0]["parent_start"] == pytest.approx(1.1)
    assert result["words"][1]["parent_end"] == pytest.approx(1.8)


def test_decoded_audio_slices_match_array():
    source = np.arange(20, dtype=np.float32)
    audio = DecodedAudio(source, 16000)
    audio.seek(4)
    assert np.array_equal(audio.read(5), source[4:9])
    audio.seek(1)
    assert np.array_equal(audio.read(2), source[1:3])


def test_empty_seed_and_out_of_bounds_window_are_explicit(tmp_path):
    path = tmp_path / "audio.wav"
    sf.write(path, np.zeros(16000, dtype=np.float32), 16000)
    with sf.SoundFile(path) as audio:
        assert align_window(FakeAligner(), {"audio_start": 0, "audio_end": 1, "seed_text": ""}, audio)["status"] == "empty_seed"
        with pytest.raises(ValueError, match="exceeds audio"):
            align_window(FakeAligner(), {"audio_start": 0, "audio_end": 2, "seed_text": "hello world"}, audio)


def test_parent_sharding_is_stable():
    assert shard_for("earnings21", "parent", 8) == shard_for("earnings21", "parent", 8)
    assert 0 <= shard_for("earnings21", "parent", 8) < 8


def test_out_of_window_timestamp_is_preserved_and_flagged(tmp_path):
    class ClippedAligner(FakeAligner):
        def align(self, audio, text, language):
            return [[SimpleNamespace(text="hello", start_time=0.9, end_time=1.2)]]

    path = tmp_path / "audio.wav"
    sf.write(path, np.zeros(16000, dtype=np.float32), 16000)
    with sf.SoundFile(path) as audio:
        result = align_window(ClippedAligner(), {"audio_start": 0, "audio_end": 1, "seed_text": "hello"}, audio)
    assert result["words"][0]["end"] == 1
    assert result["words"][0]["predicted_end"] == 1.2
    assert result["words"][0]["clipped_to_window"] is True
    assert result["quality"]["outside_audio"] == 1


def test_zip_member_locator_is_strict():
    assert parse_zip_member("az://account/data/audio.zip!17:42") == ("az://account/data/audio.zip", 17, 42)
    with pytest.raises(ValueError):
        parse_zip_member("az://account/data/audio.zip!17:0")


def test_collapsed_window_retry_preserves_core_segment_coverage():
    window = {"dataset": "earnings21", "parent_audio_path": "parent", "audio_source": "audio.zip!1:2",
              "core_segment_ids": ["s1", "s2"], "core_segment_indexes": [1, 2],
              "segments": [{"id": "s1", "seg_index": 1, "seg_start": 1, "seg_end": 2, "text": "one"},
                           {"id": "s2", "seg_index": 2, "seg_start": 2, "seg_end": 3, "text": "two words"}]}
    retries = split_retry_windows(window)
    assert [r["core_segment_ids"] for r in retries] == [["s1"], ["s2"]]
    assert [r["core_word_end"] for r in retries] == [1, 2]
    assert all(r["retry_reason"] == "collapsed_timestamps" for r in retries)


def test_resume_accepts_complete_retries_and_rejects_partial_results(tmp_path):
    key = ("earnings21", "parent", ("s1", "s2"))
    rows = [{"window": {"dataset": "earnings21", "parent_audio_path": "parent",
                        "core_segment_ids": [sid], "retry_original_core_segment_ids": ["s1", "s2"]},
             "status": "ok"} for sid in ("s1", "s2")]
    path = tmp_path / "shard.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert read_completed(path, {key}) == {key}
    path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete"):
        read_completed(path, {key})
