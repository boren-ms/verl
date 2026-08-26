import numpy as np
import soundfile as sf

from recipe.phimm.utils.audio import _is_time_chunk_spec, load_raw_audio


def test_load_raw_audio_reads_percentage_range(tmp_path):
    audio_path = tmp_path / "sample.wav"
    sf.write(audio_path, np.arange(1000, dtype=np.float32) / 1000, 16000)

    audio, sample_rate = load_raw_audio({"audio_path": f"{audio_path}#0%:10%"})

    assert sample_rate == 16000
    assert len(audio) == 100


def test_time_chunk_spec_accepts_seconds_and_percentages():
    assert _is_time_chunk_spec("sample.wav#0:1.5")
    assert _is_time_chunk_spec("sample.wav#0%:10%")
    assert not _is_time_chunk_spec("sample.wav#start:end")