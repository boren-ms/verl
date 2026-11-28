"""Random segments preserve the inner original clip on byte-addressed audio."""

import io

import numpy as np
import pytest
import soundfile as sf

from recipe.phimm.utils.audio import load_raw_audio


@pytest.mark.parametrize("outer, first, last", [("1:3", 3, 5), ("25%:75%", 3, 5)])
def test_nested_time_range_on_chunk_payload(tmp_path, outer, first, last):
    sr = 16000
    waveform = np.linspace(-0.5, 0.5, 8 * sr, dtype=np.float32)
    encoded = io.BytesIO()
    sf.write(encoded, waveform, sr, format="WAV", subtype="FLOAT")
    payload = encoded.getvalue()
    archive = tmp_path / "chunk.audio"
    archive.write_bytes(b"prefix" + payload + b"suffix")
    source = f"{archive}!6:{len(payload)}#2:6#{outer}"

    actual, actual_sr = load_raw_audio({"audio_path": source})

    assert actual_sr == sr
    np.testing.assert_array_equal(actual, waveform[first * sr : last * sr])
