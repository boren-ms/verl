import io

import numpy as np
import soundfile as sf
from recipe.phimm.utils.audio import get_audio_duration, load_raw_audio


def test_read_indexed_window_without_decoding_other_audio(tmp_path):
    rate = 16000
    signal = np.arange(rate * 4, dtype=np.int16)
    payload = io.BytesIO()
    sf.write(payload, signal, rate, format="WAV", subtype="PCM_16")
    wav = payload.getvalue()
    archive = tmp_path / "source.audio"
    # The following bytes must never affect the embedded WAV length.
    archive.write_bytes(b"before" + wav + b"following audio")
    start, end = 16385, 17817
    ref = f"{archive}!6:{len(wav)}#{start / rate}:{end / rate}"
    output, actual_rate = load_raw_audio({"audio_path": ref})
    assert actual_rate == rate
    np.testing.assert_array_equal(output, signal[start:end] / 32768)
    assert abs(get_audio_duration(ref) - (end - start) / rate) < 1e-12
