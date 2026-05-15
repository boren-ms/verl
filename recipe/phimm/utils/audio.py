from cachetools import FIFOCache, cached
import blobfile as bf
import logging
import numpy as np
from pathlib import Path
import soundfile as sf
from recipe.phimm.data.chunk import load_chunk_example

logger = logging.getLogger(__name__)


TARGET_SAMPLE_RATE = 16000


@cached(FIFOCache(maxsize=100))
def sf_read(file_path):
    """Load audio from a file."""
    # print("Audio file:", file_path)
    if not bf.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with bf.BlobFile(file_path, "rb") as f:
        audio, sr = sf.read(f)
    return audio, sr


def sf_write(file_path, audio, sr):
    """Write audio to a file."""
    fmt = Path(file_path).suffix.lstrip(".").upper() or "WAV"
    with bf.BlobFile(file_path, "wb") as f:
        sf.write(f, audio, sr, format=fmt)


def resample_audio(x, fs, target_fs=TARGET_SAMPLE_RATE):
    """Resample audio to target_fs if needed."""
    if fs != target_fs:
        import torch
        import torchaudio.functional as F

        waveform = torch.from_numpy(x.T if x.ndim > 1 else x)
        x = F.resample(waveform, fs, target_fs).numpy()
        fs = target_fs
    return x, fs


def limit_audio(x, fs, max_dur=None):
    """Resample audio to 16 kHz and limit it to max_dur seconds."""
    assert x.ndim == 1, "Only mono audio is supported."
    x, fs = resample_audio(x, fs)
    assert fs == TARGET_SAMPLE_RATE, f"Sample rate should be {TARGET_SAMPLE_RATE} Hz."

    if max_dur is not None and len(x) > fs * max_dur:
        print(f"Truncating audio {len(x) / fs:.2f} ->  {max_dur} seconds.")
        x = x[: fs * max_dur]

    min_len = int(0.16 * fs)
    pad_to_len = max(len(x), min_len)
    pad_to_len += -pad_to_len % 8
    pad_len = pad_to_len - len(x)
    if pad_len:
        x = np.pad(x, [(0, pad_len)] + [(0, 0)] * (x.ndim - 1))
    return x, fs


def _to_mono(data, source):
    if isinstance(data, np.ndarray) and data.ndim == 2:
        channels = data.shape[1]
        logger.warning("Non-mono audio (%d channels) from %s, converting to mono by averaging.", channels, source)
        data = data.mean(axis=1)
    return data


def load_raw_audio(x):
    """Load audio data from the input dictionary."""
    if (audio := x.get("audio", None)) and (sr := x.get("sr", None)):
        return _to_mono(audio, "inline audio"), sr
    if audio_path := x.get("audio_path", None) or x.get("audio_file", None):
        data, sr = sf_read(audio_path)
        return _to_mono(data, audio_path), sr
    if audio_chunk := x.get("audio_chunk", None):
        result = load_chunk_example(audio_chunk)
        if isinstance(result, list):
            result = result[0]  # "audios" chunk type returns list of (data, sr)
        data, sr = result
        return _to_mono(data, audio_chunk), sr
    raise ValueError("No audio data found in the input dictionary.")


def load_audio(x, max_dur=None):
    data, fs = load_raw_audio(x)
    return limit_audio(data, fs, max_dur=max_dur)


def load_raw_audios(x):  # x is batched
    audio_paths = x.get("audio_path", [])
    audio_chunks = x.get("audio_chunk", [])
    from itertools import zip_longest

    for audio_path, audio_chunk in zip_longest(audio_paths, audio_chunks, fillvalue=None):
        if audio_path:
            data, sr = sf_read(audio_path)
            yield _to_mono(data, audio_path), sr
        elif audio_chunk:
            result = load_chunk_example(audio_chunk)
            if isinstance(result, list):
                result = result[0]  # "audios" chunk type returns list of (data, sr)
            data, sr = result
            yield _to_mono(data, audio_chunk), sr
        else:
            raise ValueError("No audio data found in the input dictionary.")


def load_audios(x, max_dur=None):  # x is batched
    return [limit_audio(data, fs, max_dur=max_dur) for data, fs in load_raw_audios(x)]
