from cachetools import FIFOCache, cached
import blobfile as bf
import logging
import numpy as np
from pathlib import Path
import soundfile as sf
from recipe.phimm.data.chunk import load_chunk_sample, load_chunk_example
from verl.audio_cache import localize_audio_source

logger = logging.getLogger(__name__)


TARGET_SAMPLE_RATE = 16000

# Module-level chunk load mode: "cached" (default) or "sample"
_chunk_load_mode = "cached"


def _localize_remote_audio(file_path):
    return localize_audio_source(file_path)


def set_chunk_load_mode(mode: str):
    """Set the chunk loading strategy.

    Args:
        mode: "cached" to use load_chunk_example (ChunkManager, full chunk in memory),
              "sample" to use load_chunk_sample (seek-based, one connection per sample).
    """
    global _chunk_load_mode
    if mode not in ("cached", "sample"):
        raise ValueError(f"Invalid chunk_load_mode: {mode!r}. Must be 'cached' or 'sample'.")
    _chunk_load_mode = mode
    logger.info("Chunk load mode set to: %s", mode)


@cached(FIFOCache(maxsize=100))
def sf_read(file_path):
    """Load audio from a file."""
    # print("Audio file:", file_path)
    if "://" in file_path:
        return sf.read(_localize_remote_audio(file_path))

    if not bf.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with bf.BlobFile(file_path, "rb") as f:
        return sf.read(f)


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


def limit_audio(x, fs, max_dur=None, min_dur=0.16):
    """Resample audio to 16 kHz and limit it to max_dur seconds."""
    assert x.ndim == 1, "Only mono audio is supported."
    x_dur = len(x) / fs
    if x_dur < min_dur:
        print(f"Padding audio {x_dur:.2f} ->  {min_dur:.2f} seconds.")
        pad_len = int(min_dur * fs) - len(x)
        pad_mode = "edge" if len(x) > 0 else "constant"
        x = np.pad(x, (0, pad_len), mode=pad_mode)

    if max_dur is not None and x_dur > max_dur:
        print(f"Truncating audio {x_dur:.2f} ->  {max_dur} seconds.")
        x = x[: int(max_dur * fs)]

    x, fs = resample_audio(x, fs)
    assert fs == TARGET_SAMPLE_RATE, f"Sample rate should be {TARGET_SAMPLE_RATE} Hz."

    return x, fs


def _to_mono(data, source):
    if isinstance(data, np.ndarray) and data.ndim == 2:
        channels = data.shape[1]
        logger.warning("Non-mono audio (%d channels) from %s, converting to mono by averaging.", channels, source)
        data = data.mean(axis=1)
    return data


def _is_chunk_spec(path: str) -> bool:
    """Return True if ``path`` looks like a chunk spec ``file:<count>:<index>``."""
    parts = path.rsplit(":", 2)
    return len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit()


def _is_time_chunk_spec(path: str) -> bool:
    """Return True for ``file#<start>:<end>`` specs in seconds or percentages."""
    if "#" not in path:
        return False
    _, _, tail = path.rpartition("#")
    if ":" not in tail:
        return False
    s, _, e = tail.partition(":")
    try:
        _parse_range_endpoint(s)
        _parse_range_endpoint(e)
        return True
    except ValueError:
        return False


def _parse_range_endpoint(value: str):
    is_percent = value.endswith("%")
    number = float(value[:-1] if is_percent else value)
    return number, is_percent


def _load_time_chunk(spec):
    file_path, _, tail = spec.rpartition("#")
    s_str, _, e_str = tail.partition(":")
    start, start_is_percent = _parse_range_endpoint(s_str)
    end, end_is_percent = _parse_range_endpoint(e_str)
    if start_is_percent != end_is_percent:
        raise ValueError(f"Range endpoints must use the same unit: {tail!r}")
    upper_bound = 100.0 if start_is_percent else float("inf")
    if not 0 <= start < end <= upper_bound:
        unit = "percent" if start_is_percent else "seconds"
        raise ValueError(f"Invalid audio range in {unit}: {tail!r}")
    if "://" in file_path:
        readable_path = _localize_remote_audio(file_path)
    else:
        if not bf.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        readable_path = file_path

    info = sf.info(readable_path)
    sr = info.samplerate
    if start_is_percent:
        start_frame = int(info.frames * start / 100)
        stop_frame = int(info.frames * end / 100)
    else:
        start_frame = int(start * sr)
        stop_frame = int(end * sr)
    stop_frame = max(start_frame + 1, min(stop_frame, info.frames))
    audio, sr = sf.read(readable_path, start=start_frame, stop=stop_frame)
    return audio, sr


def _load_chunk(spec):
    if _is_time_chunk_spec(spec):
        return _load_time_chunk(spec)
    if _chunk_load_mode == "sample":
        result = load_chunk_sample(spec)
    else:
        result = load_chunk_example(spec)
    if isinstance(result, list):
        result = result[0]  # "audios" chunk type returns list of (data, sr)
    return result


def load_raw_audio(x):
    """Load audio data from the input dictionary."""
    if (audio := x.get("audio", None)) and (sr := x.get("sr", None)):
        return _to_mono(audio, "inline audio"), sr
    source = x.get("audio_path") or x.get("audio_file") or x.get("audio_chunk")
    if source:
        if _is_chunk_spec(source) or _is_time_chunk_spec(source):
            data, sr = _load_chunk(source)
        else:
            data, sr = sf_read(source)
        return _to_mono(data, source), sr
    raise ValueError("No audio data found in the input dictionary.")


def load_audio(x, max_dur=None, min_dur=0.16):
    data, fs = load_raw_audio(x)
    return limit_audio(data, fs, max_dur=max_dur, min_dur=min_dur)


def load_raw_audios(x):  # x is batched
    from itertools import zip_longest

    audio_paths = x.get("audio_path") or x.get("audio_file") or []
    audio_chunks = x.get("audio_chunk", [])

    for audio_path, audio_chunk in zip_longest(audio_paths, audio_chunks, fillvalue=None):
        source = audio_path or audio_chunk
        if not source:
            raise ValueError("No audio data found in the input dictionary.")
        if _is_chunk_spec(source) or _is_time_chunk_spec(source):
            data, sr = _load_chunk(source)
        else:
            data, sr = sf_read(source)
        yield _to_mono(data, source), sr


def load_audios(x, max_dur=None, min_dur=0.16):  # x is batched
    return [limit_audio(data, fs, max_dur=max_dur, min_dur=min_dur) for data, fs in load_raw_audios(x)]
