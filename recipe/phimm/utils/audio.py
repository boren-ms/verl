from cachetools import FIFOCache, cached
import blobfile as bf
from pathlib import Path
import soundfile as sf
from recipe.phimm.data.chunk import load_chunk_example


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


def limit_audio(x, fs, max_dur=None):
    """Limit the length of the audio to max_dur seconds."""
    if max_dur is not None and len(x) > fs * max_dur:
        print(f"Truncating audio {len(x) / fs:.2f} ->  {max_dur} seconds.")
        x = x[: fs * max_dur]
    n = len(x)
    n = n // 8 * 8  # make it multiple of 8
    x = x[:n]
    return x, fs


def load_raw_audio(x):
    """Load audio data from the input dictionary."""
    if (audio := x.get("audio", None)) and (sr := x.get("sr", None)):
        return audio, sr
    if audio_path := x.get("audio_path", None) or x.get("audio_file", None):
        return sf_read(audio_path)
    if audio_chunk := x.get("audio_chunk", None):
        result = load_chunk_example(audio_chunk)
        if isinstance(result, list):
            return result[0]  # "audios" chunk type returns list of (data, sr)
        return result
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
            yield sf_read(audio_path)
        elif audio_chunk:
            result = load_chunk_example(audio_chunk)
            if isinstance(result, list):
                yield result[0]  # "audios" chunk type returns list of (data, sr)
            else:
                yield result
        else:
            raise ValueError("No audio data found in the input dictionary.")


def load_audios(x, max_dur=None):  # x is batched
    return [limit_audio(data, fs, max_dur=max_dur) for data, fs in load_raw_audios(x)]
