from cachetools import FIFOCache, cached
import blobfile as bf
import soundfile as sf
from .chunk import load_chunk_example


@cached(FIFOCache(maxsize=100))
def sf_read(file_path):
    """Load audio from a file."""
    # print("Audio file:", file_path)
    if not bf.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with bf.BlobFile(file_path, "rb") as f:
        audio, sr = sf.read(f)
    return audio, sr


def limit_audio(x, fs, max_dur=30):
    """Limit the length of the audio to max_dur seconds."""
    if len(x) > fs * max_dur:
        print(f"Truncating audio {len(x) / fs:.2f} ->  {max_dur} seconds.")
        x = x[: fs * max_dur]
    return x, fs


def load_raw_audio(x):
    """Load audio data from the input dictionary."""
    if "audio" in x and "sr" in x:
        return x["audio"], x["sr"]
    elif "audio_path" in x:
        return sf_read(x["audio_path"])
    elif "audio_chunk" in x:
        return load_chunk_example(x["audio_chunk"])
    raise ValueError("No audio data found in the input dictionary.")


def load_audio(x, max_dur=30):
    data, fs = load_raw_audio(x)
    return limit_audio(data, fs, max_dur=max_dur)


def load_raw_audios(x):  # x is batched
    if "audio" in x and "sr" in x:
        return list(zip(x["audio"], x["sr"]))
    elif "audio_path" in x:
        return [sf_read(p) for p in x["audio_path"]]
    elif "audio_chunk" in x:
        return [load_chunk_example(p) for p in x["audio_chunk"]]
    raise ValueError("No audio data found in the input dictionary.")


def load_audios(x, max_dur=30):  # x is batched
    return [limit_audio(data, fs, max_dur=max_dur) for data, fs in load_raw_audios(x)]
