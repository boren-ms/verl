from cachetools import FIFOCache, cached
import blobfile as bf
import logging
import multiprocessing
import numpy as np
import os
from pathlib import Path
import queue
import soundfile as sf
import tempfile
import threading
import time
from recipe.phimm.data.chunk import load_chunk_sample, load_chunk_example

logger = logging.getLogger(__name__)


TARGET_SAMPLE_RATE = 16000
AUDIO_READ_TIMEOUT_SECONDS = 120
AUDIO_READ_ATTEMPTS = 3
AUDIO_READER_POLL_SECONDS = 0.1
bf.configure(
    connect_timeout=AUDIO_READ_TIMEOUT_SECONDS,
    read_timeout=AUDIO_READ_TIMEOUT_SECONDS,
    retry_limit=0,
)

# Module-level chunk load mode: "cached" (default) or "sample"
_chunk_load_mode = "cached"
_audio_reader_process = None
_audio_reader_requests = None
_audio_reader_lock = threading.Lock()
_audio_reader_poll_event = threading.Event()


def _retry_blob_operation(file_path, operation):
    for attempt in range(1, AUDIO_READ_ATTEMPTS + 1):
        try:
            return operation()
        except (bf.Error, OSError) as exc:
            if attempt == AUDIO_READ_ATTEMPTS:
                raise
            logger.warning(
                "Blob operation failed for %s (attempt %d/%d): %s; retrying.",
                file_path,
                attempt,
                AUDIO_READ_ATTEMPTS,
                exc,
            )
            time.sleep(attempt)


def _soundfile_read_worker(requests):
    while True:
        request = requests.get()
        if request is None:
            return

        file_path, kwargs, result_path, error_path = request
        try:
            with bf.BlobFile(file_path, "rb") as f:
                data, sample_rate = sf.read(f, **kwargs)
            temporary_path = f"{result_path}.tmp"
            with open(temporary_path, "wb") as result_file:
                np.savez(result_file, data=data, sample_rate=sample_rate)
            os.replace(temporary_path, result_path)
        except Exception as exc:
            with open(error_path, "w", encoding="utf-8") as error_file:
                error_file.write(f"{type(exc).__name__}: {exc}")


def _stop_audio_reader():
    global _audio_reader_process, _audio_reader_requests

    if _audio_reader_process is not None:
        if _audio_reader_process.is_alive():
            _audio_reader_process.terminate()
        _audio_reader_process.join(timeout=1)
    if _audio_reader_requests is not None:
        _audio_reader_requests.close()
        _audio_reader_requests.join_thread()
    _audio_reader_process = None
    _audio_reader_requests = None


def _start_audio_reader():
    global _audio_reader_process, _audio_reader_requests

    if _audio_reader_process is not None and _audio_reader_process.is_alive():
        return
    _stop_audio_reader()
    context = multiprocessing.get_context("spawn")
    _audio_reader_requests = context.Queue(maxsize=1)
    _audio_reader_process = context.Process(
        target=_soundfile_read_worker,
        args=(_audio_reader_requests,),
        daemon=True,
    )
    _audio_reader_process.start()


def _remove_audio_reader_result(result_path, error_path):
    for path in (result_path, f"{result_path}.tmp", error_path):
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def _read_soundfile_isolated(file_path, **kwargs):
    """Read Blob-backed audio out-of-process so a stalled HTTPS read can be killed."""
    with _audio_reader_lock:
        _start_audio_reader()
        descriptor, result_path = tempfile.mkstemp(prefix="verl_audio_", suffix=".npz")
        os.close(descriptor)
        os.unlink(result_path)
        error_path = f"{result_path}.error"
        try:
            _audio_reader_requests.put((file_path, kwargs, result_path, error_path), block=False)
        except queue.Full as exc:
            _stop_audio_reader()
            _remove_audio_reader_result(result_path, error_path)
            raise OSError(f"Blob audio reader is busy: {file_path}") from exc
        remaining_polls = int(AUDIO_READ_TIMEOUT_SECONDS / AUDIO_READER_POLL_SECONDS)
        while True:
            if os.path.exists(result_path):
                try:
                    with np.load(result_path) as result:
                        return result["data"], int(result["sample_rate"])
                finally:
                    _remove_audio_reader_result(result_path, error_path)
            if os.path.exists(error_path):
                try:
                    with open(error_path, encoding="utf-8") as error_file:
                        message = error_file.read()
                finally:
                    _remove_audio_reader_result(result_path, error_path)
                raise OSError(f"Blob audio read failed for {file_path}: {message}")
            if remaining_polls <= 0:
                _stop_audio_reader()
                _remove_audio_reader_result(result_path, error_path)
                logger.warning("Blob audio read timed out after %d seconds: %s", AUDIO_READ_TIMEOUT_SECONDS, file_path)
                raise OSError(f"Blob audio read exceeded {AUDIO_READ_TIMEOUT_SECONDS} seconds: {file_path}")
            _audio_reader_poll_event.wait(AUDIO_READER_POLL_SECONDS)
            remaining_polls -= 1


def _read_soundfile(file_path, **kwargs):
    if not file_path.startswith("az://"):
        return sf.read(file_path, **kwargs)

    return _retry_blob_operation(file_path, lambda: _read_soundfile_isolated(file_path, **kwargs))


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
    return _read_soundfile(file_path)


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
    """Return True if ``path`` looks like a time-range spec ``file#<start_sec>:<end_sec>``."""
    if "#" not in path:
        return False
    _, _, tail = path.rpartition("#")
    if ":" not in tail:
        return False
    s, _, e = tail.partition(":")
    try:
        float(s)
        float(e)
        return True
    except ValueError:
        return False


def _load_time_chunk(spec):
    file_path, _, tail = spec.rpartition("#")
    s_str, _, e_str = tail.partition(":")
    start_sec = float(s_str)
    end_sec = float(e_str)
    if not _retry_blob_operation(file_path, lambda: bf.exists(file_path)):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with bf.BlobFile(file_path, "rb") as f:
        info = sf.info(f)
    sr = info.samplerate
    start_frame = max(0, int(start_sec * sr))
    stop_frame = max(start_frame + 1, int(end_sec * sr))
    return _read_soundfile(file_path, start=start_frame, stop=stop_frame)


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
