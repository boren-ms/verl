"""Trim head/tail silence from audio using Silero VAD (ONNX). Supports parallel processing."""

import glob
import io
import numpy as np
import onnxruntime as ort
import soundfile as sf
import fire
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _worker(args):
    """Top-level wrapper for ProcessPoolExecutor (must be picklable)."""
    path, kwargs = args
    try:
        return SilenceTrimmer().trim_file(path, **kwargs)
    except Exception as e:
        print(f"[error] {path}: {e}")
        return None


class SilenceTrimmer:
    """Trim head/tail silence from audio files using Silero VAD (ONNX)."""

    _MODEL_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
    _MODEL_BLOB = "az://orngwus2cresco/data/boren/data/verl/silero_vad.onnx"
    _TARGET_SR = 16000

    def __init__(self, threshold: float = 0.5):
        self._threshold = threshold
        model_path = self._ensure_model()
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    @classmethod
    def _ensure_model(cls) -> Path:
        cache_dir = Path.home() / ".cache" / "silero-vad"
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / "silero_vad.onnx"
        if not path.exists():
            # Try URL first, fall back to blob (for nodes without internet)
            try:
                import urllib.request
                print(f"Downloading Silero VAD ONNX model to {path}...")
                urllib.request.urlretrieve(cls._MODEL_URL, path)
            except Exception:
                import blobfile as bf
                print(f"Downloading VAD model from blob {cls._MODEL_BLOB} ...")
                with bf.BlobFile(cls._MODEL_BLOB, "rb") as src, open(path, "wb") as dst:
                    dst.write(src.read())
        return path

    @staticmethod
    def read_audio(audio_path: str):
        """Read audio from local or az:// path. Returns (audio, sr)."""
        import blobfile as bf
        with bf.BlobFile(audio_path, "rb") as f:
            return sf.read(f, dtype="float32", always_2d=False)

    @staticmethod
    def write_audio(output_path: str, audio, sr: int):
        """Write audio to local or az:// path, inferring format from extension."""
        from recipe.phimm.utils.audio import sf_write
        sf_write(output_path, audio, sr)

    def _speech_timestamps(self, audio: np.ndarray, window: int = 512) -> list[dict]:
        sr = self._TARGET_SR
        state = np.zeros((2, 1, 128), dtype=np.float32)
        sr_tensor = np.array(sr, dtype=np.int64)
        ctx_size = 64 if sr == 16000 else 32
        ctx = np.zeros(ctx_size, dtype=np.float32)
        thr, neg_thr = self._threshold, self._threshold - 0.15
        triggered = False
        speeches, cur = [], {}

        pad = (-len(audio)) % window
        if pad:
            audio = np.pad(audio, (0, pad))

        for i in range(0, len(audio), window):
            chunk = audio[i : i + window]
            inp = np.concatenate([ctx, chunk])[np.newaxis].astype(np.float32)
            out, state = self._session.run(None, {"input": inp, "state": state, "sr": sr_tensor})[:2]
            ctx = chunk[-ctx_size:]

            if out[0][0] >= thr and not triggered:
                triggered, cur["start"] = True, i
            elif out[0][0] < neg_thr and triggered:
                cur["end"] = i + window
                if (cur["end"] - cur["start"]) / sr * 1000 >= 250:
                    speeches.append(cur)
                cur, triggered = {}, False

        if triggered:
            speeches.append({**cur, "end": len(audio)})

        # merge close segments
        if len(speeches) > 1:
            merged = [speeches[0]]
            for s in speeches[1:]:
                if (s["start"] - merged[-1]["end"]) / sr * 1000 < 100:
                    merged[-1]["end"] = s["end"]
                else:
                    merged.append(s)
            speeches = merged

        return speeches

    def trim(self, audio: np.ndarray, sr: int, head_cut_ms: int = 0, tail_cut_ms: int = 0) -> np.ndarray | None:
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # resample to 16 kHz for VAD
        tsr = self._TARGET_SR
        if sr != tsr:
            ratio = tsr / sr
            idx = np.arange(int(len(audio) * ratio)) / ratio
            fl = np.floor(idx).astype(int)
            cl = np.minimum(fl + 1, len(audio) - 1)
            audio_16k = audio[fl] * (1 - (idx - fl)) + audio[cl] * (idx - fl)
        else:
            audio_16k = audio

        timestamps = self._speech_timestamps(audio_16k)
        if not timestamps:
            return None

        n = len(audio_16k)
        hc, tc = int(head_cut_ms * tsr / 1000), int(tail_cut_ms * tsr / 1000)
        start = max(0, timestamps[0]["start"] + hc)
        end = min(n, timestamps[-1]["end"] - tc)
        end = max(start + 1, end)

        # map back to original sr
        scale = sr / tsr
        orig_start, orig_end = int(start * scale), int(end * scale)
        return audio[orig_start:orig_end]

    def trim_file(self, audio_path: str, output_path: str = None,
                  head_cut_ms: int = 0, tail_cut_ms: int = 0) -> str | None:
        """Trim silence from an audio file (supports az:// paths). Returns output path or None."""
        audio, sr = self.read_audio(audio_path)
        trimmed = self.trim(audio, sr, head_cut_ms=head_cut_ms, tail_cut_ms=tail_cut_ms)
        if trimmed is None:
            print(f"[skip] No speech: {audio_path}")
            return None
        if output_path is None:
            p = Path(audio_path)
            output_path = str(p.with_stem(p.stem + "_trimmed"))
        else:
            op = Path(output_path)
            if op.is_dir():
                output_path = str(op / Path(audio_path).name)
        self.write_audio(output_path, trimmed, sr)
        print(f"[done] {audio_path}  {len(audio)/sr:.2f}s -> {len(trimmed)/sr:.2f}s  => {output_path}")
        return output_path

    @staticmethod
    def trim_parallel(paths: list[str], jobs: int = 4, **kwargs):
        tasks = [(p, kwargs) for p in paths]
        done, failed = 0, 0
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for result in pool.map(_worker, tasks):
                if result:
                    done += 1
                else:
                    failed += 1
        print(f"\nProcessed {done + failed} files: {done} trimmed, {failed} skipped/failed")


def main(*audio: str, output: str = None, head_cut: int = 0, tail_cut: int = 0, jobs: int = 4):
    files = []
    for pattern in audio:
        expanded = glob.glob(pattern, recursive=True)
        files.extend(expanded if expanded else [pattern])

    if not files:
        print("No files specified.")
        return

    kwargs = dict(head_cut_ms=head_cut, tail_cut_ms=tail_cut)

    if len(files) == 1:
        SilenceTrimmer().trim_file(files[0], output_path=output, **kwargs)
    else:
        if output:
            if not Path(output).is_dir():
                raise ValueError("--output must be a folder for multiple files")
            kwargs["output_path"] = output
        SilenceTrimmer.trim_parallel(files, jobs=jobs, **kwargs)


if __name__ == "__main__":
    fire.Fire(main)
