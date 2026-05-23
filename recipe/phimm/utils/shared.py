from __future__ import annotations

import re
import uuid
import blobfile as bf
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse
import functools
import hashlib
import importlib.metadata
from collections.abc import Mapping, Sequence

from datasets import Dataset


def is_package_version(package_name, target_version):
    """Check if the specified package is installed with the target version."""
    try:
        version = importlib.metadata.version(package_name)
        return version == target_version
    except importlib.metadata.PackageNotFoundError:
        return False


def hash_id(d) -> str:
    return hashlib.sha256(str(d).encode("utf-8")).hexdigest()


def to_int(value, default=-1):
    """Convert a value to an integer, if possible."""
    try:
        return int(value)
    except ValueError:
        return default


def to_float(value, default=0.0):
    """Convert a value to a float, if possible."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def to_list(x, default=None):
    """Convert the input to a list."""
    if x is None:
        return default
    return x if is_list(x) else [x]


def to_range(kwargs, key):
    """Get a key from kwargs as a list range, or None if not set."""
    val = kwargs.get(key)
    return to_list(val) if val is not None else None


def in_range(val, val_range):
    """Check if val is within [lo, hi]. Returns True if val_range is None or val is None."""
    if val_range is None or val is None:
        return True
    return val_range[0] <= val <= val_range[-1]


def is_list(obj):
    return not isinstance(obj, (str, bytes)) and isinstance(obj, Sequence)


def unbatch(batch):
    """Convert a dict-of-lists to a list-of-dicts."""
    return [dict(zip(batch.keys(), vals, strict=True)) for vals in zip(*batch.values(), strict=True)]


def chkp_index(name, default=-1):
    """Extract the checkpoint index from a checkpoint directory name."""
    if not name.startswith("checkpoint-"):
        return default
    return to_int(name.split("-")[-1], default)


def find_chkps(model_dir, specified=None):
    """Find all checkpoint directories in the model directory."""
    chkps = [d for d in bf.scandir(model_dir) if d.is_dir and d.name.startswith("checkpoint-")]
    if not chkps:
        return []
    chkps = sorted(chkps, key=lambda d: chkp_index(d.name), reverse=True)
    if specified is None:
        return [chkp.path for chkp in chkps]

    if isinstance(specified, int):
        specified = [specified]

    idxs = [chkp_index(chkp.name) for chkp in reversed(chkps)]  # ascending
    chkp_indices = [i if i >= 0 else idxs[i] for i in map(to_int, specified) if -i <= len(idxs)]
    return [chkp.path for chkp in chkps if chkp_index(chkp.name) in chkp_indices]


def get_config_path(config_path=None):
    """Get the config path from command line arguments or environment variables."""
    if config_path:
        return Path(config_path).resolve()

    print("Find config from sys:", sys.argv)
    if "--config" in sys.argv:
        config_index = sys.argv.index("--config") + 1
        if config_index < len(sys.argv):
            return Path(sys.argv[config_index]).resolve()
    if "--config-name" in sys.argv:
        config_index = sys.argv.index("--config-name") + 1
        if config_index < len(sys.argv):
            return Path(sys.argv[config_index])
    return None


def run_cmd(cmd, cwd=None, check=True):
    """Run a shell command and print it."""
    if is_list(cmd):
        cmd = " ".join(cmd)
    print(f"Running: {cmd}")
    print(f"Working Dir: {cwd}")
    ret = subprocess.run(cmd, shell=True, check=check, cwd=cwd)
    print(f"Cmd: {cmd} returned: {ret.returncode}")
    return ret


def _mkdir_parent(path: str) -> None:
    if path.startswith("az://"):
        return
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)


def _ensure_can_write(output_path: str, overwrite: bool = False) -> None:
    if bf.exists(output_path) and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Set overwrite=true to replace it.")
    _mkdir_parent(output_path)


def save_dataset(dataset: Dataset, output_path: str, overwrite: bool = False) -> None:
    suffix = Path(urlparse(output_path).path).suffix.lstrip(".").lower()
    if suffix not in {"jsonl", "parquet"}:
        raise ValueError(f"Unsupported output path extension: {output_path}. Use a .jsonl or .parquet path.")

    _ensure_can_write(output_path, overwrite=overwrite)
    with bf.BlobFile(output_path, "wb") as file_obj:
        if suffix == "jsonl":
            dataset.to_json(file_obj, force_ascii=False)
        else:
            dataset.to_parquet(file_obj)


@functools.cache
def cache_remote_path(remote_path, local_path=None, cache_dir=None):
    """Sync remote to local."""
    if not remote_path.startswith("az://"):
        return remote_path

    if local_path is None:
        cache_dir = cache_dir or str(Path.home() / ".blobfile")
        local_path = str(Path(cache_dir, str(uuid.uuid4().hex), *Path(remote_path).parts[3:]))
    if not bf.exists(remote_path):
        return None
    if not bf.isdir(remote_path):
        print(f"Syncing file {remote_path} to {local_path} ...")
        bf.copy(remote_path, local_path, overwrite=True)
        return local_path

    print(f"Syncing directory {remote_path} to {local_path} ...")

    cmd = [
        "bbb",
        "sync",
        "--concurrency",
        "64",
        f"{remote_path.rstrip('/')}/",
        f"{local_path.rstrip('/')}/",
    ]
    run_cmd(cmd, check=True)
    return local_path


def is_local_path(path):
    """Check if a path is a local filesystem path."""
    return not any(path.startswith(scheme) for scheme in ["az://", "hdfs://", "s3://"])


def upload_file(local_path, remote_path, overwrite=False):
    """Upload a file from local to remote storage."""
    local_mtime = Path(local_path).stat().st_mtime
    remote_mtime = None
    if bf.exists(remote_path) and not overwrite:
        print(f"Remote file {remote_path} already exists, skipping upload.")
        return
    try:
        if bf.exists(remote_path):
            remote_mtime = bf.stat(remote_path).mtime
    except Exception as e:
        print(f"Could not stat remote file {remote_path}: {e}")
    # Copy if remote does not exist or local is newer
    if remote_mtime is None or local_mtime > remote_mtime:
        print(f"Syncing file {local_path} to {remote_path} (local newer or remote missing)")
        bf.makedirs(bf.dirname(remote_path))
        bf.copy(local_path, remote_path, overwrite=True)
    else:
        print(f"Skipping {local_path}: remote is newer or same.")


def get_value(d, key, default=None):
    if isinstance(d, Mapping) and key in d:
        return d[key]
    keys = key.split(".")
    for k in keys:
        if isinstance(d, Mapping) and k in d:
            d = d[k]
        elif isinstance(d, (list, tuple)) and k.isdigit() and int(k) < len(d):
            d = d[int(k)]
        else:
            return default
    return d


def get_values(lst, key, default=None):
    """Get a list of values from a list of dictionaries using dot notation."""
    return [get_value(d, key, default) for d in lst]


BRACKET_PATTERN = re.compile(r"<[^>]*>|\[[^\]]*\]|\{[^}]*\}|\([^)]*\)")
BRACKET_EXCLUDE = re.compile(r"^<nonspeech>$", re.IGNORECASE)


def has_brackets(text):
    text = "" if text is None else str(text)
    matches = BRACKET_PATTERN.findall(text)
    return any(not BRACKET_EXCLUDE.match(m) for m in matches)


def rank_print(*args, main=True, **kwargs):
    if main and not dist_state().is_main_process:
        return
    rank = dist_state().process_index
    print(f"[{rank}]", *args, **kwargs)


def dist_state():
    from accelerate import PartialState

    return PartialState()


def all_rank_print(*args, **kwargs):
    rank_print(*args, main=False, **kwargs)


def strip_repetitions(text, min_reps=4):
    """Truncate at first 4+ consecutive repeat of any 1-5 word n-gram."""
    words = text.split()
    for ng in range(1, 6):
        for start in range(len(words) - ng * min_reps):
            ngram = tuple(words[start : start + ng])
            reps = 0
            pos = start
            while pos + ng <= len(words) and tuple(words[pos : pos + ng]) == ngram:
                reps += 1
                pos += ng
            if reps >= min_reps:
                return " ".join(words[: start + ng])
    return text


def has_repeat(text, min_reps=4, max_ngram=5):
    """Return True if the text contains any n-gram (size 1..max_ngram) repeated >= min_reps times consecutively.

    Scans the full text, not just the tail.

    Examples (min_reps=4):
        "the the the marketing and and and and and and and and"  -> True (and x8)
        "as we have access to of of of of of of of of of of"     -> True (of x10)
        "we want to grow is it is it is it is it is it is it"    -> True ("is it" x6)
        "foo foo foo foo then more normal words after that here" -> True (foo x4)
        "this is a normal sentence."                              -> False
    """
    words = text.split() if isinstance(text, str) else list(text or [])
    n = len(words)
    if n < min_reps:
        return False
    for ng in range(1, max_ngram + 1):
        if n < ng * min_reps:
            continue
        for start in range(0, n - ng * min_reps + 1):
            ngram = tuple(words[start : start + ng])
            reps = 1
            pos = start + ng
            while pos + ng <= n and tuple(words[pos : pos + ng]) == ngram:
                reps += 1
                pos += ng
            if reps >= min_reps:
                return True
    return False


def has_repeat_error(hyp, ref=None, min_reps=4, max_ngram=5, tn_name=None, lang="english"):
    """Return True if the error portion of ``hyp`` (vs ``ref``) contains a repeated n-gram.

    Aligns ``ref`` vs ``hyp`` at the word level and runs ``has_repeat`` only on
    the hyp segments inside non-equal opcodes (insertions/substitutions). When
    ``ref`` is empty or None, falls back to scanning the full hyp. If ``tn_name``
    is given, both inputs are passed through ``text_norm(..., tn_name)`` first;
    if ``tn_name`` is None but ``lang`` is given, ``default_tn_name(lang)`` is
    used. Otherwise the caller is expected to pre-normalize.
    """
    if not hyp:
        return False
    from recipe.phimm.utils.tn import default_tn_name, text_norm

    tn_name = tn_name or default_tn_name(lang)
    hyp = text_norm(hyp, tn_name)
    ref = text_norm(ref or "", tn_name)
    if not ref:
        return has_repeat(hyp, min_reps=min_reps, max_ngram=max_ngram)
    from difflib import SequenceMatcher

    ref_words = ref.split()
    hyp_words = hyp.split()
    sm = SequenceMatcher(None, ref_words, hyp_words, autojunk=False)
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or j1 == j2:
            continue
        if has_repeat(hyp_words[j1:j2], min_reps=min_reps, max_ngram=max_ngram):
            return True
    return False


def parse_asr_response(response):
    """Extract text and language from an ASR response string.

    Supports formats like:
        <ASR_LEXICAL><lang=English><TXT>some text</TXT></ASR_LEXICAL>
        <ASR><lang=English><TXT>some text</TXT></ASR>
        "Audio Language: English.\n<ASR><lang=English><TXT>I think Andrei will pr
        simple text

    Returns a dict with 'text' and 'lang' keys.
    If no match is found, returns {'text': response, 'lang': None}.
    """
    m = re.search(r"<lang=([^>]+)>", response)
    lang = m.group(1) if m else None
    text, formatted = _parse_transcription(response)
    # new_text = strip_repetitions(text)
    return {"text": text, "lang": lang, "formatted": formatted}


def _parse_transcription(raw_text):
    """Extract clean transcription from model output.

    Returns (text, formatted) where formatted is True if <TXT> tags were found.
    """
    txt_matches = re.findall(r"<TXT>(.*?)</TXT>", raw_text, re.DOTALL)
    if txt_matches:
        return " ".join(m.strip() for m in txt_matches), True
    m = re.search(r"<TXT>(.*)", raw_text, re.DOTALL)
    if m:
        return re.sub(r"</TXT>?(?:</ASR[^>]*>)?$", "", m.group(1)).strip(), False
    m = re.search(r"Transcription:\s*(.*)", raw_text, re.DOTALL)
    if m:
        text = m.group(1).strip()
        text = re.sub(r"<\|end\|>.*", "", text).strip()
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text).strip(), False
    raw_text = re.sub(r"^Audio\s+Language:\s*\w+\s*\n?", "", raw_text).strip()
    if "<|end|>" in raw_text:
        raw_text = raw_text[: raw_text.index("<|end|>")]
    raw_text = re.sub(r"<[^>]+>", " ", raw_text)
    return re.sub(r"\s+", " ", raw_text).strip(), False
