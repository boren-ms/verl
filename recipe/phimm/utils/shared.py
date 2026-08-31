from __future__ import annotations

import re
import json
import ast
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


def _mkdir(path: str) -> None:
    if path.startswith("az://"):
        return
    Path(path).expanduser().mkdir(parents=True, exist_ok=True)


def _ensure_can_write(output_path: str, overwrite: bool = False) -> None:
    if bf.exists(output_path) and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Set overwrite=true to replace it.")
    _mkdir_parent(output_path)


def _infer_ext(output_path: str) -> str | None:
    return Path(urlparse(output_path).path).suffix.lstrip(".").lower() or None


def _write_single(dataset: Dataset, path: str, ext: str) -> None:
    with bf.BlobFile(path, "wb") as file_obj:
        if ext == "jsonl":
            dataset.to_json(file_obj, force_ascii=False)
        else:
            dataset.to_parquet(file_obj)


def save_dataset(
    dataset: Dataset,
    output_path: str,
    overwrite: bool = False,
    part_size: int | None = None,
    ext: str | None = None,
) -> None:
    """Save a HF Dataset to disk (local or az://).

    Args:
        dataset: The dataset to save.
        output_path: A file path (.jsonl/.parquet) or a folder path.
        overwrite: Whether to overwrite existing files.
        part_size: If set, split into parts of this many rows.
            When *output_path* is a file, parts are saved next to it as
            ``<stem>-00000-of-NNNNN.<ext>``.  When it is a folder, parts
            are written inside it.
        ext: Output format when *output_path* is a folder (``jsonl`` or
            ``parquet``).  Ignored when *output_path* already has a
            recognised extension.
    """
    suffix = _infer_ext(output_path)
    is_folder = suffix not in {"jsonl", "parquet"}

    if is_folder:
        ext = (ext or "jsonl").lower()
        if ext not in {"jsonl", "parquet"}:
            raise ValueError(f"Unsupported ext={ext!r}. Use 'jsonl' or 'parquet'.")
    else:
        ext = suffix

    if part_size is None or part_size <= 0 or len(dataset) <= part_size:
        # ---- single-file output ----
        if is_folder:
            output_file = f"{output_path.rstrip('/')}/data.{ext}"
        else:
            output_file = output_path
        _ensure_can_write(output_file, overwrite=overwrite)
        _write_single(dataset, output_file, ext)
        return

    # ---- multi-part output ----
    n_parts = (len(dataset) + part_size - 1) // part_size
    if is_folder:
        _mkdir(output_path)
    for i in range(n_parts):
        shard = dataset.select(range(i * part_size, min((i + 1) * part_size, len(dataset))))
        if is_folder:
            part_path = f"{output_path.rstrip('/')}/part-{i:05d}-of-{n_parts:05d}.{ext}"
        else:
            stem = Path(urlparse(output_path).path).stem
            parent = output_path[: output_path.rfind(stem)]
            part_path = f"{parent}{stem}-{i:05d}-of-{n_parts:05d}.{ext}"
        _ensure_can_write(part_path, overwrite=overwrite)
        _write_single(shard, part_path, ext)
        print(f"  Wrote part {i + 1}/{n_parts}: {part_path} ({len(shard)} rows)")


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


BRACKET_PATTERN = re.compile(
    r"<[^>]*>|\[[^\]]*\]|\{[^}]*\}|\([^)]*\)|"
    r"（[^）]*）|［[^］]*］|｛[^｝]*｝|＜[^＞]*＞|"
    r"【[^】]*】|〔[^〕]*〕|〈[^〉]*〉|《[^》]*》|「[^」]*」|『[^』]*』|〖[^〗]*〗|｟[^｠]*｠"
)
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


def has_tail_hallucination(hyp, ref=None, min_words=3, tn_name=None, lang="english"):
    """Return True if ``hyp`` ends with hallucinated insertions vs ``ref``.

    Aligns ``ref`` vs ``hyp`` at the word level. Returns True when the final
    alignment opcode is ``insert`` (or ``replace`` whose hyp side extends to
    the end of hyp) and the number of trailing hyp-only words is
    ``>= min_words``. Returns False when either side is empty. If ``tn_name``
    is given, both inputs are passed through ``text_norm(..., tn_name)`` first;
    if ``tn_name`` is None but ``lang`` is given, ``default_tn_name(lang)`` is
    used.
    """
    if not hyp or not ref:
        return False
    from recipe.phimm.utils.tn import default_tn_name, text_norm

    tn_name = tn_name or default_tn_name(lang)
    hyp = text_norm(hyp, tn_name)
    ref = text_norm(ref, tn_name)
    ref_words = ref.split()
    hyp_words = hyp.split()
    if not hyp_words or not ref_words:
        return False
    from difflib import SequenceMatcher

    sm = SequenceMatcher(None, ref_words, hyp_words, autojunk=False)
    opcodes = sm.get_opcodes()
    if not opcodes:
        return False
    tag, i1, i2, j1, j2 = opcodes[-1]
    if tag not in ("insert", "replace"):
        return False
    if j2 != len(hyp_words):
        return False
    n_extra = (j2 - j1) - (i2 - i1)
    return n_extra >= min_words


def to_keyword_list(raw_keywords):
    """Normalize a keyword field into a list of strings.

    Supports list/tuple/set, JSON/python-literal serialized strings, and scalars.
    """
    if raw_keywords is None:
        return []
    if is_list(raw_keywords) or isinstance(raw_keywords, (tuple, set)):
        return [str(x) for x in raw_keywords]
    if isinstance(raw_keywords, str):
        s = raw_keywords.strip()
        if not s:
            return []
        for parse_fn in (json.loads, ast.literal_eval):
            try:
                parsed = parse_fn(s)
            except Exception:
                continue
            if is_list(parsed) or isinstance(parsed, (tuple, set)):
                return [str(x) for x in parsed]
        return [s]
    return [str(raw_keywords)]


def has_missing_keyword(keywords, response, norm_name=None, lang="english"):
    """Return True when any normalized keyword is absent in normalized response.

    If ``norm_name`` is not provided, falls back to ``default_tn_name(lang)``.
    """
    from recipe.phimm.utils.tn import default_tn_name, text_norm

    norm_name = norm_name or default_tn_name(lang)

    keywords = to_keyword_list(keywords)
    if not keywords:
        return False

    norm_response = text_norm(str(response or ""), norm_name)
    for kw in keywords:
        norm_kw = text_norm(str(kw), norm_name).strip()
        if not norm_kw:
            continue
        if re.search(re.escape(norm_kw), norm_response) is None:
            return True
    return False


def parse_asr_response(response):
    """Extract text and language from an ASR response string.

    Supports formats like:
        <src=English><tgt=English>\nsome text
        <ASR_LEXICAL><lang=English><TXT>some text</TXT></ASR_LEXICAL>
        <ASR><lang=English><TXT>some text</TXT></ASR>
        simple text

    Returns a dict with 'text' and 'lang' keys.
    If no match is found, returns {'text': response, 'lang': None}.
    """
    m = re.search(r"<src=([^>]+)>", response) or re.search(r"<lang=([^>]+)>", response)
    lang = m.group(1) if m else None
    text, formatted = _parse_transcription(response)
    # new_text = strip_repetitions(text)
    return {"text": text, "lang": lang, "formatted": formatted}


def _parse_transcription(raw_text):
    """Extract clean transcription from model output.

    Returns (text, formatted) where formatted is True if a known task format
    was found.
    """
    task_segments = re.findall(
        r"(?:\A|\n)<src=[^>\n]+><tgt=[^>\n]+>\n(.*?)(?=\n<src=|\Z)",
        raw_text.strip(),
        re.DOTALL,
    )
    if task_segments:
        return " ".join(text.strip() for text in task_segments), True
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
