#!/usr/bin/env python3
"""
Detailed word-level error analysis for a single ASR JSONL file.

Produces:
  - summary.json          - dataset-level WER, error counts, top confusion pairs
  - error_details.csv     - per-utterance error breakdown with aligned ops
  - substitutions.csv     - ranked substitution confusion pairs
  - deletions.csv         - ranked deleted-word frequencies
  - insertions.csv        - ranked inserted-word frequencies
  - error_patterns.csv    - error rate bucketed by utterance ref-word count
  - alignment_samples.txt - human-readable alignment for the top-N worst utterances
  - report.html (opt)     - standalone visual report
"""
from __future__ import annotations

import argparse
import collections
import csv
import html
import json
import mimetypes
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import blobfile as bf
import ftfy
from whisper_normalizer.english import EnglishTextNormalizer

REPO_ROOT = Path(__file__).resolve().parents[4]
if (REPO_ROOT / "recipe").is_dir() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.phimm.utils.languages import get_language_code
from recipe.phimm.utils.open_asr_normalizer.eval_utils import normalize_for_wer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate detailed word-level error analysis from a single ASR JSONL file."
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-path", help="Path to ASR JSONL file (local or az://).")
    source.add_argument("--model", help="Model directory name for local, val_data_gen, or result_details auto-discovery.")
    p.add_argument("--dataset", default="", help="Dataset name (used in file discovery and output labeling).")
    p.add_argument(
        "--results-root",
        default="az://orngwus2cresco/data/boren/data/results/gpt-4o-mini-asr-v1",
        help="Root that contains <model>/<dataset>/result_details_*.jsonl.",
    )
    p.add_argument(
        "--val-data-root",
        default="az://orngwus2cresco/data/boren/outputs",
        help=(
            "Root for verl validation outputs. Layout: "
            "<root>/<project>/<experiment>/val_data_gen/<dataset>/<step>.jsonl."
        ),
    )
    p.add_argument("--ref-column", default="ref", help="Column name for reference text.")
    p.add_argument("--hyp-column", default="hyp", help="Column name for hypothesis text.")
    p.add_argument("--id-column", default="", help="Column for utterance ID (auto-detected if empty).")
    p.add_argument("--output-dir", default="tmp/asr-word-error-analysis", help="Where to write artifacts.")
    p.add_argument("--top-n", type=int, default=50, help="Number of worst utterances shown in alignment samples.")
    p.add_argument("--top-confusions", type=int, default=100, help="Number of top substitution pairs to report.")
    p.add_argument("--no-html", action="store_true", help="Disable the standalone HTML report (enabled by default).")
    p.add_argument("--raw-output-column", default="output", help="Column containing the raw model output to include in HTML report (default: 'output').")
    p.add_argument("--length-bucket-size", type=int, default=5, help="Ref-word-count bucket width for error_patterns.csv.")
    p.add_argument(
        "--normalizer",
        choices=("english", "openasr"),
        default="english",
        help="Text normalizer used before alignment. Use 'openasr' to match measure_wer.",
    )
    p.add_argument("--lang", default="", help="Language code/name override for --normalizer openasr.")
    p.add_argument("--lang-column", default="language", help="Row column with language name/code for --normalizer openasr.")
    p.add_argument("--case-sensitive", action="store_true", help="Do not lowercase ref/hyp before alignment.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

TIMESTAMP_RE = re.compile(r"result_details_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.jsonl$")
STEP_RE = re.compile(r"(\d+)\.jsonl$")
LOCAL_RESULTS_ROOTS = [
    Path("tmp"),
    Path.home() / "data" / "results" / "verl_word_error",
]
ID_CANDIDATES = [
    "audio_file_stem", "audio_file", "utt_id", "utterance_id", "example_id",
    "item_id", "segment_id", "id", "key", "audio_path", "path",
]
META_PROMOTED = [
    "audio_file", "audio_path", "audio_length_s", "dataset", "duration",
    "id", "sampling_rate", "text",
]
AUDIO_PATH_CANDIDATES = [
    "audio_file",
    "audio_path",
    "path",
    "source_path",
    "source",
]
ENGLISH_TEXT_NORMALIZER = EnglishTextNormalizer()


def resolve_latest(results_root: str, model: str, dataset: str) -> str:
    pattern = bf.join(results_root, model, dataset, "result_details_*.jsonl")
    matches = sorted(bf.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No result_details files found: {pattern}")

    def _ts(p: str) -> str:
        m = TIMESTAMP_RE.search(p)
        return m.group(1) if m else ""

    def _mtime(p: str) -> float:
        try:
            return bf.stat(p).mtime
        except Exception:
            return float("-inf")

    ranked = sorted(matches, key=lambda p: (_mtime(p), _ts(p), p))
    return ranked[-1]


def dataset_leaf(dataset: str) -> str:
    return dataset.rsplit("/", 1)[-1] if "/" in dataset else dataset


def extract_step(path: str) -> int:
    match = STEP_RE.search(path)
    return int(match.group(1)) if match else -1


def resolve_val_data_gen(val_data_root: str, model: str, dataset: str) -> str | None:
    ds_bare = dataset_leaf(dataset)
    pattern = bf.join(val_data_root, model, "val_data_gen", ds_bare, "*.jsonl")
    try:
        matches = sorted(bf.glob(pattern))
    except Exception:
        return None
    if not matches:
        return None
    ranked = sorted(matches, key=lambda p: (extract_step(p), p))
    return ranked[-1]


def resolve_local(model: str, dataset: str) -> str | None:
    ds_bare = dataset_leaf(dataset)
    for root in LOCAL_RESULTS_ROOTS:
        flat_path = root / model / f"{ds_bare}.jsonl"
        if flat_path.is_file():
            return str(flat_path)

        result_details_dir = root / model / dataset
        if result_details_dir.is_dir():
            matches = sorted(result_details_dir.glob("result_details_*.jsonl"))
            if matches:
                return str(matches[-1])

        val_data_dir = root / model / "val_data_gen" / ds_bare
        if val_data_dir.is_dir():
            matches = sorted(val_data_dir.glob("*.jsonl"), key=lambda p: (extract_step(str(p)), str(p)))
            if matches:
                return str(matches[-1])
    return None


def resolve_input_path(model: str, dataset: str, results_root: str, val_data_root: str) -> str:
    local_path = resolve_local(model, dataset)
    if local_path:
        return local_path

    val_path = resolve_val_data_gen(val_data_root, model, dataset)
    if val_path:
        return val_path

    return resolve_latest(results_root, model, dataset)


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with bf.BlobFile(path, "r") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON parse error on line {lineno} of {path}: {exc}") from exc
    if not rows:
        raise ValueError(f"No records in {path}")
    # promote nested meta fields
    for row in rows:
        meta = row.get("meta")
        if isinstance(meta, dict):
            for col in META_PROMOTED:
                if col not in row and col in meta:
                    row[col] = meta[col]
    # derive audio_file_stem
    for row in rows:
        if "audio_file" in row and "audio_file_stem" not in row:
            row["audio_file_stem"] = Path(str(row["audio_file"])).stem
    return rows


def remap_verl_schema(rows: list[dict], ref_col: str, hyp_col: str) -> None:
    for row in rows:
        if ref_col not in row and "gts" in row:
            row[ref_col] = row["gts"]
        if hyp_col not in row and "clean_output" in row:
            row[hyp_col] = row["clean_output"]


def detect_id_column(rows: list[dict]) -> str:
    keys = set(rows[0].keys()) if rows else set()
    for cand in ID_CANDIDATES:
        if cand in keys:
            vals = [r.get(cand) for r in rows]
            if len(set(vals)) == len(rows):
                return cand
    return ""


def clean_text(text: str) -> str:
    text = ftfy.fix_text(text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text: str) -> str:
    text = clean_text(text)
    text = ENGLISH_TEXT_NORMALIZER(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def infer_language(row: dict, lang_override: str = "", lang_column: str = "language") -> str:
    source_lang = lang_override or str(row.get(lang_column, "") or "")
    if not source_lang:
        data_source = str(row.get("data_source", "") or "")
        source_lang = data_source.rsplit("_", 1)[-1] if "_" in data_source else ""
    source_lang = source_lang.strip().lower()
    return get_language_code(source_lang or "en")


def normalize_pair(ref: str, hyp: str, row: dict, normalizer: str, lang_override: str, lang_column: str) -> tuple[str, str]:
    if normalizer == "openasr":
        lang_code = infer_language(row, lang_override=lang_override, lang_column=lang_column)
        hyp_norm, ref_norm = normalize_for_wer(clean_text(hyp), clean_text(ref), lang=lang_code)
        return ref_norm, hyp_norm
    return normalize_text(ref), normalize_text(hyp)


# ---------------------------------------------------------------------------
# Edit-distance alignment
# ---------------------------------------------------------------------------

@dataclass
class AlignOp:
    op: str        # "ok" | "sub" | "del" | "ins"
    ref_word: str  # "" for insertions
    hyp_word: str  # "" for deletions


@dataclass
class AlignResult:
    ops: list[AlignOp] = field(default_factory=list)
    ref_words: int = 0
    hyp_words: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        return self.errors / self.ref_words if self.ref_words else 0.0


def align(ref_text: str, hyp_text: str, *, case_sensitive: bool = False) -> AlignResult:
    ref_tokens = ref_text.split()
    hyp_tokens = hyp_text.split()
    n, m = len(ref_tokens), len(hyp_tokens)

    # DP table
    d = [[0] * (m + 1) for _ in range(n + 1)]
    bt: list[list[tuple[int, int, str] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
        bt[i][0] = (i - 1, 0, "del")
    for j in range(1, m + 1):
        d[0][j] = j
        bt[0][j] = (0, j - 1, "ins")
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                d[i][j] = d[i - 1][j - 1]
                bt[i][j] = (i - 1, j - 1, "ok")
                continue
            sub = d[i - 1][j - 1] + 1
            dl = d[i - 1][j] + 1
            ins = d[i][j - 1] + 1
            best = min(sub, dl, ins)
            d[i][j] = best
            if best == sub:
                bt[i][j] = (i - 1, j - 1, "sub")
            elif best == dl:
                bt[i][j] = (i - 1, j, "del")
            else:
                bt[i][j] = (i, j - 1, "ins")

    # backtrace
    ops: list[AlignOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        cell = bt[i][j]
        if cell is None:
            break
        pi, pj, tag = cell
        if tag == "ok":
            ops.append(AlignOp("ok", ref_tokens[i - 1], hyp_tokens[j - 1]))
        elif tag == "sub":
            ops.append(AlignOp("sub", ref_tokens[i - 1], hyp_tokens[j - 1]))
        elif tag == "del":
            ops.append(AlignOp("del", ref_tokens[i - 1], ""))
        else:
            ops.append(AlignOp("ins", "", hyp_tokens[j - 1]))
        i, j = pi, pj
    ops.reverse()

    res = AlignResult(ops=ops, ref_words=n, hyp_words=m)
    for op in ops:
        if op.op == "sub":
            res.substitutions += 1
        elif op.op == "del":
            res.deletions += 1
        elif op.op == "ins":
            res.insertions += 1
    return res


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def format_alignment(a: AlignResult) -> str:
    """Return a 3-line alignment string: REF / HYP / OPS."""
    ref_parts, hyp_parts, op_parts = [], [], []
    for op in a.ops:
        rw = op.ref_word or "***"
        hw = op.hyp_word or "***"
        width = max(len(rw), len(hw), 3)
        ref_parts.append(rw.ljust(width))
        hyp_parts.append(hw.ljust(width))
        label = {"ok": " ", "sub": "S", "del": "D", "ins": "I"}[op.op]
        op_parts.append(label.center(width))
    return (
        "REF: " + " ".join(ref_parts) + "\n"
        "HYP: " + " ".join(hyp_parts) + "\n"
        "OPS: " + " ".join(op_parts)
    )


@dataclass
class UtteranceResult:
    idx: int
    utt_id: str
    ref: str
    hyp: str
    ref_norm: str
    hyp_norm: str
    alignment: AlignResult
    row: dict


def analyze_all(
    rows: list[dict],
    ref_col: str,
    hyp_col: str,
    id_col: str,
    case_sensitive: bool,
    normalizer: str,
    lang_override: str,
    lang_column: str,
) -> list[UtteranceResult]:
    results: list[UtteranceResult] = []
    for i, row in enumerate(rows):
        ref = str(row.get(ref_col, "")).strip()
        hyp = str(row.get(hyp_col, "")).strip()
        uid = str(row.get(id_col, i)) if id_col else str(i)
        if case_sensitive:
            ref_norm = clean_text(ref)
            hyp_norm = clean_text(hyp)
        else:
            ref_norm, hyp_norm = normalize_pair(ref, hyp, row, normalizer, lang_override, lang_column)
        a = align(ref_norm, hyp_norm, case_sensitive=case_sensitive)
        results.append(
            UtteranceResult(
                idx=i,
                utt_id=uid,
                ref=ref,
                hyp=hyp,
                ref_norm=ref_norm,
                hyp_norm=hyp_norm,
                alignment=a,
                row=row,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

SubPair = tuple[str, str]


def aggregate(results: list[UtteranceResult]) -> dict:
    total_ref = sum(r.alignment.ref_words for r in results)
    total_errors = sum(r.alignment.errors for r in results)
    total_sub = sum(r.alignment.substitutions for r in results)
    total_del = sum(r.alignment.deletions for r in results)
    total_ins = sum(r.alignment.insertions for r in results)

    sub_pairs: collections.Counter[SubPair] = collections.Counter()
    del_words: collections.Counter[str] = collections.Counter()
    ins_words: collections.Counter[str] = collections.Counter()

    for r in results:
        for op in r.alignment.ops:
            if op.op == "sub":
                sub_pairs[(op.ref_word, op.hyp_word)] += 1
            elif op.op == "del":
                del_words[op.ref_word] += 1
            elif op.op == "ins":
                ins_words[op.hyp_word] += 1

    return {
        "total_utterances": len(results),
        "total_ref_words": total_ref,
        "total_errors": total_errors,
        "total_substitutions": total_sub,
        "total_deletions": total_del,
        "total_insertions": total_ins,
        "wer": total_errors / total_ref if total_ref else 0.0,
        "substitution_rate": total_sub / total_ref if total_ref else 0.0,
        "deletion_rate": total_del / total_ref if total_ref else 0.0,
        "insertion_rate": total_ins / total_ref if total_ref else 0.0,
        "sub_pairs": sub_pairs,
        "del_words": del_words,
        "ins_words": ins_words,
    }


def bucket_errors(results: list[UtteranceResult], bucket_size: int) -> list[dict]:
    buckets: dict[int, list[UtteranceResult]] = collections.defaultdict(list)
    for r in results:
        b = (r.alignment.ref_words // bucket_size) * bucket_size
        buckets[b].append(r)
    rows = []
    for b in sorted(buckets):
        group = buckets[b]
        ref_sum = sum(r.alignment.ref_words for r in group)
        err_sum = sum(r.alignment.errors for r in group)
        rows.append({
            "ref_word_bucket": f"{b}-{b + bucket_size - 1}",
            "utterances": len(group),
            "total_ref_words": ref_sum,
            "total_errors": err_sum,
            "bucket_wer": err_sum / ref_sum if ref_sum else 0.0,
        })
    return rows


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_summary(agg: dict, output_dir: Path, dataset: str, input_path: str) -> None:
    obj = {
        "dataset": dataset,
        "input_path": input_path,
        "total_utterances": agg["total_utterances"],
        "total_ref_words": agg["total_ref_words"],
        "total_errors": agg["total_errors"],
        "total_substitutions": agg["total_substitutions"],
        "total_deletions": agg["total_deletions"],
        "total_insertions": agg["total_insertions"],
        "wer": round(agg["wer"], 6),
        "substitution_rate": round(agg["substitution_rate"], 6),
        "deletion_rate": round(agg["deletion_rate"], 6),
        "insertion_rate": round(agg["insertion_rate"], 6),
        "top_substitution_pairs": [
            {"ref_word": rw, "hyp_word": hw, "count": c}
            for (rw, hw), c in agg["sub_pairs"].most_common(30)
        ],
        "top_deleted_words": [
            {"word": w, "count": c}
            for w, c in agg["del_words"].most_common(30)
        ],
        "top_inserted_words": [
            {"word": w, "count": c}
            for w, c in agg["ins_words"].most_common(30)
        ],
    }
    path = output_dir / "summary.json"
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    print(f"  summary          -> {path}")


def write_error_details(results: list[UtteranceResult], agg: dict, output_dir: Path) -> None:
    total_ref = agg["total_ref_words"]
    ranked = sorted(results, key=lambda r: r.alignment.errors, reverse=True)
    path = output_dir / "error_details.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "rank", "utt_id", "ref", "hyp", "ref_norm", "hyp_norm", "ref_words", "errors",
            "substitutions", "deletions", "insertions", "wer",
            "wer_contribution", "alignment_ops",
        ])
        for rank, r in enumerate(ranked, 1):
            a = r.alignment
            ops_str = " ".join(
                f"{op.op}({op.ref_word or '-'}/{op.hyp_word or '-'})"
                if op.op != "ok" else op.ref_word
                for op in a.ops
            )
            writer.writerow([
                rank, r.utt_id, r.ref, r.hyp, r.ref_norm, r.hyp_norm, a.ref_words, a.errors,
                a.substitutions, a.deletions, a.insertions,
                round(a.wer, 6),
                round(a.errors / total_ref, 8) if total_ref else 0.0,
                ops_str,
            ])
    print(f"  error_details    -> {path}")


def write_substitutions(agg: dict, output_dir: Path, top_n: int) -> None:
    path = output_dir / "substitutions.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "ref_word", "hyp_word", "count"])
        for rank, ((rw, hw), c) in enumerate(agg["sub_pairs"].most_common(top_n), 1):
            writer.writerow([rank, rw, hw, c])
    print(f"  substitutions    -> {path}")


def write_deletions(agg: dict, output_dir: Path, top_n: int) -> None:
    path = output_dir / "deletions.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "word", "count"])
        for rank, (w, c) in enumerate(agg["del_words"].most_common(top_n), 1):
            writer.writerow([rank, w, c])
    print(f"  deletions        -> {path}")


def write_insertions(agg: dict, output_dir: Path, top_n: int) -> None:
    path = output_dir / "insertions.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "word", "count"])
        for rank, (w, c) in enumerate(agg["ins_words"].most_common(top_n), 1):
            writer.writerow([rank, w, c])
    print(f"  insertions       -> {path}")


def write_error_patterns(results: list[UtteranceResult], output_dir: Path, bucket_size: int) -> None:
    rows = bucket_errors(results, bucket_size)
    path = output_dir / "error_patterns.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ref_word_bucket", "utterances", "total_ref_words", "total_errors", "bucket_wer"])
        for row in rows:
            writer.writerow([
                row["ref_word_bucket"], row["utterances"],
                row["total_ref_words"], row["total_errors"],
                round(row["bucket_wer"], 6),
            ])
    print(f"  error_patterns   -> {path}")


def write_alignment_samples(results: list[UtteranceResult], output_dir: Path, top_n: int) -> None:
    ranked = sorted(results, key=lambda r: r.alignment.errors, reverse=True)
    path = output_dir / "alignment_samples.txt"
    lines: list[str] = []
    for rank, r in enumerate(ranked[:top_n], 1):
        lines.append(
            f"=== Rank {rank} | utt_id={r.utt_id} | errors={r.alignment.errors} | WER={r.alignment.wer:.4f} ==="
        )
        lines.append(format_alignment(r.alignment))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  alignment_samples-> {path}")


def slugify(value: str) -> str:
    cleaned = [char.lower() if char.isalnum() else "-" for char in value]
    slug = "".join(cleaned).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "item"


def normalize_optional_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def get_audio_source(row: dict) -> str:
    for column in AUDIO_PATH_CANDIDATES:
        candidate = normalize_optional_text(row.get(column))
        if candidate:
            return candidate
    return ""


def guess_audio_extension(audio_source: str) -> str:
    suffix = Path(audio_source).suffix
    if suffix:
        return suffix
    return ".wav"


def copy_blobfile(src_path: str, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with bf.BlobFile(src_path, "rb") as src, dst_path.open("wb") as dst:
        while True:
            chunk = src.read(8 * 1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)


def download_ranked_audio(results: list[UtteranceResult], output_dir: Path, top_n: int) -> dict[int, Path]:
    ranked = sorted(results, key=lambda r: r.alignment.errors, reverse=True)[:top_n]
    audio_dir = output_dir / "audio"
    local_audio_by_idx: dict[int, Path] = {}
    downloaded_by_source: dict[str, Path] = {}
    for rank, result in enumerate(ranked, 1):
        audio_source = get_audio_source(result.row)
        if not audio_source:
            continue
        existing = downloaded_by_source.get(audio_source)
        if existing is None:
            stem = normalize_optional_text(result.row.get("audio_file_stem")) or result.utt_id
            filename = f"rank-{rank:03d}-{slugify(stem)}{guess_audio_extension(audio_source)}"
            destination = audio_dir / filename
            try:
                copy_blobfile(audio_source, destination)
            except (FileNotFoundError, OSError) as exc:
                print(f"  [warn] Could not download audio {audio_source}: {exc}")
                continue
            downloaded_by_source[audio_source] = destination
            existing = destination
        local_audio_by_idx[result.idx] = existing
    return local_audio_by_idx


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    return html.escape(text)


def _render_alignment_html(a: AlignResult) -> str:
    """Render alignment as flowing text with only errors highlighted inline."""
    parts: list[str] = []
    for op in a.ops:
        if op.op == "ok":
            parts.append(_esc(op.ref_word))
        elif op.op == "sub":
            parts.append(
                f'<span class="err-sub" title="sub: {_esc(op.ref_word)} → {_esc(op.hyp_word)}">'
                f'{_esc(op.ref_word)}</span>'
                f'<span class="hyp-inline">[→{_esc(op.hyp_word)}]</span>'
            )
        elif op.op == "del":
            parts.append(
                f'<span class="err-del" title="deleted">{_esc(op.ref_word)}</span>'
            )
        else:  # ins
            parts.append(
                f'<span class="err-ins" title="inserted">+{_esc(op.hyp_word)}</span>'
            )
    return " ".join(parts)


def _render_error_summary_table(a: AlignResult) -> str:
    """Build a small per-utterance error breakdown table for the details section."""
    sub_pairs: collections.Counter[tuple[str, str]] = collections.Counter()
    del_words: collections.Counter[str] = collections.Counter()
    ins_words: collections.Counter[str] = collections.Counter()
    for op in a.ops:
        if op.op == "sub":
            sub_pairs[(op.ref_word, op.hyp_word)] += 1
        elif op.op == "del":
            del_words[op.ref_word] += 1
        elif op.op == "ins":
            ins_words[op.hyp_word] += 1
    rows: list[str] = []
    for (rw, hw), c in sub_pairs.most_common(10):
        rows.append(f"<tr><td>Sub</td><td>{_esc(rw)}</td><td>{_esc(hw)}</td><td>{c}</td></tr>")
    for w, c in del_words.most_common(10):
        rows.append(f"<tr><td>Del</td><td>{_esc(w)}</td><td>&mdash;</td><td>{c}</td></tr>")
    for w, c in ins_words.most_common(10):
        rows.append(f"<tr><td>Ins</td><td>&mdash;</td><td>{_esc(w)}</td><td>{c}</td></tr>")
    if not rows:
        return "<p>No errors in this utterance.</p>"
    return (
        '<table class="err-table"><thead><tr>'
        '<th>Type</th><th>Ref</th><th>Hyp</th><th>Count</th>'
        '</tr></thead><tbody>'
        + "\n".join(rows)
        + '</tbody></table>'
    )


def render_html(
    results: list[UtteranceResult],
    agg: dict,
    output_dir: Path,
    dataset: str,
    input_path: str,
    top_n: int,
    raw_output_column: str = "",
) -> None:
    ranked = sorted(results, key=lambda r: r.alignment.errors, reverse=True)[:top_n]
    local_audio_by_idx = download_ranked_audio(results, output_dir, top_n)
    wer_pct = agg["wer"] * 100
    sub_rate = agg["substitution_rate"] * 100
    del_rate = agg["deletion_rate"] * 100
    ins_rate = agg["insertion_rate"] * 100

    # Confusion table
    confusions_html = "\n".join(
        f"<tr><td>{_esc(rw)}</td><td>{_esc(hw)}</td><td>{c}</td></tr>"
        for (rw, hw), c in agg["sub_pairs"].most_common(30)
    )

    # Top deletions table
    deletions_html = "\n".join(
        f"<tr><td>{_esc(w)}</td><td>{c}</td></tr>"
        for w, c in agg["del_words"].most_common(20)
    )

    # Top insertions table
    insertions_html = "\n".join(
        f"<tr><td>{_esc(w)}</td><td>{c}</td></tr>"
        for w, c in agg["ins_words"].most_common(20)
    )

    # Build per-utterance cards
    cards: list[str] = []
    for rank, r in enumerate(ranked, 1):
        audio_source = get_audio_source(r.row)
        local_audio = local_audio_by_idx.get(r.idx)
        audio_html = ""
        if local_audio is not None:
            rel_audio = local_audio.relative_to(output_dir).as_posix()
            mime_type, _ = mimetypes.guess_type(local_audio.name)
            source_attr = f' type="{mime_type}"' if mime_type else ""
            audio_html = (
                f'<div class="audio-block">'
                f'<audio controls preload="none">'
                f'<source src="{_esc(rel_audio)}"{source_attr}>'
                f"</audio>"
                f'<div class="audio-meta">Local: <a href="{_esc(rel_audio)}">{_esc(local_audio.name)}</a></div>'
                f"</div>"
            )
        elif audio_source:
            audio_html = f'<div class="audio-meta">Audio: <code>{_esc(audio_source)}</code> (unavailable)</div>'

        transcript_html = _render_alignment_html(r.alignment)
        err_table_html = _render_error_summary_table(r.alignment)

        raw_output_html = ""
        if raw_output_column:
            raw_val = str(r.row.get(raw_output_column, "")).strip()
            if raw_val:
                raw_output_html = (
                    f'  <details>\n'
                    f'    <summary>Raw model output</summary>\n'
                    f'    <pre class="raw-output">{_esc(raw_val)}</pre>\n'
                    f'  </details>'
                )

        cards.append(
            f"""<article class="card" id="utt-{rank}">
  <header>
    <span class="rank">#{rank}</span>
    <span class="utt-id">{_esc(r.utt_id)}</span>
    <span class="stats">
      {r.alignment.errors} errors &bull; WER {r.alignment.wer:.2%}
      &bull; S:{r.alignment.substitutions} D:{r.alignment.deletions} I:{r.alignment.insertions}
      &bull; {r.alignment.ref_words:,} ref words
    </span>
  </header>
  {audio_html}
  <details open>
    <summary>Transcript with error highlighting</summary>
    <div class="transcript">{transcript_html}</div>
  </details>
  {raw_output_html}
  <details>
    <summary>Error breakdown ({r.alignment.errors} errors)</summary>
    {err_table_html}
  </details>
</article>"""
        )
    cards_html = "\n".join(cards)

    # Navigation sidebar
    nav_items = "\n".join(
        f'<a href="#utt-{i+1}" class="nav-item">'
        f'<span class="nav-rank">#{i+1}</span> '
        f'<span class="nav-err">{r.alignment.errors} err</span> '
        f'<span class="nav-wer">{r.alignment.wer:.1%}</span>'
        f'</a>'
        for i, r in enumerate(ranked)
    )

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Word Error Analysis - {_esc(dataset)}</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;font-family:system-ui,-apple-system,sans-serif;background:#f7f8fa;color:#1a1a1a;line-height:1.6}}
.layout{{display:flex;min-height:100vh}}
.sidebar{{width:240px;background:#fff;border-right:1px solid #e0e0e0;padding:12px 0;position:sticky;top:0;height:100vh;overflow-y:auto;flex-shrink:0}}
.sidebar h3{{padding:8px 16px;margin:0;font-size:.85rem;color:#666;text-transform:uppercase;letter-spacing:.05em}}
.nav-item{{display:block;padding:6px 16px;text-decoration:none;color:#333;font-size:.8rem;border-left:3px solid transparent;transition:background .15s}}
.nav-item:hover{{background:#f0f4ff;border-left-color:#4a90d9}}
.nav-rank{{font-weight:700;color:#4a90d9}} .nav-err{{color:#c33;font-size:.75rem}} .nav-wer{{color:#666;font-size:.75rem}}
main{{flex:1;max-width:1100px;padding:24px 32px;overflow-x:hidden}}
h1{{font-size:1.5rem;margin:0 0 4px}} .subtitle{{color:#666;font-size:.9rem;margin-bottom:20px;word-break:break-all}}
h2{{font-size:1.15rem;margin-top:2rem;padding-bottom:6px;border-bottom:1px solid #e0e0e0}}
.kpi{{display:flex;gap:14px;flex-wrap:wrap;margin:16px 0}}
.kpi>div{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px 18px;min-width:130px}}
.kpi .val{{font-size:1.4rem;font-weight:700}} .kpi .lbl{{font-size:.8rem;color:#666}}
.kpi .highlight{{border-color:#4a90d9;background:#f0f4ff}}
table.data{{border-collapse:collapse;margin:12px 0;width:100%}}
table.data th,table.data td{{border:1px solid #ddd;padding:6px 10px;text-align:left;font-size:.85rem}}
table.data th{{background:#f8f8f8;position:sticky;top:0}}
table.data tr:hover{{background:#fafafa}}
.tables-row{{display:flex;gap:24px;flex-wrap:wrap}}
.tables-row>div{{flex:1;min-width:200px}}
.card{{background:#fff;border:1px solid #ddd;border-radius:10px;margin:16px 0;overflow:hidden}}
.card header{{padding:14px 18px;background:#f8f9fb;border-bottom:1px solid #eee;display:flex;align-items:center;gap:12px;flex-wrap:wrap}}
.card .rank{{font-size:1.1rem;font-weight:700;color:#4a90d9}}
.card .utt-id{{font-family:monospace;font-size:.82rem;color:#555}}
.card .stats{{font-size:.8rem;color:#666;margin-left:auto}}
details{{padding:0}}
summary{{padding:10px 18px;cursor:pointer;font-weight:600;font-size:.9rem;background:#fafbfc;border-top:1px solid #eee;user-select:none}}
summary:hover{{background:#f0f4ff}}
.transcript{{padding:16px 18px;font-size:.9rem;line-height:1.9;white-space:pre-wrap;word-wrap:break-word;max-height:600px;overflow-y:auto}}
.audio-block{{padding:8px 18px}}
.audio-block audio{{width:min(520px,100%)}}
.audio-meta{{font-size:.82rem;color:#555;padding:4px 18px;overflow-wrap:anywhere}}
.err-sub{{background:#fff3cd;border-radius:3px;padding:1px 2px;border-bottom:2px solid #fd7e14;text-decoration:line-through;opacity:.75}}
.err-del{{background:#f8d7da;border-radius:3px;padding:1px 2px;border-bottom:2px solid #dc3545;text-decoration:line-through;opacity:.7}}
.err-ins{{background:#d6ecff;border-radius:3px;padding:1px 2px;border-bottom:2px solid #007bff;font-style:italic}}
.hyp-inline{{color:#c33;font-size:.8rem;font-style:italic}}
.err-table{{width:100%;border-collapse:collapse;margin:8px 0}}
.err-table th,.err-table td{{border:1px solid #eee;padding:5px 8px;font-size:.82rem;text-align:left}}
.err-table th{{background:#f8f8f8}}
.raw-output{{padding:12px 18px;font-size:.82rem;background:#f9f9f9;border:1px solid #eee;border-radius:4px;margin:8px 18px;white-space:pre-wrap;word-wrap:break-word;max-height:400px;overflow-y:auto;font-family:monospace;line-height:1.5}}
@media(max-width:900px){{
  .sidebar{{display:none}}
  main{{padding:16px}}
  .tables-row{{flex-direction:column}}
}}
</style></head>
<body>
<div class="layout">
<nav class="sidebar">
  <h3>Utterances</h3>
  {nav_items}
</nav>
<main>
<h1>Word Error Analysis: {_esc(dataset)}</h1>
<div class="subtitle">Input: {_esc(input_path)}</div>
<div class="kpi">
  <div class="highlight"><div class="val">{wer_pct:.2f}%</div><div class="lbl">WER</div></div>
  <div><div class="val">{agg['total_utterances']:,}</div><div class="lbl">Utterances</div></div>
  <div><div class="val">{agg['total_ref_words']:,}</div><div class="lbl">Ref Words</div></div>
  <div><div class="val">{agg['total_errors']:,}</div><div class="lbl">Total Errors</div></div>
  <div><div class="val">{agg['total_substitutions']:,}</div><div class="lbl">Sub ({sub_rate:.1f}%)</div></div>
  <div><div class="val">{agg['total_deletions']:,}</div><div class="lbl">Del ({del_rate:.1f}%)</div></div>
  <div><div class="val">{agg['total_insertions']:,}</div><div class="lbl">Ins ({ins_rate:.1f}%)</div></div>
</div>

<h2>Top Substitution Confusions</h2>
<table class="data"><thead><tr><th>Ref Word</th><th>Hyp Word</th><th>Count</th></tr></thead>
<tbody>{confusions_html}</tbody></table>

<div class="tables-row">
<div>
<h2>Top Deletions</h2>
<table class="data"><thead><tr><th>Word</th><th>Count</th></tr></thead>
<tbody>{deletions_html}</tbody></table>
</div>
<div>
<h2>Top Insertions</h2>
<table class="data"><thead><tr><th>Word</th><th>Count</th></tr></thead>
<tbody>{insertions_html}</tbody></table>
</div>
</div>

<h2>Top {top_n} Worst Utterances (by error count)</h2>
{cards_html}

</main></div>
</body></html>"""

    path = output_dir / "report.html"
    path.write_text(page, encoding="utf-8")
    print(f"  report.html      -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve input
    if args.input_path:
        input_path = args.input_path
    else:
        input_path = resolve_input_path(args.model, args.dataset, args.results_root, args.val_data_root)
    print(f"Loading {input_path} ...")

    rows = load_jsonl(input_path)
    remap_verl_schema(rows, args.ref_column, args.hyp_column)
    print(f"  {len(rows)} utterances loaded")

    # Detect ID column
    id_col = args.id_column or detect_id_column(rows)
    if id_col:
        print(f"  Using ID column: {id_col}")

    # Check required columns
    for col in (args.ref_column, args.hyp_column):
        if col not in rows[0]:
            raise KeyError(f"Column '{col}' not found. Available: {sorted(rows[0].keys())}")

    # Analyze
    results = analyze_all(
        rows,
        args.ref_column,
        args.hyp_column,
        id_col,
        args.case_sensitive,
        args.normalizer,
        args.lang,
        args.lang_column,
    )
    agg = aggregate(results)

    # Print quick summary
    print(f"\n  WER = {agg['wer']:.4%}  ({agg['total_errors']}/{agg['total_ref_words']})")
    print(f"  Sub={agg['total_substitutions']}  Del={agg['total_deletions']}  Ins={agg['total_insertions']}")

    # Write outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting outputs to {output_dir}/")

    write_summary(agg, output_dir, args.dataset, input_path)
    write_error_details(results, agg, output_dir)
    write_substitutions(agg, output_dir, args.top_confusions)
    write_deletions(agg, output_dir, args.top_confusions)
    write_insertions(agg, output_dir, args.top_confusions)
    write_error_patterns(results, output_dir, args.length_bucket_size)
    write_alignment_samples(results, output_dir, args.top_n)

    if not args.no_html:
        render_html(results, agg, output_dir, args.dataset, input_path, args.top_n, args.raw_output_column)

    print("\nDone.")


if __name__ == "__main__":
    main()
