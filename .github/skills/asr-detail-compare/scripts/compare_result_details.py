#!/usr/bin/env python3
import argparse
import html
import json
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Sequence

import blobfile as bf
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if (REPO_ROOT / "recipe").is_dir() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.phimm.utils.languages import get_language_code  # noqa: E402
from recipe.phimm.utils.open_asr_normalizer.eval_utils import normalize_for_wer  # noqa: E402
from recipe.phimm.utils.open_asr_normalizer.hf_english_normalizer import (  # noqa: E402
    _HFEnglishTextNormalizer,
)


JOIN_CANDIDATES = [
    "audio_file_stem",
    "audio_file",
    "utt_id",
    "utterance_id",
    "example_id",
    "item_id",
    "segment_id",
    "id",
    "key",
    "audio_path",
    "path",
    "source_path",
    "source",
]

_english_normalizer = _HFEnglishTextNormalizer()
TIMESTAMP_PATTERN = re.compile(r"result_details_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.jsonl$")
STEP_PATTERN = re.compile(r"(\d+)\.jsonl$")
DEFAULT_RESULTS_ROOT = "az://orngwus2cresco/data/boren/data/results"
DEFAULT_VAL_DATA_ROOT = "az://orngwus2cresco/data/boren/outputs"
META_PROMOTED_COLUMNS = [
    "audio_file",
    "audio_path",
    "audio_length_s",
    "dataset",
    "duration",
    "id",
    "sampling_rate",
    "text",
]


@dataclass
class ErrorStats:
    ref_words: int
    errors: int
    substitutions: int
    deletions: int
    insertions: int

    @property
    def wer(self) -> float:
        return self.errors / max(1, self.ref_words)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two ASR result_details JSONL files on the same dataset, rank the "
            "utterances with the largest contribution to the target model's total WER, "
            "and attach baseline vs target hypotheses."
        )
    )
    baseline_source = parser.add_mutually_exclusive_group(required=True)
    baseline_source.add_argument("--baseline-path", help="Baseline result_details JSONL path.")
    baseline_source.add_argument(
        "--baseline-model",
        help="Baseline model directory name under the results root. The script resolves the latest result_details file.",
    )
    target_source = parser.add_mutually_exclusive_group(required=True)
    target_source.add_argument("--target-path", help="Target result_details JSONL path.")
    target_source.add_argument(
        "--target-model",
        help="Target model directory name under the results root. The script resolves the latest result_details file.",
    )
    parser.add_argument("--baseline-name", help="Short label for the baseline model. Defaults to the model name or path stem.")
    parser.add_argument("--target-name", help="Short label for the target model. Defaults to the model name or path stem.")
    parser.add_argument("--dataset", required=True, help="Dataset label written into outputs.")
    parser.add_argument(
        "--results-root",
        default=DEFAULT_RESULTS_ROOT,
        help=(
            "Root directory that contains <model>/<dataset>/result_details_*.jsonl. "
            f"Default: {DEFAULT_RESULTS_ROOT}"
        ),
    )
    parser.add_argument(
        "--val-data-root",
        default=DEFAULT_VAL_DATA_ROOT,
        help=(
            "Root directory for verl validation outputs. Layout: "
            "<root>/<project>/<experiment>/val_data_gen/<dataset>/<step>.jsonl. "
            f"Default: {DEFAULT_VAL_DATA_ROOT}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="tmp/asr-detail-compare",
        help="Directory where comparison artifacts will be written.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top target-error utterances to keep. Default: 20.",
    )
    parser.add_argument(
        "--join-columns",
        nargs="+",
        default=None,
        help="Optional explicit join columns. If omitted, the script auto-detects one and prefers audio_file.",
    )
    parser.add_argument(
        "--aggregate-segments-by",
        nargs="+",
        default=None,
        help="Group segmented rows by these columns and concatenate hypotheses before comparison.",
    )
    parser.add_argument(
        "--segment-index-column",
        default="seg_index",
        help="Column used to order rows during segment aggregation. Default: seg_index.",
    )
    parser.add_argument("--ref-column", default="ref", help="Reference text column. Default: ref.")
    parser.add_argument("--hyp-column", default="hyp", help="Hypothesis text column. Default: hyp.")
    parser.add_argument(
        "--write-full-csv",
        action="store_true",
        help="Also write the fully joined utterance-level comparison table.",
    )
    parser.add_argument(
        "--write-html",
        action="store_true",
        help="Also write a standalone HTML comparison for the top-ranked rows.",
    )
    parser.add_argument(
        "--audio-blob-root",
        default="az://orngwus2cresco/data/boren/data/openasr_jsonl",
        help=(
            "Azure blob root containing {dataset}/audio/{index}.wav files. "
            "When set with --write-html, downloads audio for report utterances "
            "and adds playback controls to HTML. "
            "Default: az://orngwus2cresco/data/boren/data/openasr_jsonl. "
            "Set to empty string to disable audio."
        ),
    )
    parser.add_argument(
        "--audio-local-dir",
        default=None,
        help=(
            "Local cache directory for downloaded audio files. "
            "Default: ~/data/openasr_jsonl/{dataset}/audio"
        ),
    )
    parser.add_argument(
        "--normalizer",
        choices=("auto", "english", "openasr"),
        default="auto",
        help=(
            "Text normalizer used before alignment. Default: auto (HF English for English rows, "
            "OpenASR for other languages)."
        ),
    )
    parser.add_argument("--lang", default="", help="Language code/name override for --normalizer openasr.")
    parser.add_argument("--lang-column", default="language", help="Row column with language name/code for --normalizer openasr.")
    return parser.parse_args()


def infer_label(path: str | None, model: str | None) -> str:
    if model:
        return model
    if not path:
        raise ValueError("Expected either a model name or a path when inferring the label.")
    stem = Path(path).stem
    if stem.startswith("result_details_"):
        return stem.removeprefix("result_details_")
    return stem


def extract_timestamp(path: str) -> str:
    match = TIMESTAMP_PATTERN.search(path)
    if not match:
        return ""
    return match.group(1)


def resolve_latest_result(results_root: str, model: str, dataset: str) -> str:
    pattern = bf.join(results_root, model, dataset, "result_details_*.jsonl")
    matches = sorted(bf.glob(pattern))
    if not matches:
        raise ValueError(f"No result_details files found for model={model} dataset={dataset} under {results_root}")

    ranked_matches = sorted(
        matches,
        key=lambda path: (extract_timestamp(path), path),
    )
    latest = ranked_matches[-1]
    return latest


def _extract_step(path: str) -> int:
    """Extract the numeric step from a filename like '300.jsonl'. Returns -1 on failure."""
    match = STEP_PATTERN.search(path)
    return int(match.group(1)) if match else -1


def resolve_val_data_gen(val_data_root: str, model: str, dataset: str) -> str | None:
    """Discover the latest step JSONL under val_data_gen on blob.

    Layout: <val_data_root>/<model>/val_data_gen/<dataset>/<step>.jsonl
    The model name may contain '/' to represent project/experiment.
    """
    ds_bare = dataset.rsplit("/", 1)[-1] if "/" in dataset else dataset
    pattern = bf.join(val_data_root, model, "val_data_gen", ds_bare, "*.jsonl")
    try:
        matches = sorted(bf.glob(pattern))
    except Exception:
        return None
    if not matches:
        return None
    ranked = sorted(matches, key=lambda p: (_extract_step(p), p))
    return ranked[-1]


LOCAL_RESULTS_ROOTS = [
    Path("tmp"),
    Path.home() / "data" / "results" / "verl_word_error",
]


def _resolve_local(model: str, dataset: str) -> str | None:
    """Try common local directory layouts under known results roots."""
    # Strip prefix like "openasr/" to get the bare dataset name
    ds_bare = dataset.rsplit("/", 1)[-1] if "/" in dataset else dataset
    for root in LOCAL_RESULTS_ROOTS:
        candidates = [
            # {root}/{model}/{dataset}.jsonl  (eval_openasr flat style)
            root / model / f"{ds_bare}.jsonl",
            # {root}/{model}/{dataset}/result_details_*.jsonl  (blob-mirror style)
        ]
        for c in candidates:
            if c.is_file():
                return str(c)
        # blob-mirror: {root}/{model}/{dataset}/result_details_*.jsonl — pick latest
        blob_mirror = root / model / dataset
        if blob_mirror.is_dir():
            jsonls = sorted(blob_mirror.glob("result_details_*.jsonl"))
            if jsonls:
                return str(jsonls[-1])
        # verl eval: {root}/{model}/val_data_gen/{ds_bare}/*.jsonl — pick latest
        verl_dir = root / model / "val_data_gen" / ds_bare
        if verl_dir.is_dir():
            jsonls = sorted(verl_dir.glob("*.jsonl"))
            if jsonls:
                return str(jsonls[-1])
    return None


def resolve_input_path(
    explicit_path: str | None, model: str | None, results_root: str, dataset: str,
    val_data_root: str = DEFAULT_VAL_DATA_ROOT,
) -> str:
    if explicit_path:
        return explicit_path
    if model:
        # Try local paths first (instant), then blob discovery (slow)
        local = _resolve_local(model, dataset)
        if local:
            return local
        # Try val_data_gen blob layout: <root>/<model>/val_data_gen/<dataset>/<step>.jsonl
        val_path = resolve_val_data_gen(val_data_root, model, dataset)
        if val_path:
            return val_path
        return resolve_latest_result(results_root, model, dataset)
    raise ValueError("Expected either an explicit path or a model name.")


def load_jsonl(path: str) -> pd.DataFrame:
    records: list[dict] = []
    with bf.BlobFile(path, "r") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON on line {line_number} from {path}: {exc}") from exc
    if not records:
        raise ValueError(f"No JSONL records found in {path}")
    df = pd.DataFrame(records)
    add_derived_key_columns(df)
    return df


def aggregate_segment_rows(
    df: pd.DataFrame,
    group_columns: Sequence[str],
    segment_index_column: str,
    ref_column: str,
    hyp_column: str,
) -> pd.DataFrame:
    required = [*group_columns, segment_index_column, ref_column, hyp_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Segment aggregation columns are missing: {', '.join(missing)}")

    if df.duplicated([*group_columns, segment_index_column]).any():
        raise ValueError("Segment aggregation keys are not unique.")

    records: list[dict] = []
    ordered = df.sort_values([*group_columns, segment_index_column])
    for _, group in ordered.groupby(list(group_columns), sort=False, dropna=False):
        references = [normalize_text(value) for value in group[ref_column] if normalize_text(value)]
        unique_references = list(dict.fromkeys(references))
        if len(unique_references) > 1:
            key = tuple(group.iloc[0][column] for column in group_columns)
            raise ValueError(f"Segment group {key} contains inconsistent references.")

        record = group.iloc[0].to_dict()
        record[ref_column] = unique_references[0] if unique_references else ""
        record[hyp_column] = " ".join(normalize_text(value) for value in group[hyp_column] if normalize_text(value))
        if "output" in group.columns:
            record["output"] = "\n".join(normalize_text(value) for value in group["output"] if normalize_text(value))
        record["n_segments"] = len(group)
        records.append(record)

    aggregated = pd.DataFrame(records)
    add_derived_key_columns(aggregated)
    return aggregated


def audio_file_to_stem(value: object) -> str | None:
    normalized = normalize_text(value)
    if not normalized:
        return None
    return Path(normalized).stem or None


def add_derived_key_columns(df: pd.DataFrame) -> None:
    if "meta" in df.columns:
        for column in META_PROMOTED_COLUMNS:
            if column in df.columns:
                continue
            df[column] = df["meta"].map(
                lambda value, key=column: value.get(key) if isinstance(value, dict) else None
            )
    if "audio_file" in df.columns and "audio_file_stem" not in df.columns:
        df["audio_file_stem"] = df["audio_file"].map(audio_file_to_stem)


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def normalize_asr_text(value: object) -> str:
    """Normalize text with the HF-compatible English evaluation normalizer."""
    text = normalize_text(value)
    if not text:
        return text
    return _english_normalizer(text)


def infer_language(row: pd.Series, lang_override: str = "", lang_column: str = "language", prefix: str = "") -> str:
    if lang_override:
        source_lang = lang_override.strip().lower()
        return get_language_code(source_lang or "en")

    if prefix:
        candidates = [f"{lang_column}_{prefix}", lang_column, f"data_source_{prefix}", "data_source"]
    else:
        candidates = [lang_column, "data_source"]

    source_lang = ""
    for candidate in candidates:
        value = row.get(candidate, "")
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        source_lang = str(value).strip().lower()
        if source_lang:
            break

    if source_lang and get_language_code(source_lang) == source_lang and "_" in source_lang:
        source_lang = source_lang.rsplit("_", 1)[-1]
    return get_language_code(source_lang or "en")


def normalize_asr_pair(
    ref_value: object,
    hyp_value: object,
    row: pd.Series | None = None,
    normalizer: str = "english",
    lang_override: str = "",
    lang_column: str = "language",
    prefix: str = "",
) -> tuple[str, str]:
    row_data = row if row is not None else pd.Series(dtype=object)
    lang_code = infer_language(row_data, lang_override, lang_column, prefix)
    if normalizer == "english" or (normalizer == "auto" and lang_code == "en"):
        return normalize_asr_text(ref_value), normalize_asr_text(hyp_value)
    if normalizer in ("auto", "openasr"):
        hyp_norm, ref_norm = normalize_for_wer(normalize_text(hyp_value), normalize_text(ref_value), lang=lang_code)
        return ref_norm, hyp_norm
    raise ValueError(f"Unknown normalizer: {normalizer}")


def edit_stats(ref_text: str, hyp_text: str) -> ErrorStats:
    ref_tokens = ref_text.split()
    hyp_tokens = hyp_text.split()
    rows = len(ref_tokens) + 1
    cols = len(hyp_tokens) + 1

    distance = [[0] * cols for _ in range(rows)]
    backtrace: list[list[tuple[int, int, str] | None]] = [[None] * cols for _ in range(rows)]

    for i in range(1, rows):
        distance[i][0] = i
        backtrace[i][0] = (i - 1, 0, "del")
    for j in range(1, cols):
        distance[0][j] = j
        backtrace[0][j] = (0, j - 1, "ins")

    for i in range(1, rows):
        for j in range(1, cols):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                distance[i][j] = distance[i - 1][j - 1]
                backtrace[i][j] = (i - 1, j - 1, "ok")
                continue

            delete_cost = distance[i - 1][j] + 1
            insert_cost = distance[i][j - 1] + 1
            substitute_cost = distance[i - 1][j - 1] + 1
            best_cost = min(substitute_cost, delete_cost, insert_cost)
            distance[i][j] = best_cost

            if best_cost == substitute_cost:
                backtrace[i][j] = (i - 1, j - 1, "sub")
            elif best_cost == delete_cost:
                backtrace[i][j] = (i - 1, j, "del")
            else:
                backtrace[i][j] = (i, j - 1, "ins")

    substitutions = deletions = insertions = 0
    i = len(ref_tokens)
    j = len(hyp_tokens)
    while i > 0 or j > 0:
        prev = backtrace[i][j]
        if prev is None:
            break
        prev_i, prev_j, op = prev
        if op == "sub":
            substitutions += 1
        elif op == "del":
            deletions += 1
        elif op == "ins":
            insertions += 1
        i, j = prev_i, prev_j

    errors = substitutions + deletions + insertions
    return ErrorStats(
        ref_words=len(ref_tokens),
        errors=errors,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
    )


def ensure_required_columns(df: pd.DataFrame, path: str, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def is_unique_key(df: pd.DataFrame, columns: Sequence[str]) -> bool:
    subset = df.loc[:, list(columns)]
    return subset.notna().all(axis=1).all() and not subset.duplicated().any()


def choose_join_columns(
    baseline_df: pd.DataFrame,
    target_df: pd.DataFrame,
    requested: Sequence[str] | None,
    ref_column: str,
) -> list[str]:
    if requested:
        missing = [column for column in requested if column not in baseline_df.columns or column not in target_df.columns]
        if missing:
            raise ValueError(f"Requested join columns are missing from one of the files: {', '.join(missing)}")
        return list(requested)

    common_columns = set(baseline_df.columns) & set(target_df.columns)
    for column in JOIN_CANDIDATES:
        if column in common_columns and is_unique_key(baseline_df, [column]) and is_unique_key(target_df, [column]):
            return [column]

    candidate_columns = [column for column in JOIN_CANDIDATES if column in common_columns]
    if candidate_columns and is_unique_key(baseline_df, candidate_columns) and is_unique_key(target_df, candidate_columns):
        return candidate_columns

    if ref_column in common_columns and is_unique_key(baseline_df, [ref_column]) and is_unique_key(target_df, [ref_column]):
        return [ref_column]

    if len(baseline_df) == len(target_df):
        baseline_df["__row_idx"] = range(len(baseline_df))
        target_df["__row_idx"] = range(len(target_df))
        return ["__row_idx"]

    raise ValueError(
        "Could not infer a stable join key. Pass --join-columns explicitly if the files are aligned on a custom key."
    )


def compute_metrics(
    df: pd.DataFrame,
    ref_column: str,
    hyp_column: str,
    prefix: str,
    normalizer: str,
    lang_override: str,
    lang_column: str,
) -> pd.DataFrame:
    stats = [
        edit_stats(*normalize_asr_pair(
            row[ref_column],
            row[hyp_column],
            row,
            normalizer,
            lang_override,
            lang_column,
            prefix,
        ))
        for _, row in df.iterrows()
    ]
    return pd.DataFrame(
        {
            f"{prefix}_ref_words": [item.ref_words for item in stats],
            f"{prefix}_errors": [item.errors for item in stats],
            f"{prefix}_substitutions": [item.substitutions for item in stats],
            f"{prefix}_deletions": [item.deletions for item in stats],
            f"{prefix}_insertions": [item.insertions for item in stats],
            f"{prefix}_wer": [item.wer for item in stats],
        }
    )


def slugify(value: str) -> str:
    cleaned = [char.lower() if char.isalnum() else "-" for char in value]
    slug = "".join(cleaned).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "comparison"


def copy_to_output_dir(src_path: str, output_dir: Path, prefix: str) -> Path:
    destination = output_dir / f"{slugify(prefix)}_{Path(src_path).name}"
    if not src_path.startswith("az://") and Path(src_path) == destination:
        return destination
    with bf.BlobFile(src_path, "rb") as src, destination.open("wb") as dst:
        while True:
            chunk = src.read(8 * 1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
    return destination


def format_percent(value: object) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return normalize_text(value)


def alignment_error_context(
    ref_words: list[str], hyp_words: list[str], context_words: int = 5,
) -> tuple[set[int], set[int], set[int], set[int], set[int], set[int]]:
    ref_visible: set[int] = set()
    hyp_visible: set[int] = set()
    ref_errors: set[int] = set()
    hyp_errors: set[int] = set()
    ref_gaps: set[int] = set()
    hyp_gaps: set[int] = set()

    matcher = SequenceMatcher(a=ref_words, b=hyp_words)
    opcodes = matcher.get_opcodes()
    if all(tag == "equal" for tag, *_ in opcodes):
        return set(range(len(ref_words))), set(range(len(hyp_words))), ref_errors, hyp_errors, ref_gaps, hyp_gaps

    for tag, ref_start, ref_end, hyp_start, hyp_end in opcodes:
        if tag == "equal":
            continue
        ref_errors.update(range(ref_start, ref_end))
        hyp_errors.update(range(hyp_start, hyp_end))
        if ref_start == ref_end:
            ref_gaps.add(ref_start)
        if hyp_start == hyp_end:
            hyp_gaps.add(hyp_start)
        ref_visible.update(
            range(max(0, ref_start - context_words), min(len(ref_words), ref_end + context_words))
        )
        hyp_visible.update(
            range(max(0, hyp_start - context_words), min(len(hyp_words), hyp_end + context_words))
        )

    return ref_visible, hyp_visible, ref_errors, hyp_errors, ref_gaps, hyp_gaps


def render_error_context(
    words: list[str], visible: set[int], errors: set[int], gaps: set[int], css_class: str,
) -> str:
    parts: list[str] = []
    index = 0
    while index <= len(words):
        if index in gaps:
            parts.append(
                f'<mark class="diff-change diff-gap {css_class}" tabindex="-1" '
                'title="No corresponding words">&empty;</mark>'
            )
        if index == len(words):
            break

        run_visible = index in visible or index in errors
        run_error = index in errors
        run_end = index + 1
        while run_end < len(words):
            next_visible = run_end in visible or run_end in errors
            if next_visible != run_visible or (run_visible and (run_end in errors) != run_error):
                break
            run_end += 1

        segment = " ".join(html.escape(word) for word in words[index:run_end])
        if not run_visible:
            parts.append(f'<span class="hidden-context">{segment}</span><span class="context-ellipsis">...</span>')
        elif run_error:
            parts.append(f'<mark class="diff-change {css_class}" tabindex="-1">{segment}</mark>')
        else:
            parts.append(f'<span class="diff-equal">{segment}</span>')
        index = run_end
    return " ".join(parts)


def build_comparison_html(
    rows: list[dict[str, object]], title: str, audio_map: dict[str, str] | None = None,
) -> str:
    cards: list[str] = []
    index_items: list[str] = []
    for card_index, row in enumerate(rows, start=1):
        baseline_wer = float(row["baseline_wer"])
        target_wer = float(row["target_wer"])
        if target_wer < baseline_wer:
            verdict = "target-better"
            verdict_label = "Target better"
        elif target_wer > baseline_wer:
            verdict = "target-worse"
            verdict_label = "Target worse"
        else:
            verdict = "target-same"
            verdict_label = "No change"

        audio_file_stem = normalize_text(row.get("audio_file_stem") or row.get("comparison_id"))
        ref_value = normalize_text(row.get("ref", ""))
        raw_ref_text = html.escape(normalize_text(row.get("raw_ref", row.get("ref", ""))))
        baseline_text = normalize_text(row["hyp_baseline"])
        target_text = normalize_text(row["hyp_target"])
        ref_words = ref_value.split()
        baseline_words = baseline_text.split()
        target_words = target_text.split()
        baseline_context = alignment_error_context(ref_words, baseline_words)
        target_context = alignment_error_context(ref_words, target_words)
        ref_visible = baseline_context[0] | target_context[0]
        ref_errors = baseline_context[2] | target_context[2]
        ref_gaps = baseline_context[4] | target_context[4]
        ref_text = render_error_context(ref_words, ref_visible, ref_errors, ref_gaps, "diff-reference")
        baseline_diff = render_error_context(
            baseline_words, baseline_context[1], baseline_context[3], baseline_context[5], "diff-removed"
        )
        target_diff = render_error_context(
            target_words, target_context[1], target_context[3], target_context[5], "diff-added"
        )
        has_hidden_context = any(
            len(visible) < len(words)
            for visible, words in (
                (ref_visible, ref_words),
                (baseline_context[1], baseline_words),
                (target_context[1], target_words),
            )
        )
        transcript_class = " transcript-condensed" if has_hidden_context else ""
        rank = int(row.get("rank", card_index))
        baseline_errors = int(row.get("baseline_errors", 0))
        target_errors = int(row.get("target_errors", 0))
        error_delta = int(row.get("error_delta", target_errors - baseline_errors))
        delta_label = f"{error_delta:+d}"
        index_items.append(
            f'<a href="#item-{rank}" class="index-item {verdict}">'
            f'<span class="index-rank">#{rank}</span>'
            f'<span class="index-name">{html.escape(audio_file_stem)}</span>'
            f'<span class="index-delta">{delta_label}</span>'
            "</a>"
        )

        raw_baseline_text = html.escape(str(row.get("raw_hyp_baseline", "")))
        raw_target_text = html.escape(str(row.get("raw_hyp_target", "")))

        audio_html = '<div class="audio-unavailable">Audio unavailable</div>'
        if audio_map:
            cid = normalize_text(row.get("comparison_id", ""))
            audio_src = audio_map.get(cid)
            if audio_src:
                audio_html = f'<audio controls preload="metadata" src="{html.escape(audio_src)}"></audio>'

        context_button = ""
        if has_hidden_context:
            context_button = '<button type="button" data-action="toggle-context">Show full context</button>'

        cards.append(
            f"""
                        <article class="card {verdict}{transcript_class}" id="item-{rank}" data-active-change="-1">
              <div class="card-header">
                <div>
                                    <div class="eyebrow">#{rank} &middot; audio_file_stem</div>
                  <h2>{html.escape(audio_file_stem)}</h2>
                </div>
                <div class="verdict">{verdict_label}</div>
                            </div>
                            <div class="review-toolbar">
                                <div class="audio-player">{audio_html}</div>
                                <div class="review-actions">
                                    <span class="change-status" aria-live="polite">Changes</span>
                                    <button type="button" data-action="previous-change" title="Previous changed span">Previous</button>
                                    <button type="button" data-action="next-change" title="Next changed span">Next</button>
                                    <label class="sync-toggle"><input type="checkbox" data-action="sync-scroll" checked> Sync scroll</label>
                                    {context_button}
                                </div>
                            </div>
              <div class="metrics">
                <div class="metric">
                                    <span class="label">Baseline</span>
                                    <span class="value">{format_percent(row["baseline_wer"])}</span>
                                    <span class="metric-detail">{baseline_errors} errors</span>
                </div>
                <div class="metric">
                                    <span class="label">Target</span>
                                    <span class="value">{format_percent(row["target_wer"])}</span>
                                    <span class="metric-detail">{target_errors} errors</span>
                                </div>
                                <div class="metric delta-metric">
                                    <span class="label">Error delta</span>
                                    <span class="value">{delta_label}</span>
                                    <span class="metric-detail">target &minus; baseline</span>
                </div>
              </div>
              <section class="panel ref-panel">
                                <div class="reference-block">
                                    <h3>Reference <span>normalized &middot; {len(ref_words)} words</span></h3>
                                    <p class="transcript-text">{ref_text}</p>
                                </div>
                                <details class="raw-section raw-reference">
                                    <summary>Raw reference</summary>
                                    <p class="transcript-text">{raw_ref_text}</p>
                                </details>
              </section>
                            <div class="compare-grid transcript-grid">
                                <section class="panel transcript-panel" data-side="baseline">
                                    <h3>Baseline <span>normalized</span></h3>
                                    <p class="transcript-text">{baseline_diff}</p>
                                                                        <details class="raw-section raw-hypothesis">
                                                                            <summary>Raw baseline output</summary>
                                                                            <p class="raw-hypothesis-text">{raw_baseline_text}</p>
                                                                        </details>
                </section>
                                <section class="panel transcript-panel" data-side="target">
                                    <h3>Target <span>normalized</span></h3>
                                    <p class="transcript-text">{target_diff}</p>
                                                                        <details class="raw-section raw-hypothesis">
                                                                            <summary>Raw target output</summary>
                                                                            <p class="raw-hypothesis-text">{raw_target_text}</p>
                                                                        </details>
                </section>
              </div>
            </article>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
            --bg: #f3f5f6;
            --surface: #ffffff;
            --ink: #172126;
            --muted: #647077;
            --line: #d5dadd;
            --good: #dcefe5;
            --good-ink: #17603d;
            --bad: #f7dfda;
            --bad-ink: #922f25;
            --same: #e8ecee;
            --same-ink: #58646a;
            --add: #cfead9;
            --remove: #f6d3cc;
            --reference-error: #f5e6ad;
            --focus: #126a78;
            --shadow: rgba(23, 33, 38, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
            font-family: "IBM Plex Sans", "Aptos", sans-serif;
      color: var(--ink);
            background-color: var(--bg);
            background-image: linear-gradient(rgba(23, 33, 38, 0.025) 1px, transparent 1px);
            background-size: 100% 28px;
    }}
    main {{
            width: min(1480px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 32px 0 48px;
    }}
    header {{
      margin-bottom: 24px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(2rem, 3vw, 3rem);
      line-height: 1;
    letter-spacing: 0;
    }}
    .subtitle {{
      margin: 0;
      color: var(--muted);
      font-size: 1rem;
    }}
    .card {{
            background: var(--surface);
      border: 1px solid var(--line);
            border-top-width: 5px;
            border-radius: 8px;
            box-shadow: 0 8px 24px var(--shadow);
      padding: 20px;
            margin-bottom: 24px;
            scroll-margin-top: 12px;
    }}
        .target-better {{ border-top-color: var(--good-ink); }}
        .target-worse {{ border-top-color: var(--bad-ink); }}
        .target-same {{ border-top-color: var(--same-ink); }}
        .report-index {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 8px;
            margin: 20px 0 28px;
        }}
        .index-item {{
            display: grid;
            grid-template-columns: auto minmax(0, 1fr) auto;
            align-items: center;
            gap: 10px;
            min-height: 42px;
            padding: 8px 10px;
            color: var(--ink);
            text-decoration: none;
            background: var(--surface);
            border: 1px solid var(--line);
            border-left: 4px solid var(--same-ink);
            border-radius: 6px;
        }}
        .index-item.target-better {{ border-left-color: var(--good-ink); }}
        .index-item.target-worse {{ border-left-color: var(--bad-ink); }}
        .index-item:hover {{ border-color: var(--focus); }}
        .index-rank, .index-delta {{ font-weight: 700; font-variant-numeric: tabular-nums; }}
        .index-name {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .card-header {{
      display: flex;
      gap: 16px;
      justify-content: space-between;
      align-items: start;
      margin-bottom: 14px;
    }}
    .eyebrow {{
      color: var(--muted);
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 4px;
    }}
    h2 {{
      margin: 0;
      font-size: 1.2rem;
      overflow-wrap: anywhere;
    }}
    .verdict {{
      white-space: nowrap;
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 0.9rem;
      font-weight: 700;
    }}
    .target-better .verdict {{
      background: var(--good);
      color: var(--good-ink);
    }}
    .target-worse .verdict {{
      background: var(--bad);
      color: var(--bad-ink);
    }}
    .target-same .verdict {{
      background: var(--same);
      color: var(--same-ink);
    }}
    .metrics {{
      display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .metric {{
            background: #f8fafb;
      border: 1px solid var(--line);
            border-radius: 6px;
      padding: 12px 14px;
    }}
    .label {{
      display: block;
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 4px;
    }}
    .value {{
      font-size: 1.4rem;
      font-weight: 700;
    }}
        .metric-detail {{ display: block; margin-top: 3px; color: var(--muted); font-size: 0.82rem; }}
        .target-better .delta-metric .value {{ color: var(--good-ink); }}
        .target-worse .delta-metric .value {{ color: var(--bad-ink); }}
    .ref-panel {{
      margin-bottom: 16px;
    }}
    .compare-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .panel {{
            background: #fbfcfc;
      border: 1px solid var(--line);
            border-radius: 6px;
      padding: 14px;
    }}
    .panel h3 {{
      margin: 0 0 10px;
      font-size: 1rem;
    }}
        .panel h3 span {{ color: var(--muted); font-size: 0.78rem; font-weight: 500; text-transform: uppercase; }}
    .panel p {{
      margin: 0;
      line-height: 1.75;
      font-size: 1rem;
    }}
        .diff-added, .diff-removed {{
      padding: 0.08em 0.18em;
            border-radius: 3px;
            color: inherit;
    }}
    .diff-added {{
      background: var(--add);
    }}
    .diff-removed {{
      background: var(--remove);
    }}
                .diff-reference {{ background: var(--reference-error); }}
        .diff-change.is-active {{
            outline: 3px solid var(--focus);
            outline-offset: 2px;
        }}
        .diff-gap {{ font-weight: 700; }}
                .context-ellipsis {{ display: none; color: var(--muted); font-weight: 700; }}
                .transcript-condensed .hidden-context {{ display: none; }}
                .transcript-condensed .context-ellipsis {{ display: inline; }}
        .review-toolbar {{
            position: sticky;
            top: 8px;
            z-index: 10;
            display: grid;
            grid-template-columns: minmax(260px, 1fr) auto;
            gap: 12px;
            align-items: center;
            margin: 0 -8px 16px;
            padding: 8px;
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 4px 14px var(--shadow);
    }}
    .audio-player audio {{
      width: 100%;
      height: 36px;
    }}
        .audio-unavailable {{ color: var(--muted); font-size: 0.85rem; }}
        .review-actions {{ display: flex; align-items: center; justify-content: flex-end; gap: 6px; flex-wrap: wrap; }}
        .review-actions button {{
            min-height: 34px;
            padding: 6px 10px;
            color: var(--ink);
            background: var(--surface);
            border: 1px solid #aeb7bb;
            border-radius: 5px;
            cursor: pointer;
            font: inherit;
            font-size: 0.82rem;
            font-weight: 650;
        }}
        .review-actions button:hover {{ border-color: var(--focus); color: var(--focus); }}
        .review-actions button:disabled {{ opacity: 0.45; cursor: default; }}
        .change-status {{ min-width: 104px; color: var(--muted); font-size: 0.82rem; text-align: right; }}
        .sync-toggle {{ display: flex; align-items: center; gap: 5px; color: var(--muted); font-size: 0.82rem; white-space: nowrap; }}
        .transcript-panel {{ max-height: 32rem; overflow: auto; scroll-behavior: smooth; }}
        .transcript-panel h3 {{
            position: sticky;
            top: -14px;
            z-index: 2;
            margin: -14px -14px 10px;
            padding: 12px 14px 9px;
            background: #fbfcfc;
            border-bottom: 1px solid var(--line);
        }}
        .reference-block + .raw-reference {{ margin-top: 14px; padding-top: 8px; border-top: 1px solid var(--line); }}
        .raw-reference {{ color: #39464c; }}
        .raw-reference .transcript-text {{ font-family: "IBM Plex Mono", "Cascadia Code", monospace; font-size: 0.9rem; }}
        .transcript-condensed .reference-block .transcript-text {{ max-height: 12rem; overflow: auto; }}
    .raw-section {{
      margin-top: 12px;
    }}
    .raw-section summary {{
      cursor: pointer;
      color: var(--muted);
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      padding: 6px 0;
      user-select: none;
    }}
    .raw-section summary:hover {{
      color: var(--ink);
    }}
        .raw-hypothesis {{
            margin-top: 16px;
            padding-top: 8px;
            border-top: 1px solid var(--line);
        }}
        .raw-hypothesis-text {{
            color: #39464c;
      font-family: "SF Mono", "Fira Code", "Consolas", monospace;
      font-size: 0.88rem;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    @media (max-width: 800px) {{
            .report-index {{ grid-template-columns: minmax(0, 1fr); }}
      .card-header,
      .compare-grid,
      .metrics {{
        grid-template-columns: 1fr;
        display: grid;
      }}
      .card-header {{
        align-items: stretch;
      }}
      .verdict {{
        justify-self: start;
      }}
            .review-toolbar {{ position: static; grid-template-columns: 1fr; }}
            .review-actions {{ justify-content: flex-start; }}
            .change-status {{ text-align: left; }}
            .transcript-panel {{ max-height: 24rem; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>{html.escape(title)}</h1>
            <p class="subtitle">Ranked utterance review with normalized and raw references, synchronized transcripts, and change navigation.</p>
    </header>
        <nav class="report-index" aria-label="Report items">{''.join(index_items)}</nav>
    {''.join(cards)}
  </main>
    <script>
        document.querySelectorAll(".card").forEach((card) => {{
            const targetChanges = [...card.querySelectorAll('[data-side="target"] .diff-change')];
            const baselineChanges = [...card.querySelectorAll('[data-side="baseline"] .diff-change')];
            const changes = targetChanges.length ? targetChanges : baselineChanges;
            const status = card.querySelector(".change-status");
            const previous = card.querySelector('[data-action="previous-change"]');
            const next = card.querySelector('[data-action="next-change"]');
            let activeIndex = -1;

            const updateStatus = () => {{
                status.textContent = changes.length
                    ? (activeIndex >= 0 ? `Change ${{activeIndex + 1}} of ${{changes.length}}` : `${{changes.length}} changed spans`)
                    : "No changed spans";
                previous.disabled = !changes.length;
                next.disabled = !changes.length;
            }};

            const moveToChange = (step) => {{
                if (!changes.length) return;
                changes.forEach((change) => change.classList.remove("is-active"));
                activeIndex = (activeIndex + step + changes.length) % changes.length;
                changes[activeIndex].classList.add("is-active");
                changes[activeIndex].focus({{ preventScroll: true }});
                changes[activeIndex].scrollIntoView({{ behavior: "smooth", block: "center", inline: "nearest" }});
                updateStatus();
            }};

            previous.addEventListener("click", () => moveToChange(-1));
            next.addEventListener("click", () => moveToChange(1));

            const contextButton = card.querySelector('[data-action="toggle-context"]');
            if (contextButton) {{
                contextButton.addEventListener("click", () => {{
                    const condensed = card.classList.toggle("transcript-condensed");
                    contextButton.textContent = condensed ? "Show full context" : "Compact context";
                }});
            }}

            const panes = [...card.querySelectorAll(".transcript-panel")];
            let syncing = false;
            panes.forEach((sourcePane) => {{
                sourcePane.addEventListener("scroll", () => {{
                    const syncToggle = card.querySelector('[data-action="sync-scroll"]');
                    if (!syncToggle.checked || syncing) return;
                    const sourceRange = sourcePane.scrollHeight - sourcePane.clientHeight;
                    if (sourceRange <= 0) return;
                    syncing = true;
                    const ratio = sourcePane.scrollTop / sourceRange;
                    panes.forEach((targetPane) => {{
                        if (targetPane !== sourcePane) {{
                            targetPane.scrollTop = ratio * (targetPane.scrollHeight - targetPane.clientHeight);
                        }}
                    }});
                    requestAnimationFrame(() => {{ syncing = false; }});
                }});
            }});

            updateStatus();
        }});
    </script>
</body>
</html>
"""


def download_audio_for_reports(
    report_dfs: list[pd.DataFrame],
    merged: pd.DataFrame,
    dataset: str,
    audio_blob_root: str,
    audio_local_dir: Path,
    output_dir: Path,
) -> dict[str, str]:
    """Download audio files for utterances in the reports.

    Returns a mapping from comparison_id to relative audio path (audio/{idx}.wav).
    """
    audio_idx_map = dict(
        zip(
            merged["comparison_id"].astype(str),
            merged["__audio_idx"].astype(int),
            strict=True,
        )
    )

    needed_ids: set[str] = set()
    for report_df in report_dfs:
        needed_ids.update(report_df["comparison_id"].astype(str))

    audio_local_dir.mkdir(parents=True, exist_ok=True)
    audio_output_dir = output_dir / "audio"
    audio_output_dir.mkdir(parents=True, exist_ok=True)

    # Build list of (cid, audio_idx) to download
    download_tasks: list[tuple[str, int]] = []
    for cid in sorted(needed_ids):
        audio_idx = audio_idx_map.get(cid)
        if audio_idx is not None:
            download_tasks.append((cid, audio_idx))

    def _download_one(cid: str, audio_idx: int) -> tuple[str, int, bool]:
        """Download a single audio file. Returns (cid, audio_idx, success)."""
        local_file = audio_local_dir / f"{audio_idx}.wav"
        output_file = audio_output_dir / f"{audio_idx}.wav"
        if not local_file.exists():
            blob_path = bf.join(audio_blob_root, dataset, "audio", f"{audio_idx}.wav")
            try:
                with bf.BlobFile(blob_path, "rb") as src, local_file.open("wb") as dst:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
            except Exception as exc:
                print(f"  Warning: Failed to download audio for {cid} (idx={audio_idx}): {exc}")
                return cid, audio_idx, False
        if not output_file.exists():
            shutil.copy2(local_file, output_file)
        return cid, audio_idx, True

    audio_map: dict[str, str] = {}
    n_downloaded = 0
    n_cached = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_download_one, cid, idx): (cid, idx) for cid, idx in download_tasks}
        for future in as_completed(futures):
            cid, audio_idx, ok = future.result()
            if ok:
                local_file = audio_local_dir / f"{audio_idx}.wav"
                if local_file.stat().st_mtime > 0:  # just downloaded or already cached
                    audio_map[cid] = f"audio/{audio_idx}.wav"
                if (audio_output_dir / f"{audio_idx}.wav").stat().st_size > 0:
                    n_cached += 1
                else:
                    n_downloaded += 1
    print(f"  {len(audio_map)} audio files ready ({n_downloaded} downloaded, {n_cached} cached).")

    return audio_map


def write_comparison_html(
    df: pd.DataFrame, output_path: Path, title: str, audio_map: dict[str, str] | None = None,
) -> None:
    output_path.write_text(
        build_comparison_html(df.to_dict("records"), title, audio_map), encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    baseline_path = resolve_input_path(args.baseline_path, args.baseline_model, args.results_root, args.dataset, args.val_data_root)
    target_path = resolve_input_path(args.target_path, args.target_model, args.results_root, args.dataset, args.val_data_root)
    baseline_name = args.baseline_name or infer_label(baseline_path, args.baseline_model)
    target_name = args.target_name or infer_label(target_path, args.target_model)

    baseline_df = load_jsonl(baseline_path)
    target_df = load_jsonl(target_path)

    # Auto-remap verl schema columns (gts->ref, clean_output->hyp) when needed
    for label, df in [("baseline", baseline_df), ("target", target_df)]:
        if args.ref_column not in df.columns and "gts" in df.columns:
            df[args.ref_column] = df["gts"]
        if args.hyp_column not in df.columns and "clean_output" in df.columns:
            df[args.hyp_column] = df["clean_output"]

    ensure_required_columns(baseline_df, baseline_path, [args.ref_column, args.hyp_column])
    ensure_required_columns(target_df, target_path, [args.ref_column, args.hyp_column])

    if args.aggregate_segments_by:
        baseline_df = aggregate_segment_rows(
            baseline_df,
            args.aggregate_segments_by,
            args.segment_index_column,
            args.ref_column,
            args.hyp_column,
        )
        target_df = aggregate_segment_rows(
            target_df,
            args.aggregate_segments_by,
            args.segment_index_column,
            args.ref_column,
            args.hyp_column,
        )
    baseline_df["__audio_idx"] = range(len(baseline_df))

    join_columns = choose_join_columns(baseline_df, target_df, args.join_columns, args.ref_column)
    merged = baseline_df.merge(
        target_df,
        on=join_columns,
        how="inner",
        suffixes=("_baseline", "_target"),
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("The merged comparison is empty. Check the join columns and source files.")

    ref_baseline = f"{args.ref_column}_baseline"
    ref_target = f"{args.ref_column}_target"
    hyp_baseline = f"{args.hyp_column}_baseline"
    hyp_target = f"{args.hyp_column}_target"
    ensure_required_columns(merged, "merged frame", [ref_baseline, ref_target, hyp_baseline, hyp_target])

    merged["raw_hyp_baseline"] = merged.get("output_baseline", merged[hyp_baseline])
    # output_target when both sides have 'output'; plain 'output' when only target has it
    merged["raw_hyp_target"] = merged.get("output_target", merged.get("output", merged[hyp_target]))
    merged["raw_ref"] = merged[ref_target]
    # Preserve the full original output (with <ASR> tags etc.) as raw_output
    merged["raw_output"] = merged.get("output_target", merged.get("output", pd.Series([""] * len(merged))))
    normalized_rows = [
        (
            *normalize_asr_pair(row[ref_baseline], row[hyp_baseline], row, args.normalizer, args.lang, args.lang_column, "baseline"),
            *normalize_asr_pair(row[ref_target], row[hyp_target], row, args.normalizer, args.lang, args.lang_column, "target"),
        )
        for _, row in merged.iterrows()
    ]
    merged["ref_baseline_norm"] = [row[0] for row in normalized_rows]
    merged["hyp_baseline"] = [row[1] for row in normalized_rows]
    merged["ref"] = [row[2] for row in normalized_rows]
    merged["hyp_target"] = [row[3] for row in normalized_rows]
    merged["ref_matches_baseline"] = merged["ref"] == merged["ref_baseline_norm"]

    baseline_metrics = compute_metrics(merged, ref_baseline, hyp_baseline, "baseline", args.normalizer, args.lang, args.lang_column)
    target_metrics = compute_metrics(merged, ref_target, hyp_target, "target", args.normalizer, args.lang, args.lang_column)
    merged = pd.concat([merged.reset_index(drop=True), baseline_metrics, target_metrics], axis=1)

    total_ref_words = int(merged["target_ref_words"].sum())
    total_target_errors = int(merged["target_errors"].sum())
    total_baseline_errors = int(merged["baseline_errors"].sum())
    merged["target_total_wer_contribution"] = merged["target_errors"] / max(total_ref_words, 1)
    merged["baseline_total_wer_contribution"] = merged["baseline_errors"] / max(total_ref_words, 1)
    merged["error_delta"] = merged["target_errors"] - merged["baseline_errors"]
    merged["wer_delta"] = merged["target_wer"] - merged["baseline_wer"]

    comparison_id_columns = [column for column in join_columns if column != "__row_idx"]
    if comparison_id_columns:
        merged["comparison_id"] = merged[comparison_id_columns].astype(str).agg(" | ".join, axis=1)
    else:
        merged["comparison_id"] = merged["__row_idx"].astype(str)

    report_columns = [
        "comparison_id",
        *join_columns,
        "ref",
        "raw_ref",
        "hyp_baseline",
        "hyp_target",
        "raw_hyp_baseline",
        "raw_hyp_target",
        "raw_output",
        "ref_matches_baseline",
        "baseline_ref_words",
        "baseline_errors",
        "baseline_substitutions",
        "baseline_deletions",
        "baseline_insertions",
        "baseline_wer",
        "baseline_total_wer_contribution",
        "target_ref_words",
        "target_errors",
        "target_substitutions",
        "target_deletions",
        "target_insertions",
        "target_wer",
        "target_total_wer_contribution",
        "error_delta",
        "wer_delta",
    ]

    # Report 0: Overall comparison sorted by absolute error_delta (largest changes first)
    overall_df = merged.copy()
    overall_df["abs_error_delta"] = overall_df["error_delta"].abs()
    overall_df = overall_df.sort_values(
        by=["abs_error_delta", "target_errors", "comparison_id"],
        ascending=[False, False, True],
    ).head(args.top_n)
    overall_df = overall_df.loc[:, report_columns].copy()
    overall_df.insert(0, "rank", range(1, len(overall_df) + 1))

    # Report 1: Improved utterances (error_delta < 0), sorted by baseline_errors desc
    improved_df = merged[merged["error_delta"] < 0].sort_values(
        by=["baseline_errors", "error_delta", "comparison_id"],
        ascending=[False, True, True],
    ).head(args.top_n)
    improved_df = improved_df.loc[:, report_columns].copy()
    improved_df.insert(0, "rank", range(1, len(improved_df) + 1))

    # Report 2: Degraded utterances (error_delta > 0), sorted by target_errors desc
    degraded_df = merged[merged["error_delta"] > 0].sort_values(
        by=["target_errors", "error_delta", "comparison_id"],
        ascending=[False, False, True],
    ).head(args.top_n)
    degraded_df = degraded_df.loc[:, report_columns].copy()
    degraded_df.insert(0, "rank", range(1, len(degraded_df) + 1))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = slugify(f"{args.dataset}-{baseline_name}-vs-{target_name}")
    local_baseline_jsonl = copy_to_output_dir(baseline_path, output_dir, baseline_name)
    local_target_jsonl = copy_to_output_dir(target_path, output_dir, target_name)

    reports = [
        ("overall", overall_df, f"{stem}.overall-top{args.top_n}"),
        ("improved", improved_df, f"{stem}.improved-top{args.top_n}"),
        ("degraded", degraded_df, f"{stem}.degraded-top{args.top_n}"),
    ]

    summary_outputs: dict[str, dict[str, str]] = {}

    audio_map: dict[str, str] | None = None
    if args.audio_blob_root and args.write_html:
        audio_local_dir = (
            Path(args.audio_local_dir)
            if args.audio_local_dir
            else Path.home() / "data" / "openasr_jsonl" / args.dataset / "audio"
        )
        print(f"\nDownloading audio files to {audio_local_dir} ...")
        audio_map = download_audio_for_reports(
            [overall_df, improved_df, degraded_df],
            merged,
            args.dataset,
            args.audio_blob_root,
            audio_local_dir,
            output_dir,
        )
        print(f"  {len(audio_map)} audio files ready.\n")

    for report_name, report_df, report_stem in reports:
        csv_path = output_dir / f"{report_stem}.csv"
        report_df.to_csv(csv_path, index=False)
        summary_outputs[report_name] = {"csv": str(csv_path)}
        if args.write_html:
            html_path = output_dir / f"{report_stem}.html"
            write_comparison_html(report_df, html_path, report_stem, audio_map)
            summary_outputs[report_name]["html"] = str(html_path)

    if args.write_full_csv:
        full_path = output_dir / f"{stem}.full.csv"
        merged.to_csv(full_path, index=False)

    summary = {
        "dataset": args.dataset,
        "baseline_name": baseline_name,
        "target_name": target_name,
        "baseline_model": args.baseline_model,
        "target_model": args.target_model,
        "results_root": args.results_root,
        "normalizer": args.normalizer,
        "lang": args.lang,
        "lang_column": args.lang_column,
        "baseline_path": baseline_path,
        "target_path": target_path,
        "local_baseline_jsonl": str(local_baseline_jsonl),
        "local_target_jsonl": str(local_target_jsonl),
        "join_columns": join_columns,
        "rows_compared": int(len(merged)),
        "top_n": args.top_n,
        "total_ref_words": total_ref_words,
        "baseline_total_errors": total_baseline_errors,
        "target_total_errors": total_target_errors,
        "baseline_wer": total_baseline_errors / max(total_ref_words, 1),
        "target_wer": total_target_errors / max(total_ref_words, 1),
        "error_delta": total_target_errors - total_baseline_errors,
        "improved_count": int((merged["error_delta"] < 0).sum()),
        "degraded_count": int((merged["error_delta"] > 0).sum()),
        "unchanged_count": int((merged["error_delta"] == 0).sum()),
        "reports": summary_outputs,
    }
    if args.write_full_csv:
        summary["full_csv"] = str(full_path)

    summary_path = output_dir / f"{stem}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    for report_name, report_df, _report_stem in reports:
        print(f"\n=== {report_name.upper()} (top {args.top_n}) ===")
        print(report_df.to_markdown(index=False))


if __name__ == "__main__":
    main()
