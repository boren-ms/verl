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
from whisper.normalizers.english import EnglishTextNormalizer

REPO_ROOT = Path(__file__).resolve().parents[4]
if (REPO_ROOT / "recipe").is_dir() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.phimm.utils.languages import get_language_code
from recipe.phimm.utils.open_asr_normalizer.eval_utils import normalize_for_wer


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

_whisper_normalizer = EnglishTextNormalizer()
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
        choices=("english", "openasr"),
        default="english",
        help="Text normalizer used before alignment. Use 'openasr' to match measure_wer.",
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
                lambda value: value.get(column) if isinstance(value, dict) else None
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
    """Normalize text for ASR evaluation using the Whisper English normalizer."""
    text = normalize_text(value)
    if not text:
        return text
    return _whisper_normalizer(text)


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
    if normalizer == "openasr":
        lang_code = infer_language(row if row is not None else pd.Series(dtype=object), lang_override, lang_column, prefix)
        hyp_norm, ref_norm = normalize_for_wer(normalize_text(hyp_value), normalize_text(ref_value), lang=lang_code)
        return ref_norm, hyp_norm
    return normalize_asr_text(ref_value), normalize_asr_text(hyp_value)


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


def render_word_diff(text: str, other: str, css_class: str) -> str:
    words = text.split()
    other_words = other.split()
    matcher = SequenceMatcher(a=words, b=other_words)
    parts: list[str] = []
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        segment = " ".join(html.escape(word) for word in words[i1:i2])
        if not segment:
            continue
        if tag == "equal":
            parts.append(segment)
        else:
            parts.append(f'<span class="{css_class}">{segment}</span>')
    return " ".join(parts)


def build_comparison_html(
    rows: list[dict[str, object]], title: str, audio_map: dict[str, str] | None = None,
) -> str:
    cards: list[str] = []
    for row in rows:
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
        ref_text = html.escape(normalize_text(row.get("ref", "")))
        baseline_diff = render_word_diff(normalize_text(row["hyp_baseline"]), normalize_text(row["hyp_target"]), "diff-removed")
        target_diff = render_word_diff(normalize_text(row["hyp_target"]), normalize_text(row["hyp_baseline"]), "diff-added")

        raw_baseline_text = html.escape(str(row.get("raw_hyp_baseline", "")))
        raw_target_text = html.escape(str(row.get("raw_hyp_target", "")))

        audio_html = ""
        if audio_map:
            cid = normalize_text(row.get("comparison_id", ""))
            audio_src = audio_map.get(cid)
            if audio_src:
                audio_html = f"""
              <div class="audio-player">
                <audio controls preload="none" src="{html.escape(audio_src)}"></audio>
              </div>"""

        cards.append(
            f"""
            <article class="card {verdict}">
              <div class="card-header">
                <div>
                  <div class="eyebrow">audio_file_stem</div>
                  <h2>{html.escape(audio_file_stem)}</h2>
                </div>
                <div class="verdict">{verdict_label}</div>
              </div>{audio_html}
              <div class="metrics">
                <div class="metric">
                  <span class="label">baseline_wer</span>
                  <span class="value">{format_percent(row["baseline_wer"])}</span>
                </div>
                <div class="metric">
                  <span class="label">target_wer</span>
                  <span class="value">{format_percent(row["target_wer"])}</span>
                </div>
              </div>
              <section class="panel ref-panel">
                <h3>Reference</h3>
                <p>{ref_text}</p>
              </section>
              <div class="compare-grid">
                <section class="panel">
                  <h3>hyp_baseline (normalized)</h3>
                  <p>{baseline_diff}</p>
                </section>
                <section class="panel">
                  <h3>hyp_target (normalized)</h3>
                  <p>{target_diff}</p>
                </section>
              </div>
              <details class="raw-section">
                <summary>Raw output</summary>
                <div class="compare-grid">
                  <section class="panel raw-panel">
                    <h3>Raw output (baseline)</h3>
                    <p>{raw_baseline_text}</p>
                  </section>
                  <section class="panel raw-panel">
                    <h3>Raw output (target)</h3>
                    <p>{raw_target_text}</p>
                  </section>
                </div>
              </details>
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
      --bg: #f4efe6;
      --ink: #1f1a17;
      --muted: #6f655c;
      --line: #d8cdbf;
      --good: #d7efe0;
      --good-ink: #18563b;
      --bad: #f6d9cf;
      --bad-ink: #8b2d20;
      --same: #ebe3d6;
      --same-ink: #6b5d4b;
      --add: #d8f1df;
      --remove: #f9ddd4;
      --shadow: rgba(84, 58, 28, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(188, 142, 92, 0.18), transparent 28%),
        linear-gradient(180deg, #efe5d7 0%, var(--bg) 100%);
    }}
    main {{
      width: min(1200px, calc(100vw - 32px));
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
      letter-spacing: -0.03em;
    }}
    .subtitle {{
      margin: 0;
      color: var(--muted);
      font-size: 1rem;
    }}
    .card {{
      background: rgba(255, 250, 243, 0.92);
      border: 1px solid var(--line);
      border-left-width: 8px;
      border-radius: 18px;
      box-shadow: 0 18px 40px var(--shadow);
      padding: 20px;
      margin-bottom: 18px;
      backdrop-filter: blur(6px);
    }}
    .target-better {{ border-left-color: var(--good-ink); }}
    .target-worse {{ border-left-color: var(--bad-ink); }}
    .target-same {{ border-left-color: var(--same-ink); }}
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
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .metric {{
      background: rgba(255, 255, 255, 0.65);
      border: 1px solid var(--line);
      border-radius: 14px;
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
    .ref-panel {{
      margin-bottom: 16px;
    }}
    .compare-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .panel {{
      background: rgba(255, 255, 255, 0.68);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
    }}
    .panel h3 {{
      margin: 0 0 10px;
      font-size: 1rem;
    }}
    .panel p {{
      margin: 0;
      line-height: 1.75;
      font-size: 1rem;
    }}
    .diff-added, .diff-removed {{
      padding: 0.08em 0.18em;
      border-radius: 0.3em;
    }}
    .diff-added {{
      background: var(--add);
    }}
    .diff-removed {{
      background: var(--remove);
    }}
    .audio-player {{
      margin-bottom: 14px;
    }}
    .audio-player audio {{
      width: 100%;
      height: 36px;
    }}
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
    .raw-panel {{
      background: rgba(245, 240, 232, 0.7);
      font-family: "SF Mono", "Fira Code", "Consolas", monospace;
      font-size: 0.88rem;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .raw-panel p {{
      line-height: 1.6;
    }}
    .raw-section .compare-grid {{
      margin-top: 10px;
    }}
    @media (max-width: 800px) {{
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
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>{html.escape(title)}</h1>
      <p class="subtitle">Comparison of <code>hyp_baseline</code> vs <code>hyp_target</code> with <code>audio_file_stem</code>, <code>baseline_wer</code>, and <code>target_wer</code>.</p>
    </header>
    {''.join(cards)}
  </main>
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
        zip(merged["comparison_id"].astype(str), merged["__audio_idx"].astype(int))
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
    baseline_df["__audio_idx"] = range(len(baseline_df))

    # Auto-remap verl schema columns (gts->ref, clean_output->hyp) when needed
    for label, df in [("baseline", baseline_df), ("target", target_df)]:
        if args.ref_column not in df.columns and "gts" in df.columns:
            df[args.ref_column] = df["gts"]
        if args.hyp_column not in df.columns and "clean_output" in df.columns:
            df[args.hyp_column] = df["clean_output"]

    ensure_required_columns(baseline_df, baseline_path, [args.ref_column, args.hyp_column])
    ensure_required_columns(target_df, target_path, [args.ref_column, args.hyp_column])

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
