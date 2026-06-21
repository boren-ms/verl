#!/usr/bin/env python3
"""
Entity Error Rate (EER) and Entity Word Error Rate (EWER) analysis for ASR result_details JSONL files.

EER  = entities_with_errors / total_entities  (entity-level error rate)
EWER = entity_word_errors / total_entity_words  (word-level error rate within entities)

Extracts named entities from <NE> / <NE:type> tags in the Transcription field,
aligns hypothesis against reference at word level, and computes error rates
exclusively on entity spans. Produces CSV artifacts and an optional standalone
HTML report optimized for very long transcripts with entity-only highlighting.

Produces:
  - summary.json               - dataset-level EER, EWER, entity counts, error totals
  - entity_errors.csv           - per-entity-instance error details
  - entity_substitutions.csv    - ranked entity substitution pairs
  - entity_type_breakdown.csv   - error rates by entity type
  - report.html (opt)           - standalone visual report with entity highlighting
"""
from __future__ import annotations

import argparse
import collections
import csv
import html as html_mod
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import blobfile as bf
import ftfy
from whisper_normalizer.english import EnglishTextNormalizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze entity error rate from ASR result_details JSONL with <NE> tagged Transcription."
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-path", help="Path to result_details JSONL file (local or az://).")
    source.add_argument("--model", help="Model directory name under results root (auto-discovers latest file).")
    p.add_argument("--dataset", default="", help="Dataset name (used in file discovery and output labeling).")
    p.add_argument(
        "--results-root",
        default="az://orngwus2cresco/data/boren/data/results/gpt-4o-mini-asr-v1",
        help="Root that contains <model>/<dataset>/result_details_*.jsonl.",
    )
    p.add_argument("--transcription-column", default="Transcription",
                    help="Column containing reference text with <NE> entity tags.")
    p.add_argument("--hyp-column", default="hyp", help="Column name for hypothesis text.")
    p.add_argument("--id-column", default="", help="Column for utterance ID (auto-detected if empty).")
    p.add_argument("--output-dir", default="tmp/asr-entity-error-analysis", help="Where to write artifacts.")
    p.add_argument("--top-n", type=int, default=50, help="Number of worst utterances shown in HTML report.")
    p.add_argument("--top-entities", type=int, default=100, help="Number of top entity error entries to report.")
    p.add_argument("--write-html", action="store_true", help="Also write a standalone HTML report.")
    p.add_argument("--case-sensitive", action="store_true", help="Do not lowercase ref/hyp before alignment.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

TIMESTAMP_RE = re.compile(r"result_details_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.jsonl$")
ID_CANDIDATES = [
    "audio_file_stem", "audio_file", "utt_id", "utterance_id", "example_id",
    "item_id", "segment_id", "id", "key", "audio_path", "path",
]
ENGLISH_TEXT_NORMALIZER = EnglishTextNormalizer()

# Regex for NE tags: <NE> or <NE:type>
NE_OPEN_RE = re.compile(r"<NE(?::([^>]*))?>\s*")
# Closing tag: </NE> or </NE:type>
NE_CLOSE_RE = re.compile(r"\s*</NE(?::[^>]*)?>")



def resolve_latest(results_root: str, model: str, dataset: str) -> str:
    pattern = bf.join(results_root, model, dataset, "result_details_*.jsonl")
    matches = sorted(bf.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No result_details files found: {pattern}")

    def _mtime(p: str) -> float:
        try:
            return bf.stat(p).mtime
        except Exception:
            return float("-inf")

    def _ts(p: str) -> str:
        m = TIMESTAMP_RE.search(p)
        return m.group(1) if m else ""

    ranked = sorted(matches, key=lambda p: (_mtime(p), _ts(p), p))
    return ranked[-1]


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
    # derive audio_file_stem
    for row in rows:
        if "audio_file" in row and "audio_file_stem" not in row:
            row["audio_file_stem"] = Path(str(row["audio_file"])).stem
    return rows


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
    text = html_mod.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text: str) -> str:
    text = clean_text(text)
    text = ENGLISH_TEXT_NORMALIZER(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Entity extraction from Transcription
# ---------------------------------------------------------------------------

@dataclass
class EntitySpan:
    """A named entity extracted from the Transcription field."""
    text: str               # raw entity text (between tags)
    entity_type: str        # "NE" or "NE:name" etc.
    start_word_idx: int     # start word index in clean reference (inclusive)
    end_word_idx: int       # end word index in clean reference (exclusive)
    char_start: int         # char offset in clean reference
    char_end: int           # char offset in clean reference


def extract_entities_and_clean(transcription: str) -> tuple[str, list[EntitySpan]]:
    """
    Parse <NE> and <NE:type> tags from Transcription.
    Returns (clean_text, list_of_entity_spans).
    Entity spans reference word positions in the clean (tag-free) text.
    """
    # Strip non-NE tags first so they don't affect word position computation
    transcription = re.sub(r"</?(?!NE[ >:])([a-zA-Z][^>]*)>", "", transcription)
    transcription = re.sub(r"\s+", " ", transcription).strip()

    entities: list[EntitySpan] = []
    clean_parts: list[str] = []
    current_pos = 0
    in_entity = False
    entity_type = ""
    entity_char_start = 0

    while current_pos < len(transcription):
        if not in_entity:
            # Look for opening tag
            m = NE_OPEN_RE.search(transcription, current_pos)
            if m is None:
                # No more tags - take rest of text
                clean_parts.append(transcription[current_pos:])
                break
            # Text before the tag
            clean_parts.append(transcription[current_pos:m.start()])
            entity_type = f"NE:{m.group(1)}" if m.group(1) else "NE"
            entity_char_start = sum(len(p) for p in clean_parts)
            in_entity = True
            current_pos = m.end()
        else:
            # Look for closing tag
            m = NE_CLOSE_RE.search(transcription, current_pos)
            if m is None:
                # Malformed - treat rest as entity text
                entity_text = transcription[current_pos:]
                clean_parts.append(entity_text)
                entity_char_end = sum(len(p) for p in clean_parts)
                clean_so_far = "".join(clean_parts)
                start_word = len(clean_so_far[:entity_char_start].split()) if entity_char_start > 0 else 0
                end_word = len(clean_so_far[:entity_char_end].split())
                entities.append(EntitySpan(
                    text=entity_text.strip(),
                    entity_type=entity_type,
                    start_word_idx=start_word,
                    end_word_idx=end_word,
                    char_start=entity_char_start,
                    char_end=entity_char_end,
                ))
                in_entity = False
                break
            # Entity text is between current_pos and m.start()
            entity_text = transcription[current_pos:m.start()]
            clean_parts.append(entity_text)
            entity_char_end = sum(len(p) for p in clean_parts)
            clean_so_far = "".join(clean_parts)

            # Compute word indices
            prefix_text = clean_so_far[:entity_char_start].rstrip()
            start_word = len(prefix_text.split()) if prefix_text else 0
            entity_words_text = clean_so_far[entity_char_start:entity_char_end].strip()
            n_entity_words = len(entity_words_text.split()) if entity_words_text else 0
            end_word = start_word + n_entity_words

            entities.append(EntitySpan(
                text=entity_text.strip(),
                entity_type=entity_type,
                start_word_idx=start_word,
                end_word_idx=end_word,
                char_start=entity_char_start,
                char_end=entity_char_end,
            ))
            in_entity = False
            current_pos = m.end()

    clean = "".join(clean_parts)
    # Remove extra tags that might remain (e.g. <disfluency>)
    clean = re.sub(r"</?[a-zA-Z][^>]*>", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    # Recompute word indices after tag stripping on the fully clean text
    # since non-NE tags (like <disfluency>) may shift positions
    recomputed: list[EntitySpan] = []
    clean_words = clean.split()
    last_pre_idx = 0
    last_post_idx = 0
    for ent in entities:
        ent_words = ent.text.split()
        if not ent_words:
            continue
        # Adjust hint using cumulative drift from previously matched entities
        drift = last_post_idx - last_pre_idx
        adjusted_hint = ent.start_word_idx + drift
        # Find the entity words in clean_words starting near the expected position
        found = _find_words_in_context(clean_words, ent_words, adjusted_hint)
        if found is None:
            # Fallback: forward linear scan from last found position
            scan_start = max(0, last_post_idx)
            n_ew = len(ent_words)
            for pos in range(scan_start, len(clean_words) - n_ew + 1):
                if clean_words[pos:pos + n_ew] == ent_words:
                    found = pos
                    break
            # Case-insensitive forward scan
            if found is None:
                ent_lower = [w.lower() for w in ent_words]
                for pos in range(scan_start, len(clean_words) - n_ew + 1):
                    if [w.lower() for w in clean_words[pos:pos + n_ew]] == ent_lower:
                        found = pos
                        break
        if found is not None:
            last_pre_idx = ent.start_word_idx
            last_post_idx = found
            recomputed.append(EntitySpan(
                text=ent.text,
                entity_type=ent.entity_type,
                start_word_idx=found,
                end_word_idx=found + len(ent_words),
                char_start=ent.char_start,
                char_end=ent.char_end,
            ))
        else:
            # Fallback: keep original indices if within bounds
            if ent.start_word_idx < len(clean_words):
                recomputed.append(ent)

    return clean, recomputed


def _find_words_in_context(
    text_words: list[str], target_words: list[str], hint_pos: int,
    search_radius: int = 20,
) -> Optional[int]:
    """Find target_words sequence in text_words near hint_pos."""
    n = len(target_words)
    lo = max(0, hint_pos - search_radius)
    hi = min(len(text_words) - n + 1, hint_pos + search_radius + 1)
    # Search nearest-first
    for offset in range(0, search_radius + 1):
        for sign in (0, 1):
            pos = hint_pos + (offset if sign == 0 else -offset)
            if lo <= pos < hi:
                if text_words[pos:pos + n] == target_words:
                    return pos
    # Wider fallback
    for pos in range(lo, hi):
        if text_words[pos:pos + n] == target_words:
            return pos
    # Case-insensitive fallback
    target_lower = [w.lower() for w in target_words]
    for pos in range(max(0, hint_pos - search_radius),
                     min(len(text_words) - n + 1, hint_pos + search_radius + 1)):
        if [w.lower() for w in text_words[pos:pos + n]] == target_lower:
            return pos
    return None


# ---------------------------------------------------------------------------
# Edit-distance alignment (same as asr-word-error-analysis)
# ---------------------------------------------------------------------------

@dataclass
class AlignOp:
    op: str        # "ok" | "sub" | "del" | "ins"
    ref_word: str  # "" for insertions
    hyp_word: str  # "" for deletions
    ref_idx: int   # word index in ref (-1 for insertions)
    hyp_idx: int   # word index in hyp (-1 for deletions)


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


def align(ref_text: str, hyp_text: str) -> AlignResult:
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
    ri, hj = n - 1, m - 1
    while i > 0 or j > 0:
        cell = bt[i][j]
        if cell is None:
            break
        pi, pj, tag = cell
        if tag == "ok":
            ops.append(AlignOp("ok", ref_tokens[i - 1], hyp_tokens[j - 1], i - 1, j - 1))
        elif tag == "sub":
            ops.append(AlignOp("sub", ref_tokens[i - 1], hyp_tokens[j - 1], i - 1, j - 1))
        elif tag == "del":
            ops.append(AlignOp("del", ref_tokens[i - 1], "", i - 1, -1))
        else:
            ops.append(AlignOp("ins", "", hyp_tokens[j - 1], -1, j - 1))
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
# Entity error computation
# ---------------------------------------------------------------------------

@dataclass
class EntityError:
    """Error info for a single entity instance."""
    entity_text: str
    entity_type: str
    ref_words: list[str]
    hyp_words: list[str]
    errors: int
    substitutions: int
    deletions: int
    insertions: int
    error_ops: list[AlignOp]
    utt_id: str
    utt_idx: int


@dataclass
class UtteranceResult:
    idx: int
    utt_id: str
    transcription: str       # original with tags
    ref_clean: str           # tag-stripped reference
    hyp: str
    ref_norm: str            # normalized reference
    hyp_norm: str            # normalized hypothesis
    alignment: AlignResult
    entities: list[EntitySpan]
    entity_errors: list[EntityError]
    total_entity_words: int
    total_entity_errors: int
    row: dict


def _build_ref_idx_to_entity(entities: list[EntitySpan], ref_word_count: int) -> dict[int, EntitySpan]:
    """Map each ref word index to its entity span (if any)."""
    mapping: dict[int, EntitySpan] = {}
    for ent in entities:
        for wi in range(ent.start_word_idx, min(ent.end_word_idx, ref_word_count)):
            mapping[wi] = ent
    return mapping


def compute_entity_errors(
    alignment: AlignResult,
    entities: list[EntitySpan],
    utt_id: str,
    utt_idx: int,
) -> list[EntityError]:
    """Compute errors for each entity span based on the alignment."""
    ref_to_entity = _build_ref_idx_to_entity(entities, alignment.ref_words)

    # Group alignment ops by entity
    entity_ops: dict[int, list[AlignOp]] = collections.defaultdict(list)
    # Track which entities have any alignment coverage
    entity_indices: dict[int, EntitySpan] = {}

    for op in alignment.ops:
        if op.ref_idx >= 0 and op.ref_idx in ref_to_entity:
            ent = ref_to_entity[op.ref_idx]
            ent_key = id(ent)
            entity_ops[ent_key].append(op)
            entity_indices[ent_key] = ent
        elif op.op == "ins" and op.ref_idx == -1:
            # Insertions: check if they fall between entity word positions
            # by looking at neighboring ref indices in the alignment
            pass  # Insertions not adjacent to entities are not counted

    # Also check insertions that are adjacent to entity words
    ops_list = alignment.ops
    for i, op in enumerate(ops_list):
        if op.op == "ins" and op.ref_idx == -1:
            # Check if previous or next non-insertion op belongs to an entity
            prev_ref = None
            next_ref = None
            for j in range(i - 1, -1, -1):
                if ops_list[j].ref_idx >= 0:
                    prev_ref = ops_list[j].ref_idx
                    break
            for j in range(i + 1, len(ops_list)):
                if ops_list[j].ref_idx >= 0:
                    next_ref = ops_list[j].ref_idx
                    break
            # If insertion is between two words of the same entity
            if prev_ref is not None and next_ref is not None:
                prev_ent = ref_to_entity.get(prev_ref)
                next_ent = ref_to_entity.get(next_ref)
                if prev_ent is not None and next_ent is not None and prev_ent is next_ent:
                    ent_key = id(prev_ent)
                    if op not in entity_ops[ent_key]:
                        entity_ops[ent_key].append(op)
                        entity_indices[ent_key] = prev_ent

    results: list[EntityError] = []
    for ent in entities:
        ent_key = id(ent)
        ops = entity_ops.get(ent_key, [])
        ent_ref_words = ent.text.split()
        ent_hyp_words = []
        subs = dels = ins_count = 0
        error_ops: list[AlignOp] = []

        for op in ops:
            if op.op == "ok":
                ent_hyp_words.append(op.hyp_word)
            elif op.op == "sub":
                ent_hyp_words.append(op.hyp_word)
                subs += 1
                error_ops.append(op)
            elif op.op == "del":
                dels += 1
                error_ops.append(op)
            elif op.op == "ins":
                ent_hyp_words.append(op.hyp_word)
                ins_count += 1
                error_ops.append(op)

        errors = subs + dels + ins_count
        results.append(EntityError(
            entity_text=ent.text,
            entity_type=ent.entity_type,
            ref_words=ent_ref_words,
            hyp_words=ent_hyp_words,
            errors=errors,
            substitutions=subs,
            deletions=dels,
            insertions=ins_count,
            error_ops=error_ops,
            utt_id=utt_id,
            utt_idx=utt_idx,
        ))

    return results


def analyze_all(
    rows: list[dict],
    transcription_col: str,
    hyp_col: str,
    id_col: str,
    case_sensitive: bool,
) -> list[UtteranceResult]:
    results: list[UtteranceResult] = []
    for i, row in enumerate(rows):
        transcription = str(row.get(transcription_col, "")).strip()
        hyp_raw = str(row.get(hyp_col, "")).strip()
        uid = str(row.get(id_col, i)) if id_col else str(i)

        # Extract entities and clean reference
        ref_clean, entities = extract_entities_and_clean(transcription)

        # Normalize
        if case_sensitive:
            ref_norm = clean_text(ref_clean)
            hyp_norm = clean_text(hyp_raw)
        else:
            ref_norm = normalize_text(ref_clean)
            hyp_norm = normalize_text(hyp_raw)

        # Re-map entity word indices to normalized text
        # Normalization can change word count, so we re-find entities
        norm_words = ref_norm.split()
        remapped_entities: list[EntitySpan] = []
        # Track cumulative drift between pre-norm and post-norm positions
        last_pre_norm_idx = 0
        last_post_norm_idx = 0
        for ent in entities:
            ent_text = ent.text
            if not case_sensitive:
                ent_text_norm = normalize_text(ent_text)
            else:
                ent_text_norm = clean_text(ent_text)
            ent_words = ent_text_norm.split()
            if not ent_words:
                continue
            # Adjust hint using cumulative drift from previously matched entities
            drift = last_post_norm_idx - last_pre_norm_idx
            adjusted_hint = ent.start_word_idx + drift
            found = _find_words_in_context(norm_words, ent_words, adjusted_hint, search_radius=30)
            if found is None:
                # Fallback: forward linear scan from last found position (efficient for ordered entities)
                scan_start = max(0, last_post_norm_idx)
                n_ew = len(ent_words)
                ent_lower = [w.lower() for w in ent_words]
                for pos in range(scan_start, len(norm_words) - n_ew + 1):
                    if norm_words[pos:pos + n_ew] == ent_words:
                        found = pos
                        break
                # Case-insensitive forward scan
                if found is None:
                    for pos in range(scan_start, len(norm_words) - n_ew + 1):
                        if [w.lower() for w in norm_words[pos:pos + n_ew]] == ent_lower:
                            found = pos
                            break
                # Also try backward from adjusted hint if forward scan fails
                if found is None:
                    for pos in range(min(adjusted_hint, len(norm_words) - n_ew), -1, -1):
                        if norm_words[pos:pos + n_ew] == ent_words:
                            found = pos
                            break
            if found is not None:
                last_pre_norm_idx = ent.start_word_idx
                last_post_norm_idx = found
                remapped_entities.append(EntitySpan(
                    text=ent_text_norm,
                    entity_type=ent.entity_type,
                    start_word_idx=found,
                    end_word_idx=found + len(ent_words),
                    char_start=ent.char_start,
                    char_end=ent.char_end,
                ))

        # Align
        a = align(ref_norm, hyp_norm)

        # Compute entity errors
        ent_errors = compute_entity_errors(a, remapped_entities, uid, i)

        total_ent_words = sum(e.end_word_idx - e.start_word_idx for e in remapped_entities)
        total_ent_errors = sum(e.errors for e in ent_errors)

        results.append(UtteranceResult(
            idx=i,
            utt_id=uid,
            transcription=transcription,
            ref_clean=ref_clean,
            hyp=hyp_raw,
            ref_norm=ref_norm,
            hyp_norm=hyp_norm,
            alignment=a,
            entities=remapped_entities,
            entity_errors=ent_errors,
            total_entity_words=total_ent_words,
            total_entity_errors=total_ent_errors,
            row=row,
        ))
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(results: list[UtteranceResult]) -> dict:
    total_entity_words = sum(r.total_entity_words for r in results)
    total_entity_errors = sum(r.total_entity_errors for r in results)
    total_ref_words = sum(r.alignment.ref_words for r in results)
    total_wer_errors = sum(r.alignment.errors for r in results)

    total_ent_sub = sum(e.substitutions for r in results for e in r.entity_errors)
    total_ent_del = sum(e.deletions for r in results for e in r.entity_errors)
    total_ent_ins = sum(e.insertions for r in results for e in r.entity_errors)
    total_entities = sum(len(r.entities) for r in results)
    entities_with_errors = sum(1 for r in results for e in r.entity_errors if e.errors > 0)

    # Per-type breakdown
    type_stats: dict[str, dict] = collections.defaultdict(
        lambda: {"words": 0, "errors": 0, "subs": 0, "dels": 0, "ins": 0, "count": 0, "err_count": 0}
    )
    for r in results:
        for ent, ent_err in zip(r.entities, r.entity_errors):
            t = ent.entity_type
            n_words = ent.end_word_idx - ent.start_word_idx
            type_stats[t]["words"] += n_words
            type_stats[t]["errors"] += ent_err.errors
            type_stats[t]["subs"] += ent_err.substitutions
            type_stats[t]["dels"] += ent_err.deletions
            type_stats[t]["ins"] += ent_err.insertions
            type_stats[t]["count"] += 1
            if ent_err.errors > 0:
                type_stats[t]["err_count"] += 1

    # Entity substitution pairs (entity-level)
    ent_sub_pairs: collections.Counter[tuple[str, str]] = collections.Counter()
    for r in results:
        for ent_err in r.entity_errors:
            if ent_err.errors > 0:
                ref_str = " ".join(ent_err.ref_words)
                hyp_str = " ".join(ent_err.hyp_words)
                if ref_str != hyp_str:
                    ent_sub_pairs[(ref_str, hyp_str)] += 1

    return {
        "total_utterances": len(results),
        "total_ref_words": total_ref_words,
        "total_wer_errors": total_wer_errors,
        "wer": total_wer_errors / total_ref_words if total_ref_words else 0.0,
        "total_entities": total_entities,
        "total_entity_words": total_entity_words,
        "total_entity_errors": total_entity_errors,
        "ewer": total_entity_errors / total_entity_words if total_entity_words else 0.0,
        "entity_substitutions": total_ent_sub,
        "entity_deletions": total_ent_del,
        "entity_insertions": total_ent_ins,
        "entities_with_errors": entities_with_errors,
        "eer": entities_with_errors / total_entities if total_entities else 0.0,
        "type_stats": dict(type_stats),
        "ent_sub_pairs": ent_sub_pairs,
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_summary(agg: dict, output_dir: Path, dataset: str, input_path: str) -> None:
    type_breakdown = []
    for t, s in sorted(agg["type_stats"].items()):
        type_breakdown.append({
            "entity_type": t,
            "count": s["count"],
            "words": s["words"],
            "errors": s["errors"],
            "ewer": round(s["errors"] / s["words"], 6) if s["words"] else 0.0,
            "entities_with_errors": s["err_count"],
            "eer": round(s["err_count"] / s["count"], 6) if s["count"] else 0.0,
        })

    obj = {
        "dataset": dataset,
        "input_path": input_path,
        "total_utterances": agg["total_utterances"],
        "total_ref_words": agg["total_ref_words"],
        "total_wer_errors": agg["total_wer_errors"],
        "wer": round(agg["wer"], 6),
        "total_entities": agg["total_entities"],
        "total_entity_words": agg["total_entity_words"],
        "total_entity_errors": agg["total_entity_errors"],
        "eer": round(agg["eer"], 6),
        "ewer": round(agg["ewer"], 6),
        "entity_substitutions": agg["entity_substitutions"],
        "entity_deletions": agg["entity_deletions"],
        "entity_insertions": agg["entity_insertions"],
        "entities_with_errors": agg["entities_with_errors"],
        "entity_type_breakdown": type_breakdown,
        "top_entity_substitution_pairs": [
            {"ref_entity": r, "hyp_entity": h, "count": c}
            for (r, h), c in agg["ent_sub_pairs"].most_common(30)
        ],
    }
    path = output_dir / "summary.json"
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    print(f"  summary          -> {path}")


def write_entity_errors(results: list[UtteranceResult], output_dir: Path, top_n: int) -> None:
    all_errors: list[tuple[EntityError, EntitySpan]] = []
    for r in results:
        for ent, ent_err in zip(r.entities, r.entity_errors):
            all_errors.append((ent_err, ent))

    ranked = sorted(all_errors, key=lambda x: x[0].errors, reverse=True)
    path = output_dir / "entity_errors.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "rank", "utt_id", "entity_type", "entity_ref", "entity_hyp",
            "errors", "substitutions", "deletions", "insertions",
        ])
        for rank, (ent_err, ent) in enumerate(ranked[:top_n], 1):
            writer.writerow([
                rank, ent_err.utt_id, ent.entity_type,
                " ".join(ent_err.ref_words), " ".join(ent_err.hyp_words),
                ent_err.errors, ent_err.substitutions, ent_err.deletions, ent_err.insertions,
            ])
    print(f"  entity_errors    -> {path}")


def write_entity_substitutions(agg: dict, output_dir: Path, top_n: int) -> None:
    path = output_dir / "entity_substitutions.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "ref_entity", "hyp_entity", "count"])
        for rank, ((r, h), c) in enumerate(agg["ent_sub_pairs"].most_common(top_n), 1):
            writer.writerow([rank, r, h, c])
    print(f"  entity_subs      -> {path}")


def write_entity_type_breakdown(agg: dict, output_dir: Path) -> None:
    path = output_dir / "entity_type_breakdown.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "entity_type", "count", "entities_with_errors", "eer",
            "total_words", "total_errors", "ewer",
            "substitutions", "deletions", "insertions",
        ])
        for t, s in sorted(agg["type_stats"].items()):
            eer = s["err_count"] / s["count"] if s["count"] else 0.0
            ewer = s["errors"] / s["words"] if s["words"] else 0.0
            writer.writerow([
                t, s["count"], s["err_count"], round(eer, 6),
                s["words"], s["errors"], round(ewer, 6),
                s["subs"], s["dels"], s["ins"],
            ])
    print(f"  entity_type_bkdn -> {path}")


# ---------------------------------------------------------------------------
# HTML report — optimized for very long text with entity highlighting
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    return html_mod.escape(text)


def render_html(
    results: list[UtteranceResult],
    agg: dict,
    output_dir: Path,
    dataset: str,
    input_path: str,
    top_n: int,
) -> None:
    # Rank utterances by entity error count
    ranked = sorted(results, key=lambda r: r.total_entity_errors, reverse=True)[:top_n]
    eer_pct = agg["eer"] * 100
    ewer_pct = agg["ewer"] * 100
    wer_pct = agg["wer"] * 100

    # Entity substitution confusion table
    confusions_html = "\n".join(
        f"<tr><td>{_esc(r)}</td><td>{_esc(h)}</td><td>{c}</td></tr>"
        for (r, h), c in agg["ent_sub_pairs"].most_common(30)
    )

    # Type breakdown table
    type_rows_html = "\n".join(
        f"<tr><td>{_esc(t)}</td><td>{s['count']}</td><td>{s['err_count']}</td>"
        f"<td>{s['err_count']/s['count']*100:.2f}%</td>"
        f"<td>{s['words']}</td><td>{s['errors']}</td>"
        f"<td>{s['errors']/s['words']*100:.2f}%</td>"
        f"<td>{s['subs']}</td><td>{s['dels']}</td><td>{s['ins']}</td></tr>"
        for t, s in sorted(agg["type_stats"].items()) if s["words"] > 0
    )

    # Build per-utterance cards
    cards: list[str] = []
    for rank, r in enumerate(ranked, 1):
        transcript_html = _render_transcript_with_entities(r)
        ewer_val = (r.total_entity_errors / r.total_entity_words * 100) if r.total_entity_words else 0
        ent_with_err = sum(1 for e in r.entity_errors if e.errors > 0)
        eer_val = (ent_with_err / len(r.entities) * 100) if r.entities else 0
        cards.append(
            f"""<article class="card" id="utt-{rank}">
  <header>
    <span class="rank">#{rank}</span>
    <span class="utt-id">{_esc(r.utt_id)}</span>
    <span class="stats">
      EER: {ent_with_err}/{len(r.entities)} entities ({eer_val:.1f}%) &bull;
      EWER: {r.total_entity_errors}/{r.total_entity_words} words ({ewer_val:.1f}%) &bull;
      WER: {r.alignment.wer:.2%} ({r.alignment.errors}/{r.alignment.ref_words})
    </span>
  </header>
  <details open>
    <summary>Transcript with entity highlighting ({len(r.entities)} entities)</summary>
    <div class="transcript">{transcript_html}</div>
  </details>
  <details>
    <summary>Entity error details ({sum(1 for e in r.entity_errors if e.errors > 0)} entities with errors)</summary>
    {_render_entity_error_table(r)}
  </details>
</article>"""
        )
    cards_html = "\n".join(cards)

    # Navigation sidebar
    nav_items = "\n".join(
        f'<a href="#utt-{i+1}" class="nav-item">'
        f'<span class="nav-rank">#{i+1}</span> '
        f'<span class="nav-id">{_esc(r.utt_id)}</span> '
        f'<span class="nav-err">{r.total_entity_errors} err</span>'
        f'</a>'
        for i, r in enumerate(ranked)
    )

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Entity Error Analysis - {_esc(dataset)}</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;font-family:system-ui,-apple-system,sans-serif;background:#f7f8fa;color:#1a1a1a;line-height:1.6}}
.layout{{display:flex;min-height:100vh}}
.sidebar{{width:260px;background:#fff;border-right:1px solid #e0e0e0;padding:12px 0;position:sticky;top:0;height:100vh;overflow-y:auto;flex-shrink:0}}
.sidebar h3{{padding:8px 16px;margin:0;font-size:.85rem;color:#666;text-transform:uppercase;letter-spacing:.05em}}
.nav-item{{display:block;padding:6px 16px;text-decoration:none;color:#333;font-size:.8rem;border-left:3px solid transparent;transition:background .15s}}
.nav-item:hover{{background:#f0f4ff;border-left-color:#4a90d9}}
.nav-rank{{font-weight:700;color:#4a90d9}} .nav-err{{color:#c33;font-size:.75rem}}
main{{flex:1;max-width:1100px;padding:24px 32px;overflow-x:hidden}}
h1{{font-size:1.5rem;margin:0 0 4px}} .subtitle{{color:#666;font-size:.9rem;margin-bottom:20px;word-break:break-all}}
.kpi{{display:flex;gap:14px;flex-wrap:wrap;margin:16px 0}}
.kpi>div{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px 18px;min-width:130px}}
.kpi .val{{font-size:1.4rem;font-weight:700}} .kpi .lbl{{font-size:.8rem;color:#666}}
.kpi .highlight{{border-color:#4a90d9;background:#f0f4ff}}
h2{{font-size:1.15rem;margin-top:2rem;padding-bottom:6px;border-bottom:1px solid #e0e0e0}}
table.data{{border-collapse:collapse;margin:12px 0;width:100%}}
table.data th,table.data td{{border:1px solid #ddd;padding:6px 10px;text-align:left;font-size:.85rem}}
table.data th{{background:#f8f8f8;position:sticky;top:0}}
table.data tr:hover{{background:#fafafa}}
.card{{background:#fff;border:1px solid #ddd;border-radius:10px;margin:16px 0;overflow:hidden}}
.card header{{padding:14px 18px;background:#f8f9fb;border-bottom:1px solid #eee;display:flex;align-items:center;gap:12px;flex-wrap:wrap}}
.card .rank{{font-size:1.1rem;font-weight:700;color:#4a90d9}}
.card .utt-id{{font-family:monospace;font-size:.85rem;color:#555}}
.card .stats{{font-size:.8rem;color:#666;margin-left:auto}}
details{{padding:0}}
summary{{padding:10px 18px;cursor:pointer;font-weight:600;font-size:.9rem;background:#fafbfc;border-top:1px solid #eee;user-select:none}}
summary:hover{{background:#f0f4ff}}
.transcript{{padding:16px 18px;font-size:.9rem;line-height:1.8;white-space:pre-wrap;word-wrap:break-word;max-height:600px;overflow-y:auto}}
.ent{{border-radius:3px;padding:1px 2px;position:relative}}
.ent-ok{{background:#d4edda;border-bottom:2px solid #28a745}}
.ent-err{{background:#fff3cd;border-bottom:2px solid #ffc107}}
.ent-del{{background:#f8d7da;border-bottom:2px solid #dc3545;text-decoration:line-through}}
.ent-sub{{background:#fff3cd;border-bottom:2px solid #fd7e14}}
.ent-ins{{background:#d6ecff;border-bottom:2px solid #007bff}}
.ent-tag{{font-size:.65rem;color:#555;vertical-align:super;margin-left:1px}}
.hyp-inline{{color:#c33;font-size:.8rem;font-style:italic}}
.entity-table{{width:100%;border-collapse:collapse;margin:8px 0}}
.entity-table th,.entity-table td{{border:1px solid #eee;padding:5px 8px;font-size:.82rem;text-align:left}}
.entity-table th{{background:#f8f8f8}}
.entity-table .err-row{{background:#fff8f0}}
.err-sub{{color:#e67e00}} .err-del{{color:#dc3545}} .err-ins{{color:#007bff}}
@media(max-width:900px){{
  .sidebar{{display:none}}
  main{{padding:16px}}
}}
</style></head>
<body>
<div class="layout">
<nav class="sidebar">
  <h3>Utterances</h3>
  {nav_items}
</nav>
<main>
<h1>Entity Error Analysis: {_esc(dataset)}</h1>
<div class="subtitle">Input: {_esc(input_path)}</div>
<div class="kpi">
  <div class="highlight"><div class="val">{eer_pct:.2f}%</div><div class="lbl">EER (Entity Error Rate)</div></div>
  <div class="highlight"><div class="val">{ewer_pct:.2f}%</div><div class="lbl">EWER (Entity Word Error Rate)</div></div>
  <div><div class="val">{wer_pct:.2f}%</div><div class="lbl">Word Error Rate</div></div>
  <div><div class="val">{agg['total_entities']:,}</div><div class="lbl">Total Entities</div></div>
  <div><div class="val">{agg['entities_with_errors']:,}</div><div class="lbl">Entities w/ Errors</div></div>
  <div><div class="val">{agg['total_entity_words']:,}</div><div class="lbl">Entity Words</div></div>
  <div><div class="val">{agg['total_entity_errors']:,}</div><div class="lbl">Entity Word Errors</div></div>
  <div><div class="val">{agg['entity_substitutions']:,}</div><div class="lbl">Entity Subs</div></div>
  <div><div class="val">{agg['entity_deletions']:,}</div><div class="lbl">Entity Dels</div></div>
  <div><div class="val">{agg['entity_insertions']:,}</div><div class="lbl">Entity Ins</div></div>
</div>

<h2>Entity Type Breakdown</h2>
<table class="data"><thead><tr>
  <th>Type</th><th>Count</th><th>w/ Errors</th><th>EER</th><th>Words</th><th>Word Errors</th><th>EWER</th>
  <th>Sub</th><th>Del</th><th>Ins</th>
</tr></thead><tbody>{type_rows_html}</tbody></table>

<h2>Top Entity Substitution Pairs</h2>
<table class="data"><thead><tr><th>Reference Entity</th><th>Hypothesis Entity</th><th>Count</th></tr></thead>
<tbody>{confusions_html}</tbody></table>

<h2>Top {top_n} Utterances by Entity Errors</h2>
{cards_html}

</main></div>
</body></html>"""

    path = output_dir / "report.html"
    path.write_text(page, encoding="utf-8")
    print(f"  report.html      -> {path}")


def _render_transcript_with_entities(r: UtteranceResult) -> str:
    """
    Render the full transcript text with entity spans highlighted.
    Non-entity text is shown plain. Entity text is highlighted with
    color indicating correctness: green = ok, yellow/red = error.
    Shows hyp inline for errored entities.
    """
    # Work with the normalized ref and alignment for accuracy
    ref_words = r.ref_norm.split()
    # Build ref_idx -> entity mapping
    ref_to_entity: dict[int, int] = {}  # ref word idx -> entity index
    for ei, ent in enumerate(r.entities):
        for wi in range(ent.start_word_idx, min(ent.end_word_idx, len(ref_words))):
            ref_to_entity[wi] = ei

    # Build ref_idx -> alignment op mapping
    ref_to_op: dict[int, AlignOp] = {}
    for op in r.alignment.ops:
        if op.ref_idx >= 0:
            ref_to_op[op.ref_idx] = op

    # Also track insertions adjacent to entity words
    ins_after_ref: dict[int, list[AlignOp]] = collections.defaultdict(list)
    ops_list = r.alignment.ops
    last_ref_idx = -1
    for op in ops_list:
        if op.ref_idx >= 0:
            last_ref_idx = op.ref_idx
        elif op.op == "ins" and last_ref_idx >= 0:
            ins_after_ref[last_ref_idx].append(op)

    # Render word by word
    parts: list[str] = []
    i = 0
    while i < len(ref_words):
        if i in ref_to_entity:
            ei = ref_to_entity[i]
            ent = r.entities[ei]
            ent_err = r.entity_errors[ei]

            # Render entity span
            ent_parts: list[str] = []
            for wi in range(ent.start_word_idx, min(ent.end_word_idx, len(ref_words))):
                op = ref_to_op.get(wi)
                word = _esc(ref_words[wi])
                if op is None or op.op == "ok":
                    ent_parts.append(f'<span class="ent-ok">{word}</span>')
                elif op.op == "sub":
                    ent_parts.append(
                        f'<span class="ent-sub" title="sub: {_esc(op.ref_word)} → {_esc(op.hyp_word)}">'
                        f'{word}</span>'
                        f'<span class="hyp-inline">[→{_esc(op.hyp_word)}]</span>'
                    )
                elif op.op == "del":
                    ent_parts.append(
                        f'<span class="ent-del" title="deleted">{word}</span>'
                    )
                # Add insertions after this word if they're within the entity
                for ins_op in ins_after_ref.get(wi, []):
                    if wi < ent.end_word_idx - 1 or (wi == ent.end_word_idx - 1):
                        ent_parts.append(
                            f'<span class="ent-ins" title="inserted">+{_esc(ins_op.hyp_word)}</span>'
                        )

            has_errors = ent_err.errors > 0
            css = "ent-err" if has_errors else "ent-ok"
            tag_label = _esc(ent.entity_type)
            entity_html = " ".join(ent_parts)
            parts.append(
                f'<span class="ent {css}">'
                f'{entity_html}'
                f'<span class="ent-tag">[{tag_label}]</span>'
                f'</span>'
            )
            i = min(ent.end_word_idx, len(ref_words))
        else:
            # Non-entity word - render plain
            parts.append(_esc(ref_words[i]))
            i += 1

    return " ".join(parts)


def _render_entity_error_table(r: UtteranceResult) -> str:
    """Render a table of entity errors for a single utterance."""
    rows: list[str] = []
    for ent, ent_err in zip(r.entities, r.entity_errors):
        css = ' class="err-row"' if ent_err.errors > 0 else ""
        err_detail = ""
        if ent_err.substitutions:
            err_detail += f'<span class="err-sub">S:{ent_err.substitutions}</span> '
        if ent_err.deletions:
            err_detail += f'<span class="err-del">D:{ent_err.deletions}</span> '
        if ent_err.insertions:
            err_detail += f'<span class="err-ins">I:{ent_err.insertions}</span>'
        rows.append(
            f"<tr{css}>"
            f"<td>{_esc(ent.entity_type)}</td>"
            f"<td>{_esc(' '.join(ent_err.ref_words))}</td>"
            f"<td>{_esc(' '.join(ent_err.hyp_words))}</td>"
            f"<td>{ent_err.errors}</td>"
            f"<td>{err_detail}</td>"
            f"</tr>"
        )
    return (
        '<table class="entity-table"><thead><tr>'
        '<th>Type</th><th>Reference</th><th>Hypothesis</th><th>Errors</th><th>Detail</th>'
        '</tr></thead><tbody>'
        + "\n".join(rows)
        + '</tbody></table>'
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve input
    if args.input_path:
        input_path = args.input_path
    else:
        input_path = resolve_latest(args.results_root, args.model, args.dataset)
    print(f"Loading {input_path} ...")

    rows = load_jsonl(input_path)
    print(f"  {len(rows)} utterances loaded")

    # Detect ID column
    id_col = args.id_column or detect_id_column(rows)
    if id_col:
        print(f"  Using ID column: {id_col}")

    # Check required columns
    if args.transcription_column not in rows[0]:
        raise KeyError(
            f"Column '{args.transcription_column}' not found. Available: {sorted(rows[0].keys())}"
        )
    if args.hyp_column not in rows[0]:
        raise KeyError(
            f"Column '{args.hyp_column}' not found. Available: {sorted(rows[0].keys())}"
        )

    # Analyze
    results = analyze_all(rows, args.transcription_column, args.hyp_column, id_col, args.case_sensitive)
    agg = aggregate(results)

    # Print quick summary
    print(f"\n  EER  = {agg['eer']:.4%}  ({agg['entities_with_errors']}/{agg['total_entities']})")
    print(f"  EWER = {agg['ewer']:.4%}  ({agg['total_entity_errors']}/{agg['total_entity_words']})")
    print(f"  WER  = {agg['wer']:.4%}  ({agg['total_wer_errors']}/{agg['total_ref_words']})")
    print(f"  Entities: {agg['total_entities']} total, {agg['entities_with_errors']} with errors")
    print(f"  Entity Sub={agg['entity_substitutions']}  Del={agg['entity_deletions']}  Ins={agg['entity_insertions']}")

    # Write outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting outputs to {output_dir}/")

    write_summary(agg, output_dir, args.dataset, input_path)
    write_entity_errors(results, output_dir, args.top_entities)
    write_entity_substitutions(agg, output_dir, args.top_entities)
    write_entity_type_breakdown(agg, output_dir)

    if args.write_html:
        render_html(results, agg, output_dir, args.dataset, input_path, args.top_n)

    print("\nDone.")


if __name__ == "__main__":
    main()
