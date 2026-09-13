#!/usr/bin/env python3
"""Partition a repeated long-audio transcript across segment hypotheses."""

import argparse
import json
import unicodedata
from collections import defaultdict
from pathlib import Path

from kaldialign import align


EPSILON = "<eps>"


def canonicalize(word: str) -> str:
    normalized = unicodedata.normalize("NFKC", word).casefold()
    canonical = "".join(character for character in normalized if character.isalnum())
    return canonical or normalized


def assign_reference_words(reference: str, hypotheses: list[str]) -> list[str]:
    """Return a lossless, monotonic partition of reference for hypotheses."""
    reference_words = reference.split()
    hypothesis_words = []
    hypothesis_owners = []
    for segment_index, hypothesis in enumerate(hypotheses):
        words = hypothesis.split()
        hypothesis_words.extend(words)
        hypothesis_owners.extend([segment_index] * len(words))

    if not reference_words:
        return [""] * len(hypotheses)
    if not hypothesis_words:
        result = [""] * len(hypotheses)
        result[0] = reference
        return result

    owners: list[int | None] = []
    hypothesis_index = 0
    for reference_symbol, hypothesis_symbol in align(
        [canonicalize(word) for word in reference_words],
        [canonicalize(word) for word in hypothesis_words],
        eps_symbol=EPSILON,
        merge_compounds=False,
    ):
        consumes_reference = reference_symbol != EPSILON
        consumes_hypothesis = hypothesis_symbol != EPSILON
        if consumes_reference:
            owner = hypothesis_owners[hypothesis_index] if consumes_hypothesis else None
            owners.append(owner)
        if consumes_hypothesis:
            hypothesis_index += 1

    if len(owners) != len(reference_words) or hypothesis_index != len(hypothesis_words):
        raise RuntimeError("Alignment did not consume all reference and hypothesis words")

    fill_unassigned_owners(owners, len(hypotheses))
    if any(left > right for left, right in zip(owners, owners[1:])):
        raise RuntimeError("Alignment produced non-monotonic segment assignments")

    partitions = [[] for _ in hypotheses]
    for word, owner in zip(reference_words, owners):
        partitions[owner].append(word)
    result = [" ".join(words) for words in partitions]
    if " ".join(result).split() != reference_words:
        raise RuntimeError("Aligned references do not reconstruct the original transcript")
    return result


def fill_unassigned_owners(owners: list[int | None], segment_count: int) -> None:
    index = 0
    while index < len(owners):
        if owners[index] is not None:
            index += 1
            continue
        end = index
        while end < len(owners) and owners[end] is None:
            end += 1

        left = owners[index - 1] if index else None
        right = owners[end] if end < len(owners) else None
        first_owner = 0 if left is None else left
        last_owner = segment_count - 1 if right is None else right
        if first_owner > last_owner:
            raise RuntimeError("Cannot assign a deletion span to monotonic segments")

        run_length = end - index
        owner_span = last_owner - first_owner + 1
        for offset in range(run_length):
            owners[index + offset] = first_owner + min(
                owner_span - 1, offset * owner_span // run_length
            )
        index = end


def align_records(
    records: list[dict],
    group_field: str,
    segment_index_field: str,
    reference_field: str,
    hypothesis_field: str,
    aligned_reference_field: str,
    parent_reference_field: str,
) -> None:
    groups = defaultdict(list)
    for row_index, record in enumerate(records):
        for field in (group_field, segment_index_field, reference_field, hypothesis_field):
            if field not in record:
                raise ValueError(f"Row {row_index + 1} is missing required field {field!r}")
        groups[record[group_field]].append((row_index, record))

    for group_key, members in groups.items():
        members.sort(key=lambda item: item[1][segment_index_field])
        segment_indexes = [record[segment_index_field] for _, record in members]
        if len(segment_indexes) != len(set(segment_indexes)):
            raise ValueError(f"Group {group_key!r} contains duplicate segment indexes")

        references = {record[reference_field] for _, record in members}
        if len(references) != 1:
            raise ValueError(f"Group {group_key!r} contains inconsistent references")
        parent_reference = references.pop()
        aligned_references = assign_reference_words(
            parent_reference,
            [str(record[hypothesis_field] or "") for _, record in members],
        )
        for (_, record), aligned_reference in zip(members, aligned_references):
            if parent_reference_field:
                record[parent_reference_field] = parent_reference
            record[aligned_reference_field] = aligned_reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align a full repeated transcript to monotonic segment hypotheses."
    )
    parser.add_argument("input", type=Path, help="Input JSONL path")
    parser.add_argument("output", type=Path, help="Output JSONL path")
    parser.add_argument("--group-field", default="parent_audio_path")
    parser.add_argument("--segment-index-field", default="seg_index")
    parser.add_argument("--reference-field", default="gts")
    parser.add_argument("--hypothesis-field", default="clean_output")
    parser.add_argument(
        "--aligned-reference-field",
        default="gts",
        help="Destination for each aligned segment reference. Default: overwrite gts in the output.",
    )
    parser.add_argument(
        "--parent-reference-field",
        default="parent_gts",
        help="Field used to preserve the full transcript. Pass an empty value to omit it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    with args.input.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
    if not records:
        raise ValueError(f"No records found in {args.input}")

    align_records(
        records,
        args.group_field,
        args.segment_index_field,
        args.reference_field,
        args.hypothesis_field,
        args.aligned_reference_field,
        args.parent_reference_field,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Aligned {len(records)} segments to {args.output}")


if __name__ == "__main__":
    main()