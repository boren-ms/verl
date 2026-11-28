# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Generate static biasing JSONL and dataset blocks without decoding any audio.

Run with ``python -m recipe.phimm.data.simulate_biasing --help``. Training and
validation use independent candidate pools; published RLBR validation lists
are converted without resampling or tagging their references.
"""

import argparse
import ast
import csv
import gzip
import hashlib
import html
import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import blobfile as bf
import yaml

from recipe.phimm.data.biasing import BiasingConfig, BiasingSimulator, NoEligibleKeywords, unique_phrases
from recipe.phimm.data.prompts import BIASING_POSITIONS, get_task_output

INTERNAL_DOMAINS = {
    "Gaming": "FY24Q1",
    "Insurance": "FY24Q1",
    "ScienceTech": "FY24Q1",
    "K12HigherEdu": "FY24Q4_en-US_DTEST_OfflineDataCollection",
    "Retail": "FY24Q4_en-US_DTEST_OfflineDataCollection",
}
INTERNAL_ROOT = "az://orngwus2cresco/data/boren/data/Evaluation/InhouseASR/EWER/en-US-entity-v3"


def read_jsonl(path):
    """Stream local or Blob JSONL, including gzip-compressed inputs."""
    with bf.BlobFile(str(path), "rb") as stream:
        source = gzip.GzipFile(fileobj=stream) if str(path).endswith(".gz") else stream
        for number, line in enumerate(source, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a JSON object: {path}:{number}")
                yield row


def librispeech_audio_path(path, audio_root):
    """Map historical host paths to a portable corpus root using their relative suffix."""
    prefix, separator, relative = path.replace("\\", "/").rpartition("/LibriSpeech/")
    if not separator or not prefix or not relative or ".." in relative.split("/"):
        raise ValueError(f"Unrecognized LibriSpeech audio path: {path}")
    return f"{str(audio_root).rstrip('/')}/{relative}"


def read_librispeech_tsv(path, audio_root):
    """Read RLBR's headerless id/paths/conversation TSV without executing literals."""
    with bf.BlobFile(str(path), "r") as stream:
        for number, fields in enumerate(csv.reader(stream, delimiter="\t"), 1):
            if len(fields) != 3:
                raise ValueError(f"Expected three TSV columns: {path}:{number}")
            uid, paths, conversations = fields
            paths = ast.literal_eval(paths)
            messages = ast.literal_eval(conversations)[0]["messages"]
            if len(paths) != 1 or messages[-1]["role"] != "assistant":
                raise ValueError(f"Expected one audio and an assistant reference: {path}:{number}")
            yield {
                "id": uid,
                "audio_path": librispeech_audio_path(paths[0], audio_root),
                "source_audio_path": paths[0],
                "text": messages[-1]["content"],
                "language": "English",
            }


def add_offline_output(row, *, style):
    """Format the reference; dataset loading constructs the prompt."""
    return {
        **row,
        "gt_output": get_task_output(
            style=style,
            language=row["language"],
            text=row["text"],
            components=row.get("components"),
        ),
    }


def write_manifest(rows, path, *, simulator=None, fixed=False, style="verb"):
    """Atomically save rows, counts, and a content hash; never overwrite an existing manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    counts = Counter()
    digest = hashlib.sha256()
    temporary = path.with_suffix(path.suffix + ".tmp")
    rejected_path = path.with_name("rejected_no_keywords.jsonl")
    rejected_temporary = rejected_path.with_suffix(".jsonl.tmp")
    try:
        with temporary.open("xb") as stream, rejected_temporary.open("xb") as rejected:
            for original in rows:
                row = dict(original)
                row.pop("prompt", None)
                counts["source_rows"] += 1
                if not isinstance(row.get("text"), str):
                    raise ValueError(f"Missing string text: {row.get('id')}")
                row["language"] = row.get("language") or "English"
                if fixed:
                    if "biasing_words" not in row and "distractors" not in row:
                        raise ValueError("Fixed candidate input needs biasing_words or distractors")
                    words = unique_phrases(row.get("biasing_words", row.get("distractors")) or [])
                    keywords = unique_phrases(row.get("ground_truth", row.get("keywords")) or [])
                else:
                    try:
                        words, keywords = simulator.sample(row["text"])
                    except NoEligibleKeywords:
                        rejected.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
                        counts["rejected_no_keywords"] += 1
                        continue
                row.update(biasing_words=words, keywords=keywords)
                row = add_offline_output(row, style=style)
                encoded = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
                stream.write(encoded)
                digest.update(encoded)
                counts["rows"] += 1
                counts["with_biasing"] += bool(words)
                counts["with_keywords"] += bool(keywords)
                counts["biasing_words"] += len(words)
                counts["keywords"] += len(keywords)
                audio = row.get("audio_path", "")
                for split in ("train-clean-100", "train-clean-360", "train-other-500"):
                    if f"/{split}/" in audio:
                        counts[split] += 1
                if counts["rows"] % 10000 == 0:
                    print(json.dumps({"manifest": str(path), **counts}), flush=True)
        if not counts["rows"]:
            raise ValueError("Input contains no rows")
        temporary.rename(path)
        if counts["rejected_no_keywords"]:
            rejected_temporary.rename(rejected_path)
        else:
            rejected_temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        rejected_temporary.unlink(missing_ok=True)
        raise
    summary = {"manifest": str(path.resolve()), "sha256": digest.hexdigest(), **counts}
    print(json.dumps(summary), flush=True)
    return summary


def dataset_block(path, source, *, position, style, lang_hint):
    """Load a static manifest and choose prompt placement without generating any words."""
    return {
        "dataset_name": "jsonl",
        "jsonl_paths": str(Path(path).resolve()),
        "add_task_info": {"style": style, "lang_hint": lang_hint, "biasing": position},
        "post_process": {
            "add_field": {"fields": {"data_source": source}},
            "verl_format": {"prompt_key": "prompt", "extra_keys": ["biasing_words"]},
        },
    }


def suite_dataset_config(train_path, validation, *, position, style, lang_hint, training_rows):
    """Replace the data mix and register source scorers on boren/main_2611_dev."""
    return {
        "data": {
            # The base's 1,024-token limit cannot hold 1,000 rare candidates
            # plus expanded audio. Final capacity must be checked with the model.
            "max_prompt_length": 8192,
            "train_data": [dataset_block(train_path, "asr", position=position, style=style, lang_hint=lang_hint)],
            "mix": [{"shuffle": True, "datasource": [{"name": "asr", "epoch": 1, "shuffle": True}]}],
            "val_data": [
                dataset_block(path, name, position=position if budget else None, style=style, lang_hint=lang_hint)
                for path, name, budget in validation
            ],
        },
        "reward_map": {"asr": "train_librispeech", **{name: "val_openasr_en" for _, name, _ in validation}},
        "actor_rollout_ref": {
            "actor": {"ppo_max_token_len_per_gpu": "${data.max_model_length}"},
            "ref": {"log_prob_max_token_len_per_gpu": "${data.max_model_length}"},
            "rollout": {"log_prob_max_token_len_per_gpu": "${data.max_model_length}"},
        },
        "asr": {"expected_training_rows": training_rows, "expected_training_steps": None},
    }


def prepare_rlbr_suite(args, config, common_words):
    """Prepare the full LibriSpeech 960h training manifest and six RLBR validation cohorts."""
    source = Path(args.rlbr_data_root)
    tsv = source / "LibriSpeech/asr_train_transcribe.tsv"
    factory = lambda: read_librispeech_tsv(tsv, args.audio_root)
    simulator = BiasingSimulator(config, seed=args.seed, common_words=common_words)
    simulator.prepare_word_pool(args.candidate_words)
    train_path = args.output_dir / "train/data.jsonl"
    summary = {
        "training": write_manifest(
            factory(),
            train_path,
            simulator=simulator,
            style=args.style,
        ),
        "validation": [],
        "inputs": [str(tsv)],
    }
    summary["pool_unique_phrases"] = {"English": len(simulator.pool)}
    validation = []
    for split, short in (("test-clean", "clean"), ("test-other", "other")):
        for budget in (0, 100, 500):
            path = source / f"librispeech_biasing/ref/{split}.biasing_{budget or 100}.jsonl"

            def rows(path=path, budget=budget):
                for original in read_jsonl(path):
                    row = dict(original)
                    row["source_audio_path"] = row["audio_path"]
                    row["audio_path"] = librispeech_audio_path(row["audio_path"], args.audio_root)
                    row["biasing_words"] = row.get("distractors", []) if budget else []
                    yield row

            target = args.output_dir / f"validation/{short}_{budget}.jsonl"
            result = write_manifest(rows(), target, fixed=True, style=args.style)
            summary["validation"].append(result)
            summary["inputs"].append(str(path))
            validation.append((target, f"{short}_{budget}", budget))
    if args.include_internal_validation:
        extra_results, extra_blocks = prepare_internal_validation(args)
        summary["validation"].extend(extra_results)
        validation.extend(extra_blocks)
    for position in (None, *BIASING_POSITIONS):
        blocks = suite_dataset_config(
            train_path,
            validation,
            position=position,
            style=args.style,
            lang_hint=args.lang_hint,
            training_rows=summary["training"]["rows"],
        )
        (args.output_dir / f"datasets_{position or 'none'}.yaml").write_text(
            "# @package _global_\n" + yaml.safe_dump(blocks, sort_keys=False)
        )
    return summary


def prepare_internal_validation(args):
    """Convert the five RLBR entity domains and freeze 0/100/500 candidate conditions."""
    source_root = args.internal_source_root.rstrip("/")
    rare_uri = source_root.split("/Evaluation/", 1)[0] + "/librispeech_biasing/words/all_rare_words.txt"
    cache = args.output_dir / "sources"
    cache.mkdir(exist_ok=True)
    rare_path = cache / "all_rare_words.txt"
    bf.copy(rare_uri, str(rare_path))
    rare = [line.split()[0] for line in rare_path.read_text().splitlines() if line.strip()][:500]
    results, blocks = [], []
    for domain, version in INTERNAL_DOMAINS.items():
        suffix = version if "en-US_DTEST" in version else version + "_en-US_DTEST"
        directory = f"Conversation_DomainSet_DTEST_{domain}_Entity_{suffix}"
        uri = f"{source_root}/{directory}/segments.jsonl"
        local = cache / f"{domain}.segments.jsonl"
        bf.copy(uri, str(local))
        for budget in (0, 100, 500):

            def rows(budget=budget, local=local, uri=uri):
                for original in read_jsonl(local):
                    tagged = original["Transcription"]
                    entities = list(
                        dict.fromkeys(
                            html.unescape(re.sub(r"<[^>]*>", "", value)).strip()
                            for value in re.findall(
                                r"<NE(?:\s[^>]*|:[^>]*)?>(.*?)</NE[^>]*>", tagged, flags=re.I | re.S
                            )
                        )
                    )
                    entities = [entity for entity in entities if entity]
                    words = list(dict.fromkeys(entities[:budget] + rare[: max(0, budget - len(entities[:budget]))]))
                    historical = original["WavPath"]
                    relative = historical.split("/en-US-entity-v3/", 1)
                    if len(relative) != 2:
                        raise ValueError(f"Unexpected internal audio root: {historical}")
                    yield {
                        "id": original["UUID"],
                        "audio_path": f"{source_root}/{relative[1]}",
                        "source_audio_path": historical,
                        "text": html.unescape(re.sub(r"<[^>]*>", "", tagged)).strip(),
                        "entity_transcription": tagged,
                        "keywords": entities,
                        "biasing_words": words,
                        "language": "English",
                        "source_manifest": uri,
                    }

            path = args.output_dir / f"validation/{domain}_{budget}.jsonl"
            result = write_manifest(rows(), path, fixed=True, style=args.style)
            results.append({**result, "source_manifest": uri})
            blocks.append((path, f"{domain}_{budget}", budget))
    return results, blocks


def main():
    """Generate offline training samples or convert fixed RLBR validation lists."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-manifest", nargs="+")
    source.add_argument("--rlbr-data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--negative-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-num", type=int, default=1000)
    parser.add_argument("--keyword-coverage", type=float, nargs=2, default=(0.1, 1.0), metavar=("MIN", "MAX"))
    parser.add_argument("--candidate-word-file", help="Vocabulary; first column of each line")
    parser.add_argument("--common-word-file")
    parser.add_argument("--common-word-num", type=int, default=3000)
    parser.add_argument("--fixed-candidates", action="store_true")
    parser.add_argument("--include-internal-validation", action="store_true")
    parser.add_argument("--audio-root", default="az://orngwus2cresco/data/boren/data/LibriSpeech")
    parser.add_argument("--internal-source-root", default=INTERNAL_ROOT)
    parser.add_argument("--style", choices=("verb", "lex", "display"), default="verb")
    parser.add_argument("--lang-hint", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.fixed_candidates and (args.negative_only or args.rlbr_data_root):
        parser.error("--fixed-candidates requires JSONL input and cannot use --negative-only")
    if args.common_word_num < 0:
        parser.error("--common-word-num must be nonnegative")
    config = BiasingConfig(args.max_num, tuple(args.keyword_coverage), args.negative_only)
    common_file = args.common_word_file
    if args.rlbr_data_root and not common_file:
        common_file = str(args.rlbr_data_root / "librispeech_biasing/words/all_words.count.txt")
    candidate_file = args.candidate_word_file or (common_file if args.rlbr_data_root else None)
    if not args.fixed_candidates and not candidate_file:
        parser.error("--candidate-word-file is required for JSONL sampling")

    def read_words(path, limit=None):
        if not path:
            return []
        with bf.BlobFile(path, "r") as stream:
            return [
                line.split()[0]
                for index, line in enumerate(stream)
                if (limit is None or index < limit) and line.strip()
            ]

    common_words = read_words(common_file, args.common_word_num)
    args.candidate_words = read_words(candidate_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "preparation.json").exists():
        raise FileExistsError(f"Preparation already exists: {args.output_dir}")
    if args.rlbr_data_root:
        summary = prepare_rlbr_suite(args, config, common_words)
    else:

        def rows():
            for path in args.input_manifest:
                yield from read_jsonl(path)

        simulator = BiasingSimulator(config, seed=args.seed, common_words=common_words)
        if not args.fixed_candidates:
            simulator.prepare_word_pool(args.candidate_words)
        summary = write_manifest(
            rows(), args.output_dir / "data.jsonl", simulator=simulator, fixed=args.fixed_candidates, style=args.style
        )
        summary["inputs"] = args.input_manifest
    summary["sampler"] = {
        "seed": args.seed,
        **asdict(config),
        "candidate_word_file": candidate_file,
        "common_word_file": common_file,
        "common_word_num": len(common_words),
    }
    (args.output_dir / "preparation.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
