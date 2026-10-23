"""Check real tokenizer GT round-trips before any keyword experiment launch."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from recipe.phimm.data.keyword_output import parse_keyword_output, validate_keywords
from recipe.phimm.data.prompts import get_task_output
from recipe.phimm.prepare_ls_keywords import digest
from recipe.phimm.reward.asr_response import get_hyp_text


def preflight(manifest_root, tokenizer_path, response_length=768):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False, local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("missing tokenizer EOS")
    summary = {
        "tokenizer_path": str(tokenizer_path),
        "response_length": response_length,
        "tokenizer_sha256": digest(Path(tokenizer_path) / "tokenizer.json"),
        "manifests": {},
    }
    for name in ("train_pilot10k_seed42", "dev", "internal_test", "test_clean", "test_other"):
        path = Path(manifest_root) / f"{name}.jsonl"
        lengths = []
        for line in path.open():
            row = json.loads(line)
            if not row.get("keyword_annotation_valid"):
                raise ValueError(f"{row['id']}: unvalidated labels")
            validate_keywords(row["text"], row["keywords"])
            answer = get_task_output(
                task="lang_asr_verb",
                text=row["text"],
                lang="English",
                version=2609,
                output_keywords=True,
                keywords=row["keywords"],
            )
            ids = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
            if len(ids) > response_length:
                raise ValueError(f"{row['id']}: GT length {len(ids)} exceeds {response_length}")
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            parsed = parse_keyword_output(decoded, "stop")
            if not parsed.valid or parsed.predicted_keywords != row["keywords"]:
                raise ValueError(f"{row['id']}: keyword tokenizer round-trip failed")
            if get_hyp_text(parsed.body_raw, version=2609) != row["text"]:
                raise ValueError(f"{row['id']}: transcription tokenizer round-trip failed")
            lengths.append(len(ids))
        ordered = sorted(lengths)
        summary["manifests"][name] = {
            "rows": len(lengths),
            "sha256": digest(path),
            "max_tokens": max(ordered),
            "p99_tokens": ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))],
            "roundtrip_failures": 0,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--response-length", type=int, default=768)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = preflight(args.manifest_root, args.tokenizer, args.response_length)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
