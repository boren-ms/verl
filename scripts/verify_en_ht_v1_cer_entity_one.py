"""Verify recipe/phimm/config/data/train_data/en_ht_v1_cer_entity_one.yaml.

Loads the dataset, takes the first 10 examples, prints them, and writes them
as JSON to tmp/en_ht_v1_cer_entity_one_first10.json.
"""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recipe.phimm.data.dataset import create_datasets

YAML_PATH = Path("recipe/phimm/config/data/train_data/en_ht_v1_cer_entity_one.yaml")
OUT_PATH = Path("tmp/en_ht_v1_cer_entity_one_first10.json")


def main() -> None:
    cfg = yaml.safe_load(YAML_PATH.read_text())
    # Limit the dataset early to make the verification cheap. The directory has
    # ~800 shards; restrict to a single shard for the smoke test.
    sample_shard = (
        "az://orngwus2cresco/data/boren/data/audio_trans_filter_cer05_entity/"
        "inhouse/mlang_asr_data_2025_readable/en/en_asr_ht/short_form_fix_16k/r8/"
        "Engineer_Telephony_8kHz/samples/000000.jsonl.gz"
    )
    if isinstance(cfg, list):
        for spec in cfg:
            spec["jsonl_paths"] = [sample_shard]
    elif isinstance(cfg, dict):
        cfg["jsonl_paths"] = [sample_shard]

    ds = create_datasets(cfg)
    # When the yaml is a list of one spec, datasets returns a dict.
    if isinstance(ds, dict):
        ds = next(iter(ds.values()))

    rows = []
    for i, egs in enumerate(ds):
        if i >= 10:
            break
        rows.append(egs)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
    print(f"Wrote {len(rows)} examples to {OUT_PATH}")
    print("--- First example ---")
    print(json.dumps(rows[0], ensure_ascii=False, indent=2, default=str)[:1500])


if __name__ == "__main__":
    main()
