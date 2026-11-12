"""Check the pinned pname source, checkpoint assets, scorer and real audio loader."""

import argparse
import hashlib
import json
from pathlib import Path

import blobfile as bf
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from recipe.phimm.data.prompts import get_task_output
from recipe.phimm.data.rl_dataset import RLHFDataset
from recipe.phimm.reward.asr_eval import openasr_en_eval
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local


def check(output):
    root = Path(__file__).resolve().parents[1]
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {}, {}))
    with initialize_config_dir(config_dir=str(root / "recipe/phimm/config/v2611_dev/gen"), version_base=None):
        config = compose(config_name="gen_pname_rover_v1_kwerr_2611r2")
    OmegaConf.resolve(config)
    source = config.data.gen_data[0].jsonl_paths
    with bf.BlobFile(source, "rb") as stream:
        source_bytes = stream.read()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    if source_sha != "1157db0daae53e1d102b97a3326ac6b1933effa25260a6b5fb224891c699c28f":
        raise ValueError("Pinned Rover validation source changed")
    originals = [json.loads(line) for line in source_bytes.splitlines()]
    if len(originals) != 8930 or len({row["id"] for row in originals}) != 8930:
        raise ValueError("Unexpected source cardinality")

    model = config.model.path.rstrip("/")
    with bf.BlobFile(model + "/config.json", "rb") as stream:
        model_config_bytes = stream.read()
    index_bytes = None
    if bf.exists(model + "/model.safetensors.index.json"):
        with bf.BlobFile(model + "/model.safetensors.index.json", "rb") as stream:
            index_bytes = stream.read()
        shards = sorted(set(json.loads(index_bytes)["weight_map"].values()))
    elif bf.exists(model + "/model.safetensors"):
        shards = ["model.safetensors"]
    else:
        raise ValueError("No supported HF safetensors weights found")
    weights = [{"file": name, "bytes": bf.stat(model + "/" + name).size} for name in shards]
    if not weights or not all(row["bytes"] > 0 for row in weights):
        raise ValueError("Incomplete HF model shards")
    print(f"Verified source and {len(weights)} checkpoint shards; staging model", flush=True)
    local_model = copy_to_local(config.model.path)
    tokenizer = hf_tokenizer(local_model, trust_remote_code=config.model.trust_remote_code)
    processor = hf_processor(local_model, trust_remote_code=config.model.trust_remote_code, use_fast=True)
    dataset = RLHFDataset(config.data.gen_data, tokenizer, config.data, processor, is_train=False)
    if len(dataset) != 8930:
        raise ValueError("Actual generation dataset changed membership")
    longest = max(range(len(originals)), key=lambda index: len(originals[index]["updated_text"]))
    samples = []
    for index in dict.fromkeys([0, longest]):
        row = dataset[index]
        seconds = sum(len(wave) / rate for wave, rate in row["multi_modal_data"]["audio"])
        if seconds <= 0 or row["extra_info"]["id"] != originals[index]["id"]:
            raise ValueError("Audio loading or record order failed")
        expected = row["reward_model"]["ground_truth"]
        score = openasr_en_eval(
            get_task_output("lang_asr_verb", lang="English", text=expected, version=2609),
            expected,
            version=2609,
            extra_info=row["extra_info"],
        )
        if score["n_err"] != 0 or score["nb_err"] != 0:
            raise ValueError("Exact-reference scorer check failed")
        samples.append(
            {
                "id": row["extra_info"]["id"],
                "seconds": seconds,
                "prompt_tokens_after_padding": int(row["attention_mask"].sum()),
                "reference_tokens": len(tokenizer.encode(expected)),
                "score": score,
            }
        )
    result = {
        "source": source,
        "source_sha256": source_sha,
        "records": len(dataset),
        "model": model,
        "model_config_sha256": hashlib.sha256(model_config_bytes).hexdigest(),
        "model_index_sha256": hashlib.sha256(index_bytes).hexdigest() if index_bytes is not None else None,
        "weights": weights,
        "samples": samples,
        "loader_passed": True,
        "scorer_passed": True,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    check(parser.parse_args().output)
