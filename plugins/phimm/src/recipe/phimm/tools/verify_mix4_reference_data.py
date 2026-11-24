"""Verify complete Mix4 manifests through production transforms and ASR loader."""

import argparse
import fcntl
import gzip
import hashlib
import json
import os
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import datasets
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from recipe.phimm.agentic.context_prompt import ContextPrompt, context_source
from recipe.phimm.agentic.dataset import ASRDataset, _align_source_features, _combine_datasets
from recipe.phimm.data.dataset import create_audio_dataset
from recipe.phimm.reward.asr_response import get_hyp_text

WORK = Path(__file__).resolve().parents[6]
ROOT = Path(os.environ.get("MIX4_VERIFY_ROOT", str(WORK / "tmp/mix4_reference_context")))
# Release-specific immutable manifest inventory.
PREFIX_0 = "az://orngwus2cresco/data/boren/data/verl/mix4_reference_context_20261124/data/short_orignal/"
PREFIX_1 = "az://orngwus2cresco/data/boren/data/verl/mix4_reference_context_20261124/data/short_rs23/"
PREFIX_2 = (
    "az://orngwus2cresco/data/boren/data/Training/fl_earnings_short_random_merge23_x4_reference_text_v2_20261123/"
)
PREFIX_3 = "az://orngwus2cresco/data/boren/data/Training/earnings_short_contiguous5_reference_text_v2/"
EXPECTED_SHA = {
    PREFIX_0 + "quartr_second48.jsonl.gz": ("897230fdf3245415782e566de5e2edd2e7680983a666384edc9b70ff2e83ef46"),
    PREFIX_0 + "earnings_batch1.jsonl.gz": ("b6a6d87cea538f1c78d2e703c38df5adae6fd6cc12db269ff8346ecfb13318b3"),
    PREFIX_0 + "quartr_first23.jsonl.gz": ("b18fcb2b0955f7ea4bbbe8696a4e3e6de241ca8c992d2c0156d336a41536ec5f"),
    PREFIX_0 + "quartr_final38.jsonl.gz": ("0a9be71d43702b279e97be3df65b55da07b9b4249065d7efe83568298a8716f3"),
    PREFIX_0 + "earnings_batch2.jsonl.gz": ("75836965626d6de4a7aa2d446bb8c0f9d23edd041af10ba3c20bc3ba0a9f638c"),
    PREFIX_1 + "quartr_second48.jsonl.gz": ("441da2dd6c74219e59bf2ae179d89a58a36647224dba215dd589a03870f2ad54"),
    PREFIX_1 + "earnings_batch1.jsonl.gz": ("b04b9fe654bb4a76166295cff8bca0aee80a28e965715a4e6f93961b812acbac"),
    PREFIX_1 + "quartr_first23.jsonl.gz": ("0043622bd6f4814442e0c2b31097467ebfd184a2a71f250204e128c411a84b58"),
    PREFIX_1 + "quartr_final38.jsonl.gz": ("1be90307f655ae314a0a4435ee70f7f1584f0792481b23d2dc5e6127fc443e1b"),
    PREFIX_1 + "earnings_batch2.jsonl.gz": ("334808ae4b4f32987585dcefe5afca8ea117fccca7ea8c4669712706335302b1"),
    PREFIX_2 + "source_000.jsonl": ("03801784726eaf0eecd0dbd1cb85e988489db0fbf93a76abee0c762fa59fc288"),
    PREFIX_2 + "source_001.jsonl": ("686d3c1f971fb9af7804291fc1eb3f6a8a365a0a53dd75dd34a506ec1ae2d7b9"),
    PREFIX_2 + "source_002.jsonl": ("23ec720a5e6a326de50de0db0fca3a72522033b22720ce72ed6364a77d34ec56"),
    PREFIX_2 + "source_003.jsonl": ("a393c26cc9448e19dfc83958b16aa72f13a2092d996f7e2454ed253186f24f44"),
    PREFIX_2 + "source_004.jsonl": ("82df31070ab705af1505fd44cb848597415a2830a7ec599f984d5a900179fada"),
    PREFIX_2 + "source_005.jsonl": ("e442dc994f509598b7014e66792995754dbde5dfc6aa26ca66f6f63c97e599cf"),
    PREFIX_3 + "source_000.jsonl.gz": ("0fee6d31da1b9c514303ccedf470b6786a08de28cab2596cc2dec5aa37dac123"),
    PREFIX_3 + "source_001.jsonl.gz": ("a5c5171817082deb8d2d7fc3a4792c9357fad33dac8d0bd41f5410a1e5b720a1"),
    PREFIX_3 + "source_002.jsonl.gz": ("f97f36f26ab2c622082e77779d1b4ae57f68f434a0a55c63d9a1fecae781c6c5"),
    PREFIX_3 + "source_003.jsonl.gz": ("f299bf9e9b0a3ada4fc334d548417c75bd0ae5ea911904f2c6bac62273d4fea4"),
    PREFIX_3 + "source_004.jsonl.gz": ("b0d60d367020315f53cb7003c192b59193ac0d1d548dc608a49db86def5731b5"),
}
EXPECTED = {
    "short_orignal": (25240, 25240),
    "flshort_rm23x4": (111888, 279720),
    "contig5": (24520, 122600),
    "short_rs23": (25240, 37915),
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main(variant_filter=None):
    datasets.disable_progress_bars()
    OmegaConf.register_new_resolver("eval", lambda s: eval(s, {"__builtins__": {}}, {}), replace=True)
    with initialize_config_dir(
        config_dir=str(WORK / "plugins/phimm/src/recipe/phimm/config/v2611_dev/earnings"), version_base=None
    ):
        cfg = compose(config_name="prompt_ctx/remax_mix4_reference_text_h4_bs128_ppo512_n4_gspo")
        old = compose(config_name="prompt_ctx/remax_mix4_text_h4_bs128_ppo512_n4_gspo")
    OmegaConf.resolve(cfg)
    OmegaConf.resolve(old)
    a = OmegaConf.to_container(cfg.actor_rollout_ref)
    b = OmegaConf.to_container(old.actor_rollout_ref)
    a["rollout"]["trace"]["experiment_name"] = b["rollout"]["trace"]["experiment_name"]
    assert a == b
    assert OmegaConf.to_container(cfg.algorithm) == OmegaConf.to_container(old.algorithm)
    assert OmegaConf.to_container(cfg.data.mix) == OmegaConf.to_container(old.data.mix)
    assert cfg.data.train_batch_size == 128 and cfg.trainer.total_training_steps == 2000
    assert cfg.trainer.nnodes == 4 and cfg.trainer.n_gpus_per_node == 8
    assert cfg.trainer.save_freq == 100 and cfg.trainer.test_freq == 400
    assert context_source(cfg.asr.context_aware, training=True) == "reference"
    assert context_source(cfg.asr.context_aware, training=False) == "reference"
    proof = {
        "composed_configuration": "passed",
        "unchanged_model_loss_optimizer_mixture": True,
        "training_source": "reference",
        "validation_source": "reference",
        "fourth_source_complete": True,
        "sources": [],
        "variants": {},
        "empty_reference_cases": [],
    }
    processed = []
    ROOT.mkdir(parents=True, exist_ok=True)

    def stage(item):
        uri, expected = item
        path = ROOT / "manifests" / hashlib.sha256(uri.encode()).hexdigest()[:16] / Path(uri).name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or digest(path) != expected:
            subprocess.run(["bbb", "cp", uri, str(path)], check=True, timeout=1200)
        assert digest(path) == expected, (uri, "published SHA mismatch")
        print("STAGED_SHA_PASS", uri, flush=True)
        return uri, path

    with ThreadPoolExecutor(max_workers=6) as pool:
        staged_paths = dict(pool.map(stage, EXPECTED_SHA.items()))
    for group in cfg.data.train_data:
        for conf in group:
            conf = OmegaConf.to_container(conf, resolve=True)
            variant = conf["post_process"]["add_field"]["fields"]["data_source"]
            if variant_filter and variant != variant_filter:
                continue
            remote = conf["jsonl_paths"]
            name = Path(remote).name
            path = staged_paths[remote]
            assert path.exists(), path
            # Validate the same bytes referenced by the published preset.
            sha = digest(path)
            assert sha == EXPECTED_SHA[remote]
            opener = gzip.open if path.suffix == ".gz" else open
            parents = segments = empty = 0
            ids = set()
            with opener(path, "rt") as stream:
                for line in stream:
                    row = json.loads(line)
                    refs = row["segment_reference_texts"]
                    paths = row["audio_path"]
                    assert isinstance(paths, list) and len(paths) == len(refs) and all(isinstance(x, str) for x in refs)
                    assert all(isinstance(p, str) and p for p in paths)
                    assert row["id"] not in ids, (path, row["id"])
                    ids.add(row["id"])
                    parents += 1
                    segments += len(refs)
                    empty += sum(not x for x in refs)
                    if variant == "short_rs23":
                        assert "".join(refs) == row["text"]
                        if row.get("reference_alignment_status") == "NO_POSITIVE_DURATION_REFERENCE":
                            proof["empty_reference_cases"].append(
                                {"id": row["id"], "source": row["original_data_source"], "references": refs}
                            )
            conf["jsonl_paths"] = str(path)
            ds = create_audio_dataset(**{**conf, "version": cfg.data.version})
            assert len(ds) == parents, (variant, len(ds), parents)
            adapter = ASRDataset.__new__(ASRDataset)
            adapter.ds = ds
            adapter.groups = None
            adapter.prompt_key = cfg.data.prompt_key
            for index in range(len(adapter)):
                sample = adapter[index]
                rows = sample["asr_segments"]
                refs = sample["extra_info"]["segment_reference_texts"]
                assert [s["asr_reference_text"] for s in rows] == refs
                context = ContextPrompt("text", max_history=4)
                history = []
                for seg in rows:
                    expected = "\n".join(text for text in history[-4:] if text)
                    assert context.prompt == ("Previous context: " + expected if expected else "")
                    reference = get_hyp_text(seg["asr_reference_text"])
                    history.append(reference)
                    context.observe(reference)
                if variant == "short_rs23":
                    assert sample["reward_model"]["ground_truth"] == "".join(refs)
            counts = proof["variants"].setdefault(variant, {"parents": 0, "segments": 0, "empty_segment_references": 0})
            counts["parents"] += parents
            counts["segments"] += segments
            counts["empty_segment_references"] += empty
            proof["sources"].append(
                {
                    "variant": variant,
                    "source": conf["post_process"]["add_field"]["fields"].get("original_data_source"),
                    "local_path": str(path),
                    "remote_path": remote,
                    "parents": parents,
                    "segments": segments,
                    "sha256": sha,
                    "real_loader_all_rows_verified": True,
                    "h4_previous_only_all_segments_verified": True,
                }
            )
            processed.append(ds)
            print("VERIFIED", variant, name, parents, segments, flush=True)
    for variant, (parents, segments) in EXPECTED.items():
        if not variant_filter or variant == variant_filter:
            assert (proof["variants"][variant]["parents"], proof["variants"][variant]["segments"]) == (
                parents,
                segments,
            )
    if not variant_filter:
        combined = _combine_datasets(_align_source_features(processed), cfg.data, is_train=True)
        assert len(combined) == 125000
        counts = Counter(combined["data_source"])
        assert counts == Counter(
            {"short_orignal": 25000, "flshort_rm23x4": 50000, "contig5": 25000, "short_rs23": 25000}
        ), counts
        proof.update(mixture_rows=len(combined), mixture_counts=dict(counts))
    else:
        sc_config = OmegaConf.create(OmegaConf.to_container(cfg.data, resolve=True))
        sc_config.mix = [{"shuffle": True, "datasource": [{"name": "short_rs23", "sample": 25000, "shuffle": True}]}]
        combined = _combine_datasets(_align_source_features(processed), sc_config, is_train=True)
        assert len(combined) == 25000 and set(combined["data_source"]) == {"short_rs23"}
        proof.update(sc_mixture_budget_verified=25000)
    if not variant_filter:
        validation = []
        for group in cfg.data.val_data:
            for config in group:
                conf = OmegaConf.to_container(config, resolve=True)
                uri = conf["jsonl_paths"]
                name = conf["post_process"]["add_field"]["fields"]["data_source"]
                path = ROOT / "validation" / Path(uri).name
                path.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["bbb", "cp", uri, str(path)], check=True, timeout=1200)
                conf["jsonl_paths"] = str(path)
                ds = create_audio_dataset(**{**conf, "version": cfg.data.version})
                adapter = ASRDataset.__new__(ASRDataset)
                adapter.ds = ds
                adapter.groups = None
                adapter.prompt_key = cfg.data.prompt_key
                segments = 0
                for i in range(len(adapter)):
                    rows = adapter[i]["asr_segments"]
                    context = ContextPrompt("text", max_history=4)
                    history = []
                    for row in rows:
                        reference = row["asr_reference_text"]
                        assert isinstance(reference, str)
                        expected = "\n".join(t for t in history[-4:] if t)
                        assert context.prompt == ("Previous context: " + expected if expected else "")
                        reference = get_hyp_text(reference)
                        history.append(reference)
                        context.observe(reference)
                        segments += 1
                assert (len(adapter), segments) == {
                    "earnings21": (44, 5552),
                    "earnings22": (125, 17099),
                    "coraal": (271, 22041),
                }[name]
                validation.append(
                    dict(
                        dataset=name,
                        parents=len(adapter),
                        segments=segments,
                        uri=uri,
                        sha256=digest(path),
                        reference_h4_previous_only_all_segments_verified=True,
                    )
                )
        proof["validation_loader"] = validation
    proof.update(
        published_manifest_sha_readback_verified=True,
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WORK, text=True).strip(),
    )
    proof.update(
        status="SC_TRAINING_LOADER_VERIFIED" if variant_filter else "ALL_TRAINING_DATA_VERIFIED",
        real_loader_manifest_samples=sum(v["parents"] for v in proof["variants"].values()),
        real_loader_segments=sum(v["segments"] for v in proof["variants"].values()),
        empty_reference_loader_supported=True,
    )
    assert len(proof["empty_reference_cases"]) == 4
    with (ROOT / "continuation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        (ROOT / ("sc_loader_verification.json" if variant_filter else "config_loader_verification.json")).write_text(
            json.dumps(proof, indent=2) + "\n"
        )
    print("FULL_LOADER_PASS", json.dumps(proof["variants"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["short_rs23"])
    args = parser.parse_args()
    main(args.variant)
