"""Expand audited offline manifests without decoding audio or adding task tokens.

Run with --help. Outputs are new files; existing output directories are rejected.
The input manifests must carry exact per-original pre-task metadata and indices.
"""

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path

import yaml
from recipe.phimm.tools.random_merge_expand import plan_expansion, plan_random_expansion
from recipe.phimm.tools.random_merge_list import _merge_rows


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def materialize(
    input_dir, sources_path, output_dir, blob_root, *, seed=42, rounds=4, merge_count=None, exclude_plan=None
):
    sources = json.loads(sources_path.read_text())
    if len({json.dumps(s["config"]["add_task_info"], sort_keys=True) for s in sources}) != 1:
        raise ValueError("cross-source merging requires compatible task formatting")
    originals, origins, excluded, input_hashes = [], [], [], {}
    for source in sources:
        name, recovered, old_groups = source["name"], {}, []
        path = input_dir / f"{name}.jsonl"
        input_hashes[name] = sha256(path)
        for line in path.open():
            row = json.loads(line)
            indices = row["random_merge_source_indices"]
            for index, metadata, audio, duration in zip(
                indices, row["source_metadata_json"], row["audio_path"], row["source_audio_lengths"], strict=True
            ):
                original = json.loads(metadata)
                if (
                    index in recovered
                    or original["audio_path"] != audio
                    or original["audio_length"] != duration
                    or "reward_model" in original
                ):
                    raise ValueError(f"invalid original provenance: {name}/{index}")
                recovered[index] = original
            old_groups.append(indices)
        if sorted(recovered) != list(range(len(recovered))):
            raise ValueError(f"noncontiguous original indices: {name}")
        offset = len(originals)
        excluded.extend(tuple(offset + i for i in group) for group in old_groups)
        label = source["config"]["post_process"]["add_field"]["fields"]["data_source"]
        for index in range(len(recovered)):
            originals.append(recovered[index])
            origins.append((name, index, label))
    if exclude_plan is not None:
        prior = json.loads(exclude_plan.read_text())
        if prior["original_count"] != len(originals) or prior["input_manifest_sha256"] != input_hashes:
            raise ValueError("excluded plan original provenance does not match")
        permutation = list(range(len(originals)))
        random.Random(prior["seed"]).shuffle(permutation)
        for round_index, shifts in enumerate(prior["round_offsets"]):
            for i in range(len(originals)):
                excluded.append(tuple(permutation[(i + shift) % len(originals)] for shift in shifts))
    if merge_count is None:
        plans, offsets = plan_expansion(originals, seed=seed, rounds=rounds, excluded_groups=excluded)
        prefix = "rm23"
    else:
        plans = plan_random_expansion(
            originals, seed=seed, rounds=rounds, merge_count=merge_count, excluded_groups=excluded
        )
        offsets = None
        prefix = f"rm{merge_count[0]}to{merge_count[1]}"
    # Fill ancillary keys to keep the schema stable across all source mixtures.
    keys = sorted(set().union(*(row.keys() for row in originals)))
    normalized = [{key: row.get(key) for key in keys if key != "data_source"} for row in originals]
    output_dir.mkdir(parents=True, exist_ok=False)
    counts, appearances, roles, partners = Counter(), Counter(), defaultdict(Counter), defaultdict(set)
    mixtures, pair_sources, groups, examples = Counter(), Counter(), set(), []
    config = []
    with ExitStack() as stack:
        handles = {}
        for source in sources:
            name = source["name"]
            directory = output_dir / name
            directory.mkdir()
            handles[name] = stack.enter_context((directory / "data.jsonl").open("w"))
            conf = {
                "dataset_name": "jsonl",
                "jsonl_paths": f"{blob_root}/{name}/data.jsonl",
                "add_task_info": source["config"]["add_task_info"],
                "post_process": deepcopy(source["config"]["post_process"]),
            }
            conf["post_process"]["verl_format"]["extra_keys"] = [
                "data_source",
                "random_merge_count",
                "random_merge_source_indices",
                "source_audio_lengths",
                "source_languages",
                "source_dataset_names",
                "source_row_indices",
                "source_data_sources",
                "random_merge_round",
                "random_merge_anchor_position",
            ]
            config.append(conf)
        for indices, round_index, anchor in plans:
            rows = [normalized[i] for i in indices]
            durations = [originals[i]["audio_length"] for i in indices]
            merged = _merge_rows(rows, list(indices), durations, " ", default_language="English")
            merged.update(
                id=f"{prefix}x{rounds}:{seed}:{round_index}:{anchor}",
                source_metadata_json=[json.dumps(originals[i], ensure_ascii=False, sort_keys=True) for i in indices],
                source_dataset_names=[origins[i][0] for i in indices],
                source_row_indices=[origins[i][1] for i in indices],
                source_data_sources=[origins[i][2] for i in indices],
                random_merge_round=round_index,
                random_merge_anchor_source_index=anchor,
                random_merge_anchor_position=indices.index(anchor),
            )
            handles[origins[anchor][0]].write(json.dumps(merged, ensure_ascii=False, sort_keys=True) + "\n")
            signature = tuple(sorted(indices))
            assert signature not in groups
            groups.add(signature)
            counts[origins[anchor][0]] += 1
            mixtures[len(set(merged["source_dataset_names"]))] += 1
            for pos, index in enumerate(indices):
                appearances[index] += 1
                roles[index]["first" if pos == 0 else "last" if pos == len(indices) - 1 else "middle"] += 1
                partners[index].update(i for i in indices if i != index)
                for other in indices[pos + 1 :]:
                    pair_sources[" + ".join(sorted((origins[index][0], origins[other][0])))] += 1
            if 0 in indices:
                examples.append(merged)
    n = len(originals)
    assert len(plans) == rounds * n
    assert len(appearances) == n and min(appearances.values()) >= rounds
    if merge_count is None:
        assert set(appearances.values()) == {rounds * 5 // 2}
        assert {tuple(role[k] for k in ("first", "middle", "last")) for role in roles.values()} == {
            (rounds, rounds // 2, rounds)
        }
        assert {len(p) for p in partners.values()} == {4 * rounds}
    assert not groups.intersection(tuple(sorted(g)) for g in excluded)
    config_path = output_dir / "train_data.yaml"
    config_path.write_text(
        "# Offline merge before add_task_info; do not merge again online.\n" + yaml.safe_dump(config, sort_keys=False)
    )
    (output_dir / "examples_same_original.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2))
    stats = {
        "original_count": n,
        "output_count": len(plans),
        "rounds": rounds,
        "seed": seed,
        "merge_count_histogram": dict(Counter(len(p[0]) for p in plans)),
        "parents_per_anchor_source": dict(counts),
        "component_appearances": sum(appearances.values()),
        "reuse_per_original": next(iter(appearances.values())) if len(set(appearances.values())) == 1 else None,
        "reuse_histogram": dict(Counter(appearances.values())),
        "reuse_min": min(appearances.values()),
        "reuse_max": max(appearances.values()),
        "positions_example_original0": dict(roles[0]),
        "positions_per_original": dict(roles[0]) if merge_count is None else None,
        "distinct_partners_per_original": 4 * rounds if merge_count is None else None,
        "distinct_partner_count_histogram": dict(Counter(len(partners[i]) for i in range(n))),
        "unordered_groups_unique": True,
        "old_groups_repeated": 0,
        "source_count_histogram": dict(mixtures),
        "cross_source_parents": sum(v for k, v in mixtures.items() if k > 1),
        "pair_source_counts": dict(pair_sources),
        "round_offsets": offsets,
        "merge_count_range": list(merge_count) if merge_count else [2, 3],
        "planner": "random_variable_uniform_quota" if merge_count else "balanced_circular_pairs_triples",
        "excluded_plan_sha256": sha256(exclude_plan) if exclude_plan else None,
        "original_audio_hours": math.fsum(r["audio_length"] for r in originals) / 3600,
        "expanded_audio_hours": math.fsum(originals[i]["audio_length"] * count for i, count in appearances.items())
        / 3600,
        "max_parent_audio_seconds": max(math.fsum(originals[i]["audio_length"] for i in p[0]) for p in plans),
        "input_manifest_sha256": input_hashes,
        "source_configs": sources,
        "source_configs_sha256": sha256(sources_path),
        "blob_root": blob_root,
        "output_sha256": {
            str(p.relative_to(output_dir)): sha256(p) for p in sorted(output_dir.rglob("*")) if p.is_file()
        },
    }
    (output_dir / "dataset_statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blob-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--merge-count", type=int, nargs=2, default=None)
    parser.add_argument("--exclude-plan", type=Path, default=None)
    args = parser.parse_args()
    stats = materialize(
        args.input_dir,
        args.sources,
        args.output_dir,
        args.blob_root.rstrip("/"),
        seed=args.seed,
        rounds=args.rounds,
        merge_count=args.merge_count,
        exclude_plan=args.exclude_plan,
    )
    print(
        json.dumps(
            {k: v for k, v in stats.items() if k not in ("source_configs", "output_sha256", "pair_source_counts")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
