"""Balanced oversampling must preserve provenance and diversify actual partners."""

import json
from collections import Counter, defaultdict

import datasets
import pytest
import yaml
from recipe.phimm.data.dataset import augment
from recipe.phimm.tools.offline_random_merge_expand import materialize
from recipe.phimm.tools.random_merge_expand import plan_expansion


def originals(count=101):
    return [
        dict(audio_path=f"{i}.wav", audio_length=1.5, text=f"word{i}", keywords=[str(i)], language="en")
        for i in range(count)
    ]


def test_balanced_reuse_positions_unique_partners_and_old_group_exclusion():
    rows = originals()
    previous, _ = plan_expansion(rows, seed=41)
    plans, offsets = plan_expansion(rows, excluded_groups=[p[0] for p in previous])
    assert (plans, offsets) == plan_expansion(rows, excluded_groups=[p[0] for p in previous])
    assert plans != plan_expansion(rows, seed=43)[0]
    reuse, positions, partners, groups = Counter(), defaultdict(Counter), defaultdict(set), set()
    excluded = {tuple(sorted(p[0])) for p in previous}
    for members, round_index, anchor in plans:
        assert anchor in members
        assert len(members) == 2 + round_index % 2
        signature = tuple(sorted(members))
        assert signature not in groups | excluded
        groups.add(signature)
        for pos, index in enumerate(members):
            reuse[index] += 1
            positions[index]["first" if pos == 0 else "last" if pos == len(members) - 1 else "middle"] += 1
            partners[index].update(i for i in members if i != index)
    assert len(plans) == 404
    assert len(reuse) == 101 and set(reuse.values()) == {10}
    assert all(role == {"first": 4, "last": 4, "middle": 2} for role in positions.values())
    assert all(len(value) == 16 for value in partners.values())


def test_infeasible_inputs_fail_instead_of_silently_reducing_diversity():
    with pytest.raises(ValueError, match="not enough"):
        plan_expansion(originals(16))
    with pytest.raises(ValueError, match="duration cap"):
        plan_expansion(originals(), max_audio_length=4.5)
    rows = originals()
    rows[-1]["audio_path"] = rows[0]["audio_path"]
    with pytest.raises(ValueError, match="unique"):
        plan_expansion(rows)


def test_materialized_cross_source_rows_survive_task_formatting(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    source_configs = []
    rows = originals(60)
    for s, name in enumerate(("a", "b")):
        records = []
        for local_index in range(30):
            row = rows[s * 30 + local_index]
            if s:
                row["custom"] = {"x": local_index}
            records.append(
                dict(
                    random_merge_source_indices=[local_index],
                    source_metadata_json=[json.dumps(row)],
                    audio_path=[row["audio_path"]],
                    source_audio_lengths=[row["audio_length"]],
                )
            )
        (input_dir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        source_configs.append(
            dict(
                name=name,
                config=dict(
                    add_task_info=dict(style="verb", language="English", lang_hint=True),
                    post_process=dict(
                        add_field=dict(fields=dict(data_source=name)), verl_format=dict(prompt_key="prompt")
                    ),
                ),
            )
        )
    sources_path = tmp_path / "sources.json"
    sources_path.write_text(json.dumps(source_configs))
    out = tmp_path / "output"
    stats = materialize(input_dir, sources_path, out, "az://test/new")
    assert stats["output_count"] == 240
    assert stats["cross_source_parents"] > 0
    assert stats["parents_per_anchor_source"] == {"a": 120, "b": 120}
    config = yaml.safe_load((out / "train_data.yaml").read_text())
    for source, conf in zip(("a", "b"), config, strict=True):
        ds = datasets.load_dataset("json", data_files=str(out / source / "data.jsonl"), split="train")
        formatted = augment(ds, add_task_info=conf["add_task_info"], post_process=conf["post_process"])
        assert len(formatted) == 120
        for raw, final in zip(ds, formatted, strict=True):
            originals_in_order = [json.loads(s) for s in raw["source_metadata_json"]]
            assert raw["audio_path"] == [r["audio_path"] for r in originals_in_order]
            assert raw["text"] == " ".join(r["text"] for r in originals_in_order)
            assert raw["keywords"] == [k for r in originals_in_order for k in r["keywords"]]
            assert raw["language"] == "English"
            lines = list(map(json.loads, final["reward_model"]["gt_output"].splitlines()))
            assert len(lines) == len(raw["audio_path"])
            assert {line["language"] for line in lines} == {"English"}
            assert " ".join(line["text"] for line in lines) == raw["text"]
    with pytest.raises(FileExistsError):
        materialize(input_dir, sources_path, out, "az://test/new")


@pytest.mark.parametrize("merge_variant", ["23", "15"])
def test_x4_training_pair_shares_all_controls_except_context_and_output_identity(merge_variant):
    from pathlib import Path

    import recipe.phimm
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", lambda text: eval(text, {"__builtins__": {}}, {}), replace=True)
    directory = Path(recipe.phimm.__file__).parent / "config/v2611_dev/earnings"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        configs = [
            compose(config_name=f"remax_earnings_short_random_merge{merge_variant}_x4_{suffix}")
            for suffix in ("context", "no_context")
        ]
    on, off = [OmegaConf.to_container(cfg, resolve=True) for cfg in configs]
    assert on["data"] == off["data"]
    assert on["rewards"] == off["rewards"]
    assert on["asr"]["context_aware"] is True and off["asr"]["context_aware"] is False
    for cfg in (on, off):
        assert cfg["asr"]["expected_training_rows"] == 100960
        assert cfg["trainer"]["total_training_steps"] == 800
        assert cfg["trainer"]["total_epochs"] * (100960 // cfg["data"]["train_batch_size"]) >= 800
        assert "offline_data" not in cfg
        assert "shadow/38000/qwen_hf" in cfg["actor_rollout_ref"]["model"]["path"]
        # Normalize only the run name; remaining effective differences must be the single flag.
        name = cfg["trainer"]["experiment_name"]

        def normalize(value, run_name=name):
            if isinstance(value, dict):
                return {k: normalize(v) for k, v in value.items()}
            if isinstance(value, list):
                return [normalize(v) for v in value]
            return value.replace(run_name, "RUN") if isinstance(value, str) else value

        cfg.update(normalize(cfg))
    off["asr"]["context_aware"] = True
    assert on == off
