from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from recipe.phimm.reward.asr_vote import compute_score


def test_v1h1a1_adds_language_reward_and_normalizes_by_beta_sum():
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2611_dev/dev"), version_base=None):
        original = compose(config_name="remax_dev_base0_v1h1a", overrides=[searchpath])
        config = compose(config_name="remax_dev_base0_v1h1a1", overrides=[searchpath])

    expected = OmegaConf.to_container(original, resolve=False)
    reward_kwargs = expected["reward_functions"]["librispeech"]["reward_kwargs"]
    reward_kwargs["measures"]["lang"] = {
        "delta": "abs", "beta": 0.1, "vote": "delta", "clip": [-10, 10],
    }
    reward_kwargs["reduce"]["total"] = 3.5
    assert OmegaConf.to_container(config, resolve=False) == expected
    assert config.actor_rollout_ref.rollout.gt_rollout is False
    reward = config.reward_functions.librispeech.reward_kwargs
    assert reward.measures.lang.beta == 0.1
    assert reward.reduce.total == pytest.approx(sum(measure.beta for measure in reward.measures.values()))


def test_v1h1a2_increases_word_language_weight_and_normalizes_by_beta_sum():
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2611_dev/dev"), version_base=None):
        original = compose(config_name="remax_dev_base0_v1h1a", overrides=[searchpath])
        config = compose(config_name="remax_dev_base0_v1h1a2", overrides=[searchpath])

    expected = OmegaConf.to_container(original, resolve=False)
    reward_kwargs = expected["reward_functions"]["librispeech"]["reward_kwargs"]
    reward_kwargs["measures"]["word_lang"]["beta"] = 3.0
    reward_kwargs["reduce"]["total"] = 5.4
    assert OmegaConf.to_container(config, resolve=False) == expected
    assert config.actor_rollout_ref.rollout.gt_rollout is False
    reward = config.reward_functions.librispeech.reward_kwargs
    assert reward.measures.word_lang.beta == 3.0
    assert reward.reduce.total == pytest.approx(sum(measure.beta for measure in reward.measures.values()))


def test_v1h2c_smooth_language_penalty_with_real_parser():
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2611_dev/dev"), version_base=None):
        config = compose(config_name="remax_dev_base0_v1h2c", overrides=[searchpath])
    kwargs = OmegaConf.to_container(config.reward_functions.librispeech.reward_kwargs, resolve=True)
    kwargs["text_norm"] = "identity"
    assert "word_lang" not in kwargs["measures"]
    reference = " ".join(f"word{index}" for index in range(100))

    def response(text, language):
        return f"<src=English><tgt={language}>\n<TXT>{text}</TXT>"

    correct = response(reference, "English")
    incorrect = response(reference, "French")
    baseline = compute_score(correct, reference, **kwargs)
    assert baseline["word"] == 1.0 and baseline["lang"] == 1.0
    for n_errors, weight in [(0, 1.0), (5, 0.92), (15, 0.6)]:
        text = " ".join(reference.split()[:-n_errors]) if n_errors else reference
        good = compute_score(response(text, "English"), reference, extra_info={"greedy_reward": baseline}, **kwargs)
        bad = compute_score(response(text, "French"), reference, extra_info={"greedy_reward": baseline}, **kwargs)
        assert bad["score"] - good["score"] == pytest.approx(-0.25 * weight / 3.6)
        assert bad["w_lang"] == pytest.approx(weight)
    wrong_baseline = compute_score(incorrect, reference, **kwargs)
    recovery = compute_score(correct, reference, extra_info={"greedy_reward": wrong_baseline}, **kwargs)
    unchanged = compute_score(incorrect, reference, extra_info={"greedy_reward": wrong_baseline}, **kwargs)
    assert recovery["score"] == pytest.approx(0.25 / 3.6)
    assert unchanged["score"] == 0.0


def test_v1h1a_gt_doubles_word_weight_and_normalizes_by_beta_sum():
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2611_dev/dev"), version_base=None):
        original = compose(config_name="remax_dev_base0_v1h1_gt", overrides=[searchpath])
        config = compose(config_name="remax_dev_base0_v1h1a_gt", overrides=[searchpath])

    expected = OmegaConf.to_container(original, resolve=False)
    reward_kwargs = expected["reward_functions"]["librispeech"]["reward_kwargs"]
    reward_kwargs["measures"]["word"]["beta"] = 2.0
    reward_kwargs["reduce"]["total"] = 3.4
    assert OmegaConf.to_container(config, resolve=False) == expected
    assert config.actor_rollout_ref.rollout.gt_rollout is True
    reward = config.reward_functions.librispeech.reward_kwargs
    assert reward.measures.word.beta == 2.0
    assert reward.reduce.total == pytest.approx(sum(measure.beta for measure in reward.measures.values()))


def test_v1h2_disables_only_ground_truth_rollouts():
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2611_dev/dev"), version_base=None):
        original = compose(config_name="remax_dev_base0_v1h2_gt", overrides=[searchpath])
        config = compose(config_name="remax_dev_base0_v1h2", overrides=[searchpath])

    assert original.actor_rollout_ref.rollout.gt_rollout is True
    assert config.actor_rollout_ref.rollout.gt_rollout is False
    expected = OmegaConf.to_container(original, resolve=False)
    expected["actor_rollout_ref"]["rollout"]["gt_rollout"] = False
    assert OmegaConf.to_container(config, resolve=False) == expected


@pytest.mark.parametrize(
    "config_name", ["remax_dev_base0_v0", "remax_dev_base0_v0a", "remax_dev_base0_v0b", "remax_dev_base0_v0c"]
)
def test_base0_vote_preserves_training_setup_and_configures_vote_rewards(config_name):
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2611_dev/dev"), version_base=None):
        original = compose(config_name="remax_dev_base0_r0", overrides=[searchpath])
        config = compose(config_name=config_name, overrides=[searchpath])
        v0 = compose(config_name="remax_dev_base0_v0", overrides=[searchpath])

    for section in (
        "actor_rollout_ref",
        "algorithm",
        "data",
        "trainer",
        "val_reward",
        "reward_function_by_data_source",
    ):
        assert OmegaConf.to_container(config[section], resolve=False) == OmegaConf.to_container(
            original[section], resolve=False
        )

    reward = config.reward_functions.librispeech
    assert reward.path == "recipe/phimm/reward/asr_vote.py"
    assert reward.name == "compute_score"
    kwargs = OmegaConf.to_container(reward.reward_kwargs, resolve=True)
    if config_name in ("remax_dev_base0_v0a", "remax_dev_base0_v0b", "remax_dev_base0_v0c"):
        expected = OmegaConf.to_container(v0, resolve=False)
        for name, cfg in expected["reward_functions"]["librispeech"]["reward_kwargs"]["measures"].items():
            if config_name == "remax_dev_base0_v0c" and name != "lang":
                cfg.update(tol=[-0.05, 0.02])
            else:
                cfg.update(tol=[-0.1, 0.1])
        if config_name in ("remax_dev_base0_v0b", "remax_dev_base0_v0c"):
            expected["reward_functions"]["librispeech"]["reward_kwargs"]["measures"]["lang"]["beta"] = 1.0
        assert OmegaConf.to_container(config, resolve=False) == expected
    assert kwargs["version"] == 2609
    assert kwargs["reduce"] == {"mode": "average", "total": 3}
    assert "delta" not in kwargs
    assert set(kwargs["measures"]) == set(original.reward_functions.librispeech.reward_kwargs.measures)
    for name, cfg in kwargs["measures"].items():
        full_weight = name in ("char", "word") or (
            name == "lang" and config_name in ("remax_dev_base0_v0b", "remax_dev_base0_v0c")
        )
        assert cfg["beta"] == (1.0 if full_weight else 0.25)
        assert cfg["delta"] == ("abs" if name == "lang" else "rel")
        assert cfg["tol"][0] < 0 < cfg["tol"][1]

    kwargs["text_norm"] = "identity"
    baseline = compute_score("<src=English><tgt=English>\n<TXT>cat</TXT>", "cat dog", **kwargs)
    assert baseline["score"] == 0.0
    assert not any(name.startswith("d_") for name in baseline)
    result = compute_score(
        "<src=English><tgt=English>\n<TXT>cat dog</TXT>",
        "cat dog",
        extra_info={"greedy_reward": baseline},
        **kwargs,
    )
    assert result["score"] == pytest.approx(2.25 / 3)
    assert result["d_word"] == 1.0
    assert result["d_lang"] == 0.0
    assert "d_score" not in result

    language_result = compute_score(
        "<src=English><tgt=French>\n<TXT>cat</TXT>",
        "cat dog",
        extra_info={"greedy_reward": baseline},
        **kwargs,
    )
    assert language_result["score"] == pytest.approx(-kwargs["measures"]["lang"]["beta"] / 3)
    assert language_result["v_lang"] == -1.0


def test_v0d_uses_word_edit_and_word_language_votes():
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2611_dev/dev"), version_base=None):
        original = compose(config_name="remax_dev_base0_v0c", overrides=[searchpath])
        config = compose(config_name="remax_dev_base0_v0d", overrides=[searchpath])

    expected = OmegaConf.to_container(original, resolve=False)
    measures = expected["reward_functions"]["librispeech"]["reward_kwargs"]["measures"]
    for name in list(measures):
        if name in ("word_kw", "lang"):
            del measures[name]
        else:
            measures[name].update(beta=0.25 if name.startswith("word_") else 1.0, tol=[-0.001, 0.001])
    measures["word_lang"] = {"delta": "rel", "beta": 0.5, "tol": [-0.001, 0.001]}
    assert OmegaConf.to_container(config, resolve=False) == expected

    kwargs = OmegaConf.to_container(config.reward_functions.librispeech.reward_kwargs, resolve=True)
    assert set(kwargs["measures"]) == {"char", "word", "word_del", "word_ins", "word_lang"}
    kwargs["text_norm"] = "identity"
    baseline = compute_score("<src=English><tgt=English>\n<TXT>cat</TXT>", "cat dog", **kwargs)
    assert set(baseline) == {"score", "char", "word", "word_del", "word_ins", "word_lang"}
    assert baseline["score"] == 0.0
    result = compute_score(
        "<src=English><tgt=English>\n<TXT>cat dog</TXT>",
        "cat dog",
        extra_info={"greedy_reward": baseline},
        **kwargs,
    )
    assert result["score"] == pytest.approx(2.25 / 3)
    assert {name: result[f"v_{name}"] for name in kwargs["measures"]} == {
        "char": 1.0,
        "word": 1.0,
        "word_del": 1.0,
        "word_ins": 0.0,
        "word_lang": 0.0,
    }
    language_result = compute_score(
        "<src=English><tgt=French>\n<TXT>cat</TXT>",
        "cat dog",
        extra_info={"greedy_reward": baseline},
        **kwargs,
    )
    assert language_result["score"] == pytest.approx(-0.5 / 3)
    assert language_result["v_word_lang"] == -1.0
    assert not any(name in language_result for name in ("lang", "d_lang", "v_lang"))


def test_v0e_preserves_three_measure_zero_threshold_recipe():
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2611_dev/dev"), version_base=None):
        original = compose(config_name="remax_dev_base0_v0c", overrides=[searchpath])
        config = compose(config_name="remax_dev_base0_v0e", overrides=[searchpath])

    expected = OmegaConf.to_container(original, resolve=False)
    measures = expected["reward_functions"]["librispeech"]["reward_kwargs"]["measures"]
    for name in list(measures):
        if name.startswith("word_"):
            del measures[name]
    for cfg in measures.values():
        cfg.update(tol=[0.0, 0.0])
    assert OmegaConf.to_container(config, resolve=False) == expected

    from recipe.phimm.reward.asr_vote import compute_measure_vote

    for name, cfg in config.reward_functions.librispeech.reward_kwargs.measures.items():
        for delta, expected_vote in [(-0.001, -1.0), (0.0, 0.0), (0.001, 1.0)]:
            score, _, vote = compute_measure_vote(name, 0.5 + delta, 0.5, cfg)
            assert vote == expected_vote
            assert score == expected_vote


def test_v0f_replaces_language_with_half_weight_word_edit_votes():
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2611_dev/dev"), version_base=None):
        original = compose(config_name="remax_dev_base0_v0e", overrides=[searchpath])
        config = compose(config_name="remax_dev_base0_v0f", overrides=[searchpath])

    expected = OmegaConf.to_container(original, resolve=False)
    measures = expected["reward_functions"]["librispeech"]["reward_kwargs"]["measures"]
    del measures["lang"]
    for name in ("word_ins", "word_del"):
        measures[name] = {"delta": "rel", "beta": 0.5, "tol": [0.0, 0.0]}
    assert OmegaConf.to_container(config, resolve=False) == expected

    kwargs = OmegaConf.to_container(config.reward_functions.librispeech.reward_kwargs, resolve=True)
    kwargs["text_norm"] = "identity"
    baseline = compute_score("<src=English><tgt=English>\n<TXT>cat</TXT>", "cat dog", **kwargs)
    assert set(baseline) == {"score", "char", "word", "word_ins", "word_del"}
    result = compute_score(
        "<src=English><tgt=French>\n<TXT>cat dog</TXT>",
        "cat dog",
        extra_info={"greedy_reward": baseline},
        **kwargs,
    )
    assert result["score"] == pytest.approx(2.5 / 3)
    assert result["v_word_del"] == 1.0
    assert result["v_word_ins"] == 0.0
    assert not any(name in result for name in ("lang", "d_lang", "v_lang"))

    insertion_baseline = compute_score("<TXT>cat dog bird</TXT>", "cat dog", **kwargs)
    insertion_result = compute_score(
        "<TXT>cat dog</TXT>", "cat dog", extra_info={"greedy_reward": insertion_baseline}, **kwargs
    )
    assert insertion_result["score"] == pytest.approx(2.5 / 3)
    assert insertion_result["v_word_ins"] == 1.0
    assert insertion_result["v_word_del"] == 0.0


def test_v0g_changes_only_v0f_thresholds_to_half_percent():
    from recipe.phimm.reward.asr_vote import compute_vote

    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2611_dev/dev"), version_base=None):
        original = compose(config_name="remax_dev_base0_v0f", overrides=[searchpath])
        config = compose(config_name="remax_dev_base0_v0g", overrides=[searchpath])

    expected = OmegaConf.to_container(original, resolve=False)
    for cfg in expected["reward_functions"]["librispeech"]["reward_kwargs"]["measures"].values():
        cfg.update(tol=[-0.005, 0.005])
    assert OmegaConf.to_container(config, resolve=False) == expected

    for cfg in config.reward_functions.librispeech.reward_kwargs.measures.values():
        for delta, vote in [(-0.006, -1.0), (-0.005, 0.0), (0.0, 0.0), (0.005, 0.0), (0.006, 1.0)]:
            assert compute_vote(delta, low=cfg.tol[0], high=cfg.tol[1]) == vote


@pytest.mark.parametrize("language", ["English", "French"])
def test_v2b_gt_preserves_rewards_and_uses_2609_gt_format(language):
    from datasets import Dataset

    from recipe.phimm.data.dataset import add_task_info, verl_format_ds

    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2611_dev/dev"), version_base=None):
        original = compose(config_name="remax_dev_base0_v2b", overrides=[searchpath])
        config = compose(config_name="remax_dev_base0_v2b_gt", overrides=[searchpath])

    expected = OmegaConf.to_container(original, resolve=False)
    expected["actor_rollout_ref"]["rollout"]["gt_rollout"] = True
    assert OmegaConf.to_container(config, resolve=False) == expected
    assert config.actor_rollout_ref.rollout.n == 8
    assert config.data.version == 2609

    dataset = Dataset.from_dict({"text": ["cat dog"], "language": [language], "id": ["gt-format"]})
    dataset = add_task_info(
        dataset, task="lang_asr_verb", version=config.data.version, prefix_prob=0.0, lang_hint=True
    )
    row = verl_format_ds(dataset)[0]
    assert row["reward_model"]["ground_truth"] == "cat dog"
    # add_task_info formats gt_output through get_task_output for the 2609 lang_asr_verb task.
    assert row["reward_model"]["gt_output"] == f"<src={language}><tgt={language}>\n<VERBATIM>\ncat dog"
    assert row["extra_info"]["prefix"] == ""

    result = compute_score(
        row["reward_model"]["gt_output"],
        row["reward_model"]["ground_truth"],
        extra_info=row["extra_info"],
        **OmegaConf.to_container(config.reward_functions.librispeech.reward_kwargs, resolve=True),
    )
    assert result == {"score": 0.0, "word": 0.0, "word_del": 0.0, "word_ins": 0.0, "word_kw": 0.0, "lang": 1.0}
