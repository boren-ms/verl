"""Official DAI English scoring through the native reward/count contract."""

import pytest
from recipe.phimm.agentic.scoring import aggregate_error_rates
from recipe.phimm.reward.asr_eval import dai_en_eval

score_pair = pytest.importorskip("dai_asr_i18n.scoring").score_pair


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [
        ("I paid twenty dollars (laughs).", "I paid $20."),
        ("a lot of ice cream", "alot of icecream"),
        ("", "hallucinated words"),
        ("missing speech", ""),
    ],
)
def test_official_counts_and_legacy_segment_output(reference, hypothesis):
    official = score_pair(reference, hypothesis, language="en", metric="wer")
    actual = dai_en_eval(["<src=English><tgt=English>\n<VERBATIM>\n" + hypothesis], reference)
    assert actual["n_err"] == official.counts.numerator
    assert actual["n_ref"] == official.counts.denominator
    assert actual["wer"] == official.value


def test_pooling_preserves_empty_reference_insertions():
    rows = [dai_en_eval("hello world", "hello world"), dai_en_eval("extra", "")]
    extras = {key: [row[key] for row in rows] for key in rows[0]}
    metrics = aggregate_error_rates(["test", "test"], extras)
    assert metrics["val-core/test/p_err/mean@1"] == 0.5
    assert "val-core/test/hf_p_err/mean@1" in metrics


def test_separated_eval_preset_keeps_channel_identity_and_native_dispatch():
    from pathlib import Path

    import recipe.phimm
    from hydra import compose, initialize_config_dir
    from recipe.phimm.agentic.scoring import score_response

    directory = Path(recipe.phimm.__file__).parent / "config/v2611_dev/eval"
    verl_configs = Path(__file__).resolve().parents[3] / "verl/trainer/config"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        config = compose(config_name="eval_dai_separated", overrides=[f"hydra.searchpath=[file://{verl_configs}]"])
    assert config.trainer.val_only and config.trainer.resume_mode == "disable"
    assert config.actor_rollout_ref.model.lora_rank == 0
    assert (config.trainer.nnodes, config.trainer.n_gpus_per_node) == (1, 8)
    data = config.data.val_data[0]
    assert {"clip_id", "channel", "source_revision", "svad_bounds"} <= set(data.post_process.verl_format.extra_keys)
    row = {"data_source": "dai_asr_i18n_en_public_separated", "reward_model": {"ground_truth": "hello world"}}
    assert score_response(config, row, ["hello", "world"])["n_err"] == 0
