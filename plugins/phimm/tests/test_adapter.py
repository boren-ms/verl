"""Behavioral contracts for the ASR recipe's migration to main."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from recipe.phimm.agentic.scoring import aggregate_error_rates, score_response


@pytest.fixture
def config():
    import recipe.phimm

    OmegaConf.register_new_resolver("eval", lambda text: eval(text, {"__builtins__": {}}, {}), replace=True)
    directory = Path(recipe.phimm.__file__).parent / "config/v2611_dev/main"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        return compose(config_name="remax_main_base3a_2611r2")


def test_recipe_contract(config):
    actor = config.actor_rollout_ref
    assert actor.rollout.name == "vllm"
    assert actor.model.lora_rank == 320
    assert actor.model.lora_alpha == 640
    assert actor.actor.optim.lr == 6e-7
    assert actor.actor.loss_agg_mode == "seq-mean-token-sum"
    assert actor.actor.strategy == "fsdp2"
    assert actor.rollout.n == 8
    assert actor.rollout.mode == "async"
    assert actor.rollout.calculate_log_probs
    assert actor.rollout.custom.no_repeat_ngram_window_size == 512
    assert actor.rollout.response_length == 768
    assert actor.rollout.custom.validation_response_length == 512
    assert actor.rollout.custom.validation_no_repeat_ngram_size == 15
    assert actor.rollout.engine_kwargs.vllm.gdn_prefill_backend == "triton"
    assert actor.rollout.engine_kwargs.vllm.lora_target_modules[-1] == "out_proj"
    assert actor.rollout.seed == actor.rollout.engine_kwargs.vllm.seed == 0
    assert actor.rollout.logprobs_mode == "raw_logprobs"
    assert config.data.train_batch_size == 128
    assert sum(row.sample for row in config.data.mix[0].datasource) == 55664
    assert config.trainer.total_epochs == 2
    assert config.trainer.total_training_steps is None
    assert config.trainer.save_freq == 10
    assert config.trainer.test_freq == 20
    assert config.trainer.resume_mode == "auto"
    assert not config.trainer.val_only
    assert config.algorithm.rollout_correction.rollout_is_threshold == 5
    assert not config.algorithm.rollout_correction.rollout_is_batch_normalize
    assert not config.algorithm.rollout_correction.bypass_mode
    assert config.asr.expected_training_rows == 55664
    assert config.asr.expected_training_steps == 868
    assert config.trainer.v1.trainer_mode == "sync"
    cohorts = [item for group in config.data.val_data for item in (group if OmegaConf.is_list(group) else [group])]
    assert len(cohorts) == 9


def test_corpus_counts_and_missing_fields():
    metrics = aggregate_error_rates(
        ["a", "a", "b"],
        {"n_err": [1, 0, 2], "n_ref": [1, 99, 10], "nb_err": [None, None, 1], "nb_ref": [None, None, 2]},
    )
    assert metrics["val-core/a/p_err/mean@1"] == 0.01
    assert metrics["val-core/b/p_err/mean@1"] == 0.2
    assert metrics["val-core/b/pb_err/mean@1"] == 0.5


def test_parent_grouping_preserves_source_and_segment_order():
    from recipe.phimm.agentic.dataset import group_validation_rows

    rows = [
        {"data_source": "a", "extra_info": {"audio_path": "x#2", "seg_start": 20}},
        {"data_source": "a", "extra_info": {"parent_audio_path": "x", "seg_start": 0}},
        {"data_source": "b", "extra_info": {"parent_audio_path": "x"}},
        {"data_source": "a", "extra_info": {}},
        {"data_source": "a", "extra_info": {}},
    ]
    assert group_validation_rows(rows) == [[1, 0], [2], [3], [4]]


def test_baseline_measures_drive_vote_reward(config):
    # An unseeded baseline intentionally yields score zero and raw measures;
    # sampled rollouts must receive those measures to compute the delta reward.
    row = {
        "data_source": "noisy_ls",
        "reward_model": {"ground_truth": "hello world"},
        "extra_info": {"language": "English"},
    }
    bad = "<src=English><tgt=English>\n<verbatim>hello</verbatim>"
    good = "<src=English><tgt=English>\n<verbatim>hello world</verbatim>"
    baseline = score_response(config, row, bad)
    assert baseline["score"] == 0
    info = {"language": "English", "greedy_hyp": bad, "greedy_reward": baseline}
    improved = score_response(config, row, good, extra_info=info)
    unchanged = score_response(config, row, bad, extra_info=info)
    assert improved["score"] > unchanged["score"]


def test_vllm_no_repeat_uses_configured_window(config):
    import torch
    from vllm import SamplingParams
    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

    from recipe.phimm.agentic.sampling import sampling_params_for_row

    for validate in (False, True):
        params = sampling_params_for_row(
            config.actor_rollout_ref.rollout, {"temperature": 0.0 if validate else 1.0}, validate=validate
        )
        request = SamplingParams(**params)
        NGramPerReqLogitsProcessor.validate_params(request)
        assert params["max_tokens"] == (512 if validate else 768)
        assert request.extra_args == {"ngram_size": 15, "window_size": 512}
        processor = NGramPerReqLogitsProcessor.new_req_logits_processor(None, request)
        # A match beyond the old 100-token default must now be blocked.
        tokens = list(range(1, 16)) + [99] * 150 + list(range(1, 15))
        logits = processor(tokens, torch.zeros(128))
        assert torch.isneginf(logits[15])
        assert torch.isfinite(logits[16])
        # Matches outside the configured 512-token window remain allowed.
        tokens = list(range(1, 16)) + [99] * 600 + list(range(1, 15))
        logits = processor(tokens, torch.zeros(128))
        assert torch.isfinite(logits[15])


def test_remax_advantage_masks_and_baselines():
    import torch

    from verl.trainer.ppo.core_algos import compute_remax_outcome_advantage

    rewards = torch.tensor([[0.0, 0.7, 0.0], [0.0, -0.3, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    advantages, returns = compute_remax_outcome_advantage(rewards, torch.tensor([0.0, 0.0]), mask)
    torch.testing.assert_close(advantages, torch.tensor([[0.7, 0.7, 0.0], [-0.3, -0.3, 0.0]]))
    assert torch.isfinite(returns).all()


def test_reward_component_namespaces_and_optional_values():
    from recipe.phimm.agentic.scoring import _compute_reward_metrics

    metrics = _compute_reward_metrics(
        {"wer": [0.1, None, 0.3, float("nan")], "d_wer": [-0.2, 0.0], "v_wer": [1, 0], "detail": ["text", {}]}
    )
    assert metrics["reward-core/wer/mean"] == pytest.approx(0.2)
    assert metrics["reward-delta/wer/mean"] == pytest.approx(-0.1)
    assert metrics["reward-vote/wer/mean"] == 0.5
    assert not any("detail" in key or "reward-core/d_" in key or "reward-core/v_" in key for key in metrics)


def test_training_reward_metrics_from_transfer_queue(monkeypatch):
    from types import SimpleNamespace

    from recipe.phimm.agentic import trainer
    from verl.utils import tensordict_utils as tu

    # Use the real TensorDict non-tensor representation returned by TransferQueue.
    records = [
        {"reward_extra_info": {"wer": 0.1, "d_wer": -0.2, "v_wer": 1}},
        {"reward_extra_info": {"wer": 99.0, "d_wer": 99.0, "v_wer": 99}},
        {"reward_extra_info": {"wer": 0.3, "d_wer": 0.0, "v_wer": 0}},
        {},
    ]
    data = tu.get_tensordict({"extra_fields": records})
    assert isinstance(data["extra_fields"], list)
    assert not hasattr(data["extra_fields"], "tolist")
    batch = SimpleNamespace(
        keys=["a", "padding", "b", "c"], partition_id="train", tags=[{}, {"is_padding": True}, {}, {}]
    )

    def get_fields(**kwargs):
        assert kwargs == {"keys": batch.keys, "partition_id": "train", "select_fields": ["extra_fields"]}
        return data

    def base_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        metrics["base_metric"] = 1.0

    monkeypatch.setattr(trainer.tq, "kv_batch_get", get_fields)
    monkeypatch.setattr(trainer.PPOTrainerSync, "_compute_metrics", base_metrics)
    metrics = {}
    trainer.ASRTrainerSync.__new__(trainer.ASRTrainerSync)._compute_metrics(batch, metrics, {}, 1, 0)
    assert metrics["base_metric"] == 1.0
    assert metrics["reward-core/wer/mean"] == pytest.approx(0.2)
    assert metrics["reward-delta/wer/mean"] == pytest.approx(-0.1)
    assert metrics["reward-vote/wer/mean"] == 0.5


def test_source_on_policy_tis_loss_and_gradient(config):
    import torch

    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
    from verl.utils.config import omega_conf_to_dataclass

    actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    log_probs = torch.tensor([[-2.0, -0.5, 0.0], [-4.0, -1.0, -2.0]], dtype=torch.float64, requires_grad=True)
    rollout = torch.tensor([[-5.0, -1.0, 0.0], [-3.0, -2.0, -2.0]], dtype=torch.float64)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
    advantages = torch.tensor([[0.7] * 3, [-0.3] * 3], dtype=torch.float64)
    weights = (log_probs.detach() - rollout).exp().clamp(max=5)
    actual, _ = compute_policy_loss_vanilla(
        log_probs.detach(),
        log_probs,
        advantages,
        mask,
        loss_agg_mode="seq-mean-token-sum",
        config=actor,
        rollout_is_weights=weights,
    )
    # Source dp_actor.py uses old_log_prob = log_prob.detach() on the single
    # on-policy minibatch. Its clipped ratio is one, with detached TIS weights.
    source_ratio = (log_probs - log_probs.detach()).exp()
    expected = (-advantages * source_ratio * weights * mask).sum(dim=-1).mean()
    torch.testing.assert_close(actual, expected)
    (actual_grad,) = torch.autograd.grad(actual, log_probs, retain_graph=True)
    (source_grad,) = torch.autograd.grad(expected, log_probs)
    torch.testing.assert_close(actual_grad, source_grad)
