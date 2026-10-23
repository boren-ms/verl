"""Execute trainer replay paths on CPU without importing the model stack."""

import ast
import copy
import uuid
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from verl import DataProto
from verl.trainer.ppo.bad_case_replay import BadCaseReplay, ReplayMetrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance


ROOT = Path(__file__).resolve().parents[3]


def _timer_name(node):
    if isinstance(node, ast.With):
        context = node.items[0].context_expr
        if isinstance(context, ast.Call) and context.args and isinstance(context.args[0], ast.Constant):
            return context.args[0].value
    return None


def _assigns(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets
    )


def _load_paths():
    tree = ast.parse((ROOT / "verl/trainer/ppo/ray_trainer.py").read_text())
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
    methods = {node.name: node for node in trainer.body if isinstance(node, ast.FunctionDef)}
    selected = [
        copy.deepcopy(methods[name])
        for name in ("_create_dataloader", "_get_gen_batch", "_balance_batch")
    ]
    # The factories are unused with supplied datasets and sampler. Avoid importing
    # main_ppo, whose model imports require a working torchvision installation.
    selected[0].body = [node for node in selected[0].body if not isinstance(node, ast.ImportFrom)]
    trainer.body = selected

    fit = methods["fit"]
    batch_loop = next(
        node for node in ast.walk(fit)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "batch_dict"
    )
    step_index = next(i for i, node in enumerate(batch_loop.body) if _timer_name(node) == "step")
    step = copy.deepcopy(batch_loop.body[step_index])
    stop = next(i for i, node in enumerate(step.body) if _assigns(node, "batch"))
    step.body = step.body[:stop]
    prepare = ast.parse("def prepare_generation(self, batch_dict): pass").body[0]
    prepare.body = copy.deepcopy(batch_loop.body[:step_index]) + [step]
    prepare.body += ast.parse("return batch, gen_batch_output, replay_context, metrics").body

    response_mask = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "compute_response_mask"
    )
    module = ModuleType("replay_trainer_paths")
    module.__dict__.update(
        np=np, torch=torch, deepcopy=copy.deepcopy, uuid=uuid,
        DataProto=DataProto, BadCaseReplay=BadCaseReplay, ReplayMetrics=ReplayMetrics,
        StatefulDataLoader=StatefulDataLoader, OmegaConf=OmegaConf, open_dict=open_dict,
        get_collate_fn=lambda dataset, collate: collate,
        AdvantageEstimator=SimpleNamespace(REMAX="remax"),
        marked_timer=lambda *args, **kwargs: nullcontext(),
        get_seqlen_balanced_partitions=get_seqlen_balanced_partitions,
        log_seqlen_unbalance=log_seqlen_unbalance,
        core_algos=SimpleNamespace(capture_remax_metric_baselines=lambda *args: None),
        prev_step_profile=False, curr_step_profile=False,
    )
    body = ast.parse("from __future__ import annotations").body + [trainer, response_mask, prepare]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(ROOT), "exec"), module.__dict__)
    return module


def _raw(count, start=0):
    ids = torch.arange(start, start + count)
    extra = np.empty(count, dtype=object)
    extra[:] = [{"id": int(value), "nested": [int(value)]} for value in ids]
    return DataProto.from_dict(
        tensors={
            "input_ids": ids[:, None].repeat(1, 3),
            "attention_mask": torch.ones(count, 3, dtype=torch.long),
            "position_ids": torch.arange(3).expand(count, -1).clone(),
        },
        non_tensors={"id": np.asarray(ids), "data_source": np.full(count, "test", dtype=object), "extra_info": extra},
    )


def _as_dict(batch):
    return {**batch.batch, **batch.non_tensor_batch}


def _collate(rows):
    return {
        key: torch.stack([row[key] for row in rows]) if isinstance(rows[0][key], torch.Tensor)
        else np.asarray([row[key] for row in rows], dtype=object)
        for key in rows[0]
    }


def _config(enabled=True, *, total_steps=None):
    return OmegaConf.create({
        "algorithm": {
            "adv_estimator": "remax", "bad_case_replay": {
                "enable": enabled, "replay_size": 10, "selection": {"weights": {"word": 1}},
            },
            "filter_groups": {"enable": True, "metric": "seq_reward", "mode": "variance", "max_num_gen_batches": 3},
        },
        "data": {"gen_batch_size": 100, "train_batch_size": 100, "val_batch_size": None, "dataloader_num_workers": 0},
        "actor_rollout_ref": {"actor": {"strategy": "fsdp2", "policy_loss": {}, "optim": {}}, "rollout": {"n": 2}},
        "trainer": {"total_epochs": 2, "total_training_steps": total_steps},
        "global_profiler": {"profile_continuous_steps": False},
    })


def _trainer(module, enabled=True, *, total_steps=None):
    trainer = module.RayPPOTrainer()
    trainer.config = _config(enabled, total_steps=total_steps)
    trainer.tokenizer = SimpleNamespace(
        pad_token_id=0, batch_decode=lambda rows, **kwargs: [str(row.tolist()) for row in rows]
    )
    trainer.use_critic = False
    trainer._start_profiling = lambda *args: None
    trainer.global_steps = 1
    trainer.total_training_steps = 10
    trainer.async_rollout_mode = False
    trainer.bad_case_replay = BadCaseReplay(trainer.config.algorithm.bad_case_replay, 100)
    return trainer


@pytest.mark.parametrize("enabled,fresh", [(False, 100), (True, 90)])
@pytest.mark.parametrize("explicit_steps", [None, 7])
def test_fixed_fresh_loader_preserves_nominal_validation_and_step_configuration(enabled, fresh, explicit_steps):
    module = _load_paths()
    trainer = _trainer(module, enabled, total_steps=explicit_steps)
    raw = _raw(1000)
    dataset = [{key: value[i] for key, value in _as_dict(raw).items()} for i in range(len(raw))]
    trainer._create_dataloader(dataset, dataset, _collate, SequentialSampler(dataset))
    assert trainer.train_dataloader.batch_size == fresh
    assert trainer.val_dataloaders[0].batch_size == 100
    assert trainer.config.data.gen_batch_size == trainer.config.data.train_batch_size == 100
    assert trainer.train_dataloader.drop_last
    expected_steps = explicit_steps if explicit_steps is not None else (1000 // fresh) * 2
    assert trainer.total_training_steps == expected_steps

    first = DataProto.from_single_dict(next(iter(trainer.train_dataloader)))
    combined, context = trainer.bad_case_replay.prepare(first)
    assert len(combined) == fresh
    if enabled:
        trainer.bad_case_replay.update(context, {"word": np.arange(90)})
        assert len(trainer.bad_case_replay) == 90
        second, second_context = trainer.bad_case_replay.prepare(_raw(90, start=1000))
        assert len(second) == 100
        assert second_context.replay_mask.sum() == 10
        assert trainer.train_dataloader.batch_size == 90


@pytest.mark.parametrize("enabled", [False, True])
def test_replay_requires_remax_only_when_enabled(enabled):
    module = _load_paths()
    trainer = _trainer(module, enabled)
    OmegaConf.update(trainer.config, "algorithm.adv_estimator", "grpo")
    raw = _raw(100)
    dataset = [{key: values[i] for key, values in _as_dict(raw).items()} for i in range(len(raw))]
    if enabled:
        with pytest.raises(ValueError, match="ReMax greedy baseline"):
            trainer._create_dataloader(dataset, dataset, _collate, SequentialSampler(dataset))
    else:
        trainer._create_dataloader(dataset, dataset, _collate, SequentialSampler(dataset))
        assert trainer.train_dataloader.batch_size == 100


@pytest.mark.parametrize("nominal,replay_size", [(128, 13), (130, 16)])
def test_replay_loader_rejects_batch_sizes_not_aligned_to_data_parallelism(nominal, replay_size):
    module = _load_paths()
    trainer = _trainer(module)
    trainer.config.data.gen_batch_size = trainer.config.data.train_batch_size = nominal
    trainer.config.algorithm.bad_case_replay.replay_size = replay_size
    OmegaConf.update(trainer.config, "trainer.nnodes", 2, force_add=True)
    OmegaConf.update(trainer.config, "trainer.n_gpus_per_node", 8, force_add=True)
    raw = _raw(256)
    dataset = [{key: values[i] for key, values in _as_dict(raw).items()} for i in range(len(raw))]
    with pytest.raises(ValueError, match="divisible"):
        trainer._create_dataloader(dataset, dataset, _collate, SequentialSampler(dataset))


@pytest.mark.parametrize("rows", [96, 128])
def test_balance_keeps_real_rows_and_their_metadata_aligned(rows):
    module = _load_paths()
    trainer = _trainer(module)
    trainer.actor_rollout_wg = SimpleNamespace(world_size=16)
    batch = _raw(rows)
    metrics = {}
    trainer._balance_batch(batch, metrics)
    assert len(batch) == rows
    assert set(batch.non_tensor_batch["id"].tolist()) == set(range(rows))
    for tensor, info in zip(batch.batch["input_ids"], batch.non_tensor_batch["extra_info"], strict=True):
        assert tensor[0].item() == info["id"]
    assert metrics


class _GenerationWorker:
    world_size = 16

    def __init__(self, rollouts=2):
        self.calls = []
        self.rollouts = rollouts

    def generate_sequences(self, batch):
        do_sample = batch.meta_info.get("do_sample", True) and not batch.meta_info.get("validate", False)
        self.calls.append((do_sample, len(batch)))
        if do_sample:
            groups = batch.batch["input_ids"][:, 0].reshape(-1, self.rollouts)
            assert torch.all(groups == groups[:, :1])
        return DataProto.from_dict(
            tensors={**{key: value.clone() for key, value in batch.batch.items()},
                     "responses": batch.batch["input_ids"][:, :2].clone()},
            non_tensors=copy.deepcopy(batch.non_tensor_batch), meta_info={"timing": {}},
        )


def test_shared_ppo_single_rollout_uses_original_generation_path():
    module = _load_paths()
    trainer = _trainer(module)
    trainer.config.actor_rollout_ref.rollout.n = 1
    trainer.actor_rollout_wg = _GenerationWorker(1)
    reward_sizes = []

    def baseline_reward(batch, **kwargs):
        reward_sizes.append(len(batch))
        return {"reward_tensor": torch.zeros(len(batch), 2), "reward_extra_info": {"word": np.arange(len(batch))}}

    trainer.reward_fn = baseline_reward
    batch, sampled, context, metrics = module.prepare_generation(trainer, _as_dict(_raw(90)))
    assert trainer.actor_rollout_wg.calls == [(True, 90), (False, 90)]
    assert reward_sizes == [90]
    assert len(batch) == len(context.raw_inputs) == 90
    assert len(sampled) == 90
    assert metrics["replay/candidates"] == 90
    assert len(trainer.bad_case_replay) == 90


def _load_dapo_generation_loop(module):
    """Run production generation/filter/continue and metrics, omitting GPU updates."""
    path = ROOT / "recipe/dapo/dapo_ray_trainer.py"
    tree = ast.parse(path.read_text())
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RayDAPOTrainer")
    fit = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "fit")
    outer_index = next(i for i, node in enumerate(fit.body) if isinstance(node, ast.For))
    initial_index = next(i for i, node in enumerate(fit.body) if _assigns(node, "timing_raw"))
    outer = copy.deepcopy(fit.body[outer_index])
    inner = next(node for node in outer.body if isinstance(node, ast.For))
    step_index = next(i for i, node in enumerate(inner.body) if _timer_name(node) == "step")
    step = copy.deepcopy(inner.body[step_index])

    def assigns_response_mask(node):
        return isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
            and target.slice.value == "response_mask" for target in node.targets
        )

    stop = next(i for i, node in enumerate(step.body) if assigns_response_mask(node))
    step.body = step.body[:stop]
    metrics_start = next(i for i, node in enumerate(inner.body) if _assigns(node, "gen_kept_frac"))
    logging_index = next(
        i for i, node in enumerate(inner.body[metrics_start:], start=metrics_start)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "log"
    )
    inner.body = inner.body[:step_index] + [step] + inner.body[metrics_start:logging_index]
    inner.body += ast.parse("self.logged_metrics.append(dict(metrics))").body
    runner = ast.parse("def run_dapo_generations(self): pass").body[0]
    runner.body = copy.deepcopy(fit.body[initial_index:outer_index]) + [outer]
    runner.body += ast.parse("return self.logged_metrics").body

    filter_tree = ast.parse((ROOT / "verl/trainer/ppo/filter_groups.py").read_text())
    selector = next(
        node for node in filter_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "select_prompt_uids_by_metric"
    )
    module.__dict__.update(defaultdict=defaultdict, capture_remax_metric_baselines=lambda *args: None)
    body = ast.parse("from __future__ import annotations").body + [selector, runner]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), module.__dict__)
    return module.run_dapo_generations


def test_dapo_updates_before_filter_continue_and_logs_actual_generation_totals():
    module = _load_paths()
    run = _load_dapo_generation_loop(module)
    trainer = _trainer(module)
    trainer.config.trainer.total_epochs = 1
    trainer.actor_rollout_wg = _GenerationWorker()
    trainer.train_dataloader = [_as_dict(_raw(90)), _as_dict(_raw(90, 1000))]
    trainer.logged_metrics = []
    trainer.use_rm = False
    trainer.gen_steps = 0
    events = []
    baseline_sizes = []
    original_update = trainer.bad_case_replay.update
    original_select = module.select_prompt_uids_by_metric

    def observe(context, extra):
        stats = original_update(context, extra)
        events.append(("buffer_updated", len(context.raw_inputs), len(trainer.bad_case_replay)))
        return stats

    def select(*args, **kwargs):
        assert events[-1][0] == "buffer_updated"
        events.append(("filter", len(trainer.bad_case_replay)))
        return original_select(*args, **kwargs)

    def reward(batch, **kwargs):
        if "reward_baselines" not in batch.batch:
            baseline_sizes.append(len(batch))
            scores = np.arange(len(batch), dtype=float)
            if len(baseline_sizes) == 2:
                scores[-10:] = -1  # All ten sampled replay inputs qualify again.
            return {"reward_tensor": torch.zeros(len(batch), 2), "reward_extra_info": {"word": scores}}
        values = torch.zeros(len(batch), 2)
        values[1:120:2] = 1  # Keep 60 prompt groups per generation; first 60 < target 100.
        return {"reward_tensor": values, "reward_extra_info": {"word": np.zeros(len(batch))}}

    trainer.bad_case_replay.update = observe
    module.select_prompt_uids_by_metric = select
    trainer.reward_fn = reward
    logs = run(trainer)
    assert baseline_sizes == [90, 100]
    assert events == [("buffer_updated", 90, 90), ("filter", 90), ("buffer_updated", 100, 180), ("filter", 180)]
    assert trainer.gen_steps == 1  # The production filtering branch continued once.
    assert len(logs) == 1
    metrics = logs[0]
    assert metrics["train/num_gen_batches"] == 2
    assert metrics["train/gen_kept_frac"] == pytest.approx(120 / 190)
    assert metrics["replay/prompt_count"] == 190
    assert metrics["replay/fresh_count"] == 180
    assert metrics["replay/replayed"] == 10
    assert metrics["replay/candidates"] == 190
    assert metrics["replay/reinsert_candidates"] == metrics["replay/reinserted"] == 10
    assert metrics["replay/reinsert_rate"] == 1  # Ratio of totals, not mean([0, 1]).


def test_aligned_replay_batch_uses_ordinary_generation_and_validation_baseline():
    module = _load_paths()
    run = _load_dapo_generation_loop(module)
    trainer = _trainer(module)
    trainer.config.data.gen_batch_size = trainer.config.data.train_batch_size = 128
    trainer.config.algorithm.bad_case_replay.replay_size = 16
    trainer.config.algorithm.filter_groups.enable = False
    trainer.config.actor_rollout_ref.rollout.n = 8
    trainer.config.trainer.total_epochs = 1
    OmegaConf.update(trainer.config, "trainer.nnodes", 2, force_add=True)
    OmegaConf.update(trainer.config, "trainer.n_gpus_per_node", 8, force_add=True)
    raw = _raw(224)
    dataset = [{key: value[i] for key, value in _as_dict(raw).items()} for i in range(len(raw))]
    trainer._create_dataloader(dataset, dataset, _collate, SequentialSampler(dataset))
    assert trainer.train_dataloader.batch_size == 112
    assert trainer.bad_case_replay.target_replay_count == 16

    class AlignedGenerationWorker(_GenerationWorker):
        def generate_sequences(self, batch):
            # The ordinary worker receives complete, pre-padded prompt rows.
            # No copied dispatch rows, replay masks, or baseline overrides are needed.
            assert len(batch) % self.world_size == 0
            assert batch.batch["input_ids"].shape == (len(batch), 3)
            assert batch.batch["attention_mask"].shape == (len(batch), 3)
            assert "replay_valid_mask" not in batch.batch
            assert "remax_greedy_baseline" not in batch.meta_info
            return super().generate_sequences(batch)

    trainer.actor_rollout_wg = AlignedGenerationWorker(8)
    trainer.logged_metrics = []
    trainer.use_rm = False
    trainer.gen_steps = 0
    baseline_sizes = []

    def reward(batch, **kwargs):
        if "reward_baselines" not in batch.batch:
            baseline_sizes.append(len(batch))
        return {
            "reward_tensor": torch.zeros(len(batch), 2),
            "reward_extra_info": {"word": np.arange(len(batch), dtype=float)},
        }

    trainer.reward_fn = reward
    logs = run(trainer)
    assert trainer.actor_rollout_wg.calls == [(True, 112 * 8), (False, 112), (True, 128 * 8), (False, 128)]
    assert baseline_sizes == [112, 128]
    assert [row["replay/fresh_count"] for row in logs] == [112, 112]
    assert [row["replay/replayed"] for row in logs] == [0, 16]
    assert [row["replay/prompt_count"] for row in logs] == [112, 128]
    assert [row["replay/candidates"] for row in logs] == [112, 128]
    assert [row["replay/buffer_size"] for row in logs] == [112, 224]
    assert [row["replay/reinserted"] for row in logs] == [0, 16]
    assert trainer.train_dataloader.batch_size == 112


def _load_vllm_sampling_and_gt():
    """Execute the real sampling and GT branches without importing the vLLM stack."""
    path = ROOT / "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    tree = ast.parse(path.read_text())
    method = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "generate_sequences"
    )
    start = next(i for i, node in enumerate(method.body) if _assigns(node, "do_sample"))
    stop = next(i for i, node in enumerate(method.body) if _assigns(node, "lora_requests"))
    sampling_context = next(
        node for node in method.body
        if isinstance(node, ast.With) and any(
            isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "update_sampling_params" for item in node.items
        )
    )
    runner = ast.parse("def run(self, prompts, **kwargs): pass").body[0]
    runner.body = ast.parse("batch_size = len(prompts)\nnon_tensor_batch = prompts.non_tensor_batch").body
    runner.body += copy.deepcopy(method.body[start:stop] + sampling_context.body[:2])
    runner.body += ast.parse("return kwargs, gt_responses").body
    namespace = {"logger": SimpleNamespace(info=lambda *args: None)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[runner], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["run"]


@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("do_sample", [False, True])
def test_vllm_baseline_uses_existing_validation_sampling_and_gt_guard(validate, do_sample):
    run = _load_vllm_sampling_and_gt()
    worker = SimpleNamespace(
        config=OmegaConf.create({
            "gt_rollout": True, "n": 2, "response_length": 4,
            "val_kwargs": {
                "temperature": 0.8, "top_p": 0.7, "top_k": 10,
                "repetition_penalty": 1.1, "response_length": None,
            },
        }),
        tokenizer=SimpleNamespace(encode=lambda text: [11, 12], eos_token_id=99),
    )
    batch = _raw(3).repeat(2, interleave=True)
    batch.non_tensor_batch["reward_model"] = np.asarray([{"ground_truth": "reference"}] * len(batch), dtype=object)
    batch.meta_info.update(validate=validate, do_sample=do_sample)
    kwargs, gt_responses = run(worker, batch)
    if not validate and not do_sample:
        assert kwargs == {"best_of": 1, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "temperature": 0, "n": 1}
    elif validate:
        assert kwargs["temperature"] == 0.8
        assert kwargs["top_p"] == 0.7
        assert kwargs["n"] == 1
    else:
        assert kwargs == {}
    expected_gt = {} if validate else {0: [11, 12, 99], 2: [11, 12, 99], 4: [11, 12, 99]}
    assert gt_responses == expected_gt


def _load_agent_loop_sampling():
    path = ROOT / "verl/experimental/agent_loop/agent_loop.py"
    tree = ast.parse(path.read_text())
    method = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "generate_sequences"
    )
    start = next(i for i, node in enumerate(method.body) if _assigns(node, "config"))
    stop = next(
        i for i, node in enumerate(method.body)
        if isinstance(node, ast.If) and any(
            isinstance(child, ast.Constant) and child.value == "agent_name" for child in ast.walk(node.test)
        )
    )
    runner = ast.parse("def run(self, batch): pass").body[0]
    runner.body = copy.deepcopy(method.body[start:stop]) + ast.parse("return sampling_params").body
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[runner], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["run"]


@pytest.mark.parametrize("validate", [False, True])
def test_agent_loop_baseline_uses_existing_validation_sampling(validate):
    run = _load_agent_loop_sampling()
    worker = SimpleNamespace(config=OmegaConf.create({"actor_rollout_ref": {"rollout": {
        "temperature": 0.6, "top_p": 0.9, "calculate_log_probs": True,
        "val_kwargs": {"temperature": 0.8, "top_p": 0.7},
    }}}))
    batch = _raw(1)
    batch.meta_info.update(validate=validate)
    params = run(worker, batch)
    assert params["logprobs"] is True
    assert params["temperature"] == (0.8 if validate else 0.6)
    assert params["top_p"] == (0.7 if validate else 0.9)
    assert "top_k" not in params
    assert "min_p" not in params


def _load_retention_path(trainer_path, flag):
    """Execute the trainer's real DP-size choice and core-algorithm implementation."""
    core_path = ROOT / "verl/trainer/ppo/core_algos.py"
    core_tree = ast.parse(core_path.read_text())
    functions = [
        node for node in core_tree.body if isinstance(node, ast.FunctionDef)
        and node.name in {"deduplicate_rollout_responses", "filter_nonfinite_log_probs"}
    ]
    path = ROOT / trainer_path
    trainer_tree = ast.parse(path.read_text())
    fit = next(
        node for node in ast.walk(trainer_tree) if isinstance(node, ast.FunctionDef) and node.name == "fit"
    )
    branch = next(
        node for node in ast.walk(fit)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Call)
        and isinstance(node.test.func, ast.Attribute) and node.test.func.attr == "get"
        and node.test.args and isinstance(node.test.args[0], ast.Constant) and node.test.args[0].value == flag
    )
    runner = ast.parse("def run(self, batch): pass").body[0]
    runner.body = ast.parse("metrics = {}").body + [copy.deepcopy(branch)]
    runner.body += ast.parse("return batch, metrics").body
    namespace = {"torch": torch, "defaultdict": defaultdict}
    body = ast.parse("from __future__ import annotations").body + functions + [runner]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), namespace)
    return namespace["run"]


def _retention_batch(rows):
    batch = _raw(rows)
    batch.batch["responses"] = torch.arange(rows)[:, None].repeat(1, 2)
    batch.batch["response_mask"] = torch.ones(rows, 2, dtype=torch.long)
    batch.non_tensor_batch["uid"] = np.full(rows, "same-prompt", dtype=object)
    return batch


@pytest.mark.parametrize("trainer_path", ["verl/trainer/ppo/ray_trainer.py", "recipe/dapo/dapo_ray_trainer.py"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("rows", [15, 90])
@pytest.mark.parametrize("duplicate", [False, True])
def test_trainer_dedup_uses_original_dp_alignment_with_replay(trainer_path, enabled, rows, duplicate):
    run = _load_retention_path(trainer_path, "dedup_responses")
    trainer = _trainer(_load_paths(), enabled)
    trainer.config.trainer.dedup_responses = True
    trainer.actor_rollout_wg = SimpleNamespace(world_size=16)
    batch = _retention_batch(rows)
    if duplicate:
        batch.batch["responses"][-1] = batch.batch["responses"][0]
    retained, metrics = run(trainer, batch)
    unique = rows - int(duplicate)
    expected = unique // 16 * 16
    assert len(retained) == expected
    assert retained.non_tensor_batch["id"].tolist() == list(range(expected))
    assert len(batch) == rows
    removed = rows - expected
    if removed:
        assert metrics["rollout/dedup_removed"] == removed
        assert metrics["rollout/dedup_kept"] == expected
        assert metrics["rollout/dedup_ratio"] == pytest.approx(removed / rows)
    else:
        assert retained is batch
        assert metrics == {}


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("rows", [15, 90])
@pytest.mark.parametrize("invalid", [False, True])
def test_dapo_nonfinite_uses_original_dp_alignment_with_replay(enabled, rows, invalid):
    run = _load_retention_path("recipe/dapo/dapo_ray_trainer.py", "filter_nonfinite_log_probs")
    trainer = _trainer(_load_paths(), enabled)
    trainer.config.trainer.filter_nonfinite_log_probs = True
    trainer.config.trainer.balance_batch = False
    trainer.actor_rollout_wg = SimpleNamespace(world_size=16)
    batch = _retention_batch(rows)
    batch.batch["old_log_probs"] = torch.zeros(rows, 2)
    if invalid:
        batch.batch["old_log_probs"][-1, 0] = float("nan")
    valid = rows - int(invalid)
    expected = valid // 16 * 16
    if not expected:
        with pytest.raises(ValueError, match="No complete DP batch remains"):
            run(trainer, batch)
        return
    retained, metrics = run(trainer, batch)
    assert len(retained) == expected
    assert retained.non_tensor_batch["id"].tolist() == list(range(expected))
    assert len(batch) == rows
    assert metrics["nonfinite_prob/invalid"] == int(invalid)
    assert metrics["nonfinite_prob/removed"] == rows - expected
    assert metrics["nonfinite_prob/old"] == int(invalid)
    assert metrics["nonfinite_prob/ref"] == metrics["nonfinite_prob/rollout"] == 0
    if expected == rows:
        assert retained is batch
    else:
        assert len(retained.meta_info["global_token_num"]) == expected
