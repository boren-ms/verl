import ast
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from recipe.phimm.reward.asr_vote import compute_score
from verl import DataProto
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance


def _load_reward_preparation():
    """Run the trainer's batch-to-reward path without loading the GPU/model stack."""
    root = Path(__file__).parents[3]
    path = root / "verl/trainer/ppo/ray_trainer.py"
    tree = ast.parse(path.read_text())
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
    fit = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "fit")

    def timer_name(node):
        if isinstance(node, ast.With):
            call = node.items[0].context_expr
            if isinstance(call, ast.Call) and call.args and isinstance(call.args[0], ast.Constant):
                return call.args[0].value
        return None

    step = next(node for node in ast.walk(fit) if timer_name(node) == "step")
    start = next(
        i
        for i, node in enumerate(step.body)
        if isinstance(node, ast.Assign) and ast.unparse(node.value).startswith("batch.repeat(")
    )
    stop = next(i for i, node in enumerate(step.body) if timer_name(node) == "old_log_prob")
    prepare = ast.parse("def prepare(self, batch, gen_batch_output, greedy_hyps, greedy_rewards): pass").body[0]
    prepare.body = ast.parse("timing_raw = {}; metrics = {}").body + step.body[start:stop]
    prepare.body += ast.parse("return batch").body
    trainer.body = [
        node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "_balance_batch"
    ]
    response_mask = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "compute_response_mask"
    )
    core_tree = ast.parse((root / "verl/trainer/ppo/core_algos.py").read_text())
    deduplicate = next(
        node
        for node in core_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "deduplicate_rollout_responses"
    )
    module = ModuleType("remax_reward_preparation")
    module.__dict__.update(
        np=np,
        torch=torch,
        DataProto=DataProto,
        defaultdict=defaultdict,
        AdvantageEstimator=SimpleNamespace(REMAX="remax"),
        marked_timer=lambda *args, **kwargs: nullcontext(),
        get_seqlen_balanced_partitions=get_seqlen_balanced_partitions,
        log_seqlen_unbalance=log_seqlen_unbalance,
        compute_reward=Mock(return_value=(None, {})),
        compute_reward_async=SimpleNamespace(remote=Mock()),
    )
    body = ast.parse("from __future__ import annotations").body + [trainer, response_mask, deduplicate, prepare]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("rollout_n,dedup", [(1, False), (2, False), (3, True)])
@pytest.mark.parametrize("extra_kind", ["dict", "missing", "invalid"])
@pytest.mark.parametrize("async_reward", [False, True])
def test_remax_greedy_metadata_follows_rows_through_reward_dispatch(rollout_n, dedup, extra_kind, async_reward):
    module = _load_reward_preparation()
    reference = "one two three four five six seven eight nine ten"
    words = reference.split()
    greedy_hyps = ["<src=English><tgt=English>\n" + " ".join(words[:count]) for count in (2, 9, 4, 7)]
    sampled_hyp = "<src=English><tgt=English>\n" + " ".join(words[:5])
    settings = {
        "version": 2609,
        "text_norm": "identity",
        "reduce": "sum",
        "measures": {"word": {"delta": "abs", "vote": "delta"}},
    }
    greedy_rewards = [compute_score(hyp, reference, **settings) for hyp in greedy_hyps]
    non_tensors = {"uid": np.array(list("ABCD"), dtype=object)}
    if extra_kind != "missing":
        non_tensors["extra_info"] = np.array(
            [{"source_id": i} for i in range(4)] if extra_kind == "dict" else [None] * 4, dtype=object
        )
    prompts = DataProto.from_dict(
        tensors={
            "prompt_id": torch.arange(4),
            "reward_baselines": torch.tensor([reward["score"] for reward in greedy_rewards]),
            "reward_baselines_word": torch.tensor([reward["word"] for reward in greedy_rewards]),
        },
        non_tensors=non_tensors,
    )
    responses = []
    for length in (9, 8, 2, 1):
        for i in range(rollout_n):
            response = list(range(1, length + 1))
            if dedup and i == rollout_n - 1:
                response[0] = 99
            responses.append(response + [0] * (9 - length))
    responses = torch.tensor(responses)
    sampled = DataProto.from_dict(
        tensors={
            "responses": responses,
            "attention_mask": torch.cat([torch.ones((len(responses), 1)), responses != 0], dim=-1),
        }
    )
    trainer = module.RayPPOTrainer()
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"adv_estimator": "remax"},
            "actor_rollout_ref": {"rollout": {"n": rollout_n}},
            "trainer": {"dedup_responses": dedup, "balance_batch": True},
            "reward_model": {"launch_reward_fn_async": async_reward},
        }
    )
    trainer.actor_rollout_wg = SimpleNamespace(world_size=2)
    trainer.bad_case_replay = SimpleNamespace(enabled=False)
    trainer.use_rm = False
    trainer.reward_fn = object()

    prepared = module.prepare(trainer, prompts, sampled, greedy_hyps, greedy_rewards)
    dispatched = (
        module.compute_reward_async.remote.call_args.kwargs["data"]
        if async_reward
        else module.compute_reward.call_args.args[0]
    )
    assert dispatched is prepared
    assert len(prepared) == 4 * (2 if dedup else rollout_n)
    assert prepared.batch["prompt_id"].tolist() != torch.arange(4).repeat_interleave(2 if dedup else rollout_n).tolist()
    infos = prepared.non_tensor_batch["extra_info"]
    assert len({id(info) for info in infos}) == len(infos)
    for i, info in enumerate(infos):
        prompt_id = prepared.batch["prompt_id"][i].item()
        assert info["greedy_hyp"] == greedy_hyps[prompt_id]
        assert info["greedy_reward"] == greedy_rewards[prompt_id]
        assert info["greedy_reward"]["word"] == pytest.approx(prepared.batch["reward_baselines_word"][i].item())
        if extra_kind == "dict":
            assert info["source_id"] == prompt_id
        score = compute_score(sampled_hyp, reference, extra_info=info, **settings)["score"]
        advantage = score - prepared.batch["reward_baselines"][i].item()
        assert advantage == pytest.approx(0.5 - greedy_rewards[prompt_id]["word"])
    if extra_kind == "dict":
        assert list(prompts.non_tensor_batch["extra_info"]) == [{"source_id": i} for i in range(4)]


def test_non_remax_reward_metadata_is_preserved():
    module = _load_reward_preparation()
    prompts = DataProto.from_dict(
        tensors={"prompt_id": torch.tensor([0])},
        non_tensors={"uid": np.array(["A"], dtype=object), "extra_info": np.array([{"source_id": 0}], dtype=object)},
    )
    sampled = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[1, 2], [1, 3]]),
            "attention_mask": torch.ones((2, 3)),
        }
    )
    trainer = module.RayPPOTrainer()
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"adv_estimator": "grpo"},
            "actor_rollout_ref": {"rollout": {"n": 2}},
            "trainer": {"dedup_responses": False, "balance_batch": False},
            "reward_model": {"launch_reward_fn_async": False},
        }
    )
    trainer.use_rm = False
    trainer.reward_fn = object()

    prepared = module.prepare(trainer, prompts, sampled, None, None)

    assert list(prepared.non_tensor_batch["extra_info"]) == [{"source_id": 0}] * 2
    assert module.compute_reward.call_args.args[0] is prepared
