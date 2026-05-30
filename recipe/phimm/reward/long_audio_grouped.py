# Reward manager that concatenates per-segment hyps of a long-audio recording
# (split by `svad_explode`) and scores them once per parent against the full
# reference. Each segment row in the group is then assigned the same aggregate
# result so downstream per-row averaging reduces to the per-parent score.

from __future__ import annotations

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


def _parent_key(extra_info: dict, ground_truth, fallback: str) -> str:
    if extra_info:
        for k in ("parent_audio_path", "audio_path"):
            v = extra_info.get(k)
            if v:
                return str(v).split("#", 1)[0]
    return fallback


def _seg_start(extra_info: dict) -> float:
    if not extra_info:
        return 0.0
    v = extra_info.get("seg_start")
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


@register("long_audio_grouped")
class LongAudioGroupedRewardManager(AbstractRewardManager):
    """Group segments by parent wav, concatenate hyps, score once per parent."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.max_resp_len = max_resp_len
        self.overlong_buffer_cfg = overlong_buffer_cfg

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            return data.batch["rm_scores"]

        n = len(data)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: dict[str, list] = defaultdict(list)

        # Decode every row up front so we can group + concat.
        decoded = []
        for i in range(n):
            item = data[i]
            prompt_ids = item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = item.batch["responses"]
            valid_response_length = int(item.batch["attention_mask"][prompt_length:].sum())
            valid_response_ids = response_ids[:valid_response_length]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos = self.tokenizer.eos_token
            if eos and response_str.endswith(eos):
                response_str = response_str[: -len(eos)]
            extra_info = item.non_tensor_batch.get("extra_info", None) or {}
            ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = item.non_tensor_batch[self.reward_fn_key]
            decoded.append({
                "i": i,
                "response": response_str,
                "valid_response_length": valid_response_length,
                "extra_info": extra_info,
                "ground_truth": ground_truth,
                "data_source": data_source,
                "parent": _parent_key(extra_info, ground_truth, fallback=f"__row_{i}__"),
                "seg_start": _seg_start(extra_info),
            })

        # Group by parent and score once per group.
        groups: dict[str, list[dict]] = defaultdict(list)
        for d in decoded:
            groups[d["parent"]].append(d)

        already_print_data_sources: dict[str, int] = {}

        # Per-row records, keyed by original row index, so the dumped
        # ``reward_extra_info`` columns stay aligned with the trainer's row-order
        # ``input`` / ``output`` / ``score`` lists (the group iteration order
        # below is NOT the row order). ``parent_audio_path`` + ``concat_hyp`` are
        # added so the dump can be collapsed to one row per long-audio recording.
        row_extra: dict[int, dict] = {}
        extra_keys_order: list[str] = []

        for parent, members in groups.items():
            members.sort(key=lambda m: (m["seg_start"], m["i"]))
            concat_hyp = " ".join(m["response"].strip() for m in members if m["response"].strip())
            head = members[0]
            ground_truth = head["ground_truth"]
            data_source = head["data_source"]
            extra_info = dict(head["extra_info"])
            extra_info["n_segments"] = len(members)
            extra_info["parent_audio_path"] = parent

            result = self.compute_score(
                data_source=data_source,
                solution_str=concat_hyp,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(result, dict):
                score = float(result.get("score", 0.0))
                result_dict = dict(result)
            else:
                score = float(result)
                result_dict = {"score": score}
            result_dict["parent_audio_path"] = parent
            result_dict["concat_hyp"] = concat_hyp
            if not extra_keys_order:
                extra_keys_order = list(result_dict.keys())

            for m in members:
                reward_tensor[m["i"], max(m["valid_response_length"] - 1, 0)] = score
                row_extra[m["i"]] = result_dict

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"[long_audio_grouped] parent={parent} n_segments={len(members)}")
                print(f"[long_audio_grouped][hyp] {concat_hyp[:500]}{'...' if len(concat_hyp) > 500 else ''}")
                print(f"[long_audio_grouped][ref] {str(ground_truth)[:500]}{'...' if len(str(ground_truth)) > 500 else ''}")
                print(f"[long_audio_grouped][result] {result_dict}")

        # Emit reward_extra_info columns in row order (length == n) so they stay
        # aligned with the trainer's per-row dump lists.
        for i in range(n):
            rec = row_extra.get(i, {})
            for k in extra_keys_order:
                reward_extra_info[k].append(rec.get(k))

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
