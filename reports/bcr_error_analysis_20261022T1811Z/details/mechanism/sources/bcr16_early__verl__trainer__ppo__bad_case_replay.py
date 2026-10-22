# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""In-memory replay of raw ReMax prompts, selected using greedy reward metrics."""

import copy
import hashlib
import math
import pickle
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from verl import DataProto


def _get(config, name, default=None):
    return (
        config.get(name, default)
        if isinstance(config, Mapping) or hasattr(config, "get")
        else getattr(config, name, default)
    )


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return value.copy()
        result = np.empty(value.shape, dtype=object)
        for index in np.ndindex(value.shape):
            result[index] = _cpu_copy(value[index])
        return result
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _snapshot(data):
    return DataProto.from_dict(
        tensors={key: _cpu_copy(value) for key, value in data.batch.items()} if data.batch is not None else {},
        non_tensors=_cpu_copy(data.non_tensor_batch),
        meta_info=_cpu_copy(data.meta_info),
    )


def _fingerprint(value):
    """Canonicalize nested raw inputs without depending on object addresses."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return (
            "tensor",
            str(tensor.dtype),
            tuple(tensor.shape),
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes(),
        )
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return ("objects", value.shape, tuple(_fingerprint(item) for item in value.flat))
        return ("array", str(value.dtype), value.shape, value.tobytes())
    if isinstance(value, Mapping):
        return tuple((str(key), _fingerprint(value[key])) for key in sorted(value, key=str))
    if isinstance(value, (tuple, list)):
        return tuple(_fingerprint(item) for item in value)
    return value


def _identity(row):
    values = {key: value[0] for key, value in row.non_tensor_batch.items()}
    extra = values.get("extra_info")
    extra = extra if isinstance(extra, dict) else {}
    source = values.get("data_source") or extra.get("data_source", "")
    identifiers = {}
    robust = False
    for name in (
        "id",
        "audio_id",
        "utterance_id",
        "source_id",
        "index",
        "audio_path",
        "audio_chunk",
        "audio_file",
        "parent_audio_path",
        "source_audio_path",
        "seq",
        "segment_id",
        "seg_index",
        "segment_index",
        "seg_start",
        "seg_end",
        "start_time",
        "end_time",
    ):
        value = values.get(name)
        if value is None or (isinstance(value, str) and not value):
            value = extra.get(name)
        if value is None or (isinstance(value, str) and not value):
            continue
        identifiers[name] = value
        if name in {"id", "audio_id", "utterance_id", "audio_path", "audio_chunk", "audio_file"}:
            robust |= not (isinstance(value, str) and value.lower() in {"unknown_id", "unknown", "none", "null"})
        elif name == "index":
            # Phimm and RLHFDataset synthesize index=0 when the source has none.
            # It cannot identify a row by itself.
            robust |= bool(np.asarray(value).size == 1 and np.asarray(value).item() != 0)
    if robust:
        # Parent IDs are shared by VAD/chunk children; include every available
        # audio reference and segment coordinate rather than stopping at ID.
        key = (source, identifiers)
    else:
        tensors = dict(row.batch.items()) if row.batch is not None else {}
        # Exclude batch-dependent left padding from the identity fallback.
        mask = tensors.get("attention_mask")
        if mask is not None and mask.ndim == 2:
            keep = mask[0].bool()
            tensors = {
                key: value[..., keep] if key in {"input_ids", "attention_mask", "position_ids"} else value
                for key, value in tensors.items()
            }
        key = (source, tensors, values)
    return hashlib.sha256(pickle.dumps(_fingerprint(key), protocol=5)).hexdigest()


_GENERATED_KEYS = {
    "responses",
    "response_mask",
    "uid",
    "token_level_scores",
    "token_level_rewards",
    "advantages",
    "returns",
    "old_log_probs",
    "rollout_log_probs",
    "ref_log_prob",
    "values",
    "entropys",
    "rm_scores",
    "remax_advantage_mask",
    "greedy_hyp",
    "greedy_reward",
}


def _check_raw(data):
    keys = set(data.non_tensor_batch)
    if data.batch is not None:
        keys.update(data.batch.keys())
    generated = sorted(key for key in keys if key in _GENERATED_KEYS or key.startswith("reward_baselines"))
    if generated:
        raise ValueError(f"Bad-case replay requires pre-generation inputs; found generated fields: {generated}")


def _merge_raw(parts, pad_token_id):
    """Join batches, left-padding only known prompt or full-sequence tensors."""
    parts = [_snapshot(part) for part in parts]
    tensor_keys = set(parts[0].batch.keys()) if parts[0].batch is not None else set()
    if any((set(part.batch.keys()) if part.batch is not None else set()) != tensor_keys for part in parts):
        raise ValueError("Replay and fresh inputs must have matching tensor fields")
    tensors = {}
    for key in tensor_keys:
        values = [part.batch[key] for part in parts]
        if len({(value.dtype, value.ndim) for value in values}) != 1:
            raise ValueError(f"Replay tensor {key!r} has incompatible dtype or rank")
        if len({tuple(value.shape[1:]) for value in values}) != 1:
            if key not in {"prompts", "input_ids", "attention_mask", "position_ids"} or values[0].ndim < 2:
                raise ValueError(f"Replay tensor {key!r} has incompatible non-sequence shapes")
            if len({tuple(value.shape[1:-1]) for value in values}) != 1:
                raise ValueError(f"Replay tensor {key!r} has incompatible non-sequence shapes")
            width = max(value.shape[-1] for value in values)
            fill = pad_token_id if key in {"prompts", "input_ids"} else 0
            values = [F.pad(value, (width - value.shape[-1], 0), value=fill) for value in values]
        tensors[key] = torch.cat(values, dim=0)
    non_tensors = {}
    for key in dict.fromkeys(key for part in parts for key in part.non_tensor_batch):
        values = [part.non_tensor_batch.get(key) for part in parts]
        if all(value is not None for value in values) and len({value.shape[1:] for value in values}) == 1:
            non_tensors[key] = np.concatenate(values)
        else:
            rows = [
                None if value is None else value[i]
                for part, value in zip(parts, values, strict=True)
                for i in range(len(part))
            ]
            non_tensors[key] = np.empty(len(rows), dtype=object)
            non_tensors[key][:] = rows
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=parts[0].meta_info)


def concat_replay_batches(parts: list[DataProto], pad_token_id: int) -> DataProto:
    """Accumulate filtered rollouts with different padded prompt lengths.

    Full input/attention/position sequences and prompt-only tensors receive left
    padding. Response-only tensors retain their fixed response width and values.
    Like DataProto.concat, metadata belongs to the first input batch.
    """
    if not parts:
        raise ValueError("Cannot concatenate an empty list of replay batches")
    return _merge_raw(parts, pad_token_id)


@dataclass
class ReplayContext:
    """An isolated raw snapshot aligned with one generation batch's greedy scores."""

    raw_inputs: DataProto
    identities: list[str]
    replay_mask: np.ndarray
    fresh_count: int
    target_replay_count: int


@dataclass
class _Entry:
    inputs: DataProto
    score: float
    metrics: dict[str, float]


class BadCaseReplay:
    """Pop difficult raw inputs and retain the globally lowest-scoring observations.

    State deliberately lives only in this object: checkpoint resume starts empty.
    Call ``prepare`` before assigning training UIDs or mutating input fields, and
    ``update`` once greedy rewards are available, before any filtering/reordering.
    """

    def __init__(self, config, nominal_batch_size, *, pad_token_id=0):
        self.enabled = bool(_get(config, "enable", False))
        if isinstance(nominal_batch_size, bool) or not isinstance(nominal_batch_size, int) or nominal_batch_size <= 0:
            raise ValueError("Bad-case replay nominal_batch_size must be a positive integer")
        self.capacity = _get(config, "capacity", 400)
        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int) or self.capacity <= 0:
            raise ValueError("Bad-case replay capacity must be a positive integer")
        self.add_ratio = self._ratio(_get(config, "add_ratio", 0.2), "add_ratio", inclusive=True)
        replay_ratio = self._ratio(_get(config, "replay_ratio", 0.1), "replay_ratio", inclusive=False)
        self.sampling_strategy = _get(config, "sampling_strategy", "random")
        if self.sampling_strategy not in {"random", "worst"}:
            raise ValueError("Bad-case replay sampling_strategy must be 'random' or 'worst'")
        selection = _get(config, "selection", {})
        weights = _get(selection, "weights", {"word": 0.7, "char": 0.3})
        if not hasattr(weights, "items") or not weights:
            raise ValueError("Bad-case replay selection.weights must be a nonempty metric mapping")
        self.weights = {}
        for key, weight in weights.items():
            if not isinstance(key, str) or not key:
                raise ValueError("Bad-case replay metric names must be nonempty strings")
            try:
                number = float(weight)
            except (TypeError, ValueError) as exc:
                raise ValueError("Bad-case replay weights must be finite and non-negative") from exc
            if not math.isfinite(number) or number < 0:
                raise ValueError("Bad-case replay weights must be finite and non-negative")
            self.weights[key] = number
        total = sum(self.weights.values())
        if not math.isfinite(total) or total <= 0:
            raise ValueError("Bad-case replay selection weights must have a finite positive sum")
        self.weights = {key: weight / total for key, weight in self.weights.items() if weight > 0}
        self.target_replay_count = math.floor(nominal_batch_size * replay_ratio) if self.enabled else 0
        self.fresh_batch_size = nominal_batch_size - self.target_replay_count
        self.pad_token_id = pad_token_id
        self._rng = np.random.default_rng(_get(config, "seed", 42))
        self._entries: dict[str, _Entry] = {}

    @staticmethod
    def _ratio(value, name, *, inclusive):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Bad-case replay {name} must be a finite ratio") from exc
        if not math.isfinite(number) or number < 0 or (number > 1 if inclusive else number >= 1):
            raise ValueError(f"Bad-case replay {name} must lie in [0, {'1]' if inclusive else '1)'}")
        return number

    def __len__(self):
        return len(self._entries)

    def prepare(self, fresh: DataProto) -> tuple[DataProto, ReplayContext | None]:
        if not self.enabled:
            return fresh, None
        _check_raw(fresh)
        if len(fresh) != self.fresh_batch_size:
            raise ValueError(f"Bad-case replay expects {self.fresh_batch_size} fresh prompts, got {len(fresh)}")
        count = min(self.target_replay_count, len(self))
        keys = list(self._entries)
        if self.sampling_strategy == "random":
            chosen = [keys[index] for index in self._rng.choice(len(keys), size=count, replace=False)]
        else:
            chosen = sorted(keys, key=lambda key: self._entries[key].score)[:count]
        replay = [self._entries[key].inputs for key in chosen]
        combined = _merge_raw([fresh, *replay], self.pad_token_id) if replay else fresh
        snapshot = _snapshot(combined)
        identities = [_identity(snapshot[i : i + 1]) for i in range(len(snapshot))]
        for key in chosen:
            del self._entries[key]
        replay_mask = np.arange(len(combined)) >= len(fresh)
        return combined, ReplayContext(snapshot, identities, replay_mask, len(fresh), self.target_replay_count)

    def _scores(self, size, baseline_extra):
        metrics = {}
        scores = np.zeros(size, dtype=np.float64)
        for key, weight in self.weights.items():
            value = baseline_extra.get(key)
            if value is None:
                raise ValueError(f"Bad-case replay selection metric {key!r} is missing from greedy rewards")
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            try:
                vector = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Bad-case replay metric {key!r} must contain numeric scalar values") from exc
            if vector.shape != (size,):
                raise ValueError(f"Bad-case replay metric {key!r} must have shape ({size},), got {vector.shape}")
            metrics[key] = vector
            scores += weight * vector
        return scores, metrics

    def update(self, context, baseline_extra, baseline_reward_tensor=None) -> dict[str, float]:
        if not self.enabled or context is None:
            return {}
        size = len(context.raw_inputs)
        scores, metrics = self._scores(size, baseline_extra or {})
        finite = np.isfinite(scores)
        valid = np.flatnonzero(finite)
        count = min(math.floor(size * self.add_ratio), len(valid))
        candidates = valid[np.argsort(scores[valid], kind="stable")[:count]]
        latest = {context.identities[index]: index for index in valid}
        # Refresh only observations that still reside in the buffer. Popped rows
        # re-enter solely via the current generation's bottom-fraction candidates.
        old_keys = set(self._entries)
        candidate_keys = [context.identities[index] for index in candidates]
        merged_keys = old_keys | set(candidate_keys)
        for key in merged_keys:
            if key in latest:
                index = latest[key]
                self._entries[key] = _Entry(
                    _snapshot(context.raw_inputs[index : index + 1]),
                    float(scores[index]),
                    {name: float(values[index]) for name, values in metrics.items()},
                )
        # Iterate new candidate IDs in selection order so equal-score ties remain
        # stable and reproducible rather than following hash-set iteration order.
        ordered_keys = [key for key in self._entries if key in old_keys]
        ordered_keys.extend(dict.fromkeys(candidate_keys))
        ordered_keys = list(dict.fromkeys(ordered_keys))
        ranked = sorted(ordered_keys, key=lambda key: self._entries[key].score)
        retained = ranked[: self.capacity]
        evicted = len(ranked) - len(retained)
        self._entries = {key: self._entries[key] for key in retained}
        replay_candidates = [index for index in candidates if context.replay_mask[index]]
        reinserted = len({context.identities[index] for index in replay_candidates} & set(retained))
        replayed = int(context.replay_mask.sum())
        stats = {
            "replay/target_count": context.target_replay_count,
            "replay/replayed": replayed,
            "replay/fresh_count": context.fresh_count,
            "replay/prompt_count": size,
            "replay/candidates": len(candidates),
            "replay/invalid_scores": int((~finite).sum()),
            "replay/deduplicated": len(old_keys) + len(candidates) - len(merged_keys),
            "replay/evicted": evicted,
            "replay/buffer_size": len(self),
            "replay/reinsert_candidates": len(replay_candidates),
            "replay/reinserted": reinserted,
            "replay/reinsert_rate": reinserted / replayed if replayed else 0.0,
            "replay/score_count": len(valid),
        }
        if len(valid):
            stats.update(
                {
                    "replay/score_min": float(scores[valid].min()),
                    "replay/score_mean": float(scores[valid].mean()),
                    "replay/score_max": float(scores[valid].max()),
                }
            )
        return stats


class ReplayMetrics:
    """Aggregate generation counts across DAPO filtering retries."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._counts = {}
        self._score_sum = 0.0
        self._score_min = math.inf
        self._score_max = -math.inf

    def update(self, stats):
        for key, value in stats.items():
            if key in {"replay/reinsert_rate", "replay/score_min", "replay/score_mean", "replay/score_max"}:
                continue
            self._counts[key] = value if key == "replay/buffer_size" else self._counts.get(key, 0) + value
        if stats.get("replay/score_count", 0):
            self._score_sum += stats["replay/score_mean"] * stats["replay/score_count"]
            self._score_min = min(self._score_min, stats["replay/score_min"])
            self._score_max = max(self._score_max, stats["replay/score_max"])

    def as_dict(self):
        if not self._counts:
            return {}
        result = dict(self._counts)
        denominator = result.get("replay/replayed", 0)
        result["replay/reinsert_rate"] = result.get("replay/reinserted", 0) / denominator if denominator else 0.0
        if result.get("replay/score_count", 0):
            result.update(
                {
                    "replay/score_min": self._score_min,
                    "replay/score_mean": self._score_sum / result["replay/score_count"],
                    "replay/score_max": self._score_max,
                }
            )
        return result
