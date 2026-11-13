"""Keep the source recipe's scorer dispatch and corpus-count aggregation."""

import importlib
from collections import defaultdict
from functools import cache

import numpy as np


@cache
def load_score_function(path, name):
    module = importlib.import_module(path.removesuffix(".py").replace("/", "."))
    return getattr(module, name)


def score_response(config, row, response, *, validate=False, extra_info=None):
    section = config.get("val_reward", {}) if validate else config
    registrations = section.get("reward_functions", {})
    mapping = section.get("reward_function_by_data_source", {})
    source = row["data_source"]
    name = mapping.get(source, source)
    if name not in registrations:
        raise ValueError(f"No {'validation' if validate else 'training'} scorer for {source!r}")
    spec = registrations[name]
    fn = load_score_function(spec["path"], spec["name"])
    return fn(
        solution_str=response,
        ground_truth=row["reward_model"]["ground_truth"],
        extra_info=dict(extra_info if extra_info is not None else row.get("extra_info", {}) or {}),
        **dict(spec.get("reward_kwargs", {})),
    )


def aggregate_error_rates(data_sources, extra_infos):
    """Use sum(errors)/sum(reference units), never mean(utterance WER)."""
    totals = defaultdict(lambda: defaultdict(float))
    prefixes = {"n_": "p_", "nb_": "pb_", "nu_": "pu_", "dter_n_": "dter_p_", "eer_n_": "eer_p_", "nc_": "pc_"}
    for i, source in enumerate(data_sources):
        for key, values in extra_infos.items():
            value = values[i]
            if value is not None and isinstance(value, (int, float)):
                totals[source][key] += value
    metrics = {}
    for source, counts in totals.items():
        for prefix, output_prefix in prefixes.items():
            denominator = counts.get(prefix + "ref", 0)
            if denominator <= 0:
                continue
            for key, value in counts.items():
                if key.startswith(prefix) and key != prefix + "ref":
                    name = output_prefix + key[len(prefix) :]
                    section = "val-core" if name.endswith(("p_err", "pb_err")) else "val-aux"
                    metrics[f"{section}/{source}/{name}/mean@1"] = value / denominator
    return metrics


# Reused from the source branch: verl/trainer/ppo/ray_trainer.py.
def _compute_reward_metrics(reward_extra_infos: dict[str, list]) -> dict[str, float]:
    """Aggregate reward means into core, delta, and vote tracking namespaces."""
    metrics = {}
    for key, values in reward_extra_infos.items():
        numeric_values = np.asarray(
            [value for value in values if isinstance(value, (bool, int, float, np.bool_, np.number))],
            dtype=np.float64,
        )
        numeric_values = numeric_values[np.isfinite(numeric_values)]
        if numeric_values.size == 0:
            continue
        if key.startswith("d_"):
            metrics[f"reward-delta/{key.removeprefix('d_')}/mean"] = float(np.mean(numeric_values))
        elif key.startswith("v_"):
            metrics[f"reward-vote/{key.removeprefix('v_')}/mean"] = float(np.mean(numeric_values))
        else:
            metrics[f"reward-core/{key}/mean"] = float(np.mean(numeric_values))
        metrics[f"reward-aux/{key}/std"] = float(np.std(numeric_values))
        metrics[f"reward-aux/{key}/max"] = float(np.max(numeric_values))
        metrics[f"reward-aux/{key}/min"] = float(np.min(numeric_values))
    return metrics
