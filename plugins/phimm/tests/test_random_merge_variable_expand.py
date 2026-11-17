"""Variable-count offline augmentation keeps singletons and whole-audio order."""

from collections import Counter
from itertools import combinations

import datasets
import pytest
from recipe.phimm.data.dataset import augment
from recipe.phimm.tools.random_merge_expand import plan_random_expansion
from recipe.phimm.tools.random_merge_list import _merge_rows


def rows(n=300):
    return [
        dict(audio_path=f"{i}.wav", audio_length=2.0, text=f"word{i}", keywords=[str(i)], language="en")
        for i in range(n)
    ]


def test_uniform_quota_unique_singletons_and_partners_reproducible():
    originals = rows()
    excluded = [(0,), (1, 2), (3, 4, 5)]
    plans = plan_random_expansion(originals, excluded_groups=excluded)
    assert plans == plan_random_expansion(originals, excluded_groups=excluded)
    assert plans != plan_random_expansion(originals, seed=44, excluded_groups=excluded)
    assert Counter(len(members) for members, _, _ in plans) == {k: 240 for k in range(1, 6)}
    anchored, used_pairs, used_groups, reused = Counter(), set(), set(), Counter()
    for members, round_index, anchor in plans:
        assert anchor in members and 0 <= round_index < 4
        assert len(set(members)) == len(members)
        signature = tuple(sorted(members))
        assert signature not in used_groups and signature not in excluded
        used_groups.add(signature)
        for pair in combinations(signature, 2):
            assert pair not in used_pairs
            used_pairs.add(pair)
        anchored[anchor] += 1
        reused.update(members)
    assert len(anchored) == 300 and set(anchored.values()) == {4}
    assert sum(reused.values()) == 3600 and min(reused.values()) >= 4


def test_singleton_and_five_segments_keep_task_components():
    originals = rows()
    plans = plan_random_expansion(originals)
    samples = []
    for count in (1, 5):
        members, _, _ = next(plan for plan in plans if len(plan[0]) == count)
        samples.append(
            _merge_rows([originals[i] for i in members], list(members), [2.0] * count, " ", default_language="English")
        )
    formatted = augment(
        datasets.Dataset.from_list(samples),
        add_task_info=dict(style="verb", language="English", lang_hint=True),
        post_process=dict(verl_format=dict(prompt_key="prompt")),
    )
    for raw, final in zip(samples, formatted, strict=True):
        assert isinstance(raw["audio_path"], list)
        assert final["reward_model"]["ground_truth"] == raw["text"]
        assert final["reward_model"]["gt_output"].count("<src=English>") == len(raw["audio_path"])


def test_impossible_quota_duration_and_unique_inputs_fail():
    with pytest.raises(ValueError, match="singleton quota"):
        plan_random_expansion(rows(), rounds=6)
    with pytest.raises(ValueError, match="duration cap"):
        plan_random_expansion(rows(), max_audio_length=10)
    data = rows()
    data[-1]["audio_path"] = data[0]["audio_path"]
    with pytest.raises(ValueError, match="unique"):
        plan_random_expansion(data)
