import pytest

from verl.workers.rollout.gt_rollout import generate_with_gt_responses, prepare_gt_responses


class Tokenizer:
    eos_token_id = 999

    def encode(self, text, add_special_tokens=True):
        assert not add_special_tokens
        return list(text.encode())


def group(n=8):
    row = {
        "output_protocol": "asr-keywords-v1",
        "gt_group_id": "audio-id",
        "ground_truth": "rare",
        "gt_output": "rare\n<KEYWORDS>\nrare\n",
    }
    return [dict(row) for _ in range(n)]


def test_complete_gt_only_one_candidate_and_eos():
    rows = group() + [{**r, "gt_group_id": "second"} for r in group()]
    result = prepare_gt_responses(rows, Tokenizer(), 8, 128, purpose="train_sampling")
    assert list(result) == [0, 8]
    assert bytes(result[0][:-1]).decode() == rows[0]["gt_output"]
    assert result[0][-1] == 999


@pytest.mark.parametrize(
    "overrides", [{"do_sample": False}, {"validate": True}, {"purpose": "evaluation"}, {"purpose": "baseline"}]
)
def test_baseline_and_all_evaluation_get_no_gt(overrides):
    kwargs = {"purpose": "train_sampling", **overrides}
    assert prepare_gt_responses(group(), Tokenizer(), 8, 128, **kwargs) == {}


@pytest.mark.parametrize("failure", ["length", "missing", "group", "purpose", "partial"])
def test_invalid_gt_fails_before_generation(failure):
    rows = group()
    kwargs = {"purpose": "train_sampling"}
    budget = 128
    if failure == "length":
        budget = 5
    if failure == "missing":
        rows[0].pop("gt_output")
    if failure == "group":
        rows[1]["gt_group_id"] = "other"
    if failure == "purpose":
        kwargs = {}
    if failure == "partial":
        rows.pop()
    with pytest.raises(ValueError):
        prepare_gt_responses(rows, Tokenizer(), 8, budget, **kwargs)


def test_native_finish_reasons_survive_stop_without_eos_and_gt_scoring():
    from types import SimpleNamespace

    class Engine:
        def generate(self, **kwargs):
            return [
                SimpleNamespace(outputs=[SimpleNamespace(token_ids=[42], finish_reason="length")]),
                SimpleNamespace(outputs=[SimpleNamespace(token_ids=[43], finish_reason="stop")]),
                SimpleNamespace(outputs=[SimpleNamespace(token_ids=[44], finish_reason="length")]),
            ]

    responses, probs, reasons = generate_with_gt_responses(
        Engine(),
        [{"prompt_token_ids": [1]}] * 3,
        SimpleNamespace(max_tokens=10),
        {0: [7, 999]},
        False,
        return_finish_reasons=True,
    )
    assert responses == [[7, 999], [43], [44]]
    assert probs == []
    assert reasons == ["stop", "stop", "length"]
