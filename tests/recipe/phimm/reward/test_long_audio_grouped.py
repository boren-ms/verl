import numpy as np
import torch

from verl import DataProto
from recipe.phimm.reward.long_audio_grouped import LongAudioGroupedRewardManager


class _Tokenizer:
    eos_token = None

    def __init__(self, responses):
        self.responses = responses

    def decode(self, token_ids, skip_special_tokens=True):
        return self.responses[int(token_ids[0])]


def _run_manager(responses, extra_info, version=None):
    score_calls = []

    def compute_score(**kwargs):
        score_calls.append(kwargs)
        return {"score": 1.0, "wer": 0.0, "n_err": 0, "n_ref": 2}

    data = DataProto.from_dict(
        tensors={
            "prompts": torch.ones((2, 1), dtype=torch.long),
            "responses": torch.tensor([[1], [2]], dtype=torch.long),
            "attention_mask": torch.ones((2, 2), dtype=torch.long),
        },
        non_tensors={
            "extra_info": np.array(extra_info, dtype=object),
            "reward_model": np.array(
                [{"ground_truth": "hello world"}, {"ground_truth": "hello world"}],
                dtype=object,
            ),
            "data_source": np.array(["openml", "openml"], dtype=object),
        },
    )
    manager = LongAudioGroupedRewardManager(
        tokenizer=_Tokenizer(responses),
        num_examine=0,
        compute_score=compute_score,
        version=version,
    )

    result = manager(data, return_dict=True)

    return score_calls, result


def test_manager_parses_2607_responses_before_merging():
    responses = {
        1: (
            "Audio Language: English.\n"
            "<ASR_VERBATIM><lang=English><TXT>world</TXT></ASR_VERBATIM>"
        ),
        2: (
            "Audio Language: English.\n"
            "<ASR_VERBATIM><lang=English><TXT>hello</TXT></ASR_VERBATIM>"
        ),
    }
    extra_info = [
        {"parent_audio_path": "parent.wav", "seg_start": 10.0},
        {"parent_audio_path": "parent.wav", "seg_start": 0.0},
    ]

    score_calls, result = _run_manager(responses, extra_info, version=2607)

    assert len(score_calls) == 1
    assert score_calls[0]["solution_str"] == "hello\nworld"
    assert result["reward_extra_info"]["wer"] == [0.0, 0.0]


def test_manager_uses_current_response_format_by_default():
    responses = {
        1: "<src=English><tgt=English>\nhello",
        2: "<src=English><tgt=English>\nworld",
    }
    extra_info = [
        {"parent_audio_path": "parent.wav", "seg_start": 0.0},
        {"parent_audio_path": "parent.wav", "seg_start": 10.0},
    ]

    score_calls, _ = _run_manager(responses, extra_info)

    assert len(score_calls) == 1
    assert score_calls[0]["solution_str"] == "hello\nworld"
