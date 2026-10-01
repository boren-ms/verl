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


def _run_manager(responses, extra_info, version=None, data_sources=None):
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
            "data_source": np.array(data_sources if data_sources is not None else ["openml", "openml"], dtype=object),
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
    assert result["reward_extra_info"]["wer"] == [None, 0.0]


def test_manager_uses_2609_response_format():
    responses = {
        1: "<src=English><tgt=English>\n<TXT>hello</TXT>",
        2: "<src=English><tgt=English>\n<TXT>world</TXT>",
    }
    extra_info = [
        {"parent_audio_path": "parent.wav", "seg_start": 0.0},
        {"parent_audio_path": "parent.wav", "seg_start": 10.0},
    ]

    score_calls, _ = _run_manager(responses, extra_info, version=2609)

    assert len(score_calls) == 1
    assert score_calls[0]["solution_str"] == "hello\nworld"


def test_manager_keeps_tasks_with_shared_parent_separate():
    responses = {
        1: "<src=English><tgt=English>\n<TXT>Hello, world.</TXT>",
        2: "<src=English><tgt=English>\n<LEXICAL><TXT>hello world</TXT>",
    }
    extra_info = [
        {"parent_audio_path": "parent.wav", "seg_start": 0.0},
        {"parent_audio_path": "parent.wav", "seg_start": 0.0},
    ]

    score_calls, result = _run_manager(
        responses,
        extra_info,
        version=2609,
        data_sources=["earning_chunk_verb", "earning_chunk_lex"],
    )

    assert len(score_calls) == 2
    assert [call["data_source"] for call in score_calls] == ["earning_chunk_verb", "earning_chunk_lex"]
    assert [call["solution_str"] for call in score_calls] == ["Hello, world.", "hello world"]
    assert all(call["extra_info"]["n_segments"] == 1 for call in score_calls)
    assert all(call["extra_info"]["parent_audio_path"] == "parent.wav" for call in score_calls)
    assert result["reward_extra_info"]["wer"] == [0.0, 0.0]
