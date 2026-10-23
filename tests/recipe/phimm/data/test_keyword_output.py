from pathlib import Path

from datasets import Dataset
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from recipe.phimm.data.dataset import add_task_info, verl_format_ds
from recipe.phimm.data.keyword_output import PROTOCOL, parse_keyword_output, target_keywords
from recipe.phimm.data.prompts import get_task_output, get_task_prompt
from recipe.phimm.reward.asr_keywords import compute_score, eval_score, word_error_counts


@pytest.mark.parametrize(
    "text,keys",
    [
        ("asked jean valjean fauchelevent replied", ["fauchelevent"]),
        ("the words mated and intermingled", ["mated", "intermingled"]),
        ("when i was a young man", []),
    ],
)
@pytest.mark.parametrize("prefix_prob", [0.0, 1.0])
def test_dataset_gt_roundtrip_and_reward(text, keys, prefix_prob):
    source = Dataset.from_list(
        [
            {
                "id": "sample",
                "text": text,
                "keywords": keys,
                "keyword_annotation_valid": True,
                "audio_path": "sample.flac",
            }
        ]
    )
    task = add_task_info(
        source, task="lang_asr_verb", version=2609, output_keywords=True, lang_hint=True, prefix_prob=prefix_prob
    )
    row = verl_format_ds(task)[0]
    answer = row["reward_model"]["gt_output"]
    parsed = parse_keyword_output(row["extra_info"]["prefix"] + answer, "stop")
    assert parsed.valid and parsed.predicted_keywords == keys
    assert row["reward_model"]["ground_truth"] == text
    assert row["reward_model"]["gt_group_id"] == "sample"
    kwargs = dict(
        extra_info=row["extra_info"],
        version=2609,
        measures={"word": {"beta": 1}},
        reduce={"mode": "average", "total": 1},
    )
    score = compute_score(answer, text, **kwargs)
    assert score["n_err"] == 0 and score["kw_list_f1"] == 1
    assert score["score"] == pytest.approx(1.6)
    assert score["extra_info"]["parsed_transcription"] == text
    if not keys:
        assert answer.endswith("\n<KEYWORDS>\n\n")


@pytest.mark.parametrize(
    "tail",
    [
        "\n<keywords>\nrare\n",
        "\n<Keywords>\nrare\n",
        "\n<KEYWORDS>\nrare, rare\n",
        "\n<KEYWORDS>\nrare,\n",
        "\n<KEYWORDS>\nrare\nexplanation",
        "\n<KEYWORDS>\nrare\n<KEYWORDS>\nrare",
        " <KEYWORDS>\nrare\n",
        '\n<KEYWORDS>\n"Washington, D.C."\n',
    ],
)
def test_malformed_blocks_are_not_hidden_or_rewarded(tail):
    body = get_task_output(task="lang_asr_verb", text="a rare word", version=2609)
    extra = {"output_protocol": PROTOCOL, "keywords": ["rare"], "keyword_annotation_valid": True}
    result = eval_score(body + tail, "a rare word", extra_info=extra)
    assert result["kw_block_fmt"] == result["kw_list_f1"] == 0
    assert result["n_err"] > 0


@pytest.mark.parametrize(
    "reason,status", [("stop", "valid_empty_normalized"), ("length", "truncated"), (None, "incomplete")]
)
def test_empty_tail_requires_termination_evidence(reason, status):
    result = parse_keyword_output("body\n<KEYWORDS>", reason)
    assert result.keyword_block_status == status


def test_missing_and_empty_targets_are_distinct():
    extra = {"output_protocol": PROTOCOL, "keywords": [], "keyword_annotation_valid": True}
    assert eval_score("the man", "the man", extra_info=extra)["kw_list_f1"] == 0
    body = get_task_output(task="lang_asr_verb", text="the man", version=2609, output_keywords=True, keywords=[])
    assert eval_score(body, "the man", extra_info=extra)["kw_list_f1"] == 1
    with pytest.raises(ValueError):
        get_task_output(text="the man", version=2609, output_keywords=True, keywords=None)


def test_truncated_nonempty_block_is_not_success():
    extra = {
        "output_protocol": PROTOCOL,
        "keywords": ["rare"],
        "keyword_annotation_valid": True,
        "generation_finish_reason": "length",
    }
    body = get_task_output(task="lang_asr_verb", text="rare", version=2609, output_keywords=True, keywords=["rare"])
    assert eval_score(body, "rare", extra_info=extra)["kw_list_f1"] == 0


def test_target_order_deterministic_and_old_task_unchanged():
    assert target_keywords("Rare the word rare intermingled", {"the", "word"}) == ["rare", "intermingled"]
    old = get_task_prompt(task="lang_asr_verb", lang="English", version=2609)
    assert "<KEYWORDS>" not in old
    assert get_task_prompt(task="lang_asr_verb", lang="English", version=2609, output_keywords=True).startswith(old)
    with pytest.raises(ValueError):
        get_task_output(task="lang_asr_verb", text="rare", version=2607, output_keywords=True, keywords=["rare"])


def test_bwer_uwer_partition_counts_and_token_boundaries():
    # bar is not a keyword occurrence inside barn; each reference occurrence counts.
    result = word_error_counts("bar barn bar", "bar wrong", ["bar"])
    assert result == {"n_err": 2, "n_ref": 3, "nb_err": 1, "nb_ref": 2, "nu_err": 1, "nu_ref": 1}
    inserted = word_error_counts("bar the", "bar bar the extra", ["bar"])
    assert inserted["n_err"] == 2 and inserted["nb_err"] == 1 and inserted["nu_err"] == 1
    assert inserted["n_err"] == inserted["nb_err"] + inserted["nu_err"]
    assert inserted["n_ref"] == inserted["nb_ref"] + inserted["nu_ref"]


def test_wrong_list_is_not_filtered_against_reference():
    body = get_task_output(task="lang_asr_verb", text="rare", version=2609) + "\n<KEYWORDS>\nrare, hallucination\n"
    extra = {"output_protocol": PROTOCOL, "keywords": ["rare"], "keyword_annotation_valid": True}
    result = eval_score(body, "rare", extra_info=extra)
    assert result["kw_list_precision"] == 0.5 and result["kw_list_recall"] == 1
    assert result["n_err"] == 0


@pytest.mark.parametrize(
    "config,joint,ngt",
    [
        ("baseline_eval/eval_base_ls", False, False),
        ("baseline_eval/eval_joint_prompt_ls", True, False),
        ("training/remax_ls_joint_pilot", True, True),
        ("evaluation/eval_joint_checkpoint_ls", True, False),
    ],
)
def test_keyword_config_composes_without_official_train_leak(config, joint, ngt):
    root = Path(__file__).resolve().parents[4] / "recipe/phimm/config"
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(config_dir=str(root), version_base=None):
        cfg = compose(
            config_name="2609_kw/" + config,
            overrides=(
                ["actor_rollout_ref.model.path=/tmp/exported_trained_model"] if config.startswith("evaluation/") else []
            ),
        )
        OmegaConf.resolve(cfg)
    assert cfg.keyword_experiment.output_keywords is joint
    assert cfg.actor_rollout_ref.rollout.gt_rollout is ngt
    assert cfg.val_reward.reward_manager == "asr_keywords"
    assert cfg.actor_rollout_ref.rollout.val_kwargs.response_length == cfg.data.max_response_length
    assert cfg.trainer.save_freq == 10
    if ngt:
        assert "pilot" in cfg.data.train_data.jsonl_paths
        assert all("test_" not in ds.jsonl_paths for ds in cfg.data.val_data)
    else:
        assert cfg.trainer.val_only


@pytest.mark.parametrize("gt", ["rare", "<src=English><tgt=English>\n<VERBATIM>\nwrong\n<KEYWORDS>\nrare\n"])
def test_corrupted_joint_gt_cannot_fall_back_or_silently_train(gt):
    ds = Dataset.from_list(
        [
            {
                "id": "sample",
                "text": "rare",
                "keywords": ["rare"],
                "prompt": "transcribe",
                "keyword_annotation_valid": True,
                "output_protocol": PROTOCOL,
                "gt_output": gt,
            }
        ]
    )
    with pytest.raises(ValueError, match="round-trip"):
        verl_format_ds(ds)
