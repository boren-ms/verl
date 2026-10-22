from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from recipe.phimm.reward.asr_response import get_asr_text, get_hyp_text, parse_task_output
from recipe.phimm.reward.asr_vote import compute_score
from recipe.phimm.utils.tn import text_norm


HALLUCINATION = "this is a hallucination"
MALFORMED_SUFFIXES = [
    pytest.param(f"<src=English>\n{HALLUCINATION}", id="missing-target"),
    pytest.param(f"<src=English><tgt=English>{HALLUCINATION}", id="missing-newline"),
    pytest.param(f"<src=\n{HALLUCINATION}", id="unclosed-source"),
    pytest.param(
        f"<src=\n{HALLUCINATION}\n<src=English><tgt=English>\nmore words",
        id="unclosed-source-before-valid-header",
    ),
]


@pytest.mark.parametrize("version", [2609, "2609"])
@pytest.mark.parametrize("prefix", ["<verbatim>hello world</verbatim>", "", "<TXT></TXT>", "<nonspeech>"])
@pytest.mark.parametrize("suffix", MALFORMED_SUFFIXES)
def test_hyp_text_preserves_malformed_suffix_without_accepting_its_header(version, prefix, suffix):
    response = f"<src=English><tgt=English>\n{prefix}\n{suffix}"
    parsed = parse_task_output(response, version=version)

    assert parsed == [
        {"src": "English", "tgt": "English", "text": prefix},
        {"src": None, "tgt": None, "text": suffix},
    ]
    assert get_asr_text(parsed) == f"{prefix} {suffix}"
    hyp = get_hyp_text(response, version=version)
    assert HALLUCINATION in hyp
    if "hello world" in prefix:
        assert hyp.startswith("hello world ")
    if "more words" in suffix:
        assert "more words" in hyp


def test_hyp_text_preserves_words_attached_to_a_malformed_header_after_empty_text():
    response = "<src=Chinese><tgt=Chinese>\n<TXT></TXT>\n<src=Chinese><tgt=Chinese>凭空添加的文本"

    assert "凭空添加的文本" in get_hyp_text(response, version=2609)


@pytest.fixture(scope="module")
def l2_reward_kwargs():
    root = Path(__file__).resolve().parents[4]
    config_root = root / "recipe/phimm/config"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2611_dev/dev"), version_base=None):
        config = compose(config_name="remax_dev_base0_v1h2a_fix_L2", overrides=[searchpath])
    return OmegaConf.to_container(config.reward_functions.librispeech.reward_kwargs, resolve=True)


@pytest.mark.parametrize("suffix", MALFORMED_SUFFIXES)
def test_l2_reward_penalizes_hallucinations_after_malformed_headers(l2_reward_kwargs, suffix):
    reference = "hello world"
    correct = "<src=English><tgt=English>\n<verbatim>hello world</verbatim>"
    baseline = compute_score(correct, reference, **l2_reward_kwargs)
    extra_info = {"language": "English", "greedy_reward": baseline}
    plain = compute_score(correct + "\n" + HALLUCINATION, reference, extra_info=extra_info, **l2_reward_kwargs)
    malformed = compute_score(correct + "\n" + suffix, reference, extra_info=extra_info, **l2_reward_kwargs)

    normalized = text_norm(get_hyp_text(correct + "\n" + suffix, version=2609), name="openasr_en")
    assert HALLUCINATION in normalized
    if "more words" in suffix:
        assert "more words" in normalized
    assert malformed["word"] <= plain["word"]
    assert malformed["char"] <= plain["char"]
    assert malformed["d_word"] < 0
    assert malformed["d_char"] < 0
    assert malformed["score"] <= plain["score"] < 0
    assert malformed["lang"] == 0.0
    assert malformed["fmt"] == 0.5


def test_hyp_text_preserves_valid_multilingual_segments():
    response = "<src=English><tgt=English>\n<verbatim>hello</verbatim>\n<src=Chinese><tgt=Chinese>\n<TXT>你好</TXT>"

    assert get_hyp_text(response, version=2609) == "hello 你好"


def test_hyp_text_keeps_valid_multilingual_prefix_clean_when_a_later_header_fails():
    response = (
        "<src=English><tgt=English>\n<verbatim>hello</verbatim>\n"
        "<src=Chinese><tgt=Chinese>\n<TXT>你好</TXT>\n"
        f"<src=\n{HALLUCINATION}"
    )

    assert get_hyp_text(response, version=2609).startswith("hello 你好 ")
    assert HALLUCINATION in get_hyp_text(response, version=2609)


def test_hyp_text_keeps_fallback_for_an_entirely_invalid_response():
    response = "<VERBATIM>\n<TXT>hello world</TXT>"

    assert parse_task_output(response, version=2609) == [{"src": None, "tgt": None, "text": response}]
    assert get_hyp_text(response, version=2609) == "hello world"
