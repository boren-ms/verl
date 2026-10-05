import pytest

from recipe.phimm.reward.asr_measure import (
    _parse_response,
    check_fmt,
    check_lang,
    compute_score,
    compute_kw_acc,
    lang_score,
    reduce_scores,
)
from recipe.phimm.reward.asr_response import get_asr_text, get_hyp_text, parse_task_output


def test_accepts_code_switch_output():
    output = (
        "<src=Chinese><tgt=Chinese>\n<TXT>祖父叶与良。</TXT>\n"
        "<src=Italian><tgt=Italian>\n<TXT>E, inoltre, attore.</TXT>"
    )
    task_output = parse_task_output(output, version=2609)

    assert task_output == [
        {"src": "Chinese", "tgt": "Chinese", "text": "<TXT>祖父叶与良。</TXT>"},
        {"src": "Italian", "tgt": "Italian", "text": "<TXT>E, inoltre, attore.</TXT>"},
    ]
    assert check_fmt(task_output, version=2609)
    assert check_lang(task_output, "Chinese Italian") == 1.0


def test_code_switch_language_reward_requires_exact_match():
    task_output = parse_task_output(
        "<src=Chinese><tgt=Chinese>\n<TXT>祖父叶与良。</TXT>",
        version=2609,
    )

    assert check_lang(task_output, "Chinese Italian") == 0.0


def test_language_reward_uses_header_with_raw_text():
    task_output = parse_task_output(
        "<src=English><tgt=English>\nplain text",
        version=2609,
    )

    assert not check_fmt(task_output, version=2609)
    assert check_lang(task_output, "English") == 1.0


def test_code_switch_language_reward_uses_headers_with_later_raw_text():
    task_output = parse_task_output(
        "<src=English><tgt=English>\n<TXT>hello</TXT>\n"
        "<src=Chinese><tgt=Chinese>\nmissing wrapper",
        version=2609,
    )

    assert not check_fmt(task_output, version=2609)
    assert check_lang(task_output, "English Chinese") == 1.0


def test_accepts_code_switch_output_without_first_header():
    output = (
        "<TXT>祖父叶与良。</TXT>\n"
        "<src=Italian><tgt=Italian>\n<TXT>E, inoltre, attore.</TXT>"
    )

    task_output = parse_task_output(output, version=2609)

    assert task_output == [
        {"src": None, "tgt": None, "text": "<TXT>祖父叶与良。</TXT>"},
        {"src": "Italian", "tgt": "Italian", "text": "<TXT>E, inoltre, attore.</TXT>"},
    ]
    assert check_fmt(task_output, version=2609)


def test_format_and_language_ignore_source_language():
    output = "<src=English><tgt=French>\n<TXT>Bonjour</TXT>"
    task_output = parse_task_output(output, version=2609)

    assert check_fmt(task_output, version=2609)
    assert check_lang(task_output, "French") == 1.0


@pytest.mark.parametrize("mode_tag", ["LEXICAL", "verbatim", "ASR_READABLE"])
def test_parse_task_output_preserves_segment_mode_tags(mode_tag):
    output = (
        f"<src=English><tgt=English> \n<{mode_tag}>\n"
        "<TXT>OK OK i think if you have</TXT>"
    )

    task_output = parse_task_output(output, version=2609)

    assert task_output == [
        {
            "src": "English",
            "tgt": "English",
            "text": f"<{mode_tag}>\n<TXT>OK OK i think if you have</TXT>",
        }
    ]
    assert check_fmt(task_output, version=2609)


def test_parse_task_output_accepts_text_without_language_header():
    task_output = parse_task_output("<VERBATIM>\n<TXT>She's pregnant.</TXT>", version=2609)

    assert task_output == [
        {
            "src": None,
            "tgt": None,
            "text": "<VERBATIM>\n<TXT>She's pregnant.</TXT>",
        }
    ]
    assert check_fmt(task_output, version=2609)
    assert check_lang(task_output, "English") == 0.0


def test_parse_response_uses_text_without_language_header():
    result = _parse_response(
        "<VERBATIM>\n<TXT>She's pregnant.</TXT>",
        ground_truth="She's pregnant.",
        language="English",
        version=2609,
    )

    assert result["word"] == 1.0
    assert result["fmt"] == 1.0
    assert result["lang"] == 0.0


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("plain text", [{"src": None, "tgt": None, "text": "plain text"}]),
        (
            "<VERBATIM>\nplain text",
            [{"src": None, "tgt": None, "text": "<VERBATIM>\nplain text"}],
        ),
        (
            "<TXT>missing closing tag",
            [{"src": None, "tgt": None, "text": "<TXT>missing closing tag"}],
        ),
        (
            "<src=English><tgt=English>\nplain text",
            [{"src": "English", "tgt": "English", "text": "plain text"}],
        ),
        (
            "<src=English><tgt=English>\n<VERBATIM>\nplain text",
            [{"src": "English", "tgt": "English", "text": "<VERBATIM>\nplain text"}],
        ),
    ],
)
def test_parse_task_output_preserves_raw_2609_text(output, expected):
    result = parse_task_output(output, version=2609)

    assert result == expected
    assert not check_fmt(result, version=2609)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<TXT>speech</TXT>", True),
        ("<nonspeech>", True),
        ("<VERBATIM>\n<TXT>speech</TXT>", True),
        ("speech", False),
        ("<TXT>missing closing tag", False),
        ("<nonspeech>speech", False),
        (None, False),
    ],
)
def test_check_fmt_requires_2609_text_wrapper(text, expected):
    task_output = [{"src": None, "tgt": None, "text": text}]

    assert check_fmt(task_output, version=2609) is expected


def test_check_fmt_keeps_legacy_text_behavior():
    task_output = [{"src": "English", "tgt": "English", "text": "speech"}]

    assert check_fmt(task_output, version=2607)


@pytest.mark.parametrize(
    ("output", "version", "expected"),
    [
        (
            None,
            2609,
            [{"src": None, "tgt": None, "text": None}],
        ),
        (
            "<src=English><tgt=English>\nplain text",
            2609,
            [{"src": "English", "tgt": "English", "text": "plain text"}],
        ),
        (
            "<ASR><lang=English><TXT>missing close",
            2607,
            [{"src": None, "tgt": None, "text": None}],
        ),
    ],
)
def test_parse_task_output_always_returns_segment_dicts(output, version, expected):
    assert parse_task_output(output, version=version) == expected


def test_parse_response_uses_structured_task_output():
    result = _parse_response(
        "<src=English><tgt=English>\n<TXT>hello world</TXT>",
        ground_truth="hello world",
        language="English",
        version=2609,
    )

    assert result["word"] == 1.0
    assert result["lang"] == 1.0
    assert result["fmt"] == 1.0


def test_get_asr_text_uses_task_output():
    task_output = parse_task_output(
        "<src=English><tgt=English>\n<VERBATIM>\n<TXT>hello</TXT>\n"
        "<src=Chinese><tgt=Chinese>\n<TXT>你好</TXT>",
        version=2609,
    )

    assert get_asr_text(task_output) == "hello 你好"


def test_get_hyp_text_removes_model_markup():
    output = "<ASR><lang=English><TXT><NONSPEECH>Hello<sep> world</TXT></lang></ASR>"

    assert get_hyp_text(output, version=2609) == "Hello world"


def test_parse_task_output_accepts_exact_2609_nonspeech():
    assert parse_task_output("<nonspeech>", version=2609) == [
        {"src": None, "tgt": None, "text": "<nonspeech>"}
    ]


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (
            "<TXT><nonspeech></TXT>",
            [{"src": None, "tgt": None, "text": "<TXT><nonspeech></TXT>"}],
        ),
        (
            "<src=English><tgt=English>\n<TXT><nonspeech></TXT>",
            [{"src": "English", "tgt": "English", "text": "<TXT><nonspeech></TXT>"}],
        ),
    ],
)
def test_parse_task_output_accepts_wrapped_2609_nonspeech(output, expected):
    assert parse_task_output(output, version=2609) == expected


@pytest.mark.parametrize(
    ("output", "version", "expected"),
    [
        ("<nonspeech>", 2609, 0.0),
        ("<TXT><nonspeech></TXT>", 2609, 0.0),
        ("<src=English><tgt=English>\n<TXT><nonspeech></TXT>", 2609, 1.0),
        ("<src=English><tgt=French>\n<TXT><nonspeech></TXT>", 2609, 0.0),
        ("<ASR><lang=English><TXT><nonspeech></TXT></ASR>", 2607, 1.0),
        ("<ASR><lang=French><TXT><nonspeech></TXT></ASR>", 2607, 0.0),
    ],
)
def test_nonspeech_language_score_depends_only_on_language_tags(output, version, expected):
    task_output = parse_task_output(output, version=version)

    assert check_fmt(task_output, version=version)
    assert check_lang(task_output, "English") == expected
    assert lang_score(output, language="English", version=version) == {
        "score": expected,
        "p_lang": expected,
    }

    result = _parse_response(output, ground_truth="", language="English", version=version)

    assert result["lang"] == expected
    assert result["fmt"] == 1.0
    assert result["word"] == 1.0


@pytest.mark.parametrize("mode_tag", ["verbatim", "READABLE"])
def test_get_hyp_text_removes_asr_mode_markup(mode_tag):
    output = f"<{mode_tag}>Hello world</{mode_tag}>"

    assert get_hyp_text(output, version=2607) == "Hello world"


@pytest.mark.parametrize("punctuation", [".", "。"])
def test_get_hyp_text_removes_empty_punctuation(punctuation):
    assert get_hyp_text(f"<nonspeech>{punctuation}", version=2609) == ""


@pytest.mark.parametrize("tag", ["ASR", "ASR_LEXICAL", "ASR_VERBATIM", "ASR_READABLE"])
def test_parse_task_output_defaults_to_2607_response_format(tag):
    output = f"Audio Language: English.\n<{tag}><lang=English><TXT>hello world</TXT></{tag}>"

    task_output = parse_task_output(output)

    assert task_output == [{"src": "English", "tgt": "English", "text": "hello world"}]
    assert check_fmt(task_output, version=2607)
    assert check_lang(task_output, "English") == 1.0


def test_parse_task_output_falls_back_to_2607_for_unknown_version():
    output = "Audio Language: English.\n<ASR><lang=English><TXT>hello</TXT></ASR>"

    assert parse_task_output(output, version="unknown") == [
        {"src": "English", "tgt": "English", "text": "hello"}
    ]


def test_parse_task_output_accepts_2607_code_switch_response():
    output = (
        "Audio Language: English and Chinese.\n"
        "<ASR><lang=English><TXT>hello</TXT><lang=Chinese><TXT>你好</TXT></ASR>"
    )

    task_output = parse_task_output(output, version=2607)

    assert task_output == [
        {"src": "English", "tgt": "English", "text": "hello"},
        {"src": "Chinese", "tgt": "Chinese", "text": "你好"},
    ]
    assert check_fmt(task_output, version=2607)
    assert check_lang(task_output, "English Chinese") == 1.0


def test_parse_response_uses_2607_response_text():
    result = _parse_response(
        "Audio Language: English.\n<ASR><lang=English><TXT>hello world</TXT></ASR>",
        ground_truth="hello world",
        language="English",
        version=2607,
    )

    assert result["word"] == 1.0
    assert result["lang"] == 1.0
    assert result["fmt"] == 1.0


def test_parse_response_accepts_2607_response_without_audio_language():
    result = _parse_response(
        "<ASR><lang=English><TXT>hello world</TXT></ASR>",
        ground_truth="hello world",
        language="English",
        version=2607,
    )

    assert result["word"] == 1.0
    assert result["lang"] == 1.0
    assert result["fmt"] == 1.0

    task_output = parse_task_output(
        "<ASR><lang=English><TXT>hello world</TXT></ASR>",
        version=2607,
    )
    assert task_output == [{"src": None, "tgt": "English", "text": "hello world"}]


@pytest.mark.parametrize(
    ("scores", "kwargs", "expected"),
    [
        ([1.0, 0.5], {"total": 4}, 0.375),
        ([1.0, 0.5], {}, 0.75),
        ([1.0, 0.5], {"total": 1.5}, 1.0),
        ([-1.0, 0.5], {"total": 4}, -0.125),
        ([], {"total": 4}, 0.0),
        ([], {}, 0.0),
    ],
)
def test_average_reduction(scores, kwargs, expected):
    assert reduce_scores(scores, mode="average", **kwargs) == pytest.approx(expected)


@pytest.mark.parametrize("total", [0, -1, float("nan"), float("inf"), float("-inf"), True, "4", None])
@pytest.mark.parametrize("scores", [[], [1.0]])
def test_average_reduction_rejects_invalid_total(scores, total):
    with pytest.raises(ValueError, match="positive, finite"):
        reduce_scores(scores, mode="average", total=total)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("sum", 1.5), ("mean", 0.75), ("multiply", 0.5), ("geometric", 0.5**0.5), ("harmonic", 2 / 3)],
)
def test_existing_reductions_ignore_additional_kwargs(mode, expected):
    assert reduce_scores([1.0, 0.5], mode=mode, total=4) == pytest.approx(expected)
    assert reduce_scores([], mode=mode, total=4) == 0.0


@pytest.mark.parametrize("reduction", ["average", {"mode": "average", "total": 4}])
def test_compute_score_forwards_average_total(reduction):
    result = compute_score(
        "<ASR><lang=English><TXT>hello world</TXT></ASR>",
        ground_truth="hello world",
        language="English",
        version=2607,
        reduce=reduction,
        total=4,
        gamma=2,
        measures={"word": {"beta": 1.0}, "lang": {"beta": 0.5}, "fmt": {"beta": 0.2}},
    )

    assert result["score"] == pytest.approx(((1.0 + 0.5 + 0.2) / 4) ** 2)


@pytest.mark.parametrize("reduction", ["average", {"mode": "average", "total": 4}])
def test_compute_score_average_preserves_cut(reduction):
    result = compute_score(
        "<ASR><lang=English><TXT>hello world",
        ground_truth="hello world",
        language="English",
        version=2607,
        reduce=reduction,
        total=4,
        measures={"word": {"beta": 1.0}, "fmt": {"beta": 0.5, "cut": 0.0}},
    )

    assert result["fmt"] == 0.0
    assert result["score"] == 0.0


@pytest.mark.parametrize(
    ("reduction", "expected"),
    [
        ({"mode": "average", "total": 4}, 0.375),
        ({"mode": "average"}, 0.75),
        ({"mode": "mean"}, 0.75),
        ({"mode": "sum"}, 1.5),
        ({}, 1.5),
    ],
)
def test_compute_score_mapping_reduction(reduction, expected):
    result = compute_score(
        "<ASR><lang=English><TXT>hello world</TXT></ASR>",
        ground_truth="hello world",
        language="English",
        version=2607,
        reduce=reduction,
        measures={"word": {"beta": 1.0}, "lang": {"beta": 0.5}},
    )

    assert result["score"] == pytest.approx(expected)


def test_compute_score_mapping_total_overrides_reward_kwargs():
    result = compute_score(
        "<ASR><lang=English><TXT>hello world</TXT></ASR>",
        ground_truth="hello world",
        language="English",
        version=2607,
        reduce={"mode": "average", "total": 4},
        total=2,
        measures={"word": {"beta": 1.0}},
    )

    assert result["score"] == pytest.approx(0.25)


def test_compute_score_cut_zeros_reward_at_threshold():
    result = compute_score(
        "<ASR><lang=English><TXT>hello world",
        ground_truth="hello world",
        language="English",
        version=2607,
        reduce="mean",
        measures={
            "char": {"beta": 0.5},
            "word": {"beta": 0.5},
            "lang": {"beta": 1.0, "cut": 0.0},
            "fmt": {"beta": 1.0, "cut": 0.0},
        },
    )

    assert result["fmt"] == 0.0
    assert result["lang"] == 0.0
    assert result["score"] == 0.0


def test_compute_score_cut_preserves_reward_above_threshold():
    result = compute_score(
        "<ASR><lang=English><TXT>hello world</TXT></ASR>",
        ground_truth="hello world",
        language="English",
        version=2607,
        reduce="mean",
        measures={
            "word": {"beta": 1.0},
            "lang": {"beta": 1.0, "cut": 0.5},
            "fmt": {"beta": 1.0, "cut": 0.5},
        },
    )

    assert result["score"] == 1.0


def test_compute_score_does_not_gate_measure_without_cut():
    result = compute_score(
        "<ASR><lang=English><TXT>hello world",
        ground_truth="hello world",
        language="English",
        version=2607,
        reduce="mean",
        measures={
            "word": {"beta": 1.0},
            "fmt": {"beta": 1.0},
        },
    )

    assert result["fmt"] == 0.0
    assert result["score"] > 0.0


@pytest.mark.parametrize(
    ("reference", "hypothesis", "keywords", "expected"),
    [
        ("the quick brown fox", "the quick brown fox", ["brown"], 1.0),
        ("the quick brown fox", "the quick blue fox", ["brown"], 0.0),
        ("the quick brown fox", "the quick fox", ["brown"], 0.0),
        ("the quick brown fox", "the quick brown brown fox", ["brown"], 0.0),
        ("new york state", "new york state", ["new york"], 1.0),
        ("new york state", "new state", ["new york"], 0.5),
        ("the quick brown fox", "the quick blue fox", None, 1.0),
    ],
)
def test_compute_kw_acc(reference, hypothesis, keywords, expected):
    result = compute_kw_acc(reference, hypothesis, keywords)

    assert result["accuracy"] == expected


def test_compute_kw_acc_reports_error_and_reference_counts():
    result = compute_kw_acc(
        "new york state",
        "new state",
        ["new york"],
    )

    assert result == {"accuracy": 0.5, "n_err": 1, "n_ref": 2}


def test_parse_response_reports_keyword_accuracy():
    result = _parse_response(
        "the quick blue fox",
        ground_truth="the quick brown fox",
        extra_info={"keywords": ["brown"]},
    )

    assert result["keyword"] == 0.0


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("<src=English><tgt=English>\n<TXT>Hello</TXT>", 1.0),
        ("<src=French><tgt=French>\n<TXT>Bonjour</TXT>", 0.0),
        ("<src=English><tgt=English>Hello", 0.0),
    ],
)
def test_lang_score_reports_only_p_lang(output, expected):
    assert lang_score(output, language="English", version=2609) == {
        "score": expected,
        "p_lang": expected,
    }