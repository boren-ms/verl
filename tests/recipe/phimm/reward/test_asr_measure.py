import pytest

from recipe.phimm.reward.asr_measure import (
    _parse_response,
    check_fmt,
    check_lang,
    compute_score,
    compute_kw_acc,
    compute_think_keyword_f2,
    lang_score,
)
from recipe.phimm.reward.asr_response import (
    get_asr_text,
    get_hyp_text,
    parse_task_output,
    parse_think_output,
)


def test_accepts_code_switch_output():
    output = (
        "<src=Chinese><tgt=Chinese>\n<TXT>祖父叶与良。</TXT>\n"
        "<src=Italian><tgt=Italian>\n<TXT>E, inoltre, attore.</TXT>"
    )
    task_output = parse_task_output(output, version=2609)

    assert task_output == [
        {"src": "Chinese", "tgt": "Chinese", "text": "祖父叶与良。"},
        {"src": "Italian", "tgt": "Italian", "text": "E, inoltre, attore."},
    ]
    assert check_fmt(task_output)
    assert check_lang(task_output, "Chinese Italian") == 1.0


def test_accepts_code_switch_output_without_first_header():
    output = (
        "<TXT>祖父叶与良。</TXT>\n"
        "<src=Italian><tgt=Italian>\n<TXT>E, inoltre, attore.</TXT>"
    )

    task_output = parse_task_output(output, version=2609)

    assert task_output == [
        {"src": None, "tgt": None, "text": "祖父叶与良。"},
        {"src": "Italian", "tgt": "Italian", "text": "E, inoltre, attore."},
    ]
    assert check_fmt(task_output)


def test_format_and_language_ignore_source_language():
    output = "<src=English><tgt=French>\n<TXT>Bonjour</TXT>"
    task_output = parse_task_output(output, version=2609)

    assert check_fmt(task_output)
    assert check_lang(task_output, "French") == 1.0


@pytest.mark.parametrize("mode_tag", ["LEXICAL", "verbatim", "ASR_READABLE"])
def test_parse_task_output_removes_asr_mode_tags(mode_tag):
    output = (
        f"<src=English><tgt=English> \n<{mode_tag}>\n"
        "<TXT>OK OK i think if you have</TXT>"
    )

    task_output = parse_task_output(output, version=2609)

    assert task_output == [
        {"src": "English", "tgt": "English", "text": "OK OK i think if you have"}
    ]
    assert check_fmt(task_output)


def test_parse_task_output_accepts_text_without_language_header():
    task_output = parse_task_output("<VERBATIM>\n<TXT>She's pregnant.</TXT>", version=2609)

    assert task_output == [{"src": None, "tgt": None, "text": "She's pregnant."}]
    assert check_fmt(task_output)
    assert check_lang(task_output, "English") == 1.0


def test_parse_response_uses_text_without_language_header():
    result = _parse_response(
        "<VERBATIM>\n<TXT>She's pregnant.</TXT>",
        ground_truth="She's pregnant.",
        language="English",
        version=2609,
    )

    assert result["word"] == 1.0
    assert result["fmt"] == 1.0
    assert result["lang"] == 1.0


@pytest.mark.parametrize(
    "output",
    [
        "plain text",
        "<VERBATIM>\nplain text",
        "<TXT>missing closing tag",
        "<src=English><tgt=English>\nplain text",
        "<src=English><tgt=English>\n<VERBATIM>\nplain text",
    ],
)
def test_parse_task_output_marks_2609_text_without_complete_txt_wrapper_invalid(output):
    result = parse_task_output(output, version=2609)

    assert isinstance(result, list)
    assert all(set(segment) == {"src", "tgt", "text"} for segment in result)
    assert any(segment["text"] is None for segment in result)
    assert not check_fmt(result)


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
            [{"src": "English", "tgt": "English", "text": None}],
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
        "<src=English><tgt=English>\n<TXT>hello</TXT>\n"
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
            [{"src": None, "tgt": None, "text": "<nonspeech>"}],
        ),
        (
            "<src=English><tgt=English>\n<TXT><nonspeech></TXT>",
            [{"src": "English", "tgt": "English", "text": "<nonspeech>"}],
        ),
    ],
)
def test_parse_task_output_accepts_wrapped_2609_nonspeech(output, expected):
    assert parse_task_output(output, version=2609) == expected


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
    assert check_fmt(task_output)
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
    assert check_fmt(task_output)
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


def test_think_prefix_is_separated_from_2607_asr_output():
    output = (
        "<think>rare phrase,surname</think>\n"
        "Audio Language: English.\n"
        "<ASR><lang=English><TXT>hello world</TXT></ASR>"
    )

    assert parse_think_output(output) == {
        "valid": True,
        "keywords": ["rare phrase", "surname"],
        "response": (
            "Audio Language: English.\n"
            "<ASR><lang=English><TXT>hello world</TXT></ASR>"
        ),
    }
    assert get_hyp_text(output, version=2607) == "hello world"
    assert parse_task_output(output, version=2607) == [
        {"src": "English", "tgt": "English", "text": "hello world"}
    ]


@pytest.mark.parametrize(
    ("output", "keywords", "expected_f2", "expected_format"),
    [
        ("<think>Alpha,Beta</think>\ntranscription", ["alpha", "beta"], 1.0, True),
        ("<think>alpha</think>\ntranscription", ["alpha", "beta"], 5 / 9, True),
        (
            "<think>alpha,beta,false alarm</think>\ntranscription",
            ["alpha", "beta"],
            10 / 11,
            True,
        ),
        ("<think></think>\ntranscription", [], 1.0, True),
        ("transcription only", ["alpha"], 0.0, False),
        ("<think>alpha</think><think>beta</think>\ntranscription", ["alpha"], 0.0, False),
    ],
)
def test_compute_think_keyword_f2(output, keywords, expected_f2, expected_format):
    result = compute_think_keyword_f2(output, keywords=keywords, text_norm="hf_english")

    assert result["f2"] == pytest.approx(expected_f2)
    assert result["format"] is expected_format


def test_parse_response_reports_think_keyword_metrics():
    result = _parse_response(
        (
            "<think>alpha,beta,false alarm</think>\n"
            "Audio Language: English.\n"
            "<ASR><lang=English><TXT>alpha beta</TXT></ASR>"
        ),
        ground_truth="alpha beta",
        extra_info={"keywords": ["alpha", "beta"], "language": "English"},
        version=2607,
        text_norm="hf_english",
        think="keyword",
    )

    assert result["think_keyword"] == pytest.approx(10 / 11)
    assert result["think_precision"] == pytest.approx(2 / 3)
    assert result["think_recall"] == 1.0
    assert result["think_fmt"] == 1.0
    assert result["word"] == 1.0


def test_compute_score_gates_missing_think_format():
    result = compute_score(
        "Audio Language: English.\n<ASR><lang=English><TXT>alpha</TXT></ASR>",
        ground_truth="alpha",
        extra_info={"keywords": ["alpha"], "language": "English"},
        version=2607,
        think="keyword",
        reduce="mean",
        measures={
            "word": {"beta": 1.0},
            "think_keyword": {"beta": 1.0},
            "think_fmt": {"beta": 0.1, "cut": 0.5},
        },
    )

    assert result["word"] == 1.0
    assert result["think_keyword"] == 0.0
    assert result["think_fmt"] == 0.0
    assert result["score"] == 0.0


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