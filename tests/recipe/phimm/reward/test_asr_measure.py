import pytest

from recipe.phimm.reward.asr_measure import (
    _parse_response,
    check_fmt,
    check_lang,
    compute_bracket_del,
    compute_deltas,
    compute_openml_acc,
    compute_score,
    compute_kw_score,
    lang_score,
    reduce_scores,
)
from recipe.phimm.reward.asr_response import get_asr_text, get_hyp_text, parse_task_output


@pytest.mark.parametrize("baseline", [None, {}])
def test_compute_deltas_skips_missing_baseline(baseline):
    assert compute_deltas({"score": 0.5, "word": 1.0}, baseline) == {}


def test_compute_deltas_subtracts_only_shared_scores_without_mutation():
    raw_scores = {"score": 0.5, "word": 0.25, "char": 0.75, "fmt": 1.0}
    baseline = {"score": 0.25, "word": 0.75, "char": 0.75, "other": 1.0}

    assert compute_deltas(raw_scores, baseline) == {
        "d_score": 0.25,
        "d_word": -0.5,
        "d_char": 0.0,
    }
    assert raw_scores == {"score": 0.5, "word": 0.25, "char": 0.75, "fmt": 1.0}
    assert baseline == {"score": 0.25, "word": 0.75, "char": 0.75, "other": 1.0}


@pytest.mark.parametrize("scale", ["constant", "char", "word"])
@pytest.mark.parametrize("cut", [None, 0.5])
def test_compute_score_returns_deltas_from_greedy_reward(scale, cut):
    config = {"beta": 0.5, "gamma": 2.0}
    if cut is not None:
        config["cut"] = cut
    kwargs = {
        "version": 2609,
        "text_norm": "identity",
        "scale": scale,
        "gamma": 2.0,
        "reduce": {"mode": "average", "total": 2},
        "measures": {"word": config},
    }
    baseline = compute_score("<TXT>cat dog</TXT>", ground_truth="cat dog", **kwargs)
    raw = compute_score("<TXT>ct dog</TXT>", ground_truth="cat dog", **kwargs)
    extra_info = {"greedy_reward": baseline}
    result = compute_score("<TXT>ct dog</TXT>", ground_truth="cat dog", extra_info=extra_info, **kwargs)

    assert {key: result[key] for key in raw} == raw
    assert set(result) == set(raw) | {f"d_{key}" for key in raw}
    for key, value in raw.items():
        assert result[f"d_{key}"] == pytest.approx(value - baseline[key])
    assert result["d_word"] == -0.5
    assert extra_info == {"greedy_reward": baseline}


@pytest.mark.parametrize("extra_info", [None, {}, {"greedy_reward": None}, {"greedy_reward": {}}])
def test_compute_score_omits_deltas_without_baseline(extra_info):
    result = compute_score(
        "<TXT>hello</TXT>",
        ground_truth="hello",
        version=2609,
        extra_info=extra_info,
        measures={"word": {"beta": 1.0}},
    )

    assert result["score"] == 1.0
    assert not any(key.startswith("d_") for key in result)


@pytest.mark.parametrize("measures", [None, {}, {"word": {}}, {"word": {"beta": 0.0}, "fmt": {}}])
@pytest.mark.parametrize("scale", ["constant", "char", "word"])
@pytest.mark.parametrize("with_baseline", [False, True])
def test_compute_score_returns_only_configured_raw_metrics(measures, scale, with_baseline):
    extra_info = {}
    if with_baseline:
        extra_info["greedy_reward"] = {"score": 0.25, "word": 0.75, "fmt": 0.5, "char": 0.9}
    result = compute_score(
        "<TXT>ct dog</TXT>",
        ground_truth="cat dog",
        version=2609,
        text_norm="identity",
        measures=measures,
        scale=scale,
        extra_info=extra_info,
    )

    raw_keys = {"score", *(measures or {})}
    expected_keys = raw_keys | ({f"d_{key}" for key in raw_keys} if with_baseline else set())
    assert set(result) == expected_keys
    if measures:
        assert result["word"] == 0.5
        if "fmt" in measures:
            assert result["fmt"] == 0.0
    else:
        assert result["score"] == 0.0
    if with_baseline:
        for key in raw_keys:
            assert result[f"d_{key}"] == pytest.approx(result[key] - extra_info["greedy_reward"][key])


@pytest.mark.parametrize(
    ("hypothesis", "bracket", "bracket_del"),
    [
        ("hello world", 1.0, 1.0),
        ("hello [background noise] world", 0.0, 2 / 4),
        ("hello <background noise> world", 0.0, 2 / 4),
        ("hello (background noise) world", 0.0, 2 / 4),
        ("one [two three] <four> (five six) seven", 0.0, 2 / 7),
        ("[all words removed]", 0.0, 0.0),
        ("<all words removed>", 0.0, 0.0),
        ("(all words removed)", 0.0, 0.0),
        ("hello [] <> world", 0.0, 2 / 4),
        ("hello () world", 0.0, 2 / 3),
        ("hello {background noise} world", 0.0, 2 / 4),
        ("hello [unfinished words", 1.0, 1.0),
        ("hello [background noise> world", 1.0, 1.0),
        ("hello <background noise] world", 1.0, 1.0),
        ("hello <nonspeech> world", 1.0, 1.0),
        ("hello <NONSPEECH> world", 1.0, 1.0),
        ("hello[noise]world", 0.0, 1.0),
        ("hello\t[background\nnoise]\tworld", 0.0, 2 / 4),
        ("", 1.0, 1.0),
        (" \t\n", 1.0, 1.0),
    ],
)
def test_compute_openml_acc_bracket_metrics(hypothesis, bracket, bracket_del):
    assert compute_bracket_del(hypothesis) == pytest.approx(bracket_del)
    result = compute_openml_acc(hypothesis, "hello world", "english")

    assert result["bracket"] == bracket
    assert result["bracket_del"] == pytest.approx(bracket_del)


@pytest.mark.parametrize(
    "pair",
    [
        "<>", "[]", "{}", "()", "（）", "［］", "｛｝", "＜＞",
        "【】", "〔〕", "〈〉", "《》", "「」", "『』", "〖〗", "｟｠",
    ],
)
@pytest.mark.parametrize(
    ("template", "expected"),
    [
        ("hello {left}background noise{right} world", 0.5),
        ("{left}all words removed{right}", 0.0),
        ("hello {left}{right} world", 2 / 3),
        ("hello {left}unfinished words", 1.0),
    ],
)
def test_bracket_del_supports_all_bracket_styles(pair, template, expected):
    hypothesis = template.format(left=pair[0], right=pair[1])
    assert compute_bracket_del(hypothesis) == pytest.approx(expected)
    result = compute_openml_acc(hypothesis, "hello world", "english")

    assert result["bracket_del"] == pytest.approx(expected)
    assert result["bracket"] == (1.0 if expected == 1.0 else 0.0)


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("<TXT>hello [background noise] world</TXT>", 0.5),
        ("<TXT>hello world</TXT>", 1.0),
        ("<nonspeech>", 1.0),
        ("<TXT>[background noise]</TXT>", 0.0),
        ("<TXT>hello 【background noise】 world</TXT>", 0.5),
        ("<TXT>hello {background noise} world</TXT>", 0.5),
    ],
)
def test_parse_response_reports_bracket_del_after_markup_cleanup(output, expected):
    result = _parse_response(output, ground_truth="hello world", version=2609)

    assert result["bracket_del"] == pytest.approx(expected)


@pytest.mark.parametrize("cut", [None, 0.5])
@pytest.mark.parametrize("pair", ["[]", "{}", "（）", "【】"])
def test_compute_score_supports_bracket_del(cut, pair):
    config = {"beta": 0.5}
    if cut is not None:
        config["cut"] = cut
    result = compute_score(
        f"<TXT>hello {pair[0]}background noise{pair[1]} world</TXT>",
        ground_truth="hello world",
        version=2609,
        measures={"bracket_del": config},
    )

    assert result["bracket_del"] == 0.5
    assert result["score"] == (0.25 if cut is None else 0.0)


def test_bracket_del_does_not_change_reward_unless_configured():
    result = compute_score(
        "<TXT>hello [background noise] world</TXT>",
        ground_truth="hello world",
        version=2609,
        measures={"bracket": {"beta": 1.0}},
    )

    assert result["bracket"] == 0.0
    assert "bracket_del" not in result
    assert result["score"] == 0.0


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
    assert check_lang(task_output, "Chinese Italian") == {"acc": 1.0, "n_err": 0, "n_ref": 2}


def test_code_switch_language_reward_gives_partial_credit():
    task_output = parse_task_output(
        "<src=Chinese><tgt=Chinese>\n<TXT>祖父叶与良。</TXT>",
        version=2609,
    )

    assert check_lang(task_output, "Chinese Italian") == {"acc": 0.5, "n_err": 1, "n_ref": 2}


@pytest.mark.parametrize(
    ("reference", "predictions", "expected_acc", "expected_errors", "expected_references"),
    [
        ("English Chinese", ["English", "Chinese"], 1.0, 0, 2),
        ("English Chinese", ["English"], 0.5, 1, 2),
        ("English Chinese", ["English", "French"], 0.5, 1, 2),
        ("English Chinese", ["Chinese", "English"], 0.0, 2, 2),
        ("English Chinese French", ["English", "German", "French"], 2 / 3, 1, 3),
        ("English Chinese", ["English", "Chinese", "French"], 0.5, 1, 2),
        ("English", ["English", "French", "Chinese"], -1.0, 2, 1),
        ("English English", ["English"], 0.5, 1, 2),
        ("English", ["English", "English"], 0.0, 1, 1),
        ("en_zh", ["English", "Mandarin"], 1.0, 0, 2),
        ("English Chinese", ["en_zh"], 1.0, 0, 2),
        ("English Chinese", [None, "English"], 0.5, 1, 2),
        ("English Chinese", [], 0.0, 2, 2),
        ("", [], 1.0, 0, 0),
        (None, [], 1.0, 0, 0),
        ("", ["English"], 0.0, 1, 0),
    ],
)
def test_check_lang_returns_ordered_edit_details(
    reference, predictions, expected_acc, expected_errors, expected_references,
):
    task_output = [{"src": None, "tgt": language, "text": None} for language in predictions]
    result = check_lang(task_output, reference)

    assert set(result) == {"acc", "n_err", "n_ref"}
    assert result["acc"] == pytest.approx(expected_acc)
    assert result["n_err"] == expected_errors
    assert result["n_ref"] == expected_references


@pytest.mark.parametrize(
    ("predictions", "expected_acc", "expected_errors"),
    [
        (["English"], 0.5, 1),
        (["English", "French"], 0.5, 1),
        (["Chinese", "English"], 0.0, 2),
        (["English", "Chinese", "French"], 0.5, 1),
    ],
)
def test_language_rewards_use_ordered_accuracy_and_error_count(predictions, expected_acc, expected_errors):
    solution = "\n".join(f"<src={lang}><tgt={lang}>\n<TXT>word</TXT>" for lang in predictions)
    kwargs = {"version": 2609, "text_norm": "identity", "language": "English Chinese"}
    parsed = _parse_response(solution, ground_truth="one two three four", **kwargs)
    score = compute_score(
        solution, ground_truth="one two three four",
        measures={"lang": {}, "word_lang": {"beta": 1.0}}, **kwargs,
    )

    assert parsed["lang"] == expected_acc
    assert parsed["word_lang"] == 1.0 - expected_errors / 4
    assert score == {
        "score": expected_acc + 1.0 - expected_errors / 4,
        "lang": expected_acc,
        "word_lang": 1.0 - expected_errors / 4,
    }
    assert lang_score(solution, **kwargs) == {"score": expected_acc, "p_lang": expected_acc}


def test_language_reward_uses_header_with_raw_text():
    task_output = parse_task_output(
        "<src=English><tgt=English>\nplain text",
        version=2609,
    )

    assert check_fmt(task_output, version=2609)
    assert check_lang(task_output, "English")["acc"] == 1.0


def test_code_switch_language_reward_uses_headers_with_later_raw_text():
    task_output = parse_task_output(
        "<src=English><tgt=English>\n<TXT>hello</TXT>\n"
        "<src=Chinese><tgt=Chinese>\nmissing wrapper",
        version=2609,
    )

    assert check_fmt(task_output, version=2609)
    assert check_lang(task_output, "English Chinese")["acc"] == 1.0


def test_rejects_code_switch_output_without_first_header():
    output = (
        "<TXT>祖父叶与良。</TXT>\n"
        "<src=Italian><tgt=Italian>\n<TXT>E, inoltre, attore.</TXT>"
    )

    task_output = parse_task_output(output, version=2609)

    assert task_output == [
        {"src": None, "tgt": None, "text": None},
    ]
    assert not check_fmt(task_output, version=2609)
    assert check_lang(task_output, "Italian")["acc"] == 0.0


def test_format_and_language_ignore_source_language():
    output = "<src=English><tgt=French>\n<TXT>Bonjour</TXT>"
    task_output = parse_task_output(output, version=2609)

    assert check_fmt(task_output, version=2609)
    assert check_lang(task_output, "French")["acc"] == 1.0


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


def test_parse_task_output_rejects_text_without_language_header():
    task_output = parse_task_output("<VERBATIM>\n<TXT>She's pregnant.</TXT>", version=2609)

    assert task_output == [
        {
            "src": None,
            "tgt": None,
            "text": None,
        }
    ]
    assert not check_fmt(task_output, version=2609)
    assert check_lang(task_output, "English")["acc"] == 0.0


def test_parse_response_scores_text_but_rejects_format_without_language_header():
    result = _parse_response(
        "<VERBATIM>\n<TXT>She's pregnant.</TXT>",
        ground_truth="She's pregnant.",
        language="English",
        version=2609,
    )

    assert result["word"] == 1.0
    assert result["fmt"] == 0.0
    assert result["lang"] == 0.0


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (
            "<src=English><tgt=English>\n<TXT>missing closing tag",
            [{"src": "English", "tgt": "English", "text": "<TXT>missing closing tag"}],
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
    assert check_fmt(result, version=2609)


@pytest.mark.parametrize(
    "output",
    [
        "",
        " \t\n ",
        "plain text",
        "<TXT>plain text</TXT>",
        "<VERBATIM>\nplain text",
        "<nonspeech>plain text",
        "plain text<nonspeech>",
        "<TXT><nonspeech></TXT>",
        "<src=English>\nplain text",
        "<tgt=English>\nplain text",
        "<src=English><tgt=English>plain text",
        "<src=><tgt=English>\nplain text",
    ],
)
def test_parse_task_output_requires_2609_language_header(output):
    task_output = parse_task_output(output, version=2609)

    assert task_output == [{"src": None, "tgt": None, "text": None}]
    assert not check_fmt(task_output, version=2609)


@pytest.mark.parametrize(
    ("src", "tgt"),
    [(" English ", " English "), (" ", "English"), ("English", " ")],
)
def test_parse_task_output_strips_2609_header_language_values(src, tgt):
    task_output = parse_task_output(f"<src={src}><tgt={tgt}>\nplain text", version=2609)

    assert task_output == [{"src": src.strip(), "tgt": tgt.strip(), "text": "plain text"}]
    assert check_fmt(task_output, version=2609)


def test_parse_task_output_rejects_later_missing_target_header():
    task_output = parse_task_output(
        "<src=English><tgt=English>\nhello\n<src=Chinese>\n你好",
        version=2609,
    )

    assert task_output == [
        {"src": "English", "tgt": "English", "text": "hello"},
        {"src": None, "tgt": None, "text": None},
    ]
    assert not check_fmt(task_output, version=2609)
    assert check_lang(task_output, "English")["acc"] == 1.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<TXT>speech</TXT>", True),
        ("<nonspeech>", True),
        ("<VERBATIM>\n<TXT>speech</TXT>", True),
        ("speech", True),
        ("<VERBATIM>\nspeech", True),
        ("<TXT>missing closing tag", True),
        ("speech</TXT>", True),
        ("<nonspeech>speech", True),
        (None, False),
    ],
)
def test_check_fmt_accepts_optional_2609_text_wrapper(text, expected):
    task_output = [{"src": "English", "tgt": "English", "text": text}]

    assert check_fmt(task_output, version=2609) is expected


@pytest.mark.parametrize("version", [2607, 2609, "2609"])
@pytest.mark.parametrize(
    "task_output",
    [
        [],
        [{"src": None, "tgt": None, "text": None}],
        [
            {"src": "English", "tgt": "English", "text": "speech"},
            {"src": None, "tgt": None, "text": None},
        ],
    ],
)
def test_check_fmt_rejects_missing_parsed_text(task_output, version):
    assert not check_fmt(task_output, version=version)


@pytest.mark.parametrize(
    "output",
    [
        "<src=English><tgt=English>\nhello world",
        "<src=English><tgt=English>\n<TXT>hello world</TXT>",
        "<src=English><tgt=English>\n<VERBATIM>\nhello world",
    ],
)
def test_parse_response_accepts_optional_2609_text_wrapper(output):
    result = _parse_response(
        output,
        ground_truth="hello world",
        language="English",
        version=2609,
        text_norm="identity",
    )

    assert result["fmt"] == 1.0
    assert result["word"] == 1.0


def test_check_fmt_keeps_legacy_text_behavior():
    task_output = [{"src": "English", "tgt": "English", "text": "speech"}]

    assert check_fmt(task_output, version=2607)


@pytest.mark.parametrize(
    ("output", "version", "expected"),
    [
        (
            None,
            2607,
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

    assert get_asr_text(task_output) == "<VERBATIM>\n<TXT>hello</TXT> <TXT>你好</TXT>"


@pytest.mark.parametrize(
    "text",
    [
        "hello world",
        "<TXT>hello world</TXT>",
        "<VERBATIM>\nhello world",
        "<ASR_READABLE>\n<TXT>hello world</TXT>",
        "<TXT>hello world",
        "hello world</TXT>",
    ],
)
def test_get_asr_text_preserves_markup_for_hyp_text_cleanup(text):
    task_output = parse_task_output(f"<src=English><tgt=English>\n{text}", version=2609)

    assert get_asr_text(task_output) == text
    assert get_hyp_text(f"<src=English><tgt=English>\n{text}", version=2609) == "hello world"


@pytest.mark.parametrize(
    ("output", "version"),
    [
        ("<src=English><tgt=English>\n<TXT></TXT>", 2609),
        ("<src=English><tgt=English>\n<VERBATIM>\n<TXT></TXT>", 2609),
        ("<ASR><lang=English><TXT></TXT></ASR>", 2607),
    ],
)
def test_get_hyp_text_preserves_valid_empty_text(output, version):
    assert check_fmt(parse_task_output(output, version=version), version=version)
    assert get_hyp_text(output, version=version) == ""


def test_get_hyp_text_removes_model_markup():
    output = "<ASR><lang=English><TXT><NONSPEECH>Hello<sep> world</TXT></lang></ASR>"

    assert get_hyp_text(output, version=2609) == "Hello world"


@pytest.mark.parametrize("output", ["<nonspeech>", "<NONSPEECH>", " \n<NonSpeech>\t "])
def test_parse_task_output_accepts_standalone_2609_nonspeech(output):
    task_output = parse_task_output(output, version=2609)

    assert task_output == [{"src": None, "tgt": None, "text": "<nonspeech>"}]
    assert check_fmt(task_output, version=2609)
    assert get_hyp_text(output, version=2609) == ""
    result = _parse_response(output, ground_truth="", language="English", version=2609)
    assert result["fmt"] == 1.0
    assert result["word"] == 1.0
    assert result["lang"] == 0.0


def test_parse_task_output_accepts_2609_nonspeech_with_header():
    assert parse_task_output("<src=English><tgt=English>\n<nonspeech>", version=2609) == [
        {"src": "English", "tgt": "English", "text": "<nonspeech>"}
    ]


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (
            "<TXT><nonspeech></TXT>",
            [{"src": None, "tgt": None, "text": None}],
        ),
        (
            "<src=English><tgt=English>\n<TXT><nonspeech></TXT>",
            [{"src": "English", "tgt": "English", "text": "<TXT><nonspeech></TXT>"}],
        ),
    ],
)
def test_parse_task_output_requires_header_for_wrapped_2609_nonspeech(output, expected):
    assert parse_task_output(output, version=2609) == expected


@pytest.mark.parametrize(
    ("output", "version", "expected", "expected_fmt"),
    [
        ("<nonspeech>", 2609, 0.0, 1.0),
        ("<TXT><nonspeech></TXT>", 2609, 0.0, 0.0),
        ("<src=English><tgt=English>\n<nonspeech>", 2609, 1.0, 1.0),
        ("<src=English><tgt=English>\n<TXT><nonspeech></TXT>", 2609, 1.0, 1.0),
        ("<src=English><tgt=French>\n<TXT><nonspeech></TXT>", 2609, 0.0, 1.0),
        ("<ASR><lang=English><TXT><nonspeech></TXT></ASR>", 2607, 1.0, 1.0),
        ("<ASR><lang=French><TXT><nonspeech></TXT></ASR>", 2607, 0.0, 1.0),
    ],
)
def test_nonspeech_language_score_depends_only_on_language_tags(output, version, expected, expected_fmt):
    task_output = parse_task_output(output, version=version)

    assert float(check_fmt(task_output, version=version)) == expected_fmt
    assert check_lang(task_output, "English")["acc"] == expected
    assert lang_score(output, language="English", version=version) == {
        "score": expected,
        "p_lang": expected,
    }

    result = _parse_response(output, ground_truth="", language="English", version=version)

    assert result["lang"] == expected
    assert result["fmt"] == expected_fmt
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
    assert check_lang(task_output, "English")["acc"] == 1.0


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
    assert check_lang(task_output, "English Chinese")["acc"] == 1.0


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


@pytest.mark.parametrize("scale", ["constant", "char", "word"])
@pytest.mark.parametrize("gamma", [1.0, 2.0])
@pytest.mark.parametrize(
    "reduction",
    ["sum", "mean", "multiply", "geometric", "harmonic", {"mode": "average", "total": 4}],
)
def test_compute_score_scales_after_reduction_and_global_gamma(scale, gamma, reduction):
    kwargs = {
        "version": 2609,
        "text_norm": "identity",
        "reduce": reduction,
        "gamma": gamma,
        "measures": {"fmt": {"beta": 0.4}, "char": {"beta": 0.6, "gamma": 2.0}},
    }
    baseline = compute_score("<TXT>ct dog</TXT>", ground_truth="cat dog", **kwargs)
    result = compute_score("<TXT>ct dog</TXT>", ground_truth="cat dog", scale=scale, **kwargs)

    assert result["char"] == pytest.approx(5 / 6)
    assert "word" not in result
    factor = {"constant": 1.0, "char": 5 / 6, "word": 0.5}[scale]
    assert result["score"] == pytest.approx(baseline["score"] * factor)
    assert {key: value for key, value in result.items() if key != "score"} == {
        key: value for key, value in baseline.items() if key != "score"
    }


@pytest.mark.parametrize("scale", ["char", "word"])
@pytest.mark.parametrize(
    ("reference", "hypothesis", "accuracy"),
    [("a", "a b c d", -2.0), ("a", "b", 0.0), ("", "", 1.0), ("", "a b", -1.0)],
)
def test_compute_score_scale_uses_raw_accuracy_without_measure(scale, reference, hypothesis, accuracy):
    result = compute_score(
        f"<src=English><tgt=English>\n<TXT>{hypothesis}</TXT>",
        ground_truth=reference,
        version=2609,
        text_norm="identity",
        scale=scale,
        measures={"fmt": {"beta": 0.5}},
    )

    assert scale not in result
    assert result["score"] == pytest.approx(accuracy * 0.5)


@pytest.mark.parametrize("scale", ["constant", "char", "word"])
def test_compute_score_scale_preserves_cut(scale):
    result = compute_score(
        "<TXT>a b c d</TXT>",
        ground_truth="a",
        version=2609,
        text_norm="identity",
        scale=scale,
        measures={"fmt": {"beta": 1.0}, "word": {"cut": 0.0}},
    )

    assert result["word"] == -2.0
    assert result["score"] == 0.0


@pytest.mark.parametrize("scale", ["constant", "char", "word"])
def test_compute_score_scale_with_no_measures(scale):
    result = compute_score("<TXT>ct dog</TXT>", ground_truth="cat dog", version=2609, scale=scale)

    assert result["score"] == 0.0


@pytest.mark.parametrize("scale", ["unknown", "fmt", 1, None, {}, ["word"]])
def test_compute_score_rejects_invalid_scale(scale):
    with pytest.raises(ValueError, match="Unsupported score scale"):
        compute_score("<TXT>hello</TXT>", ground_truth="hello", scale=scale)


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [("hello", "hello"), ("hello", "wrong"), ("", ""), ("hello", ""), ("", "hello")],
)
def test_fixed_metric_is_always_one(reference, hypothesis):
    output = f"<TXT>{hypothesis}</TXT>"
    kwargs = {"version": 2609, "text_norm": "identity"}
    parsed = _parse_response(output, ground_truth=reference, **kwargs)
    result = compute_score(output, ground_truth=reference, **kwargs)
    weighted = compute_score(output, ground_truth=reference, measures={"fixed": {"beta": 0.5}}, **kwargs)

    assert parsed["fixed"] == 1.0
    assert result == {"score": 0.0}
    assert weighted["fixed"] == 1.0
    assert weighted["score"] == 0.5


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
        ("the quick brown fox", "the quick blue fox", ["brown"], 0.75),
        ("the quick brown fox", "the quick fox", ["brown"], 0.75),
        ("the quick brown fox", "the quick brown brown fox", ["brown"], 0.75),
        ("new york state", "new york state", ["new york"], 1.0),
        ("new york state", "new state", ["new york"], 2 / 3),
        ("the quick brown fox", "the quick blue fox", None, 1.0),
    ],
)
def test_compute_kw_score(reference, hypothesis, keywords, expected):
    result = compute_kw_score(reference, hypothesis, keywords)

    assert result["word_kw"] == pytest.approx(expected)
    assert result["keyword"] == pytest.approx(
        1.0 - result["kw_nerr"] / max(result["kw_nref"], 1)
    )
    assert set(result) == {"word_kw", "keyword", "kw_nerr", "kw_nref"}


def test_compute_kw_score_reports_error_and_reference_counts():
    result = compute_kw_score(
        "new york state",
        "new state",
        ["new york"],
    )

    assert result["word_kw"] == pytest.approx(2 / 3)
    assert result["keyword"] == 0.5
    assert result["kw_nerr"] == 1
    assert result["kw_nref"] == 2


def test_compute_kw_score_without_keywords_preserves_evaluation_counts():
    assert compute_kw_score("the quick brown fox", "the quick blue fox") == {
        "word_kw": 1.0, "keyword": 1.0, "kw_nerr": 0, "kw_nref": 0,
    }


def test_parse_response_reports_keyword_rewards():
    result = _parse_response(
        "the quick blue fox",
        ground_truth="the quick brown fox",
        extra_info={"keywords": ["brown"]},
    )

    assert result["word_kw"] == 0.75
    assert result["keyword"] == 0.0


@pytest.mark.parametrize(
    ("reference", "hypothesis", "keywords", "expected"),
    [
        ("the quick brown fox", "the quick brown fox", ["brown"], 1.0),
        ("the quick brown fox", "the quick blue fox", ["brown"], 0.75),
        ("the quick brown fox", "the quick fox", ["brown"], 0.75),
        ("the quick brown fox", "the quick brown brown fox", ["brown"], 0.75),
        ("new york state", "new state", ["new york"], 2 / 3),
        ("new york state", "state", ["new york", "york"], 1 / 3),
        ("the quick brown fox", "a slow brown fox", ["brown"], 1.0),
        ("the quick brown fox", "the quick blue fox", None, 1.0),
        ("the quick brown fox", "the quick blue fox", [], 1.0),
        ("hello world", "hello brown world", ["brown"], 0.5),
        ("", "", ["brown"], 1.0),
        ("", "brown", ["brown"], 0.0),
        ("brown", "brown brown brown brown", ["brown"], -2.0),
    ],
)
def test_parse_response_keyword_uses_whole_utterance(reference, hypothesis, keywords, expected):
    keyword_result = compute_kw_score(reference, hypothesis, keywords, text_norm="identity")
    result = _parse_response(
        f"<TXT>{hypothesis}</TXT>",
        ground_truth=reference,
        version=2609,
        text_norm="identity",
        extra_info={"keywords": keywords},
    )

    assert result["word_kw"] == pytest.approx(expected)
    assert result["word_kw"] == keyword_result["word_kw"]
    assert result["keyword"] == keyword_result["keyword"]
    assert result["keyword"] == pytest.approx(
        1.0 - keyword_result["kw_nerr"] / max(keyword_result["kw_nref"], 1)
    )


@pytest.mark.parametrize(
    ("reference", "hypothesis", "keywords", "expected_counts"),
    [
        ("the quick brown fox", "the quick blue fox", ["brown"], (1, 1)),
        ("new york state", "new state", ["new york"], (1, 2)),
        ("hello world", "hello brown world", ["brown"], (1, 0)),
        ("hello world", "hello world", None, (0, 0)),
    ],
)
def test_keyword_evaluation_reports_counts_without_rewards(
    monkeypatch, reference, hypothesis, keywords, expected_counts,
):
    from recipe.phimm.reward import asr_eval

    monkeypatch.setattr(
        asr_eval, "measure_openasr_en_wer",
        lambda hyp, ref: {"wer": 0.25, "n_err": 1, "n_ref": len(ref.split())},
    )
    result = asr_eval.openasr_en_eval(
        hypothesis, reference,
        version=2609, extra_info={"keywords": keywords},
    )

    assert "word_kw" not in result
    assert "keyword" not in result
    assert "kw_acc" not in result
    assert (result["nb_err"], result["nb_ref"]) == expected_counts
    assert result["score"] == 0.75


def test_keyword_uses_normalized_reference_word_count():
    result = _parse_response(
        "<TXT>hello blue world</TXT>",
        ground_truth="Hello, BROWN world!",
        version=2609,
        text_norm="lower",
        extra_info={"keywords": ["BROWN"]},
    )

    assert result["word_kw"] == pytest.approx(2 / 3)


def test_compute_score_keyword_is_configurable_and_reports_greedy_delta():
    result = compute_score(
        "<TXT>the quick blue fox</TXT>",
        ground_truth="the quick brown fox",
        version=2609,
        text_norm="identity",
        extra_info={"keywords": ["brown"], "greedy_reward": {"score": 2.0, "word_kw": 1.0}},
        measures={"word_kw": {"beta": 2.0}},
    )

    assert result == {"score": 1.5, "word_kw": 0.75, "d_score": -0.5, "d_word_kw": -0.25}


def test_compute_score_keyword_only_accepts_raw_2609_text():
    result = compute_score(
        "new state",
        ground_truth="new york state",
        version=2609,
        text_norm="identity",
        extra_info={"keywords": ["new york"]},
        measures={"keyword": {"beta": 2.0}},
    )

    assert result == {"score": 1.0, "keyword": 0.5}


def test_compute_score_omits_unconfigured_keyword():
    result = compute_score(
        "<TXT>the quick blue fox</TXT>",
        ground_truth="the quick brown fox",
        version=2609,
        text_norm="identity",
        extra_info={"keywords": ["brown"]},
        measures={"word": {}},
    )

    assert result == {"score": 0.75, "word": 0.75}


@pytest.mark.parametrize(
    ("solution", "reference", "expected_lang", "expected_fmt", "word_lang", "word_fmt"),
    [
        ("<src=English><tgt=English>\n<TXT>hello world</TXT>", "hello world", 1.0, 1.0, 1.0, 1.0),
        ("<src=French><tgt=French>\n<TXT>hello world</TXT>", "hello world", 0.0, 1.0, 0.5, 1.0),
        (
            "<src=English><tgt=English>\n<TXT>hello world</TXT>\n<src=broken",
            "hello world", 1.0, 0.0, 1.0, 0.5,
        ),
        ("", "one two three four", 0.0, 0.0, 0.75, 0.75),
        ("", "", 0.0, 0.0, 0.0, 0.0),
        ("<src=English><tgt=English>\n<TXT>hello</TXT>", "", 1.0, 1.0, 1.0, 1.0),
        (
            "<src=English><tgt=English>\n<TXT>hello</TXT>\n"
            "<src=French><tgt=French>\n<TXT>bonjour</TXT>\n"
            "<src=Chinese><tgt=Chinese>\n<TXT>你好</TXT>",
            "hello", -1.0, 1.0, -1.0, 1.0,
        ),
    ],
)
def test_word_lang_and_word_fmt_normalize_utterance_errors(
    solution, reference, expected_lang, expected_fmt, word_lang, word_fmt,
):
    result = _parse_response(
        solution, ground_truth=reference, version=2609,
        text_norm="identity", extra_info={"language": "English"},
    )

    assert result["lang"] == expected_lang
    assert result["fmt"] == expected_fmt
    assert result["word_lang"] == word_lang
    assert result["word_fmt"] == word_fmt


def test_word_lang_and_word_fmt_use_normalized_reference_words():
    result = _parse_response(
        "", ground_truth="one   two\nthree four",
        version=2609, text_norm="identity", language="English",
    )

    assert result["word_lang"] == 0.75
    assert result["word_fmt"] == 0.75


def test_compute_score_word_lang_and_word_fmt_support_weights_and_deltas():
    result = compute_score(
        "", ground_truth="one two three four",
        version=2609, text_norm="identity", language="English",
        measures={"word_lang": {"beta": 2.0}, "word_fmt": {"beta": 1.0}},
        extra_info={"greedy_reward": {"score": 3.0, "word_lang": 1.0, "word_fmt": 1.0}},
    )

    assert result == {
        "score": 2.25,
        "word_lang": 0.75,
        "word_fmt": 0.75,
        "d_score": -0.75,
        "d_word_lang": -0.25,
        "d_word_fmt": -0.25,
    }


def test_compute_score_omits_unconfigured_word_lang_and_word_fmt():
    result = compute_score(
        "", ground_truth="hello world", version=2609, text_norm="identity",
        measures={"lang": {}, "fmt": {}},
    )

    assert result == {"score": 0.0, "lang": 0.0, "fmt": 0.0}


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected"),
    [
        ("a b c", "a b c", (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
        ("a b c", "a c", (2 / 3, 1.0, 2 / 3, 1.0, 2 / 3, 2 / 3)),
        ("a b c", "a b c d", (1.0, 2 / 3, 1.0, 2 / 3, 2 / 3, 2 / 3)),
        ("a b c", "a x c", (1.0, 1.0, 1.0, 1.0, 2 / 3, 2 / 3)),
        ("a b c", "a c d e", (2 / 3, 1 / 3, 2 / 3, 1 / 3, 0.0, 0.0)),
        ("cat dog", "ct dog", (1.0, 1.0, 5 / 6, 1.0, 0.5, 5 / 6)),
        ("cat dog", "cart dog", (1.0, 1.0, 1.0, 5 / 6, 0.5, 5 / 6)),
        ("", "", (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
        ("", "a b", (1.0, -1.0, 1.0, -1.0, -1.0, -1.0)),
        ("a b", "", (0.0, 1.0, 0.0, 1.0, 0.0, 0.0)),
        ("a", "a b c d", (1.0, -2.0, 1.0, -2.0, -2.0, -2.0)),
    ],
)
def test_parse_response_reports_edit_accuracies(reference, hypothesis, expected):
    result = _parse_response(
        f"<TXT>{hypothesis}</TXT>",
        ground_truth=reference,
        language="English",
        version=2609,
        text_norm="identity",
    )

    metrics = ("word_del", "word_ins", "char_del", "char_ins", "word", "char")
    assert tuple(result[name] for name in metrics) == pytest.approx(expected)


def test_parse_response_edit_accuracies_use_text_normalization():
    result = _parse_response(
        "<TXT>HELLO WORLD!</TXT>",
        ground_truth="hello, world.",
        language="English",
        version=2609,
        text_norm="simple",
    )

    for name in ("word_del", "word_ins", "char_del", "char_ins", "word", "char"):
        assert result[name] == 1.0


@pytest.mark.parametrize(
    ("metric", "hypothesis", "expected"),
    [
        ("word_del", "a c", 2 / 3),
        ("word_ins", "a b c d", 2 / 3),
        ("char_del", "a c", 2 / 3),
        ("char_ins", "a b c d", 2 / 3),
    ],
)
@pytest.mark.parametrize("cut", [None, 1 - 1 / 3])
def test_compute_score_supports_edit_accuracies(metric, hypothesis, expected, cut):
    config = {"beta": 0.5}
    if cut is not None:
        config["cut"] = cut
    result = compute_score(
        f"<TXT>{hypothesis}</TXT>",
        ground_truth="a b c",
        language="English",
        version=2609,
        text_norm="identity",
        measures={metric: config},
    )

    assert result[metric] == pytest.approx(expected)
    assert result["score"] == pytest.approx(0.0 if cut is not None else 0.5 * expected)


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