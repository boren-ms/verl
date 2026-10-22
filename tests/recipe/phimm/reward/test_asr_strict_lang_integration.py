import pytest

from recipe.phimm.reward import asr_measure, asr_vote


CORRECT_2609 = "<src=English><tgt=English>\n<TXT>hello world</TXT>"
WRONG_SOURCE_2609 = "<src=French><tgt=English>\n<TXT>hello world</TXT>"
LEGACY_2607 = "<ASR><lang=English><TXT>hello world</TXT></ASR>"
CORRECT_2607 = "Audio Language: English.\n" + LEGACY_2607
WRONG_SOURCE_2607 = "Audio Language: French.\n" + LEGACY_2607
UNKNOWN_SOURCE_2607 = "Audio Language: Klingon.\n" + LEGACY_2607
LEGACY_PARSER_KWARGS = [
    pytest.param({}, id="default"),
    pytest.param({"version": None}, id="none"),
    pytest.param({"version": 2607}, id="2607-int"),
    pytest.param({"version": "2607"}, id="2607-string"),
    pytest.param({"version": "unknown"}, id="unknown"),
]


@pytest.mark.parametrize("version", [2609, "2609"])
def test_lang_score_validates_source_and_prefers_example_language(version):
    kwargs = {"version": version, "language": "French", "extra_info": {"language": "English"}}

    assert asr_measure.lang_score(CORRECT_2609, **kwargs) == {"score": 1.0, "p_lang": 1.0}
    assert asr_measure.lang_score(WRONG_SOURCE_2609, **kwargs) == {"score": 0.0, "p_lang": 0.0}


def test_parse_response_penalizes_source_without_changing_lexical_accuracy():
    result = asr_measure._parse_response(
        WRONG_SOURCE_2609, "hello world", version=2609, text_norm="identity"
    )

    assert result["word"] == result["char"] == result["fmt"] == 1.0
    assert result["lang"] == 0.0
    assert result["word_lang"] == pytest.approx(0.99)


def test_compute_score_propagates_strict_language_and_composite_measures():
    result = asr_measure.compute_score(
        WRONG_SOURCE_2609,
        "hello world",
        version=2609,
        text_norm="identity",
        measures={"lang": {"beta": 0.2}, "word_lang": {}, "word": {}, "fmt": {}},
    )

    assert result == pytest.approx(
        {"score": 2.99, "lang": 0.0, "word_lang": 0.99, "word": 1.0, "fmt": 1.0}
    )


@pytest.mark.parametrize("score_type", ["acc", "err"])
@pytest.mark.parametrize("sample_correct", [False, True])
def test_vote_roundtrip_preserves_source_language_delta(score_type, sample_correct):
    kwargs = {
        "version": 2609,
        "text_norm": "identity",
        "measures": {"word_lang": {"score": score_type, "delta": "abs", "vote": "delta"}},
    }
    greedy_output = WRONG_SOURCE_2609 if sample_correct else CORRECT_2609
    sample_output = CORRECT_2609 if sample_correct else WRONG_SOURCE_2609
    greedy = asr_vote.compute_score(greedy_output, "hello world", **kwargs)
    original_greedy = greedy.copy()
    result = asr_vote.compute_score(
        sample_output, "hello world", extra_info={"greedy_reward": greedy}, **kwargs
    )
    delta = 0.01 if sample_correct else -0.01
    accuracy = 1.0 if sample_correct else 0.99

    assert result == pytest.approx(
        {
            "score": delta,
            "word_lang": accuracy - (1.0 if score_type == "err" else 0.0),
            "d_word_lang": delta,
            "v_word_lang": delta,
        }
    )
    assert greedy == original_greedy


def test_public_scorer_accepts_aliases_mixed_identifiers_and_repeated_positions():
    response = "<src=EN_zh en><tgt=English Mandarin ENGLISH>\n<TXT>hello world</TXT>"
    result = asr_measure.compute_score(
        response,
        "hello world",
        version=2609,
        text_norm="identity",
        extra_info={"language": "English_Chinese English"},
        measures={"lang": {}, "word_lang": {}},
    )

    assert result == {"score": 2.0, "lang": 1.0, "word_lang": 1.0}


def test_public_scorer_keeps_partial_credit_for_one_wrong_source_position():
    response = "<src=en fr><tgt=en de>\n<TXT>hello world</TXT>"
    result = asr_measure._parse_response(
        response,
        "hello world",
        version=2609,
        text_norm="identity",
        extra_info={"language": "English_German"},
    )

    assert result["word"] == 1.0
    assert result["lang"] == 0.5
    assert result["word_lang"] == pytest.approx(0.995)


@pytest.mark.parametrize(
    ("reference", "src", "tgt"),
    [
        ("Chinese", "Chinese", "zh"),
        ("Cantonese", "Cantonese", "yue"),
        ("Mandarin", "zh", "Mandarin"),
        ("zh", "Chinese", "zh"),
        ("yue", "yue", "Cantonese"),
    ],
)
def test_public_scorers_accept_existing_names_codes_and_aliases(reference, src, tgt):
    response = f"<src={src}><tgt={tgt}>\n<TXT>你好</TXT>"
    kwargs = {"version": 2609, "language": reference, "text_norm": "identity"}

    assert asr_measure.lang_score(response, **kwargs) == {"score": 1.0, "p_lang": 1.0}
    result = asr_measure.compute_score(response, "你好", measures={"lang": {}}, **kwargs)
    assert result == {"score": 1.0, "lang": 1.0}


@pytest.mark.parametrize("identifier", ["中文", "粤语"])
@pytest.mark.parametrize("field", ["src", "tgt", "reference"])
def test_public_scorer_rejects_chinese_script_language_identifiers(identifier, field):
    languages = {"src": "Chinese", "tgt": "Chinese", "reference": "Chinese"}
    languages[field] = identifier
    response = f"<src={languages['src']}><tgt={languages['tgt']}>\n<TXT>你好</TXT>"

    assert asr_measure.lang_score(response, version=2609, language=languages["reference"]) == {
        "score": 0.0, "p_lang": 0.0
    }


def test_vote_penalizes_chinese_cantonese_header_mismatch():
    kwargs = {
        "version": 2609,
        "language": "Chinese",
        "text_norm": "identity",
        "measures": {"lang": {"delta": "abs", "vote": "delta"}},
    }
    greedy = asr_vote.compute_score("<src=Chinese><tgt=Chinese>\n你好", "你好", **kwargs)
    result = asr_vote.compute_score(
        "<src=Cantonese><tgt=Chinese>\n你好",
        "你好",
        extra_info={"greedy_reward": greedy},
        **kwargs,
    )

    assert result == {"score": -1.0, "lang": 0.0, "d_lang": -1.0, "v_lang": -1.0}


def test_legacy_scoring_distinguishes_chinese_and_cantonese():
    response = "Audio Language: Cantonese.\n<ASR><lang=Cantonese><TXT>你好</TXT></ASR>"

    assert asr_measure.parse_task_output(response, version=2607) == [
        {"src": "Cantonese", "tgt": "Cantonese", "text": "你好"}
    ]

    assert asr_measure.lang_score(response, version=2607, language="Chinese") == {
        "score": 0.0, "p_lang": 0.0
    }


def test_public_scorers_score_a_malformed_trailing_segment_as_text_and_a_language_error():
    response = CORRECT_2609 + "\n<src=English>\n<TXT>extra</TXT>"
    kwargs = {"version": 2609, "text_norm": "identity"}

    assert asr_measure.lang_score(response, **kwargs) == {"score": 0.0, "p_lang": 0.0}
    result = asr_measure._parse_response(response, "hello world", **kwargs)
    assert result["word"] < 1.0
    assert result["lang"] == 0.0
    assert result["word_lang"] == pytest.approx(result["word"])


def test_matching_unknown_headers_cannot_earn_language_credit():
    response = "<src=Klingon><tgt=Klingon>\n<TXT>hello world</TXT>"

    assert asr_measure.lang_score(response, version=2609, language="Klingon") == {
        "score": 0.0,
        "p_lang": 0.0,
    }


def test_language_measure_remains_a_header_check_for_wrong_language_body():
    response = "<src=German><tgt=German>\n<TXT>This new pattern is being created.</TXT>"

    assert asr_measure.lang_score(response, version=2609, language="German") == {
        "score": 1.0,
        "p_lang": 1.0,
    }


@pytest.mark.parametrize("parser_kwargs", LEGACY_PARSER_KWARGS)
@pytest.mark.parametrize(
    ("response", "source", "expected_lang"),
    [
        pytest.param(CORRECT_2607, "English", 1.0, id="correct"),
        pytest.param(WRONG_SOURCE_2607, "French", 0.0, id="wrong-source"),
        pytest.param(UNKNOWN_SOURCE_2607, "Klingon", 0.0, id="unknown-source"),
        pytest.param(LEGACY_2607, None, 0.0, id="missing-source"),
    ],
)
def test_legacy_parser_public_scorers_require_a_supported_matching_source(
    parser_kwargs, response, source, expected_lang
):
    kwargs = {**parser_kwargs, "text_norm": "identity"}
    assert asr_measure.parse_task_output(response, **parser_kwargs) == [
        {"src": source, "tgt": "English", "text": "hello world"}
    ]
    assert asr_measure.lang_score(response, **kwargs) == {
        "score": expected_lang, "p_lang": expected_lang
    }
    parsed = asr_measure._parse_response(response, "hello world", **kwargs)
    expected_word_lang = 1.0 if expected_lang else 0.99
    assert parsed["word"] == parsed["char"] == parsed["fmt"] == 1.0
    assert parsed["lang"] == expected_lang
    assert parsed["word_lang"] == pytest.approx(expected_word_lang)
    measured = asr_measure.compute_score(
        response, "hello world", measures={"lang": {}, "word_lang": {}}, **kwargs
    )
    assert measured == pytest.approx({
        "score": expected_lang + expected_word_lang,
        "lang": expected_lang,
        "word_lang": expected_word_lang,
    })


@pytest.mark.parametrize("parser_kwargs", LEGACY_PARSER_KWARGS)
@pytest.mark.parametrize("score_type", ["acc", "err"])
@pytest.mark.parametrize("sample_correct", [False, True])
def test_legacy_parser_vote_roundtrip_penalizes_missing_source(parser_kwargs, score_type, sample_correct):
    kwargs = {
        **parser_kwargs,
        "text_norm": "identity",
        "measures": {
            "lang": {"score": score_type, "delta": "abs", "vote": "delta"},
            "word_lang": {"score": score_type, "delta": "abs", "vote": "delta"},
        },
    }
    greedy_output = LEGACY_2607 if sample_correct else CORRECT_2607
    sample_output = CORRECT_2607 if sample_correct else LEGACY_2607
    greedy = asr_vote.compute_score(greedy_output, "hello world", **kwargs)
    offset = 1.0 if score_type == "err" else 0.0
    assert greedy == pytest.approx({
        "score": 0.0,
        "lang": float(not sample_correct) - offset,
        "word_lang": (0.99 if sample_correct else 1.0) - offset,
    })
    original_greedy = greedy.copy()
    result = asr_vote.compute_score(
        sample_output, "hello world", extra_info={"greedy_reward": greedy}, **kwargs
    )
    lang_delta = 1.0 if sample_correct else -1.0
    word_lang_delta = 0.01 if sample_correct else -0.01
    assert result == pytest.approx({
        "score": (lang_delta + word_lang_delta) / 2,
        "lang": float(sample_correct) - offset,
        "word_lang": (1.0 if sample_correct else 0.99) - offset,
        "d_lang": lang_delta,
        "v_lang": lang_delta,
        "d_word_lang": word_lang_delta,
        "v_word_lang": word_lang_delta,
    })
    assert greedy == original_greedy


@pytest.mark.parametrize("parser_kwargs", LEGACY_PARSER_KWARGS)
def test_legacy_parser_scoring_retains_ordered_mixed_language_partial_credit(parser_kwargs):
    response = (
        "Audio Language: English and French.\n"
        "<ASR><lang=English><TXT>hello</TXT><lang=German><TXT>world</TXT></ASR>"
    )
    assert asr_measure.parse_task_output(response, **parser_kwargs) == [
        {"src": "English", "tgt": "English", "text": "hello"},
        {"src": "French", "tgt": "German", "text": "world"},
    ]
    result = asr_measure._parse_response(
        response,
        "hello world",
        **parser_kwargs,
        text_norm="identity",
        extra_info={"language": "English_German"},
    )

    assert result["word"] == result["char"] == result["fmt"] == 1.0
    assert result["lang"] == 0.5
    assert result["word_lang"] == pytest.approx(0.995)


@pytest.mark.parametrize("parser_kwargs", LEGACY_PARSER_KWARGS)
@pytest.mark.parametrize("surplus_source", ["Klingon", "French"])
def test_legacy_parser_retains_surplus_source_languages_for_strict_scoring(parser_kwargs, surplus_source):
    response = f"Audio Language: English and {surplus_source}.\n" + LEGACY_2607

    assert asr_measure.parse_task_output(response, **parser_kwargs) == [
        {"src": f"English {surplus_source}", "tgt": "English", "text": "hello world"}
    ]
    assert asr_measure.lang_score(response, **parser_kwargs, language="English") == {
        "score": 0.0, "p_lang": 0.0
    }
    result = asr_measure._parse_response(
        response, "hello world", **parser_kwargs, text_norm="identity", language="English"
    )

    assert result["word"] == result["char"] == result["fmt"] == 1.0
    assert result["lang"] == 0.0
    assert result["word_lang"] == pytest.approx(0.99)


@pytest.mark.parametrize("parser_kwargs", LEGACY_PARSER_KWARGS)
def test_legacy_parser_surplus_source_preserves_partial_credit_for_two_targets(parser_kwargs):
    response = (
        "Audio Language: English, French and German.\n"
        "<ASR><lang=English><TXT>hello</TXT><lang=French><TXT>world</TXT></ASR>"
    )

    assert asr_measure.parse_task_output(response, **parser_kwargs) == [
        {"src": "English", "tgt": "English", "text": "hello"},
        {"src": "French German", "tgt": "French", "text": "world"},
    ]
    result = asr_measure._parse_response(
        response,
        "hello world",
        **parser_kwargs,
        text_norm="identity",
        language="English_French",
    )

    assert result["word"] == result["char"] == result["fmt"] == 1.0
    assert result["lang"] == 0.5
    assert result["word_lang"] == pytest.approx(0.995)


@pytest.mark.parametrize("parser_kwargs", LEGACY_PARSER_KWARGS)
def test_legacy_parser_matching_unknown_names_cannot_earn_language_credit(parser_kwargs):
    response = "Audio Language: Klingon.\n<ASR><lang=Klingon><TXT>hello world</TXT></ASR>"
    assert asr_measure.parse_task_output(response, **parser_kwargs) == [
        {"src": "Klingon", "tgt": "Klingon", "text": "hello world"}
    ]

    assert asr_measure.lang_score(response, **parser_kwargs, language="Klingon") == {
        "score": 0.0, "p_lang": 0.0
    }


@pytest.mark.parametrize(
    ("response", "version_kwargs", "expected_bad"),
    [
        pytest.param(CORRECT_2609, {}, False, id="default-correct-2609"),
        pytest.param(WRONG_SOURCE_2609, {}, True, id="default-wrong-source-2609"),
        pytest.param(CORRECT_2609, {"version": 2609}, False, id="correct-2609"),
        pytest.param(WRONG_SOURCE_2609, {"version": 2609}, True, id="wrong-source-2609"),
        pytest.param(CORRECT_2607, {"version": 2607}, False, id="correct-2607"),
        pytest.param(WRONG_SOURCE_2607, {"version": 2607}, True, id="wrong-source-2607"),
        pytest.param(UNKNOWN_SOURCE_2607, {"version": 2607}, True, id="unknown-source-2607"),
        pytest.param(LEGACY_2607, {"version": 2607}, True, id="missing-source-2607"),
        pytest.param(CORRECT_2607, {"version": None}, False, id="none-correct-2607"),
        pytest.param(WRONG_SOURCE_2607, {"version": None}, True, id="none-wrong-source-2607"),
        pytest.param(CORRECT_2607, {"version": "unknown"}, False, id="unknown-correct-2607"),
        pytest.param(WRONG_SOURCE_2607, {"version": "unknown"}, True, id="unknown-wrong-source-2607"),
    ],
)
def test_dataset_bad_language_filter_preserves_version_aware_parsing(response, version_kwargs, expected_bad):
    from recipe.phimm.data.dataset import _is_bad_lang

    assert _is_bad_lang({"raw_response": response, **version_kwargs, "language": "English"}) is expected_bad


def test_dataset_bad_language_filter_rejects_chinese_cantonese_header_mismatch():
    from recipe.phimm.data.dataset import _is_bad_lang

    assert _is_bad_lang({
        "raw_response": "<src=Cantonese><tgt=Chinese>\n你好",
        "version": 2609,
        "language": "Chinese",
    })


@pytest.mark.parametrize("identifier", ["中文", "粤语"])
def test_dataset_bad_language_filter_rejects_chinese_script_language_identifiers(identifier):
    from recipe.phimm.data.dataset import _is_bad_lang

    assert _is_bad_lang({
        "raw_response": f"<src={identifier}><tgt=Chinese>\n你好",
        "version": 2609,
        "language": "Chinese",
    })
