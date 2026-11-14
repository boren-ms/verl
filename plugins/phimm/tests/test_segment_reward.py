"""Check joint text rewards and independent 2609 response checks."""

import importlib

import pytest

from recipe.phimm.reward.asr_measure import _parse_response, check_lang, compute_score, lang_score
from recipe.phimm.reward.asr_response import get_hyp_text, parse_task_output
from recipe.phimm.reward.asr_vote import compute_score as vote_score

SCORERS = [
    ("asr_eval", "openasr_eval"),
    ("asr_eval", "openasr_en_eval"),
    ("asr_eval", "openasr_hi_in_eval"),
    ("asr_eval", "non_speech_eval"),
    ("asr_eval", "foreign_language_eval"),
    ("asr_measure", "compute_score"),
    ("asr_measure", "lang_score"),
    ("asr_vote", "compute_score"),
    ("openasr_expansion", "openasr_expansion_eval"),
    ("asr_inhouse_measure", "eval_score"),
]


def response(text, language="English"):
    return f"<src={language}><tgt={language}>\n<verbatim>{text}</verbatim>"


def test_joint_text_metrics_cross_segment_boundaries():
    parsed = _parse_response(
        [response("Hello,"), response("World!")],
        "Hello, World!",
        extra_info={"language": "English", "keywords": ["Hello World"]},
    )
    for metric in ("word", "char", "word_kw", "char_kw", "keyword", "punc", "cap", "lang", "fmt"):
        assert parsed[metric] == 1.0, metric


@pytest.mark.parametrize("text", [response("hello world"), "", "<nonspeech>", "<verbatim"])
def test_single_response_list_preserves_string_scores(text):
    assert _parse_response([text], "hello world") == _parse_response(text, "hello world")


@pytest.mark.parametrize(
    ("second", "language", "fmt"),
    [
        (response("world"), 1.0, 1.0),
        (response("world", "French"), 0.5, 1.0),
        ("", 0.5, 0.5),
        ("<verbatim", 0.5, 0.5),
        ("world", 0.5, 1.0),
    ],
)
def test_response_boundaries_preserve_invalid_checks(second, language, fmt):
    responses = [response("hello"), second]
    parsed = _parse_response(responses, "hello world")
    assert parsed["lang"] == language
    assert parsed["fmt"] == fmt
    assert parsed["word_fmt"] == 1 - (1 - fmt) / 2
    assert parsed["word_lang"] == pytest.approx(1 - (1 - min(parsed["word"], 0.99)) * (1 - language))
    assert parsed["char_lang"] == pytest.approx(1 - (1 - parsed["char"]) * (1 - language))
    assert lang_score(responses)["score"] == language


def test_repeated_headers_within_one_response_still_count_as_language_errors():
    first = "\n".join([response("hello"), response("there")])
    parsed = _parse_response([first, response("world")], "hello there world")
    assert parsed["word"] == 1
    assert parsed["lang"] == 0.5


@pytest.mark.parametrize("language", ["English_Chinese", "zh_en", "Chinese_English"])
@pytest.mark.parametrize(
    "responses, expected",
    [
        ([response("hello"), response("world", "Chinese")], 1.0),
        ([response("hello", "Chinese"), response("world")], 1.0),
        ([response("hello"), response("world")], 1.0),
        ([response("hello"), response("world", "French")], 0.5),
        ([response("hello"), ""], 0.5),
        ([response("hello"), "world"], 0.5),
        ([response("hello"), "<src=English><tgt=Chinese>\nworld"], 0.5),
        ([response("hello"), response("world", "Cantonese")], 0.5),
        ([response("hello"), response("world", "Unknown")], 0.5),
    ],
)
def test_multilingual_reference_accepts_any_allowed_language_per_response(language, responses, expected):
    options = {"extra_info": {"language": language}}
    assert lang_score(responses, **options) == {"score": expected, "p_lang": expected}
    parsed = _parse_response(responses, "hello world", **options)
    assert parsed["lang"] == expected
    assert compute_score(responses, "hello world", measures={"lang": {}}, **options)["score"] == expected
    baseline = vote_score(responses, "hello world", measures={"lang": {"vote": "abs"}}, **options)
    assert (
        vote_score(
            responses,
            "hello world",
            measures={"lang": {"vote": "abs"}},
            extra_info={"language": language, "greedy_reward": baseline},
        )["score"]
        == expected
    )


def test_multilingual_reference_checks_every_header_in_a_mixed_response():
    valid = "\n".join([response("hello"), response("world", "Chinese")])
    invalid = "\n".join([response("hello"), response("world", "French")])
    assert check_lang(parse_task_output(valid), "English_Chinese") == {"acc": 1, "n_err": 0, "n_ref": 2}
    assert check_lang(parse_task_output(invalid), "English_Chinese") == {"acc": 0.5, "n_err": 1, "n_ref": 2}
    assert check_lang([], "English_Chinese") == {"acc": 0, "n_err": 1, "n_ref": 1}
    assert lang_score([response("hello", "Unknown")], language="Unknown_Unsupported")["score"] == 0


def test_format_checks_weight_independent_responses_equally():
    first = "\n".join([response("hello"), response("there")])
    assert _parse_response([first, ""], "hello there")["fmt"] == 0.5


def test_segment_order_affects_joint_text_alignment():
    assert _parse_response([response("world"), response("hello")], "hello world")["word"] < 1


def test_vote_reward_uses_joint_greedy_metrics_and_averaged_checks():
    options = {
        "reduce": "sum",
        "measures": {
            "word": {"vote": "delta"},
            "lang": {"vote": "abs", "beta": 0.1},
            "fmt": {"vote": "abs", "beta": 0.1},
        },
    }
    baseline = vote_score([response("hello"), ""], "hello world", **options)
    assert baseline == {"score": 0, "word": 0.5, "lang": 0.5, "fmt": 0.5}
    result = vote_score(
        [response("hello"), response("world")],
        "hello world",
        extra_info={"greedy_reward": baseline},
        **options,
    )
    assert result["score"] == pytest.approx(0.7)
    assert result["d_word"] == 0.5
    assert result["d_lang"] == result["d_fmt"] == 0.5
    assert result["v_lang"] == result["v_fmt"] == 1


def test_measure_reward_accepts_segment_list():
    result = compute_score(
        [response("hello"), response("world", "French")],
        "hello world",
        measures={"word": {}, "lang": {}, "fmt": {}},
        reduce="mean",
    )
    assert result == {"score": pytest.approx(2.5 / 3), "word": 1, "lang": 0.5, "fmt": 1}


def test_openasr_validation_scores_full_text_and_cross_segment_keywords():
    from recipe.phimm.reward.asr_eval import openasr_en_eval, openasr_eval

    outputs = [response("hello"), response("world")]
    extra = {"language": "English", "keywords": ["hello world"]}
    result = openasr_en_eval(outputs, "hello world", extra_info=extra)
    assert result["score"] == 1
    assert result["n_err"] == 0
    assert result["n_ref"] == 2
    assert result["nb_err"] == 0
    assert result["nb_ref"] == 2
    assert openasr_eval(outputs, "hello world", extra_info=extra)["n_err"] == 0
    incomplete = openasr_en_eval([response("hello"), ""], "hello world", extra_info=extra)
    assert incomplete["score"] == 0.5
    assert incomplete["n_err"] == 1
    assert incomplete["n_ref"] == 2


@pytest.mark.parametrize(
    ("language", "texts", "reference"),
    [
        ("Chinese", ["你好", "世界"], "你好世界"),
        ("French", ["bonjour", "monde"], "bonjour monde"),
        ("Vietnamese", ["xin", "chào"], "xin chào"),
    ],
)
def test_expansion_validation_accepts_segments(language, texts, reference):
    from recipe.phimm.reward.openasr_expansion import openasr_expansion_eval

    result = openasr_expansion_eval([response(text, language) for text in texts], reference, language=language)
    assert result["score"] == 1
    assert result["n_err"] == 0
    assert result["n_ref"] > 0


def test_hindi_validation_passes_joint_text_and_full_lattice(monkeypatch):
    from recipe.phimm.reward import asr_eval

    captured = []
    lattice = [["hello"], ["world"]]

    def measure(hyp_text, reference_lattice, lang):
        captured.append((hyp_text, reference_lattice, lang))
        return {"wer": 0.0, "n_err": 0, "n_ref": 2}

    monkeypatch.setattr(asr_eval, "measure_openasr_oiwer", measure)
    result = asr_eval.openasr_hi_in_eval(
        [response("hello", "Hindi"), response("world", "Hindi")],
        "hello world",
        extra_info={"language": "Hindi", "lattice": lattice},
    )
    assert captured == [("hello world", lattice, "hi")]
    assert result["score"] == 1


def test_inhouse_validation_scores_parent_text_once(monkeypatch):
    from recipe.phimm.reward import asr_inhouse_measure

    captured = []
    entity_inputs = []

    def measure(ref, hyp, *, locale):
        captured.append((ref, hyp, locale))
        return 0, 2, 0.0, {}

    monkeypatch.setattr(asr_inhouse_measure, "ensure_pack_dir", lambda *args: None)
    monkeypatch.setattr(asr_inhouse_measure, "_compute_dter", measure)

    def measure_entities(ref, hyp, **kwargs):
        entity_inputs.append((ref, hyp))
        return 0, 1, 0.0

    monkeypatch.setattr(asr_inhouse_measure, "_compute_eer", measure_entities)
    result = asr_inhouse_measure.eval_score([response("hello"), response("world")], "hello world", compute_eer=True)
    assert captured == [("hello world", "hello world", "en-us")]
    assert entity_inputs == [("hello world", "hello world")]
    assert result["score"] == 1
    assert result["dter_n_ref"] == 2
    assert result["eer_n_ref"] == 1


@pytest.mark.parametrize(("second", "score"), [("<nonspeech>", 1), (response("hello"), 0)])
def test_non_speech_validation_checks_all_segments(second, score):
    from recipe.phimm.reward.asr_eval import non_speech_eval

    result = non_speech_eval(["<nonspeech>", second], "")
    assert result["score"] == score
    assert result["n_ref"] == 1


@pytest.mark.parametrize(("second", "score"), [(response("<foreign_language>"), 1), (response("hello"), 0)])
def test_foreign_language_validation_cannot_hide_invalid_segment(second, score):
    from recipe.phimm.reward.asr_eval import foreign_language_eval

    result = foreign_language_eval([response("<foreign_language>"), second], "")
    assert result["score"] == score
    assert result["n_ref"] == 1


@pytest.mark.parametrize(("module", "name"), SCORERS)
def test_all_scorers_preserve_single_response_and_accept_ordered_lists(module, name, monkeypatch):
    from recipe.phimm.reward import asr_eval, asr_inhouse_measure

    # Keep external Hindi/SpeechInsight engines outside this input-contract test.
    # Dedicated tests above verify the actual joint text passed to those engines.
    monkeypatch.setattr(asr_eval, "measure_openasr_oiwer", lambda *args: {"wer": 0, "n_err": 0, "n_ref": 2})
    monkeypatch.setattr(asr_inhouse_measure, "ensure_pack_dir", lambda *args: None)
    monkeypatch.setattr(asr_inhouse_measure, "_compute_dter", lambda *args, **kwargs: (0, 2, 0, {}))
    scorer = getattr(importlib.import_module(f"recipe.phimm.reward.{module}"), name)
    language = "French" if module == "openasr_expansion" else "English"
    options: dict = {"language": language, "version": 2609}
    if module in {"asr_measure", "asr_vote"}:
        options["measures"] = {"word": {}, "lang": {}, "fmt": {}}
    single = response("hello world", language)
    assert scorer([single], "hello world", **options) == scorer(single, "hello world", **options)
    result = scorer([response("hello", language), response("world", language)], "hello world", **options)
    assert isinstance(result["score"], int | float)


@pytest.mark.parametrize(("module", "name"), SCORERS)
@pytest.mark.parametrize(("value", "error"), [([], ValueError), ([None], TypeError), (["hello", 1], TypeError)])
def test_all_scorers_reject_invalid_segment_lists(module, name, value, error):
    scorer = getattr(importlib.import_module(f"recipe.phimm.reward.{module}"), name)
    with pytest.raises(error):
        scorer(value, "hello world", language="English")


@pytest.mark.parametrize(("value", "error"), [([], ValueError), ([None], TypeError), (["hello", 1], TypeError)])
def test_invalid_segment_lists_fail_explicitly(value, error):
    with pytest.raises(error):
        vote_score(value, "hello")
    with pytest.raises(error):
        get_hyp_text(value)


@pytest.mark.parametrize("version", [None, 2609, "2609"])
def test_only_2609_parser_defaults_and_explicit_version(version):
    assert get_hyp_text(response("hello"), version=version) == "hello"
    assert get_hyp_text("<src=English><tgt=English>\n<TXT>hello</TXT>", version=version) == "hello"
    assert get_hyp_text("<nonspeech>", version=version) == ""


@pytest.mark.parametrize("version", [2607, "2607", 1234])
def test_unsupported_model_versions_fail(version):
    with pytest.raises(ValueError, match="only model version 2609"):
        parse_task_output(response("hello"), version=version)
    with pytest.raises(ValueError, match="only model version 2609"):
        vote_score([response("hello")], "hello", version=version)
