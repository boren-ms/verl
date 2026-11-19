"""Diarization reward contracts: speaker permutation, timing, alignment, pooling."""

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from recipe.phimm.agentic.scoring import aggregate_error_rates, score_response
from recipe.phimm.reward.diarization_eval import diarization_eval, measure_der, measure_wder


def segment(speaker, start, end):
    return {"speaker": speaker, "start_s": start, "end_s": end}


def words(*items):
    return [{"speaker": speaker, "text": text} for speaker, text in items]


def reference():
    return {
        "language": "English",
        "reference_segments": [segment("A", 0, 1), segment("B", 1, 2)],
        "reference_words": words(("A", "hello"), ("B", "world")),
        "evaluation_duration_s": 2.0,
    }


def test_speaker_permutation_is_perfect_for_both_metrics():
    prediction = {
        "segments": [segment("Y", 0, 1), segment("X", 1, 2)],
        "words": words(("Y", "hello"), ("X", "world")),
    }
    result = diarization_eval([json.dumps(prediction)], extra_info=reference())
    assert result["score"] == 1
    assert result["der"] == result["wder"] == 0
    assert result["der_n_ref"] == result["wder_n_ref"] == 2


def test_der_miss_false_alarm_confusion_and_overlap():
    ref = [segment("A", 0, 2), segment("B", 1, 3)]
    result = measure_der(ref, [segment("X", 0, 4)])
    assert result == {
        "der": 0.75,
        "der_n_ref": 4,
        "der_n_err": 3,
        "der_n_miss": 1,
        "der_n_fa": 1,
        "der_n_confusion": 1,
    }


def test_der_unions_same_speaker_without_filling_gaps_and_clips_duration():
    ref = [segment("A", 0, 1), segment("A", 0.5, 2), segment("A", 3, 5)]
    result = measure_der(ref, [segment("X", 0, 8)], evaluation_duration_s=4)
    assert result["der_n_ref"] == 3
    assert result["der_n_fa"] == 1
    assert result["der"] == pytest.approx(1 / 3)


def test_zero_reference_retains_false_alarms_for_pooling():
    result = measure_der([], [segment("X", 0, 2)])
    assert result["der"] is None
    assert result["der_n_err"] == 2
    assert result["der_n_ref"] == 0


def test_wder_speaker_switch_includes_substitutions_but_excludes_insertions_and_deletions():
    ref = words(("A", "hello"), ("A", "red"), ("B", "blue"), ("B", "world"))
    hyp = words(("X", "hello"), ("Y", "green"), ("Y", "blue"), ("Y", "world"), ("Z", "extra"))
    result = measure_wder(ref, hyp)
    assert result == {"wder": 0.25, "wder_n_err": 1, "wder_n_ref": 4}
    assert measure_wder(ref, hyp[:1])["wder_n_ref"] == 1


def test_wder_normalization_preserves_speaker_for_expanded_words():
    ref = words(("A", "Hello!"), ("B", "world"))
    hyp = words(("X", "hello"), ("Y", "WORLD."))
    assert measure_wder(ref, hyp)["wder"] == 0
    assert measure_wder(ref, [])["wder"] is None


@pytest.mark.parametrize("prediction", ["plain transcript", "[]", "{}", '{"text":"hello"}', "null"])
def test_missing_diarization_is_unavailable_not_perfect(prediction):
    result = diarization_eval(prediction, extra_info=reference())
    assert result["score"] == 0
    assert result["der"] is result["wder"] is None
    assert result["der_unavailable"] == result["wder_unavailable"] == 1


def test_explicit_empty_predictions_count_as_missed_speech():
    result = diarization_eval('{"segments":[],"words":[]}', extra_info=reference())
    assert result["der"] == 1
    assert result["der_n_miss"] == 2
    assert result["der_unavailable"] == result["wder_unavailable"] == 0
    assert result["wder"] is None


def test_partial_support_and_selected_wder_reward():
    prediction = json.dumps({"words": words(("X", "hello"), ("Y", "world"))})
    result = diarization_eval(prediction, extra_info=reference(), reward_metric="wder")
    assert result["score"] == 1
    assert result["der"] is None
    assert result["wder"] == 0


@pytest.mark.parametrize("start,end", [(-1, 2), (2, 1), (float("nan"), 2), (0, float("inf")), (True, 2)])
def test_invalid_reference_raises_but_invalid_prediction_is_unavailable(start, end):
    bad = [segment("A", start, end)]
    info = reference() | {"reference_segments": bad}
    with pytest.raises(ValueError):
        diarization_eval("{}", extra_info=info)
    result = diarization_eval(json.dumps({"segments": bad}), extra_info=reference())
    assert result["der_unavailable"] == 1


def test_multiple_responses_do_not_join_local_speakers_and_timestamps():
    result = diarization_eval(['{"segments":[]}', '{"segments":[]}'], extra_info=reference())
    assert result["der"] is None


def test_svad_restores_local_timestamps_and_keeps_one_global_speaker_mapping():
    info = reference() | {"svad_bounds": [[0, 1], [1, 2]]}
    responses = [
        json.dumps({"segments": [segment("X", 0, 1)], "words": words(("X", "hello"))}),
        json.dumps({"segments": [segment("Y", 0, 1)], "words": words(("Y", "world"))}),
    ]
    result = diarization_eval(responses, extra_info=info)
    assert result["der"] == result["wder"] == 0
    # Resetting every segment to speaker X must not be rescued by local mapping.
    responses[1] = json.dumps({"segments": [segment("X", 0, 1)], "words": words(("X", "world"))})
    result = diarization_eval(responses, extra_info=info)
    assert result["der"] == result["wder"] == 0.5


def test_svad_single_segment_nonzero_offset_and_local_duration_clipping():
    info = {
        "reference_segments": [segment("A", 10, 11)],
        "reference_words": words(("A", "hello")),
        "evaluation_duration_s": 12,
        "svad_bounds": [[10, 11]],
    }
    prediction = json.dumps({"segments": [segment("X", 0, 5)], "words": words(("X", "hello"))})
    result = diarization_eval([prediction], extra_info=info)
    assert result["der"] == result["wder"] == 0
    assert result["der_n_ref"] == 1


def test_svad_does_not_remove_reference_speech_in_dropped_gaps():
    info = {
        "reference_segments": [segment("A", 0, 4)],
        "reference_words": words(("A", "hello"), ("A", "world")),
        "evaluation_duration_s": 4,
        "svad_bounds": [[0, 1], [3, 4]],
    }
    predictions = [
        json.dumps({"segments": [segment("X", 0, 1)], "words": words(("X", text))}) for text in ["hello", "world"]
    ]
    result = diarization_eval(predictions, extra_info=info)
    assert result["der_n_ref"] == 4
    assert result["der_n_miss"] == 2
    assert result["der"] == 0.5


def test_svad_partial_metric_support_requires_all_segment_predictions():
    info = reference() | {"svad_bounds": [[0, 1], [1, 2]]}
    predictions = [
        json.dumps({"segments": [segment("X", 0, 1)], "words": words(("X", "hello"))}),
        json.dumps({"words": words(("Y", "world"))}),
    ]
    result = diarization_eval(predictions, extra_info=info, reward_metric="wder")
    assert result["wder"] == 0 and result["score"] == 1
    assert result["der"] is None and result["der_unavailable"] == 1
    assert diarization_eval(predictions[:1], extra_info=info)["wder"] is None
    assert diarization_eval([predictions[0], "invalid JSON"], extra_info=info)["wder"] is None


@pytest.mark.parametrize("bounds", [[], [[0, 2], [1, 2]], [[-1, 0]], [[0, 3]], [[0]], [[0, float("nan")]]])
def test_svad_malformed_reference_bounds_raise(bounds):
    with pytest.raises(ValueError):
        diarization_eval("{}", extra_info=reference() | {"svad_bounds": bounds})


def test_text_segments_use_the_same_speaker_and_decoding_bounds():
    info = reference() | {"svad_bounds": [[0, 1], [1, 2]]}
    result = diarization_eval(["hello", "world"], extra_info=info, prediction_format="text")
    assert result["single_speaker_baseline"] == 1
    assert result["der"] == result["wder"] == 0.5
    assert result["der_n_confusion"] == 1
    assert result["der_n_miss"] == result["der_n_fa"] == 0
    assert result["score"] == 0.5


def test_text_segments_strip_asr_markup_and_empty_nonspeech_outputs():
    info = reference() | {"svad_bounds": [[0, 1], [1, 2]]}
    outputs = ["<nonspeech>", "<src=English><tgt=English>\n<verbatim>world</verbatim>"]
    result = diarization_eval(outputs, extra_info=info, prediction_format="text", version=2609)
    assert result["der_n_miss"] == 1
    assert result["der_n_confusion"] == 0
    assert result["der_n_fa"] == 0
    assert result["wder_n_ref"] == 1
    assert result["wder"] == 0


def test_text_single_speaker_example_counts_overlap_confusion_and_silence():
    info = {
        "reference_segments": [segment("A", 0, 2), segment("B", 1, 3)],
        "reference_words": words(("A", "hello"), ("B", "world")),
        "evaluation_duration_s": 4,
        "svad_bounds": [[0, 1], [1, 3], [3, 4]],
    }
    result = diarization_eval(["hello", "world", "noise"], extra_info=info, prediction_format="text")
    assert result["der"] == 0.75
    assert result["der_n_miss"] == result["der_n_fa"] == result["der_n_confusion"] == 1


def test_text_prediction_does_not_use_reference_labels_or_reference_times():
    from recipe.phimm.reward.diarization_eval import _text_recording_prediction

    info = reference()
    result = _text_recording_prediction(["hello", "world"], [[5, 6], [8, 9]], info, speaker="speaker_0", version=2609)
    assert result == {
        "segments": [segment("speaker_0", 5, 6), segment("speaker_0", 8, 9)],
        "words": words(("speaker_0", "hello"), ("speaker_0", "world")),
    }
    assert (
        _text_recording_prediction(["hello", "world"], [[5, 6], [8, 9]], {}, speaker="speaker_0", version=2609)
        == result
    )


def test_text_empty_predictions_remain_available_and_count_missed_speech():
    result = diarization_eval(
        ["", ""], extra_info=reference() | {"svad_bounds": [[0, 1], [1, 2]]}, prediction_format="text"
    )
    assert result["der"] == 1
    assert result["der_unavailable"] == result["wder_unavailable"] == 0
    assert result["wder"] is None
    assert result["score"] == 0


def test_text_without_timestamps_can_score_wder_but_not_der():
    info = reference()
    info.pop("evaluation_duration_s")
    result = diarization_eval("hello world", extra_info=info, prediction_format="text")
    assert result["der"] is None
    assert result["wder"] == 0.5


def test_text_mode_rejects_missing_segment_outputs():
    result = diarization_eval(
        "hello", extra_info=reference() | {"svad_bounds": [[0, 1], [1, 2]]}, prediction_format="text"
    )
    assert result["der"] is result["wder"] is None


@pytest.mark.parametrize("metadata", [{"evaluation_duration_s": 2}, {"seg_start": 0, "seg_end": 2}])
def test_text_single_segment_falls_back_to_recorded_duration_or_decoder_bounds(metadata):
    info = reference()
    info.pop("evaluation_duration_s")
    result = diarization_eval("hello world", extra_info=info | metadata, prediction_format="text")
    assert result["der"] == result["wder"] == 0.5


def test_text_evaluation_reuses_existing_asr_wer_counts_without_changing_der_reward():
    from recipe.phimm.reward.asr_eval import measure_openasr_en_wer

    info = reference() | {"svad_bounds": [[0, 1], [1, 2]]}
    result = diarization_eval(
        ["Hello!", "earth"], "hello world", extra_info=info, prediction_format="text", compute_wer=True
    )
    expected = measure_openasr_en_wer("Hello! earth", "hello world")
    assert {key: result[key] for key in ("wer", "n_err", "n_ref")} == {
        key: expected[key] for key in ("wer", "n_err", "n_ref")
    }
    assert result["score"] == 1 - result["der"]


def test_pooling_uses_speaker_seconds_and_aligned_words():
    results = [
        diarization_eval('{"segments":[],"words":[]}', extra_info=reference()),
        diarization_eval(
            json.dumps({"segments": [segment("X", 0, 98)], "words": words(("X", "hello"))}),
            extra_info={
                "reference_segments": [segment("A", 0, 98)],
                "reference_words": words(("A", "hello")),
            },
        ),
    ]
    extras = {key: [result.get(key) for result in results] for key in set().union(*results)}
    pooled = aggregate_error_rates(["a", "a"], extras)
    assert pooled["val-core/a/der_p_err/mean@1"] == 0.02
    assert pooled["val-core/a/wder_p_err/mean@1"] == 0


def test_pooling_does_not_report_partial_corpus_as_complete():
    result = aggregate_error_rates(
        ["a", "a", "b"],
        {
            "der_n_err": [0, None, 1],
            "der_n_ref": [2, None, 2],
            "der_unavailable": [0, 1, 0],
            "wder_n_err": [0, 1, 1],
            "wder_n_ref": [2, 2, 2],
            "wder_unavailable": [0, 0, 0],
        },
    )
    assert "val-core/a/der_p_err/mean@1" not in result
    assert result["val-core/a/wder_p_err/mean@1"] == 0.25
    assert result["val-core/b/der_p_err/mean@1"] == 0.5


@pytest.mark.parametrize(
    "dataset,manifest",
    [
        ("dai_asr_i18n_public_en", "data.jsonl"),
        ("dai_asr_i18n_public_en_validation_smoke", "data-validation-smoke.jsonl"),
        ("dai_asr_i18n_public_en_internal_test", "data-internal-test.jsonl"),
    ],
)
def test_native_eval_config_dispatches_to_diarization_and_preserves_row_fields(dataset, manifest):
    import recipe.phimm

    OmegaConf.register_new_resolver("eval", lambda text: eval(text, {"__builtins__": {}}, {}), replace=True)
    directory = Path(recipe.phimm.__file__).parent / "config/v2611_dev/eval"
    verl_configs = Path(__file__).resolve().parents[3] / "verl/trainer/config"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        config = compose(
            config_name="eval_diarization",
            overrides=[f"hydra.searchpath=[file://{verl_configs}]", f"data/val_data/diarization={dataset}"],
        )
    assert config.trainer.val_only
    assert "val_openasr_en" in config.rewards
    assert config.reward_map.dai_asr_i18n_en_public == "val_diarization"
    data = config.data.val_data[0]
    assert data.dataset_name == "jsonl"
    assert data.jsonl_paths.endswith("/" + manifest)
    assert {"reference_segments", "reference_words", "evaluation_duration_s", "svad_bounds", "svad_source"} <= set(
        data.post_process.verl_format.extra_keys
    )
    assert "/svad30/" in data.jsonl_paths
    assert data.add_task_info.style == "verb"
    assert "add_field" not in data.post_process
    assert config.rewards.val_diarization.reward_kwargs.prediction_format == "text"
    row = {
        "data_source": "dai_asr_i18n_en_public",
        "reward_model": {"ground_truth": "hello world"},
        "extra_info": reference() | {"svad_bounds": [[0, 1], [1, 2]]},
    }
    assert score_response(config, row, ["hello", "world"])["score"] == 0.5
