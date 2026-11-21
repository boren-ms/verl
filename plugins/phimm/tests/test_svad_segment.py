"""SVAD preparation keeps one full-reference parent and uses existing audio readers."""

import json

import datasets
import numpy as np
import pytest
import soundfile as sf
from omegaconf import OmegaConf
from recipe.phimm.agentic.dataset import ASRDataset
from recipe.phimm.tools.offline_audio import materialize
from recipe.phimm.tools.svad_segment import prepare_manifest, segment_row, svad_segment
from recipe.phimm.utils.audio import _is_chunk_spec, _is_time_chunk_spec, load_raw_audio

pytest.importorskip("onnxruntime")
pytest.importorskip("scipy")


def test_parent_reference_and_selectors_reload_with_existing_reader(tmp_path):
    path = tmp_path / "audio.wav"
    audio = np.linspace(-0.2, 0.2, 8000, dtype=np.float32)
    sf.write(path, np.column_stack((audio, np.zeros_like(audio))), 8000, subtype="FLOAT")
    original = {"audio_path": str(path), "id": "parent", "text": "full original reference", "keywords": ["original"]}
    result = segment_row(original)
    assert original["audio_path"] == str(path)
    assert result["text"] == original["text"]
    assert result["id"] == "parent" and result["keywords"] == ["original"]
    assert len(result["audio_path"]) == 1
    spec = result["audio_path"][0]
    assert _is_time_chunk_spec(spec) and not _is_chunk_spec(spec)
    decoded, sr = load_raw_audio({"audio_path": spec})
    assert sr == 8000
    np.testing.assert_array_equal(decoded, audio / 2)
    assert list(tmp_path.iterdir()) == [path]


def test_offline_preparation_releases_decoded_parent_cache(tmp_path):
    from recipe.phimm.utils.audio import sf_read

    path = tmp_path / "audio.wav"
    sf.write(path, np.zeros(4000), 16000)
    segment_row({"audio_path": str(path), "text": "parent"})
    assert len(sf_read.cache) == 0


def test_prepared_dataset_uses_selectors_instead_of_inline_parent_audio(tmp_path):
    path = tmp_path / "audio.wav"
    sf.write(path, np.zeros(4000), 16000)
    ds = datasets.Dataset.from_list(
        [
            {
                "audio_path": str(path),
                "text": "full reference",
                "audio": {"array": [0.5] * 16, "sampling_rate": 16000},
            }
        ]
    )
    prepared = svad_segment(ds)
    assert "audio" not in prepared.column_names
    assert prepared[0]["text"] == "full reference"
    assert len(load_raw_audio({"audio_path": prepared[0]["audio_path"][0]})[0]) == 4000


def test_silence_fallback_has_bounded_nonempty_parent_list(tmp_path):
    path = tmp_path / "silent.wav"
    sf.write(path, np.zeros(16000 * 4 + 8000), 16000)
    row = {"audio_path": str(path), "text": "parent reference"}
    result = segment_row(row, max_len_sec=2)
    assert result["svad_method"] == "fixed_fallback"
    assert result["svad_bounds"] == [(0, 2), (2, 4), (4, 4.5)]
    assert result["text"] == row["text"]
    for spec in result["audio_path"]:
        audio, sr = load_raw_audio({"audio_path": spec})
        assert 0 < len(audio) / sr <= 2
    with pytest.raises(RuntimeError, match="no retained speech"):
        segment_row(row, max_len_sec=2, empty_policy="error")


def test_svad_context_overlaps_are_removed_and_maximum_enforced(tmp_path, monkeypatch):
    from recipe.phimm.tools import svad_segment as module

    path = tmp_path / "audio.wav"
    sf.write(path, np.linspace(-0.1, 0.1, 16000 * 6), 16000)

    class Chunker:
        def chunk(self, audio, sr):
            return [(-0.3, 3.2), (3.0, 6.4)]

    monkeypatch.setattr(module, "_chunker", lambda *args: Chunker())
    result = segment_row({"audio_path": str(path), "text": "one reference"}, max_len_sec=2)
    bounds = result["svad_bounds"]
    assert bounds[0][0] == 0 and bounds[-1][1] == 6
    assert sum(end - start for start, end in bounds) == pytest.approx(6)
    assert all(start < end <= start + 2 for start, end in bounds)
    assert all(first[1] <= second[0] for first, second in zip(bounds, bounds[1:], strict=False))
    assert all(len(load_raw_audio({"audio_path": spec})[0]) <= 32000 for spec in result["audio_path"])


def test_preparation_streams_parent_rows_in_same_order_for_worker_counts(tmp_path):
    path = tmp_path / "source.wav"
    sf.write(path, np.zeros(32000), 16000)
    source = tmp_path / "input.jsonl"
    source.write_text(
        "".join(json.dumps({"audio_path": "source.wav", "text": f"reference {i}"}) + "\n" for i in range(3))
    )
    outputs = []
    for workers in (1, 2):
        destination = tmp_path / f"output{workers}.jsonl"
        counts = prepare_manifest(src=source, dst=destination, n_workers=workers, max_len_sec=0.7)
        assert counts["parents"] == 3 and counts["segments"] == 9
        outputs.append(destination.read_text())
    assert outputs[0] == outputs[1]
    assert [json.loads(line)["text"] for line in outputs[0].splitlines()] == [f"reference {i}" for i in range(3)]
    assert sorted(tmp_path.glob("*.wav")) == [path]


def test_failed_audio_and_existing_destination_do_not_replace_manifest(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps({"audio_path": "missing.wav"}) + "\n")
    destination = tmp_path / "output.jsonl"
    with pytest.raises(RuntimeError, match="missing.wav"):
        prepare_manifest(src=source, dst=destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".output.jsonl.*"))
    destination.write_text("existing\n")
    with pytest.raises(FileExistsError):
        prepare_manifest(src=source, dst=destination)
    assert destination.read_text() == "existing\n"


def test_offline_snapshot_keeps_full_reference_and_loads_asr_segments(tmp_path):
    path = tmp_path / "audio.wav"
    sf.write(path, np.zeros(32000), 16000)
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({"audio_path": str(path), "text": "full original reference", "id": "parent"}) + "\n")
    source = {
        "dataset_name": "jsonl",
        "jsonl_paths": str(raw),
        "add_task_info": {"style": "verb", "language": "English"},
        "post_process": {"verl_format": {"prompt_key": "prompt"}},
    }
    config = OmegaConf.create(
        {
            "data": {
                "train_files": "train",
                "val_files": "val",
                "version": 2609,
                "train_data": [source],
                "val_data": [source],
            },
            "asr": {"context_aware": True},
            "offline_data": {"svad_segment": {"max_len_sec": 0.7}},
        }
    )
    before = OmegaConf.to_container(config, resolve=True)
    snapshot = materialize(config, tmp_path / "prepared")
    assert OmegaConf.to_container(config, resolve=True) == before
    assert "offline_data" not in snapshot
    assert snapshot.data.val_data == config.data.val_data
    train = ASRDataset("train", None, snapshot.data)
    assert len(train) == 1
    parent = train[0]
    assert parent["reward_model"]["ground_truth"] == "full original reference"
    assert len(parent["asr_segments"]) == 3
    assert all(segment["asr_full_reference"] for segment in parent["asr_segments"])
    assert len(ASRDataset("val", None, snapshot.data)[0]["asr_segments"]) == 1
    assert sorted(tmp_path.rglob("*.wav")) == [path]


@pytest.mark.parametrize(
    "options", [{"max_len_sec": float("nan")}, {"min_seg_sec": 0}, {"num_threads": 0}, {"empty_policy": "unknown"}]
)
def test_invalid_options(options):
    with pytest.raises(ValueError):
        svad_segment(datasets.Dataset.from_list([{"audio_path": "A.wav"}]), **options)


def test_mutually_exclusive_offline_plans(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        materialize(OmegaConf.create({"offline_data": {"svad_segment": {}, "random_segment": {}}}), tmp_path / "out")


@pytest.mark.parametrize("direct", [False, True], ids=["earnings", "long"])
@pytest.mark.parametrize(
    "preset, dataset, rows, batch, epochs, steps",
    [
        ("remax_earnings_long_svad30_text_h4", "earnings_long_svad30", 739, 4, 2, 368),
        ("remax_earnings_short_contiguous10m_text_h4", "earnings_contiguous10m", 806, 4, 2, 402),
        ("remax_earnings_short_contiguous_full_text_h4", "earnings_contiguous_full", 180, 2, 4, 360),
    ],
)
def test_long_recipes_preserve_earnings_controls_and_use_text_history4(
    direct, preset, dataset, rows, batch, epochs, steps
):
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from recipe.phimm.agentic.context_prompt import ContextPrompt
    from recipe.phimm.utils.offline_data import active_audio_transforms

    directory = Path(__file__).resolve().parents[1] / "src/recipe/phimm/config/v2611_dev/earnings"
    OmegaConf.register_new_resolver("eval", lambda text: eval(text, {"__builtins__": {}}, {}), replace=True)
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        baseline = compose(config_name="base_earning")
        reference = compose(config_name="prompt_ctx/remax_earnings_short_contiguous5_text_history4")
    with initialize_config_dir(config_dir=str(directory / "long" if direct else directory), version_base=None):
        candidate = compose(config_name=preset if direct else f"long/{preset}")
    OmegaConf.resolve(candidate)
    assert candidate.trainer.experiment_name == preset
    assert candidate.asr.context_aware.enable is True
    assert candidate.asr.context_aware == reference.asr.context_aware
    assert candidate.asr.context_aware.mode == "text"
    assert candidate.asr.context_aware.max_history == 4
    options = candidate.asr.context_aware
    context = ContextPrompt(
        options.mode,
        max_history=options.max_history,
        max_terms=options.max_terms,
        filter_common=options.filter_common,
        filter_numbers=options.filter_numbers,
        filter_fillers=options.filter_fillers,
        extra_common_words=options.common_words,
    )
    for text in ("OpenAI.", "NVIDIA.", "Microsoft.", "Qwen.", "Anthropic."):
        context.observe(text)
    assert context.context == "NVIDIA.\nMicrosoft.\nQwen.\nAnthropic."
    for key in (
        "actor_rollout_ref.model",
        "algorithm",
        "rewards",
        "data.val_data",
        "actor_rollout_ref.rollout.n",
        "actor_rollout_ref.rollout.temperature",
        "actor_rollout_ref.rollout.val_kwargs",
    ):
        assert OmegaConf.select(candidate, key) == OmegaConf.select(baseline, key), key
    actor = OmegaConf.to_container(candidate.actor_rollout_ref.actor, resolve=True)
    baseline_actor = OmegaConf.to_container(baseline.actor_rollout_ref.actor, resolve=True)
    assert actor.pop("ppo_mini_batch_size") == batch
    baseline_actor.pop("ppo_mini_batch_size")
    assert actor.pop("ppo_max_token_len_per_gpu") == candidate.data.max_model_length
    baseline_actor.pop("ppo_max_token_len_per_gpu")
    assert actor == baseline_actor
    rollout = candidate.actor_rollout_ref.rollout
    assert rollout.prompt_length == candidate.data.max_prompt_length == reference.data.max_prompt_length == 4096
    assert rollout.max_model_len == candidate.data.max_model_length == reference.data.max_model_length == 4864
    assert rollout.max_model_len == rollout.prompt_length + rollout.response_length
    assert rollout.enable_chunked_prefill is True
    assert rollout.max_num_batched_tokens == 8192
    assert candidate.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu == rollout.max_model_len
    assert candidate.data.train_batch_size == candidate.data.gen_batch_size == batch
    assert candidate.asr.expected_training_rows == rows
    assert sum(source.sample for group in candidate.data.mix for source in group.datasource) == rows
    assert candidate.trainer.total_training_steps is None
    assert candidate.trainer.total_epochs == epochs
    assert candidate.trainer.resume_mode == "disable" and candidate.trainer.resume_from_path is None
    assert candidate.trainer.nnodes == 2 and candidate.trainer.n_gpus_per_node == 8
    assert batch * rollout.n % (candidate.trainer.nnodes * candidate.trainer.n_gpus_per_node) == 0
    batches = candidate.asr.expected_training_rows // candidate.data.train_batch_size
    assert batches * candidate.trainer.total_epochs == candidate.asr.expected_training_steps == steps
    assert active_audio_transforms(candidate) == {}
    assert len(candidate.data.train_data) == 5
    for source in candidate.data.train_data:
        assert "pre_process" not in source
        assert f"{dataset}_20261117_v1/" in source.jsonl_paths
        name = source.post_process.add_field.fields.data_source
        assert candidate.reward_map[name] == "train_librispeech"
