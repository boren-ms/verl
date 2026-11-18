"""Exercise native Hydra overrides and the raw-row data path without a GPU stack."""

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf, open_dict


@pytest.fixture
def compose_recipe():
    import recipe.phimm

    directory = Path(recipe.phimm.__file__).parent / "config/v2611_dev"
    OmegaConf.register_new_resolver("eval", lambda text: eval(text, {"__builtins__": {}}, {}), replace=True)

    def load(kind="main", overrides=(), name=None):
        name = name or ("remax_main_base3a_2611r2" if kind == "main" else "eval_openfull_zip")
        with initialize_config_dir(config_dir=str(directory / kind), version_base=None):
            return compose(config_name=name, overrides=list(overrides))

    return load


@pytest.mark.parametrize("context", [True, False])
def test_fl_earnings_training_sources_have_registered_scorers(compose_recipe, context):
    suffix = "context" if context else "no_context"
    config = compose_recipe(kind="earnings", name=f"remax_fl_earnings_short_random_merge23_x4_{suffix}")
    OmegaConf.resolve(config)
    sources = {source.post_process.add_field.fields.data_source for group in config.data.train_data for source in group}
    assert "quartr_fl_segments" in sources
    for source in sources:
        scorer = config.reward_map.get(source, source)
        assert scorer in config.rewards, f"No scorer for {source!r}"
    assert config.asr.context_aware.enable is context


def test_context_openlong_preserves_full_suite_and_evaluation_settings(compose_recipe):
    overrides = ["actor_rollout_ref.model.path=/models/sft38000", "trainer.experiment_name=context-test"]
    base = compose_recipe(kind="eval", name="eval_openlong_zip", overrides=overrides)
    config = compose_recipe(kind="eval", name="eval_openlong_context_zip", overrides=overrides)
    OmegaConf.resolve(config)
    assert config.data.val_data == base.data.val_data
    assert len(config.data.val_data[0]) == 3
    assert all(source.val_batch_size == -1 for source in config.data.val_data[0])
    assert config.trainer.val_only and config.trainer.resume_mode == "disable"
    assert config.actor_rollout_ref.model.lora_rank == 0
    assert config.asr.context_aware.enable and not base.asr.context_aware.enable
    assert config.actor_rollout_ref.rollout.prompt_length == 2048
    assert config.data.max_model_length == 2816


def test_context_training_preserves_recipe_and_enables_ordered_segments(compose_recipe):
    base = compose_recipe()
    config = compose_recipe(overrides=["+asr_training=context_aware", "trainer.experiment_name=context-train"])
    OmegaConf.resolve(config)
    assert config.asr.context_aware.enable and not base.asr.context_aware.enable
    assert list(config.offline_data.random_segment.segment_count) == [1, 2]
    assert config.offline_data.random_segment.segment_len is None
    assert config.data.max_audio_dur is None
    assert config.actor_rollout_ref.rollout.prompt_length == 2048
    assert config.data.max_model_length == 2816
    assert not config.trainer.val_only
    for field in ("train_data", "val_data", "mix", "train_batch_size"):
        assert config.data[field] == base.data[field]
    assert config.rewards == base.rewards
    assert config.reward_map == base.reward_map
    assert config.algorithm == base.algorithm
    assert config.trainer.default_hdfs_dir.endswith("/context-train")


@pytest.mark.parametrize("tag, mode, bounds", [("sc", "segment_count", [1, 2]), ("sl", "segment_len", [5, 30])])
def test_base3a_segment_variants_inherit_training_and_validation(compose_recipe, tag, mode, bounds):
    import recipe.phimm
    from recipe.phimm.agentic.dataset import _flatten_data_confs
    from recipe.phimm.utils.offline_data import active_audio_transforms

    base = compose_recipe()
    name = f"remax_main_base3a_2611r2_{tag}"
    config = compose_recipe(kind="rand_segment", name=name)
    OmegaConf.resolve(config)
    registry_path = Path(recipe.phimm.__file__).parent / "config/data/train_data/offline_audio/preparation.json"
    registry = json.loads(registry_path.read_text())
    preparation = next(entry for entry in registry["configs"] if entry["name"] == name)
    other_mode = "segment_len" if mode == "segment_count" else "segment_count"
    assert preparation["options"][mode] == bounds
    assert preparation["options"][other_mode] is None
    assert preparation["options"]["seed"] == 42
    assert not active_audio_transforms(config)
    assert "offline_data" not in config
    sources = list(_flatten_data_confs(config.data.train_data))
    assert [source.jsonl_paths for source in sources] == [
        registry["tasks"][task]["uri"] for task in preparation["tasks"]
    ]
    assert all(not source.get("pre_process") for source in sources)
    assert config.data.max_audio_dur is None
    for field in ("val_data", "mix", "train_batch_size"):
        assert config.data[field] == base.data[field]
    assert config.rewards == base.rewards
    assert config.reward_map == base.reward_map
    assert config.actor_rollout_ref.model.path == base.actor_rollout_ref.model.path
    assert config.actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.mode == 0
    assert config.asr.expected_training_rows == base.asr.expected_training_rows == 55664
    assert config.asr.expected_training_steps == base.asr.expected_training_steps == 868
    assert config.trainer.experiment_name == name
    assert config.trainer.default_hdfs_dir.endswith(f"/{name}")


def test_base3a_async_inherits_base_except_trainer(compose_recipe):
    base = compose_recipe()
    name = "remax_main_base3a_2611r2_async"
    config = compose_recipe(kind="async", name=name)
    assert config.trainer.use_v1
    assert config.trainer.v1.trainer_mode == "colocate_async"
    assert config.trainer.v1.trainer_class == "recipe.phimm.agentic.trainer.ASRTrainerColocateAsync"
    assert config.trainer.v1.colocate_async.num_warmup_batches == 1
    assert config.asr.colocate_queue_on_driver
    assert config.trainer.nnodes == base.trainer.nnodes == 2
    assert config.trainer.experiment_name == name
    assert config.trainer.default_hdfs_dir.endswith(f"/{name}")
    assert config.trainer.default_local_dir.endswith(f"/{name}")
    base.trainer.experiment_name = name
    base.trainer.v1.trainer_mode = "colocate_async"
    base.trainer.v1.trainer_class = "recipe.phimm.agentic.trainer.ASRTrainerColocateAsync"
    with open_dict(base.asr):
        base.asr.colocate_queue_on_driver = True
    assert OmegaConf.to_container(config, resolve=True) == OmegaConf.to_container(base, resolve=True)


def test_native_overrides_propagate_from_any_working_directory(compose_recipe, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config = compose_recipe(
        overrides=[
            "trainer.project_name=test-project",
            "trainer.experiment_name=test-run",
            "actor_rollout_ref.model.path=/models/audio",
            "actor_rollout_ref.model.lora_rank=16",
            "data.train_batch_size=64",
            "data.max_prompt_length=512",
            "data.max_response_length=256",
            "actor_rollout_ref.rollout.n=2",
            "actor_rollout_ref.rollout.seed=7",
        ]
    )
    OmegaConf.resolve(config)
    actor = config.actor_rollout_ref
    assert actor.actor._target_ == "verl.workers.config.FSDPActorConfig"
    assert actor.rollout._target_ == "verl.workers.config.RolloutConfig"
    assert actor.actor.ppo_mini_batch_size == config.data.gen_batch_size == 64
    assert actor.actor.rollout_n == actor.ref.rollout_n == 2
    assert actor.rollout.prompt_length == 512
    assert actor.rollout.response_length == 256
    assert actor.rollout.max_num_batched_tokens == 32 * (512 + 256)
    assert actor.model.lora_alpha == 32
    assert actor.rollout.engine_kwargs.vllm.seed == 7
    assert actor.rollout.engine_kwargs.vllm.compilation_config.mode == 0
    assert actor.rollout.trace.experiment_name == "test-run"
    assert actor.rollout.prometheus.served_model_name == "/models/audio"
    assert config.trainer.default_hdfs_dir.endswith("/test-project/test-run")
    assert config.asr.validation_remote_dir == config.trainer.default_hdfs_dir + "/val_data_gen"
    assert config.trainer.validation_data_dir == config.trainer.default_local_dir + "/val_data_gen"
    assert "source_only_keys" not in config.asr


def test_native_schema_rejects_legacy_and_misspelled_settings(compose_recipe):
    for override in ("actor_rollout_ref.actor.tis_imp_ratio_cap=8", "trainer.save_frequence=20"):
        with pytest.raises(ConfigCompositionException):
            compose_recipe(overrides=[override])


def test_evaluation_composes_native_base_without_training_manifests(compose_recipe):
    config = compose_recipe(
        "eval",
        [
            "trainer.experiment_name=eval-test",
            "actor_rollout_ref.model.path=/models/merged",
            "trainer.nnodes=3",
        ],
    )
    OmegaConf.resolve(config)
    assert config.trainer.val_only
    assert config.trainer.resume_mode == "disable"
    assert config.actor_rollout_ref.model.lora_rank == 0
    assert config.actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.mode == 0
    assert "mix" not in config.data
    assert "val_reward" not in config
    assert "expected_training_steps" not in config.asr
    assert config.data.val_batch_size == 768
    assert config.data.train_data == config.data.val_data
    configs = [item for group in config.data.val_data for item in (group if OmegaConf.is_list(group) else [group])]
    assert len(configs) == 35
    sources = {item.post_process.add_field.fields.data_source for item in configs}
    assert {"eslo_telephony_fr", "accueil_ubs_fr"} <= sources
    for source in sources:
        scorer = config.reward_map.get(source, source)
        assert scorer.startswith("val_"), source
        assert scorer in config.rewards, source


def test_evaluation_requires_explicit_model_and_experiment(compose_recipe):
    config = compose_recipe("eval")
    assert config.actor_rollout_ref.model.path is None
    assert config.trainer.experiment_name is None


def test_raw_dataset_mixes_without_loading_audio_or_tokenizing(monkeypatch):
    import datasets
    from recipe.phimm.agentic import dataset

    sources = {
        name: datasets.Dataset.from_list(
            [
                {
                    "data_source": name,
                    "prompt": [{"role": "user", "content": f"{name}-{i}<audio>"}],
                    "extra_info": {"audio_path": f"{name}-{i}"},
                }
                for i in range(4)
            ]
        )
        for name in ("a", "b")
    }
    monkeypatch.setattr(dataset, "create_audio_dataset", lambda name, version: sources[name])
    config = OmegaConf.create(
        {
            "train_files": "train",
            "val_files": "val",
            "version": 2609,
            "shuffle": False,
            "train_batch_size": 2,
            "train_data": [{"name": "a"}, [{"name": "b"}]],
            "val_data": [{"name": "a"}],
            "mix": [
                {
                    "shuffle": True,
                    "datasource": [
                        {"name": "a", "sample": 3, "shuffle": True},
                        {"name": "b", "sample": 2, "shuffle": True},
                    ],
                }
            ],
        }
    )
    # Any dataset-side audio preprocessing/tokenization would fail with these sentinels.
    train = dataset.ASRDataset("train", tokenizer=object(), processor=object(), config=config)
    # Golden order obtained from the integration commit's sampler, before refactoring.
    assert [row["prompt"][0]["content"] for row in train.ds] == [
        "b-2<audio>",
        "a-1<audio>",
        "b-1<audio>",
        "a-2<audio>",
        "a-3<audio>",
    ]
    assert len(train) == 5
    row = train[0]
    assert row["raw_prompt"] == row["prompt"]
    assert row["asr_segments"] == [dict(train.ds[0])]
    assert not row["asr_validate"]
    assert "input_ids" not in row and "multi_modal_inputs" not in row
    val = dataset.ASRDataset("val", tokenizer=object(), config=config, max_samples=2)
    assert len(val) == 2
    assert val[0]["asr_validate"]


def test_jsonl_pipeline_preserves_tags_keywords_prompt_and_ranked_limit(tmp_path):
    from recipe.phimm.data.dataset import create_audio_dataset

    path = tmp_path / "rows.jsonl"
    rows = [
        {"audio_path": "audios/one.flac", "display": "Hello<NE> Morgan Stanley</NE>."},
        {"audio_path": "audios/two.flac", "display": "Second row"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = create_audio_dataset(
        dataset_name="jsonl",
        jsonl_paths=str(path),
        version=2609,
        pre_process={
            "path_map": {"field": "audio_path", "src_part": "audios/", "dst_part": "/local/"},
            "rename_fields": {"mappings": {"text": "display"}},
            "clean_tagged_text": {},
        },
        add_task_info={"style": "verb", "language": "English", "lang_hint": True},
        post_process={
            "input_egs_limit": 1,
            "add_field": {"fields": {"data_source": "earn_fy27"}},
            "verl_format": {"prompt_key": "prompt"},
        },
    )
    assert len(result) == 1
    row = result[0]
    assert row["audio_path"] == "/local/one.flac"
    assert row["data_source"] == "earn_fy27"
    assert row["reward_model"]["ground_truth"] == "Hello Morgan Stanley."
    assert row["extra_info"]["keywords"] == ["Morgan Stanley"]
    assert row["extra_info"]["language"] == "English"
    assert "prefix" not in row["extra_info"]
    assert row["prompt"][0]["role"] == "user"
    assert "<audio>" in row["prompt"][0]["content"]


def test_incomplete_eval_fails_before_initializing_gpu_workers(compose_recipe):
    from recipe.phimm.main_asr_remax import main

    with pytest.raises(ValueError, match="model.path"):
        main.__wrapped__(compose_recipe("eval"))
    with pytest.raises(ValueError, match="experiment_name"):
        main.__wrapped__(compose_recipe("eval", ["actor_rollout_ref.model.path=/models/audio"]))


def test_removed_transform_is_rejected():
    import datasets
    from recipe.phimm.data.dataset import process_ds

    with pytest.raises(ValueError, match="Unsupported Base3a data transforms"):
        process_ds(datasets.Dataset.from_list([{"text": "hello"}]), svad_explode={"enabled": True})


@pytest.mark.parametrize("async_mode", [False, True])
def test_entrypoint_sends_resolved_config_to_fresh_worker(compose_recipe, monkeypatch, tmp_path, async_mode):
    import pickle
    import subprocess
    import sys
    from types import ModuleType

    import ray
    from recipe.phimm.main_asr_remax import main

    received = []
    options = []
    runner_module = ModuleType("verl.trainer.main_ppo")
    runner_module.TaskRunnerV1 = object()

    def run_ppo(config, **kwargs):
        received.append(config)
        options.append(kwargs)

    runner_module.run_ppo = run_ppo
    monkeypatch.setitem(sys.modules, "verl.trainer.main_ppo", runner_module)
    monkeypatch.setenv("VLLM_CUDART_SO_PATH", "/test/libcudart.so.13")
    monkeypatch.setattr(ray.util, "get_node_ip_address", lambda: "10.0.0.1")
    config = compose_recipe(
        kind="async" if async_mode else "main",
        name="remax_main_base3a_2611r2_async" if async_mode else None,
        overrides=[
            "trainer.experiment_name=fresh-worker-test",
            f"trainer.default_local_dir={tmp_path}",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=3",
        ],
    )
    main.__wrapped__(config)
    assert len(received) == 1
    assert received[0].actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu == 6
    if async_mode:
        assert received[0].transfer_queue.backend.SimpleStorage.required_node_resource == "node:10.0.0.1"
        assert options[0]["task_runner_options"] == {"resources": {"node:10.0.0.1": 0.001}}
    else:
        assert "task_runner_options" not in options[0]
    # A new interpreter models Ray's process-local resolver registry. It must
    # resolve the serialized config without importing the recipe entrypoint.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import pickle, sys; from omegaconf import OmegaConf; "
            "assert not OmegaConf.has_resolver('eval'); "
            "config = pickle.loads(sys.stdin.buffer.read()); "
            "assert OmegaConf.to_container(config, resolve=True)"
            "['actor_rollout_ref']['ref']['log_prob_micro_batch_size_per_gpu'] == 6",
        ],
        input=pickle.dumps(received[0]),
        check=True,
        capture_output=True,
    )


@pytest.mark.parametrize("kind", ["zip", "audio"])
def test_audio_byte_readers_preserve_payload_and_detect_truncation(tmp_path, kind):
    from verl.audio_chunk import read_chunk_audio
    from verl.audio_zip import read_zip_audio

    reader = read_zip_audio if kind == "zip" else read_chunk_audio
    path = tmp_path / f"archive.{kind}"
    path.write_bytes(b"header" + b"encoded audio" + b"next item")
    assert reader(f"{path}!6:13") == b"encoded audio"
    with pytest.raises(EOFError, match="Truncated"):
        reader(f"{path}!6:100")


def test_audio_cache_mapping_preserves_byte_and_time_selectors(monkeypatch, tmp_path):
    from verl import audio_cache

    monkeypatch.setattr(audio_cache, "LOCAL_DATA_ROOT", tmp_path)
    source = "az://orngwus2cresco/data/corpus/audio.zip!128:4096#1.0:2.0"
    assert audio_cache.local_audio_source(source) == f"{tmp_path}/corpus/audio.zip!128:4096#1.0:2.0"


@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("window", [100, 512])
def test_sampling_applies_configured_no_repeat_window(compose_recipe, validate, window):
    from recipe.phimm.agentic.sampling import sampling_params_for_row

    # Override validation independently to catch accidental use of the training value.
    field = "validation_no_repeat_ngram_window_size" if validate else "no_repeat_ngram_window_size"
    config = compose_recipe(overrides=[f"actor_rollout_ref.rollout.custom.{field}={window}"])
    original = {"temperature": 0.0 if validate else 1.0}
    params = sampling_params_for_row(config.actor_rollout_ref.rollout, original, validate=validate)
    assert params["extra_args"] == {"ngram_size": 15, "window_size": window}
    assert "ngram_window" not in params["extra_args"]
    assert params["max_tokens"] == (512 if validate else 768)
    assert params["stop"] == ["<|im_end|>"]
    assert "extra_args" not in original


def test_jsonl_loading_reads_updated_manifest_without_dataset_disk_cache(tmp_path, monkeypatch):
    import datasets
    from recipe.phimm.data.dataset import create_audio_dataset

    def forbidden(*args, **kwargs):
        pytest.fail("Dataset disk caching must not be used")

    monkeypatch.setattr(datasets.Dataset, "load_from_disk", forbidden)
    monkeypatch.setattr(datasets.Dataset, "save_to_disk", forbidden)
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"text": "first revision"}) + "\n")
    assert create_audio_dataset(dataset_name="jsonl", jsonl_paths=str(path))[0]["text"] == "first revision"
    path.write_text(json.dumps({"text": "second revision"}) + "\n")
    assert create_audio_dataset(dataset_name="jsonl", jsonl_paths=str(path))[0]["text"] == "second revision"


@pytest.mark.parametrize(
    "kind, name, mode, bounds",
    [
        ("earnings", "remax_earnings_short_sc2", "segment_count", [3, 3]),
        ("earnings", "remax_earnings_short_sc2a", "segment_count", [1, 3]),
        ("earnings", "remax_earnings_short_fl5", "segment_len", [2, 30]),
        ("rand_segment", "remax_main_base3a_2611r2_sc1a", "segment_count", [2, 2]),
        ("async", "remax_main_base3a_2611r2_async_sc_bs64_ppo4", "segment_count", [1, 2]),
    ],
)
def test_merged_segment_recipes_load_their_prepared_partitions(compose_recipe, kind, name, mode, bounds):
    import recipe.phimm
    from recipe.phimm.agentic.dataset import _flatten_data_confs, _prepare_data_confs

    config = compose_recipe(kind=kind, name=name)
    base = (
        compose_recipe(kind="earnings", name="remax_earnings_short_baseline")
        if kind == "earnings"
        else compose_recipe()
    )
    OmegaConf.resolve(config)
    registry_path = Path(recipe.phimm.__file__).parent / "config/data/train_data/offline_audio/preparation.json"
    registry = json.loads(registry_path.read_text())
    preparation = next(entry for entry in registry["configs"] if entry["name"] == name)
    assert preparation["options"][mode] == bounds
    assert not preparation["options"]["even_split"]
    if name.endswith("fl5"):
        assert preparation["options"]["min_segments"] == 2
    assert "offline_data" not in config
    assert config.data.get("random_segment") is None
    assert config.data.get("random_merge") is None
    sources = _prepare_data_confs(config.data, is_train=True)
    assert [source["jsonl_paths"] for source in sources] == [
        registry["tasks"][task]["uri"] for task in preparation["tasks"]
    ]
    assert all(not source.get("pre_process") for source in sources)
    assert all(registry["tasks"][task]["remote_readback_verified"] for task in preparation["tasks"])
    assert config.data.mix == base.data.mix
    assert config.data.val_data == base.data.val_data
    assert len(sources) == len(list(_flatten_data_confs(base.data.train_data)))
    assert config.data.max_audio_dur is None
