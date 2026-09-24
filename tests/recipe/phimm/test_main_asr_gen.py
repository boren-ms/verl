import ast
import io
import json
import logging
import math
import os
import queue
import re
import threading
import traceback
import uuid
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import blobfile as bf
import datasets
import numpy as np
import pytest
import soundfile as sf
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).parents[3]


def _read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


@pytest.fixture
def generation_io():
    path = PROJECT_ROOT / "recipe/phimm/main_asr_gen.py"
    module = ast.parse(path.read_text())
    definitions = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_part_index", "_jsonl_num_rows", "_resume_state_from_output"}
    ]
    namespace = {"bf": bf, "json": json, "os": os, "re": re}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.fixture
def dataset_namespace():
    # Follow the dataset CPU tests' AST loading pattern to avoid model/GPU imports.
    path = PROJECT_ROOT / "recipe/phimm/data/rl_dataset.py"
    module = ast.parse(path.read_text())
    definitions = [
        node for node in module.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name != "main"
    ]
    module = ast.Module(
        body=ast.parse("from __future__ import annotations").body + definitions,
        type_ignores=[],
    )
    sources = {
        "short": datasets.Dataset.from_dict({"audio_path": ["bad.wav"]}),
        "long": datasets.Dataset.from_dict({"audio_path": ["good0.wav", "good1.wav", "good2.wav"]}),
    }
    namespace = {
        "Dataset": object,
        "Sequence": Sequence,
        "datasets": datasets,
        "sf": sf,
        "logger": logging.getLogger(__name__),
        "get_num_proc": lambda value: value,
        "create_audio_dataset": lambda **kwargs: sources[kwargs["name"]],
        "ds_conf": [{"name": "short"}, {"name": "long"}],
        "config": SimpleNamespace(data=OmegaConf.create({"num_proc": None})),
        "tokenizer": None,
        "processor": None,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def _generation_dataset(namespace):
    path = PROJECT_ROOT / "recipe/phimm/main_asr_gen.py"
    module = ast.parse(path.read_text())
    main_task = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main_task")
    call = next(
        node
        for node in ast.walk(main_task)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "RLHFDataset"
    )
    return eval(compile(ast.Expression(call), str(path), "eval"), namespace)


def test_generation_preserves_all_examples_from_unequal_sources(dataset_namespace):
    dataset = _generation_dataset(dataset_namespace)

    assert len(dataset) == 4
    assert list(dataset.ds["audio_path"]) == ["bad.wav", "good0.wav", "good1.wav", "good2.wav"]


def test_generation_does_not_substitute_unreadable_audio(dataset_namespace):
    attempted = []

    def load_audio(row, max_dur):
        attempted.append(row["audio_path"])
        if row["audio_path"] == "bad.wav":
            raise FileNotFoundError("bad.wav")
        raise AssertionError("Generation must not substitute another sample")

    dataset_namespace["load_audio"] = load_audio
    dataset = _generation_dataset(dataset_namespace)

    with pytest.raises(RuntimeError, match="Unable to load audio after 1 attempt"):
        dataset[0]
    assert attempted == ["bad.wav"]


def test_training_still_uses_interleaving_by_default(dataset_namespace):
    dataset = dataset_namespace["RLHFDataset"](dataset_namespace["ds_conf"], None, dataset_namespace["config"].data)

    assert dataset.is_training
    assert list(dataset.ds["audio_path"]) == ["bad.wav", "good0.wav"]


@pytest.fixture
def generation_pipeline(tmp_path):
    path = PROJECT_ROOT / "recipe/phimm/main_asr_gen.py"
    module = ast.parse(path.read_text())
    scoring_definitions = [
        node
        for node in module.body
        if (
            isinstance(node, ast.ImportFrom)
            and node.module in {
                "recipe.phimm.reward.asr_eval",
                "recipe.phimm.reward.asr_response",
                "verl.trainer.ppo.reward",
            }
        )
        or (isinstance(node, ast.FunctionDef) and node.name in {"_load_generation_scoring", "log_examples"})
    ]
    main_task = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main_task")
    start = next(
        index
        for index, node in enumerate(main_task.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "tn_err" for target in node.targets)
    )
    scoring_setup = next(
        node
        for node in main_task.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_load_generation_scoring"
    )
    main_task.body = [scoring_setup, *main_task.body[start:]]
    main_task.decorator_list = []
    main_task.args.args.extend([ast.arg(arg="left_egs"), ast.arg(arg="split_idx")])
    module = ast.fix_missing_locations(ast.Module(body=[*scoring_definitions, main_task], type_ignores=[]))

    class Progress:
        def __init__(self, **kwargs):
            self.updates = 0
            self.closed = False

        def update(self, count):
            self.updates += count

        def close(self):
            self.closed = True

    progress = Progress()

    def prepare(batch):
        extra_info = batch.get("extra_info", [{}])[0] or {}
        return SimpleNamespace(batch=batch["prompt"], non_tensor_batch={}, row_id=extra_info.get("id"))

    def generate(data):
        return [
            SimpleNamespace(
                batch={
                    "prompts": np.array([1]),
                    "attention_mask": np.array([1, 1]),
                    "responses": np.array([2]),
                }
            )
            for _ in data.batch
        ]

    namespace = {
        "bf": bf,
        "json": json,
        "np": np,
        "uuid": uuid,
        "logging": logging,
        "OmegaConf": OmegaConf,
        "Dataset": datasets.Dataset,
        "Sequence": datasets.Sequence,
        "Value": datasets.Value,
        "DataProto": SimpleNamespace(from_single_dict=prepare),
        "pad_dataproto_to_divisor": lambda data, divisor: (data, 0),
        "unpad_dataproto": lambda data, pad_size: data,
        "wg": SimpleNamespace(world_size=1, generate_sequences=generate),
        "tokenizer": SimpleNamespace(decode=lambda *args, **kwargs: "hello"),
        "num_examine": 0,
        "reward_kwargs": {},
        "output_dir": str(tmp_path),
        "tqdm": lambda **kwargs: progress,
        "split_size": 2,
    }
    exec(compile(module, str(path), "exec"), namespace)

    def run(loader, total=4, already_saved=0, split_index=0, scorer_config=None):
        namespace.update(
            dataloader=loader,
            total_batches=math.ceil(total),
            start_batch_idx=already_saved,
            total_egs=total,
        )
        outcome = []
        output = io.StringIO()

        def capture_print(*args, **kwargs):
            print(*args, **{**kwargs, "file": output})

        namespace["print"] = capture_print

        def target():
            try:
                config = OmegaConf.create({
                    "data": {"prefetch_depth": 1},
                    "custom_reward_function": {
                        "reward_kwargs": namespace["reward_kwargs"],
                        **(scorer_config or {}),
                    },
                })
                namespace["main_task"](config, already_saved, split_index)
            except BaseException as exc:
                outcome.append(exc)

        existing_threads = set(threading.enumerate())
        runner = threading.Thread(target=target, daemon=True)
        runner.start()
        runner.join(timeout=10)
        assert not runner.is_alive(), "Generation pipeline deadlocked"
        assert not [
            thread
            for thread in threading.enumerate()
            if thread not in existing_threads and thread.name.startswith("asr-generation-")
        ], "Generation workers were left running"
        assert progress.closed
        return SimpleNamespace(error=outcome[0] if outcome else None, output=output.getvalue())

    return SimpleNamespace(run=run, namespace=namespace, output_dir=tmp_path, progress=progress)


def _generation_batches(count):
    for index in range(count):
        yield {
            "prompt": [[{"content": "Transcribe"}]],
            "reward_model": [{"ground_truth": "hello"}],
            "audio_path": [f"{index}.wav"],
            "extra_info": [{"id": index}],
        }


@pytest.mark.parametrize(
    "scorer_config",
    [
        None,
        {"path": None, "name": "unused"},
        {"path": "recipe/phimm/reward/asr_eval.py", "name": "openasr_eval"},
    ],
    ids=["unconfigured", "null-path", "explicit-default"],
)
def test_generation_defaults_to_openasr_and_preserves_metadata(generation_pipeline, scorer_config):
    from recipe.phimm.reward.asr_eval import openasr_eval

    examples = [
        ("Spanish", "hola mundo", "hola", ["mundo"]),
        ("German", "die 100 dollar", "die one hundred dollars", []),
        ("English", "hello world", "hello world", ["hello"]),
        ("English", "I have twenty dollars", "I have $20", ["dollars"]),
    ]
    rows = list(_generation_batches(len(examples)))
    responses = []
    expected = []
    for row, (language, reference, hypothesis, keywords) in zip(rows, examples, strict=True):
        row["reward_model"][0]["ground_truth"] = reference
        extra_info = row["extra_info"][0]
        extra_info.update(language=language, keywords=keywords)
        response = f"Audio Language: {language}.\n<ASR><lang={language}><TXT>{hypothesis}</TXT></ASR>"
        responses.append(response)
        expected.append(openasr_eval(response, reference, extra_info=extra_info, version=2607))
    batch = {key: [row[key][0] for row in rows] for key in rows[0]}
    decoded = iter(responses)
    namespace = generation_pipeline.namespace
    namespace["tokenizer"].decode = lambda *args, **kwargs: next(decoded)
    assert namespace["openasr_eval"] is openasr_eval
    namespace["reward_kwargs"] = {"version": 2607}
    namespace["num_examine"] = 1

    result = generation_pipeline.run([batch], total=len(examples), scorer_config=scorer_config)

    assert result.error is None
    saved = datasets.Dataset.from_list(_read_jsonl(generation_pipeline.output_dir / "part-000.jsonl"))
    assert "extra_info" not in saved.column_names
    assert list(saved["language"]) == [example[0] for example in examples]
    assert list(saved["keywords"]) == [example[3] for example in examples]
    assert list(saved["raw_response"]) == responses
    assert list(saved["response"]) == [example[2] for example in examples]
    for actual, score in zip(saved, expected, strict=True):
        assert set(score) == {"score", "wer", "n_err", "n_ref"}
        assert {key: actual[key] for key in score} == score
    assert list(saved["wer"]) == [0.5, 1.0, 0.0, 0.0]
    assert list(saved["n_err"]) == [1, 3, 0, 0]
    assert list(saved["n_ref"]) == [2, 3, 2, 3]
    assert not any(key.startswith("p_") for key in saved.column_names)
    assert not any("edge" in key for key in saved.column_names)
    assert "WER: 100.00%" in result.output
    assert "Overall wer: 40.00% [4/10] on 4 generated samples" in result.output
    assert "edge_wer" not in result.output


@pytest.mark.parametrize(
    "scorer_config,extra_metrics",
    [
        (
            {"path": "recipe/phimm/reward/asr_eval.py", "name": "openasr_en_eval"},
            {"kw_acc", "nb_err", "nb_ref"},
        ),
        (
            {"path": "recipe/phimm/reward/asr_edge.py", "name": "eval_score"},
            {"p_fmt", "p_lang", "p_kw_missing"},
        ),
    ],
    ids=["keyword-metrics", "response-checks"],
)
def test_generation_saves_configured_measurements_without_edge_fields(
    generation_pipeline, scorer_config, extra_metrics
):
    rows = list(_generation_batches(1))
    rows[0]["reward_model"][0]["ground_truth"] = "hello world"
    rows[0]["extra_info"][0].update(language="English", keywords=["world"])
    response = "Audio Language: English.\n<ASR><lang=English><TXT>hello</TXT></ASR>"
    namespace = generation_pipeline.namespace
    namespace["tokenizer"].decode = lambda *args, **kwargs: response
    namespace["reward_kwargs"] = {"version": 2607}

    result = generation_pipeline.run(rows, total=1, scorer_config=scorer_config)

    assert result.error is None
    record, = _read_jsonl(generation_pipeline.output_dir / "part-000.jsonl")
    assert extra_metrics <= record.keys()
    assert record["keywords"] == ["world"]
    assert record["response"] == "hello"
    assert record["n_err"] == 1 and record["n_ref"] == 2 and record["wer"] == 0.5
    assert not any("edge" in key for key in record)
    assert "Overall wer: 50.00% [1/2]" in result.output


@pytest.mark.parametrize("path", [None, "recipe/phimm/reward/asr_eval.py"])
def test_generation_reward_kwargs_configure_scoring_and_response_version(generation_pipeline, path):
    rows = list(_generation_batches(1))
    rows[0]["reward_model"][0]["ground_truth"] = "ice cream"
    response = "Audio Language: English.\n<ASR><lang=English><TXT>icecream</TXT></ASR>"
    namespace = generation_pipeline.namespace
    namespace["tokenizer"].decode = lambda *args, **kwargs: response
    versions = []
    parse = namespace["get_hyp_text"]

    def parse_response(text, version):
        versions.append(version)
        return parse(text, version=version)

    namespace["get_hyp_text"] = parse_response
    scorer_config = {
        "path": path,
        "name": "openasr_eval",
        "reward_kwargs": {"version": 2607, "merge_compounds": False},
    }

    result = generation_pipeline.run(rows, total=1, scorer_config=scorer_config)

    assert result.error is None
    record, = _read_jsonl(generation_pipeline.output_dir / "part-000.jsonl")
    assert record["response"] == "icecream"
    assert record["n_err"] == record["n_ref"] == 2
    assert record["wer"] == 1.0
    assert versions == [2607]


@pytest.mark.parametrize("config", [{}, {"custom_reward_function": None}, {"custom_reward_function": {}}])
def test_generation_scoring_without_configuration_uses_default(generation_pipeline, config):
    namespace = generation_pipeline.namespace

    score_fn, reward_kwargs = namespace["_load_generation_scoring"](OmegaConf.create(config))

    assert score_fn is namespace["openasr_eval"]
    assert reward_kwargs == {}


@pytest.mark.parametrize(
    "scorer_config,error,message",
    [
        ({"path": "missing_measure_function.py", "name": "score"}, FileNotFoundError, "not found"),
        (
            {"path": "recipe/phimm/reward/asr_eval.py", "name": "missing_measure_function"},
            AttributeError,
            "not found",
        ),
    ],
)
def test_generation_invalid_scorer_configuration_fails_explicitly(
    generation_pipeline, scorer_config, error, message
):
    config = OmegaConf.create({"data": {}, "custom_reward_function": scorer_config})
    with pytest.raises(error, match=message):
        generation_pipeline.namespace["_load_generation_scoring"](config)
    assert not list(generation_pipeline.output_dir.glob("part-*.jsonl"))


@pytest.mark.parametrize("score", [None, 0.5, {"wer": 0.5}, {"n_err": 1}, {"n_ref": 2}])
def test_generation_rejects_measurements_without_error_counts(generation_pipeline, score):
    generation_pipeline.namespace["openasr_eval"] = lambda *args, **kwargs: score

    result = generation_pipeline.run(_generation_batches(1), total=1)

    assert isinstance(result.error, ValueError)
    assert "must return a dict containing n_err and n_ref" in str(result.error)
    assert "All Done" not in result.output
    assert not list(generation_pipeline.output_dir.glob("part-*.jsonl"))


@pytest.mark.parametrize("merge_compounds", [None, True, False])
def test_generation_openasr_respects_compound_normalization(generation_pipeline, merge_compounds):
    rows = list(_generation_batches(1))
    rows[0]["reward_model"][0]["ground_truth"] = "ice cream"
    response = "Audio Language: English.\n<ASR><lang=English><TXT>icecream</TXT></ASR>"
    namespace = generation_pipeline.namespace
    namespace["tokenizer"].decode = lambda *args, **kwargs: response
    namespace["reward_kwargs"] = {"version": 2607}
    if merge_compounds is not None:
        namespace["reward_kwargs"]["merge_compounds"] = merge_compounds

    result = generation_pipeline.run(rows, total=1)

    assert result.error is None
    record, = _read_jsonl(generation_pipeline.output_dir / "part-000.jsonl")
    assert record["response"] == "icecream"
    assert record["raw_response"] == response
    expected = (
        {"score": 0.0, "wer": 1.0, "n_err": 2, "n_ref": 2}
        if merge_compounds is False
        else {"score": 1.0, "wer": 0.0, "n_err": 0, "n_ref": 1}
    )
    assert {key: record[key] for key in expected} == expected


@pytest.mark.parametrize("metadata", ["missing", None, {}])
def test_generation_scoring_handles_missing_metadata(generation_pipeline, metadata):
    rows = list(_generation_batches(1))
    if metadata == "missing":
        rows[0].pop("extra_info")
    else:
        rows[0]["extra_info"] = [metadata]
    received = []

    def score(*args, **kwargs):
        received.append(kwargs)
        return {"n_err": 0, "n_ref": 1}

    generation_pipeline.namespace["openasr_eval"] = score
    generation_pipeline.namespace["reward_kwargs"] = {"version": 2607, "language": "Spanish"}

    result = generation_pipeline.run(rows, total=1)

    assert result.error is None
    assert received == [{"version": 2607, "language": "Spanish", "extra_info": {}}]


def test_generation_propagates_producer_failure(generation_pipeline):
    error = OSError("dataset read failed")

    def broken_loader():
        raise error
        yield

    result = generation_pipeline.run(broken_loader())

    assert result.error is error
    assert "broken_loader" in [frame.name for frame in traceback.extract_tb(result.error.__traceback__)]
    assert "All Done" not in result.output
    assert "Overall wer" not in result.output
    assert not list(generation_pipeline.output_dir.iterdir())


def test_generation_propagates_batch_preparation_failure(generation_pipeline):
    error = ValueError("invalid batch")

    def prepare(batch):
        raise error

    generation_pipeline.namespace["DataProto"].from_single_dict = prepare
    result = generation_pipeline.run(_generation_batches(4))

    assert result.error is error
    assert "All Done" not in result.output


@pytest.mark.parametrize("failure_stage", ["decode", "score", "write", "final_write"])
def test_generation_propagates_consumer_failure(generation_pipeline, failure_stage):
    error = OSError(f"{failure_stage} failed")
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise error

    namespace = generation_pipeline.namespace
    total = 20
    if failure_stage == "decode":
        namespace["tokenizer"].decode = fail
    elif failure_stage == "score":
        namespace["openasr_eval"] = fail
    else:
        namespace["bf"] = SimpleNamespace(makedirs=bf.makedirs, BlobFile=fail)
        if failure_stage == "final_write":
            total = 1
    result = generation_pipeline.run(_generation_batches(total), total=total)

    assert result.error is error
    assert calls == [1], "A failed consumer operation must not be retried during cleanup"
    assert "All Done" not in result.output


def test_generation_failure_unblocks_full_preparation_queue(generation_pipeline):
    error = RuntimeError("generation failed")
    third_batch_prepared = threading.Event()
    namespace = generation_pipeline.namespace
    prepare = namespace["DataProto"].from_single_dict

    def mark_prepared(batch):
        data = prepare(batch)
        if data.row_id == 2:
            third_batch_prepared.set()
        return data

    def fail(data):
        assert third_batch_prepared.wait(timeout=5)
        raise error

    namespace["DataProto"].from_single_dict = mark_prepared
    namespace["wg"].generate_sequences = fail
    result = generation_pipeline.run(_generation_batches(20), total=20)

    assert result.error is error
    assert "All Done" not in result.output
    assert not list(generation_pipeline.output_dir.iterdir())


def test_consumer_failure_unblocks_full_output_queue(generation_pipeline, monkeypatch):
    error = ValueError("scoring failed")
    full_output_queue = threading.Event()
    queues = []

    class ObservedQueue(queue.Queue):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            queues.append(self)

        def put(self, item, *args, **kwargs):
            if len(queues) >= 2 and self is queues[1] and self.full():
                full_output_queue.set()
            return super().put(item, *args, **kwargs)

    def fail(*args, **kwargs):
        assert full_output_queue.wait(timeout=5)
        raise error

    monkeypatch.setattr(queue, "Queue", ObservedQueue)
    generation_pipeline.namespace["openasr_eval"] = fail
    result = generation_pipeline.run(_generation_batches(20), total=20)

    assert result.error is error
    assert full_output_queue.is_set()
    assert "All Done" not in result.output


def test_late_producer_failure_preserves_saved_prefix_without_flushing_buffer(generation_pipeline):
    error = OSError("later dataset read failed")
    third_batch_scored = threading.Event()

    def mark_scored(ds, **kwargs):
        if ds[0]["id"] == 2:
            third_batch_scored.set()

    def broken_loader():
        yield from _generation_batches(3)
        assert third_batch_scored.wait(timeout=5)
        raise error

    generation_pipeline.namespace["log_examples"] = mark_scored
    result = generation_pipeline.run(broken_loader(), total=5)

    assert result.error is error
    assert "All Done" not in result.output
    parts = list(generation_pipeline.output_dir.glob("part-*.jsonl"))
    assert [part.name for part in parts] == ["part-000.jsonl"]
    saved = datasets.Dataset.from_list(_read_jsonl(parts[0]))
    assert list(saved["id"]) == [0, 1]


@pytest.mark.parametrize("already_saved,split_index", [(0, 0), (2, 1)])
def test_generation_success_preserves_output_and_flushes_last_split(generation_pipeline, already_saved, split_index):
    rows = list(_generation_batches(5))[already_saved:]
    result = generation_pipeline.run(rows, total=5, already_saved=already_saved, split_index=split_index)

    assert result.error is None
    assert "Saved 5/5 [100.00%] samples." in result.output
    assert "All Done" in result.output
    assert f"on {5 - already_saved} generated samples" in result.output
    assert not list(generation_pipeline.output_dir.glob("*.parquet"))
    parts = sorted(generation_pipeline.output_dir.glob("part-*.jsonl"))
    assert [part.name for part in parts] == [f"part-{index:03d}.jsonl" for index in range(split_index, 3)]
    saved = datasets.Dataset.from_list([row for part in parts for row in _read_jsonl(part)])
    assert list(saved["id"]) == list(range(already_saved, 5))
    assert list(saved["response"]) == ["hello"] * (5 - already_saved)
    assert generation_pipeline.progress.updates == 5 - already_saved


@pytest.mark.parametrize("split_size", [1, 2, 3])
@pytest.mark.parametrize(
    "keywords",
    [
        [[], ["hello"]],
        [["hello"], []],
        [None, ["hello"]],
        [[], []],
        [[None], ["hello"]],
    ],
    ids=["empty-first", "empty-last", "null-column", "all-empty", "null-element"],
)
def test_generation_preserves_keyword_lists_across_batches(generation_pipeline, keywords, split_size):
    rows = list(_generation_batches(len(keywords)))
    for index, (row, values) in enumerate(zip(rows, keywords, strict=True)):
        row["extra_info"][0].update(keywords=values, optional_count=None if index == 0 else 1)
    generation_pipeline.namespace["split_size"] = split_size

    result = generation_pipeline.run(rows, total=len(rows))

    assert result.error is None
    assert "All Done" in result.output
    parts = sorted(generation_pipeline.output_dir.glob("part-*.jsonl"))
    assert len(parts) == math.ceil(len(rows) / split_size)
    saved = [row for part in parts for row in _read_jsonl(part)]
    assert [row["id"] for row in saved] == [0, 1]
    assert [row["keywords"] for row in saved] == keywords
    assert [row["optional_count"] for row in saved] == [None, 1]
    assert isinstance(saved[1]["optional_count"], int)
    assert [row["response"] for row in saved] == ["hello", "hello"]


def test_generation_jsonl_preserves_unicode_newlines_and_measurements(generation_pipeline):
    text = "M\u00fcnchen\n\u6771\u4eac"
    rows = list(_generation_batches(1))
    rows[0]["reward_model"][0]["ground_truth"] = text
    rows[0]["extra_info"][0].update(
        keywords=["M\u00fcnchen"],
        metadata={"speaker": "Jos\u00e9", "active": True},
    )
    generation_pipeline.namespace["tokenizer"].decode = lambda *args, **kwargs: text
    generation_pipeline.namespace["openasr_eval"] = lambda *args, **kwargs: {
        "n_err": 1,
        "n_ref": 3,
        "wer": 1 / 3,
    }

    result = generation_pipeline.run(rows, total=1)

    assert result.error is None
    path = generation_pipeline.output_dir / "part-000.jsonl"
    contents = path.read_text(encoding="utf-8")
    assert contents.endswith("\n") and len(contents.splitlines()) == 1
    assert "M\u00fcnchen" in contents and "\u6771\u4eac" in contents
    record, = _read_jsonl(path)
    assert record["text"] == record["response"] == record["raw_response"] == text
    assert record["keywords"] == ["M\u00fcnchen"]
    assert record["metadata"] == {"speaker": "Jos\u00e9", "active": True}
    assert record["n_err"] == 1
    assert record["n_ref"] == 3
    assert record["wer"] == 1 / 3
    assert "Overall wer: 33.33% [1/3]" in result.output
    assert not any("edge" in key for key in record)
    assert "edge_wer" not in result.output


@pytest.mark.parametrize(
    "path,expected",
    [
        ("part-000.jsonl", 0),
        ("/output/part-1000.jsonl", 1000),
        ("az://account/container/part-002.jsonl", 2),
        ("part-002.parquet", None),
        ("part-000.jsonl.tmp", None),
        ("part-abc.jsonl", None),
        ("details.jsonl", None),
    ],
)
def test_generation_part_index(generation_io, path, expected):
    assert generation_io._part_index(path) == expected


def test_generation_resumes_from_written_jsonl(generation_pipeline, generation_io):
    first = generation_pipeline.run(_generation_batches(2), total=2)
    assert first.error is None
    first_part = generation_pipeline.output_dir / "part-000.jsonl"
    original = first_part.read_bytes()
    state = generation_io._resume_state_from_output(str(generation_pipeline.output_dir), 5, 1, True)
    assert state == (2, 1)

    second = generation_pipeline.run(
        list(_generation_batches(5))[2:], total=5, already_saved=state[0], split_index=state[1]
    )

    assert second.error is None
    assert first_part.read_bytes() == original
    parts = sorted(generation_pipeline.output_dir.glob("part-*.jsonl"))
    saved = [row for part in parts for row in _read_jsonl(part)]
    assert [row["id"] for row in saved] == list(range(5))
    assert generation_io._resume_state_from_output(str(generation_pipeline.output_dir), 5, 1, True) == (5, 3)


def test_generation_resume_supports_jsonl(generation_io, tmp_path):
    (tmp_path / "part-000.jsonl").write_text('{"id":0}\n{"id":1}\n')
    (tmp_path / "part-001.jsonl").write_text('{"id":2}\n{"id":3}\n')
    assert generation_io._resume_state_from_output(str(tmp_path), 5, 2, True) == (4, 2)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("with_jsonl", [False, True])
def test_generation_rejects_legacy_output_directory(generation_io, tmp_path, enabled, with_jsonl):
    legacy = tmp_path / "part-000.parquet"
    legacy.write_bytes(b"not read as Parquet")
    if with_jsonl:
        (tmp_path / "part-001.jsonl").write_text('{"id":1}\n')
    with pytest.raises(ValueError, match="Only JSONL output is supported.*Use a new output_path"):
        generation_io._resume_state_from_output(str(tmp_path), 5, 1, enabled)
    assert legacy.read_bytes() == b"not read as Parquet"


def test_generation_resume_empty_output(generation_io, tmp_path):
    assert generation_io._resume_state_from_output(str(tmp_path), 5, 2, True) == (0, 0)


def test_generation_resume_disabled_ignores_existing_output(generation_io, tmp_path):
    (tmp_path / "part-000.jsonl").write_text("incomplete")
    assert generation_io._resume_state_from_output(str(tmp_path), 5, 2, False) == (0, 0)


def test_generation_resume_complete_partial_batch(generation_io, tmp_path):
    (tmp_path / "part-000.jsonl").write_text('{"id":0}\n{"id":1}\n{"id":2}\n')
    assert generation_io._resume_state_from_output(str(tmp_path), 3, 2, True) == (3, 1)


@pytest.mark.parametrize("total,batch_size,match", [(5, 2, "not aligned"), (2, 2, "exceeding")])
def test_generation_resume_rejects_incompatible_counts(generation_io, tmp_path, total, batch_size, match):
    (tmp_path / "part-000.jsonl").write_text('{"id":0}\n{"id":1}\n{"id":2}\n')
    with pytest.raises(ValueError, match=match):
        generation_io._resume_state_from_output(str(tmp_path), total, batch_size, True)


@pytest.mark.parametrize(
    "contents,match",
    [
        ('{"id":0}\n{"id":', "Invalid JSONL record.*:2"),
        ('{"id":0}\n\n', "Invalid JSONL record.*:2"),
        ("[]\n", "Expected a JSON object"),
        ("", "empty output split"),
    ],
)
def test_generation_resume_rejects_invalid_jsonl(generation_io, tmp_path, contents, match):
    (tmp_path / "part-000.jsonl").write_text(contents)
    with pytest.raises(ValueError, match=match):
        generation_io._resume_state_from_output(str(tmp_path), 5, 1, True)


def test_generation_resume_rejects_duplicate_indices(generation_io, tmp_path):
    (tmp_path / "part-000.jsonl").write_text('{"id":0}\n')
    (tmp_path / "part-0.jsonl").write_text('{"id":0}\n')
    with pytest.raises(ValueError, match="Duplicate output split index"):
        generation_io._resume_state_from_output(str(tmp_path), 5, 1, True)


def test_generation_resume_rejects_missing_split(generation_io, tmp_path):
    (tmp_path / "part-000.jsonl").write_text('{"id":0}\n')
    (tmp_path / "part-002.jsonl").write_text('{"id":2}\n')
    with pytest.raises(ValueError, match="Missing output split 1"):
        generation_io._resume_state_from_output(str(tmp_path), 5, 1, True)
