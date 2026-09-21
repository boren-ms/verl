import ast
import io
import logging
import math
import queue
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
    main_task = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main_task")
    start = next(
        index
        for index, node in enumerate(main_task.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "batches" for target in node.targets)
    )
    main_task.body = main_task.body[start:]
    main_task.decorator_list = []
    main_task.args.args.extend([ast.arg(arg="left_egs"), ast.arg(arg="split_idx")])
    module = ast.fix_missing_locations(ast.Module(body=[main_task], type_ignores=[]))

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
        "np": np,
        "uuid": uuid,
        "logging": logging,
        "Dataset": datasets.Dataset,
        "Sequence": datasets.Sequence,
        "Value": datasets.Value,
        "concatenate_datasets": datasets.concatenate_datasets,
        "DataProto": SimpleNamespace(from_single_dict=prepare),
        "pad_dataproto_to_divisor": lambda data, divisor: (data, 0),
        "unpad_dataproto": lambda data, pad_size: data,
        "wg": SimpleNamespace(world_size=1, generate_sequences=generate),
        "tokenizer": SimpleNamespace(decode=lambda *args, **kwargs: "hello"),
        "eval_score": lambda *args, **kwargs: {"n_err": 0, "n_ref": 1, "n_edge": 0},
        "get_hyp_text": lambda text, **kwargs: text,
        "log_examples": lambda *args, **kwargs: None,
        "num_examine": 0,
        "wer_kwargs": {},
        "output_dir": str(tmp_path),
        "tqdm": lambda **kwargs: progress,
        "split_size": 2,
    }
    exec(compile(module, str(path), "exec"), namespace)

    def run(loader, total=4, already_saved=0, split_index=0):
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
                namespace["main_task"](SimpleNamespace(data={"prefetch_depth": 1}), already_saved, split_index)
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


def test_generation_scores_each_rows_language_and_keywords(generation_pipeline):
    from recipe.phimm.reward.asr_edge import eval_score
    from recipe.phimm.reward.asr_response import get_hyp_text

    examples = [
        ("Spanish", "hola mundo", "hola", ["mundo"]),
        ("German", "die 100 dollar", "die one hundred dollars", []),
        ("English", "hello world", "hello world", ["hello"]),
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
        expected.append(eval_score(response, reference, extra_info=extra_info, version=2607))
    batch = {key: [row[key][0] for row in rows] for key in rows[0]}
    decoded = iter(responses)
    namespace = generation_pipeline.namespace
    namespace["tokenizer"].decode = lambda *args, **kwargs: next(decoded)
    namespace["eval_score"] = eval_score
    namespace["get_hyp_text"] = get_hyp_text
    namespace["wer_kwargs"] = {"version": 2607}

    result = generation_pipeline.run([batch], total=len(examples))

    assert result.error is None
    saved = datasets.Dataset.from_parquet(str(generation_pipeline.output_dir / "part-000.parquet"))
    assert "extra_info" not in saved.column_names
    assert list(saved["language"]) == [example[0] for example in examples]
    assert list(saved["keywords"]) == [example[3] for example in examples]
    assert list(saved["raw_response"]) == responses
    assert list(saved["response"]) == [example[2] for example in examples]
    for actual, score in zip(saved, expected, strict=True):
        assert {key: actual[key] for key in score} == score
    assert list(saved["wer"]) == [0.5, 1.0, 0.0]
    assert list(saved["p_lang"]) == [1.0, 1.0, 1.0]
    assert list(saved["p_kw_missing"]) == [1.0, 0.0, 0.0]


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
        return {"n_err": 0, "n_ref": 1, "n_edge": 0}

    generation_pipeline.namespace["eval_score"] = score
    generation_pipeline.namespace["wer_kwargs"] = {"version": 2607, "language": "Spanish"}

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
        namespace["eval_score"] = fail
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
    generation_pipeline.namespace["eval_score"] = fail
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
    parts = list(generation_pipeline.output_dir.glob("part-*.parquet"))
    assert [part.name for part in parts] == ["part-000.parquet"]
    saved = datasets.Dataset.from_parquet(str(parts[0]))
    assert list(saved["id"]) == [0, 1]


@pytest.mark.parametrize("already_saved,split_index", [(0, 0), (2, 1)])
def test_generation_success_preserves_output_and_flushes_last_split(generation_pipeline, already_saved, split_index):
    rows = list(_generation_batches(5))[already_saved:]
    result = generation_pipeline.run(rows, total=5, already_saved=already_saved, split_index=split_index)

    assert result.error is None
    assert "Saved 5/5 [100.00%] samples." in result.output
    assert "All Done" in result.output
    parts = sorted(generation_pipeline.output_dir.glob("part-*.parquet"))
    assert [part.name for part in parts] == [f"part-{index:03d}.parquet" for index in range(split_index, 3)]
    saved = datasets.concatenate_datasets([datasets.Dataset.from_parquet(str(part)) for part in parts])
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
def test_generation_preserves_keyword_schema_across_batches(generation_pipeline, keywords, split_size):
    rows = list(_generation_batches(len(keywords)))
    for index, (row, values) in enumerate(zip(rows, keywords, strict=True)):
        row["extra_info"][0].update(keywords=values, optional_count=None if index == 0 else 1)
    generation_pipeline.namespace["split_size"] = split_size

    result = generation_pipeline.run(rows, total=len(rows))

    assert result.error is None
    assert "All Done" in result.output
    parts = sorted(generation_pipeline.output_dir.glob("part-*.parquet"))
    assert len(parts) == math.ceil(len(rows) / split_size)
    saved_parts = [datasets.Dataset.from_parquet(str(part)) for part in parts]
    for saved_part in saved_parts:
        assert saved_part.features["keywords"].feature == datasets.Value("string")
    saved = datasets.concatenate_datasets(saved_parts)
    assert list(saved["id"]) == [0, 1]
    assert list(saved["keywords"]) == keywords
    assert list(saved["optional_count"]) == [None, 1]
    assert saved.features["optional_count"] == datasets.Value("int64")
    assert list(saved["response"]) == ["hello", "hello"]
