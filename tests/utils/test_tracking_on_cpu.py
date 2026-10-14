# Copyright 2026 Microsoft Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from verl.utils.tracking import Tracking, ValidationGenerationsLogger


@pytest.fixture
def wandb(monkeypatch):
    for name in (
        "WANDB_RUN_ID",
        "WANDB_MODE",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
        "WANDB_ORGANIZATION",
        "WANDB_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    module = MagicMock()
    module.Api.return_value.projects.return_value = [SimpleNamespace(name="project")]
    module.Api.return_value.runs.return_value = []
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


def init_run(wandb, project="project", experiment="experiment", config=None, backend="wandb"):
    tracking = Tracking(project, experiment, default_backend=backend, config=config)
    kwargs = wandb.init.call_args.kwargs
    tracking.log({"reward": 1.0}, step=10)
    wandb.log.assert_called_with(data={"reward": 1.0}, step=10)
    del tracking
    wandb.finish.assert_called_with(exit_code=0)
    return kwargs


def test_same_experiment_reuses_stable_id(wandb):
    first = init_run(wandb)
    second = init_run(wandb)
    assert first["id"] == second["id"]
    assert len(first["id"]) == 32
    assert first["resume"] == second["resume"] == "allow"
    assert first["entity"] == "genai"
    assert first["project"] == "project"
    assert first["name"] == "experiment"


def test_run_identity_is_scoped_to_entity_project_and_name(wandb, monkeypatch):
    ids = {
        init_run(wandb)["id"],
        init_run(wandb, project="other-project")["id"],
        init_run(wandb, experiment="other-experiment")["id"],
    }
    monkeypatch.setenv("WANDB_ENTITY", "other-entity")
    ids.add(init_run(wandb)["id"])
    assert len(ids) == 4


def test_existing_run_is_resumed(wandb):
    wandb.Api.return_value.runs.return_value = [SimpleNamespace(id="existing-id")]
    assert init_run(wandb)["id"] == "existing-id"
    wandb.Api.return_value.projects.assert_called_once_with(entity="genai")
    wandb.Api.return_value.runs.assert_called_once_with(
        path="genai/project",
        filters={"display_name": "experiment"},
        order="-created_at",
        per_page=1,
    )


def test_new_project_skips_run_lookup(wandb):
    wandb.Api.return_value.projects.return_value = []
    assert init_run(wandb)["resume"] == "allow"
    wandb.Api.return_value.runs.assert_not_called()


def test_explicit_run_id_takes_precedence(wandb, monkeypatch):
    monkeypatch.setenv("WANDB_RUN_ID", "explicit-id")
    assert init_run(wandb)["id"] == "explicit-id"
    wandb.Api.assert_not_called()


@pytest.mark.parametrize("mode", ["offline", "dryrun", "disabled"])
def test_offline_modes_skip_server_lookup(wandb, monkeypatch, mode):
    monkeypatch.setenv("WANDB_MODE", mode)
    first = init_run(wandb)
    assert init_run(wandb)["id"] == first["id"]
    wandb.Api.assert_not_called()


@pytest.mark.parametrize("method", ["projects", "runs"])
def test_lookup_failure_does_not_create_duplicate(wandb, method):
    getattr(wandb.Api.return_value, method).side_effect = RuntimeError("lookup failed")
    with pytest.raises(RuntimeError, match="lookup failed"):
        Tracking("project", "experiment", default_backend="wandb")
    wandb.init.assert_not_called()


def test_auth_proxy_and_config_are_preserved(wandb, monkeypatch):
    config = {"trainer": {"wandb_api_key": "configured-key", "wandb_proxy": "http://proxy"}}
    monkeypatch.setenv("WANDB_API_KEY", "environment-key")
    monkeypatch.setenv("WANDB_ORGANIZATION", "https://wandb.example")
    kwargs = init_run(wandb, config=config)
    wandb.login.assert_called_once_with(host="https://wandb.example", key="environment-key", relogin=True)
    wandb.Settings.assert_called_once_with(https_proxy="http://proxy")
    assert kwargs["settings"] is wandb.Settings.return_value
    assert kwargs["config"] is config


def test_long_self_hosted_key_uses_environment(wandb, monkeypatch):
    key = "local-" + "x" * 50
    init_run(wandb, config={"trainer": {"wandb_api_key": key}})
    wandb.login.assert_not_called()
    assert os.environ["WANDB_API_KEY"] == key
    assert os.environ["WANDB_BASE_URL"] == "https://msaip.wandb.io"
    monkeypatch.delenv("WANDB_API_KEY")
    monkeypatch.delenv("WANDB_BASE_URL")


def test_deprecated_tracking_backend_reuses_id(wandb):
    with pytest.warns(DeprecationWarning, match="tracking"):
        kwargs = init_run(wandb, backend="tracking")
    assert kwargs["id"] == init_run(wandb)["id"]


def test_non_wandb_backend_does_not_initialize_wandb(wandb):
    tracking = Tracking("project", "experiment", default_backend=[])
    assert tracking.logger == {}
    wandb.init.assert_not_called()
    wandb.Api.assert_not_called()


@pytest.mark.parametrize("with_ground_truth", [False, True])
def test_validation_wandb_table_preserves_samples_and_history(wandb, with_ground_truth):
    class Table:
        def __init__(self, columns, data=None):
            self.columns = columns
            self.data = list(data or [])

        def add_data(self, *row):
            assert len(row) == len(self.columns)
            self.data.append(list(row))

    wandb.Table.side_effect = Table
    samples = [("input-a", "output-a", 0.5), ("input-b", "output-b", 0.75)]
    if with_ground_truth:
        samples = [(*samples[0], "reference-a"), (*samples[1], None)]
    logger = ValidationGenerationsLogger()
    for step in [10, 20]:
        logger.log(["wandb"], samples, step)
    table = wandb.log.call_args.args[0]["val/generations"]
    fields = ["input", "output", "score"] + (["ground_truth"] if with_ground_truth else [])
    assert table.columns == ["step"] + [f"{field}_{i}" for i in [1, 2] for field in fields]
    assert table.data == [[step, *samples[0], *samples[1]] for step in [10, 20]]
    assert wandb.log.call_args.kwargs == {"step": 20}


@pytest.mark.parametrize("backend", ["swanlab", "mlflow", "clearml", "tensorboard", "vemlp_wandb"])
@pytest.mark.parametrize("with_ground_truth", [False, True])
def test_validation_backends_include_optional_ground_truth(monkeypatch, backend, with_ground_truth):
    module = MagicMock()
    monkeypatch.setitem(sys.modules, backend, module)
    sample = ("input", "output", 0.5)
    if with_ground_truth:
        sample += ("reference",)
    expected = {"input": "input", "output": "output", "score": 0.5}
    if with_ground_truth:
        expected["ground_truth"] = "reference"
    logger = ValidationGenerationsLogger()
    if backend == "tensorboard":
        logger.writer = MagicMock()
    elif backend == "mlflow":
        artifacts = []
        module.log_artifact.side_effect = lambda path: artifacts.append(json.loads(Path(path).read_text()))
    elif backend == "vemlp_wandb":
        monkeypatch.setitem(sys.modules, "volcengine_ml_platform", SimpleNamespace(wandb=module))
        logger._log_generations_to_wandb = MagicMock()
    logger.log([backend], [sample], 10)
    if backend == "swanlab":
        module.echarts.Table.return_value.add.assert_called_once_with(
            headers=["step", *expected], rows=[[10, *sample]]
        )
    elif backend == "mlflow":
        assert artifacts == [[expected]]
    elif backend == "clearml":
        call = module.Task.current_task.return_value.get_logger.return_value.report_table.call_args
        assert call.kwargs["table_plot"].to_dict(orient="records") == [{"step": 10, **expected}]
    elif backend == "tensorboard":
        text = logger.writer.add_text.call_args.args[1]
        assert ("**Ground truth:** reference" in text) == with_ground_truth
        logger.writer.add_text.assert_called_once_with("val/generations", text, 10)
    else:
        logger._log_generations_to_wandb.assert_called_once_with([sample], 10, module)


def validation_sample_trainer(limit=2):
    path = Path(__file__).parents[2] / "verl/trainer/ppo/ray_trainer.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_maybe_log_val_generations"
    )
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    trainer = SimpleNamespace(
        config=SimpleNamespace(trainer=SimpleNamespace(log_val_generations=limit, logger=["wandb"])),
        validation_generations_logger=MagicMock(),
        global_steps=10,
    )
    return trainer, namespace["_maybe_log_val_generations"]


def test_validation_sample_selection_keeps_ground_truth_aligned():
    trainer, log_samples = validation_sample_trainer()
    inputs = ["c", "a", "b"]
    outputs = [f"output-{value}" for value in inputs]
    scores = [0.3, 0.1, 0.2]
    gts = [f"reference-{value}" for value in inputs]
    log_samples(trainer, inputs, outputs, scores)
    original = trainer.validation_generations_logger.log.call_args.args[1]
    log_samples(trainer, inputs, outputs, scores, gts=gts)
    selected = trainer.validation_generations_logger.log.call_args.args[1]
    assert [sample[:3] for sample in selected] == original
    assert len(selected) == 2
    assert all(sample[3] == f"reference-{sample[0]}" for sample in selected)
    trainer.validation_generations_logger.log.assert_called_with(["wandb"], selected, 10)


def test_validation_sample_selection_rejects_misaligned_references():
    trainer, log_samples = validation_sample_trainer()
    with pytest.raises(ValueError, match="shorter"):
        log_samples(trainer, ["input"], ["output"], [0.5], gts=[])
    trainer.validation_generations_logger.log.assert_not_called()


def test_disabled_validation_sample_logging_does_not_require_references():
    trainer, log_samples = validation_sample_trainer(limit=0)
    log_samples(trainer, ["input"], ["output"], [0.5], gts=[])
    trainer.validation_generations_logger.log.assert_not_called()
