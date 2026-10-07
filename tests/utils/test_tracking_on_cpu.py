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

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from verl.utils.tracking import Tracking


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
