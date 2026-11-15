"""Exercise the submission boundary without connecting to a Ray cluster."""

import importlib.util
import shlex
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def submitter(monkeypatch):
    script = Path(__file__).resolve().parents[1] / "submit.py"
    spec = importlib.util.spec_from_file_location("phimm_submit", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    client = Mock()
    client.submit_job.return_value = "test-job"
    ray = ModuleType("ray")
    ray.init = Mock()
    ray.shutdown = Mock()
    ray._private = SimpleNamespace(
        worker=SimpleNamespace(
            global_worker=SimpleNamespace(node=SimpleNamespace(address_info={"webui_url": "host:8265"}))
        )
    )
    jobs = ModuleType("ray.job_submission")
    jobs.JobSubmissionClient = Mock(return_value=client)
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.job_submission", jobs)
    return module, ray, client


def test_training_selects_config_and_synced_sources(submitter, monkeypatch):
    module, ray, client = submitter
    monkeypatch.setenv("PYTHONPATH", "/existing")
    override = "trainer.experiment_name='two words; $(touch unwanted)'"
    module.main(
        [
            "--submission-id",
            "test-job",
            "data.shuffle=false",
            "--config-path",
            "config/v2611_dev/long",
            "--config-name",
            "remax_main_base3a_2611r2_sl_long5",
            override,
        ]
    )
    kwargs = client.submit_job.call_args.kwargs
    repository = Path(module.__file__).resolve().parents[2]
    assert shlex.split(kwargs["entrypoint"]) == [
        "cd",
        str(repository),
        "&&",
        "exec",
        "python",
        "-u",
        "-m",
        "recipe.phimm.main_asr_remax",
        "--config-name",
        "remax_main_base3a_2611r2_sl_long5",
        "--config-path",
        "config/v2611_dev/long",
        "data.shuffle=false",
        override,
    ]
    assert kwargs["runtime_env"]["env_vars"]["PYTHONPATH"].split(":") == [
        str(repository / "plugins/phimm/src"),
        str(repository / "plugins/qwen35_audio/src"),
        str(repository),
        "/existing",
    ]
    ray.shutdown.assert_called_once()


def test_eval_defaults_experiment_to_submission_id(submitter):
    module, ray, client = submitter
    module.main(["--submission-id", "test-job", "--evaluate", "actor_rollout_ref.model.path=/model"])
    command = shlex.split(client.submit_job.call_args.kwargs["entrypoint"])
    assert "recipe.phimm.main_asr_eval" in command
    assert "trainer.experiment_name=test-job" in command
    assert "actor_rollout_ref.model.path=/model" in command
    ray.shutdown.assert_called_once()


def test_submission_failure_disconnects_without_retry(submitter):
    module, ray, client = submitter
    client.submit_job.side_effect = RuntimeError("submission rejected")
    with pytest.raises(RuntimeError, match="submission rejected"):
        module.main(["--submission-id", "test-job"])
    client.submit_job.assert_called_once()
    ray.shutdown.assert_called_once()
