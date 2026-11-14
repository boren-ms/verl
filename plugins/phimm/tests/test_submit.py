"""The submit CLI preserves recipe overrides before connecting to Ray."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("phimm_submit", Path(__file__).parents[1] / "submit.py")
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)


def test_dry_run_uses_original_base3a(capsys):
    submit.main(["--name", "base3a-image-test", "--dry-run"])
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["submission_id"] == "base3a-image-test"
    assert "--config-name remax_main_base3a_2611r2 " in receipt["entrypoint"]
    assert "trainer.experiment_name=base3a-image-test" in receipt["entrypoint"]
    assert "random_segment" not in receipt["entrypoint"]


def test_eval_and_legacy_id_preserve_explicit_overrides():
    args = submit.parse_args(
        [
            "--submission-id",
            "eval-test",
            "--evaluate",
            "actor_rollout_ref.model.path=/models/checkpoint with spaces",
            "trainer.experiment_name=explicit-name",
        ]
    )
    command = submit.command_for(args)
    assert command[3] == "recipe.phimm.main_asr_eval"
    assert command[5] == "eval_openfull_zip"
    assert command.count("trainer.experiment_name=explicit-name") == 1
    assert "trainer.experiment_name=eval-test" not in command
    assert "actor_rollout_ref.model.path=/models/checkpoint with spaces" in command


def test_unsynced_node_fails_before_installing():
    with pytest.raises(RuntimeError, match="Sync this checkout"):
        submit.prepare_node(str(Path(__file__).resolve().parents[3]), "different-source-hash")
