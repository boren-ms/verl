"""Periodic exports must prove completed checkpoints without claiming finality."""

import copy

import pytest
from recipe.phimm.sft.export_hf import validate_training_evidence


def evidence():
    return {
        "submission_id": "train",
        "status": "RUNNING",
        "target_optimizer_step": 32260,
        "last_completed_optimizer_step": 1100,
        "checkpoint": {"step": 1000, "complete_expected_files": 98},
        "nonfinite_metrics": [],
        "teacher_forced_validation": [{"step": 1000}, {"step": 1100}],
    }


def test_periodic_export_accepts_committed_snapshot_during_training_but_final_export_does_not():
    monitor = evidence()
    validate_training_evidence(monitor, step=1000, submission_id="train", world_size=32, intermediate=True)
    with pytest.raises(AssertionError):
        validate_training_evidence(monitor, step=1000, submission_id="train", world_size=32)


@pytest.mark.parametrize(
    "change",
    [
        {"status": "FAILED"},
        {"submission_id": "other"},
        {"last_completed_optimizer_step": 999},
        {"checkpoint": {"step": 900, "complete_expected_files": 98}},
        {"checkpoint": {"step": 1000, "complete_expected_files": 26}},
        {"teacher_forced_validation": [{"step": 900}]},
        {"nonfinite_metrics": ["nan"]},
    ],
)
def test_periodic_export_rejects_unverified_snapshot(change):
    monitor = {**copy.deepcopy(evidence()), **change}
    with pytest.raises(AssertionError):
        validate_training_evidence(monitor, step=1000, submission_id="train", world_size=32, intermediate=True)


def test_final_export_still_requires_exact_successful_target_and_final_validation():
    monitor = evidence()
    monitor.update(
        status="SUCCEEDED",
        target_optimizer_step=1000,
        last_completed_optimizer_step=1000,
        teacher_forced_validation=[{"step": 1000}],
    )
    validate_training_evidence(monitor, step=1000, submission_id="train", world_size=32)
