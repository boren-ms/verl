import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github/skills/asr-detail-compare/scripts/compare_result_details.py"
)
SPEC = importlib.util.spec_from_file_location("compare_result_details", SCRIPT_PATH)
compare_result_details = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(compare_result_details)


def test_html_highlights_baseline_to_target_changes_not_reference_errors():
    row = {
        "comparison_id": "sample",
        "ref": "the reference wording",
        "raw_ref": "The reference wording.",
        "hyp_baseline": "shared model wording",
        "hyp_target": "shared changed wording",
        "raw_hyp_baseline": "shared model wording",
        "raw_hyp_target": "shared changed wording",
        "baseline_wer": 2 / 3,
        "target_wer": 2 / 3,
        "baseline_errors": 2,
        "target_errors": 2,
        "error_delta": 0,
    }

    report = compare_result_details.build_comparison_html([row], "Comparison")

    assert '<mark class="diff-change diff-removed" tabindex="-1">model</mark>' in report
    assert '<mark class="diff-change diff-added" tabindex="-1">changed</mark>' in report
    assert '<p class="transcript-text">the reference wording</p>' in report
    assert "diff-reference" not in report