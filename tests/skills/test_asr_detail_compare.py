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
        "baseline_substitutions": 2,
        "baseline_deletions": 0,
        "baseline_insertions": 0,
        "target_errors": 2,
        "target_substitutions": 2,
        "target_deletions": 0,
        "target_insertions": 0,
        "error_delta": 0,
    }

    report = compare_result_details.build_comparison_html([row], "Comparison")

    assert (
        '<mark class="aligned-token word-wrong diff-change diff-removed" '
        'data-align="ref-1" tabindex="0">model</mark>'
    ) in report
    assert (
        '<mark class="aligned-token word-wrong diff-change diff-added" '
        'data-align="ref-1" tabindex="0">changed</mark>'
    ) in report
    assert '<span class="aligned-token reference-token" data-align="ref-1" tabindex="0">reference</span>' in report
    assert '<mark class="aligned-token word-correct" data-align="ref-2" tabindex="0">wording</mark>' in report
    assert 'data-error-color="word-wrong" checked> Substitution</label>' in report
    assert 'data-error-color="word-deleted" checked> Deleted</label>' in report
    assert '--wrong: #e8ecee;' in report
    assert '--wrong-ink: #39464c;' in report
    assert '--deleted: #d9eaf7;' in report
    assert '--deleted-ink: #245b78;' in report
    assert '.word-deleted { background: var(--deleted); color: var(--deleted-ink); }' in report
    assert 'class="legend-swatch model-change">Model change</span>' in report
    assert 'class="error-count word-wrong">Sub 2</span>' in report
    assert 'class="error-count word-deleted">Del 0</span>' in report
    assert 'class="error-count word-inserted">Ins 0</span>' in report
    assert 'class="error-count word-number-error">Num 0</span>' in report
    assert 'title="Number errors overlap the edit categories"' in report
    assert '.word-correct { background: transparent; color: inherit; }' in report
    assert '.reference-token { background: transparent; color: inherit; }' in report
    assert '.reference-panel .alignment-gap { background: transparent; color: var(--muted); }' in report
    assert 'const selectAligned = (token) =>' in report
    assert '.transcript-condensed .hidden-context.is-aligned { display: inline; }' in report
    assert '.compare-grid {' in report
    assert 'display: flex;' in report
    assert 'data-after="reference" role="separator"' in report
    assert 'data-after="baseline" role="separator"' in report
    assert 'resizer.addEventListener("pointerdown", (event) =>' in report
    assert 'document.addEventListener("pointermove", move);' in report
    assert 'resizer.addEventListener("keydown", (event) =>' in report
    assert 'resizer.addEventListener("dblclick", resetPanelWidths);' in report
    assert 'const minimumWidth = Math.min(180, combinedWidth / 2);' in report
    assert 'const setReportPanelVisibility = (side, visible) =>' in report
    assert 'setReportPanelVisibility(toggle.dataset.panelToggle, toggle.checked);' in report
    assert 'const setReportSyncScroll = (enabled) =>' in report
    assert 'setReportSyncScroll(event.currentTarget.checked);' in report
    assert 'const setReportFullContext = (showFull) =>' in report
    assert 'setReportFullContext(card.classList.contains("transcript-condensed"));' in report
    assert 'const setReportErrorColor = (errorClass, enabled) =>' in report
    assert 'setReportErrorColor(toggle.dataset.errorColor, toggle.checked);' in report
    assert 'body.color-word-wrong-off .word-wrong' in report
    assert 'body.color-word-deleted-off .word-deleted' in report
    assert 'aria-label="Visible transcript panels for all utterances"' in report
    assert '.audio-player { display: flex; flex: 0 0 auto; align-items: center; white-space: nowrap; }' in report
    assert 'width: 260px;' in report
    assert 'overflow-x: auto;' in report
    assert 'flex-wrap: nowrap;' in report
    assert '.metrics {' in report
    assert 'margin-bottom: 12px;' in report
    assert '.metric {' in report
    assert 'white-space: nowrap;' in report
    assert '.error-breakdown { display: inline-flex; flex-wrap: nowrap;' in report
    assert '.compare-grid,\n      .metrics {' not in report
    assert '.transcript-panel { height: 32rem;' in report
    assert '.transcript-panel { height: 24rem; }' in report
    assert '.reference-block .transcript-text { max-height:' not in report
    assert 'pane.scrollTop += itemRect.top - paneRect.top - (pane.clientHeight - itemRect.height) / 2;' in report
    assert report.index('data-side="reference"') < report.index('data-side="baseline"') < report.index('data-side="target"')


def test_compound_alignment_is_correct_and_links_all_reference_words():
    keys, correct, missing, insertions = compare_result_details.align_hypothesis_to_reference(
        ["ice", "cream", "today"], ["icecream", "today"]
    )

    assert keys == ["ref-0 ref-1", "ref-2"]
    assert correct == [True, True]
    assert missing == {}
    assert insertions == {}


def test_compact_context_keeps_only_words_surrounding_errors():
    correct = [True] * 20
    correct[10] = False

    visible = compare_result_details.error_context_visibility(correct, {})

    assert visible == set(range(5, 16))


def test_html_highlights_inserted_model_words_in_yellow():
    row = {
        "comparison_id": "sample",
        "ref": "one two",
        "raw_ref": "one two",
        "hyp_baseline": "one extra two",
        "hyp_target": "one two",
        "raw_hyp_baseline": "one extra two",
        "raw_hyp_target": "one two",
        "baseline_wer": 0.5,
        "target_wer": 0.0,
        "baseline_errors": 1,
        "target_errors": 0,
        "error_delta": -1,
    }

    report = compare_result_details.build_comparison_html([row], "Comparison")

    assert 'class="aligned-token word-inserted diff-change diff-removed"' in report
    assert '>extra</mark>' in report
    assert 'data-error-color="word-inserted" checked> Inserted</label>' in report
    assert '.word-inserted { background: var(--inserted); color: var(--inserted-ink); }' in report


def test_html_highlights_number_errors_in_red():
    row = {
        "comparison_id": "sample",
        "ref": "revenue was 2024 dollars",
        "raw_ref": "Revenue was 2024 dollars",
        "hyp_baseline": "revenue was 2025 dollars",
        "hyp_target": "revenue was 2024 dollars",
        "raw_hyp_baseline": "Revenue was 2025 dollars",
        "raw_hyp_target": "Revenue was 2024 dollars",
        "baseline_wer": 0.25,
        "target_wer": 0.0,
        "baseline_errors": 1,
        "target_errors": 0,
        "error_delta": -1,
    }

    report = compare_result_details.build_comparison_html([row], "Comparison")

    assert 'class="aligned-token word-number-error diff-change diff-removed"' in report
    assert '>2025</mark>' in report
    assert 'data-error-color="word-number-error" checked> Number error</label>' in report
    assert '.word-number-error { background: var(--number-error); color: var(--number-error-ink); }' in report
    assert '--number-error: #f8d9d4;' in report
    assert '--number-error-ink: #8d2921;' in report
    assert 'class="error-count word-number-error">Num 1</span>' in report


def test_baseline_only_html_hides_target_and_visible_panels_fill_grid():
    row = {
        "comparison_id": "sample",
        "ref": "reference words",
        "raw_ref": "Reference words",
        "hyp_baseline": "baseline words",
        "raw_hyp_baseline": "baseline words",
        "baseline_wer": 0.5,
        "baseline_errors": 1,
    }

    report = compare_result_details.build_comparison_html(
        [row], "Baseline", show_target=False
    )

    assert 'data-side="reference"' in report
    assert 'data-side="baseline"' in report
    assert 'class="panel transcript-panel panel-hidden panel-unavailable" data-side="target"' in report
    assert 'data-panel-toggle="reference" checked' in report
    assert 'data-panel-toggle="baseline" checked' in report
    assert 'data-panel-toggle="target" disabled' in report
    assert '<span class="label">Target</span>' not in report
    assert 'class="legend-swatch model-change"' not in report
    assert '.transcript-panel.panel-hidden { display: none; }' in report
    assert 'resetCardPanelLayout(card);' in report
    assert 'setReportPanelVisibility(toggle.dataset.panelToggle, toggle.checked);' in report


def test_reference_target_html_hides_synthetic_baseline_panel():
    row = {
        "comparison_id": "sample",
        "ref": "reference words",
        "raw_ref": "Reference words",
        "hyp_baseline": "reference words",
        "hyp_target": "target words",
        "raw_hyp_baseline": "Reference words",
        "raw_hyp_target": "target words",
        "baseline_wer": 0.0,
        "target_wer": 0.5,
        "baseline_errors": 0,
        "target_errors": 1,
        "error_delta": 1,
    }

    report = compare_result_details.build_comparison_html(
        [row], "Target", show_baseline=False
    )

    assert 'data-side="reference"' in report
    assert 'class="panel transcript-panel panel-hidden panel-unavailable" data-side="baseline"' in report
    assert 'data-side="target"' in report
    assert 'data-panel-toggle="reference" checked' in report
    assert 'data-panel-toggle="baseline" disabled' in report
    assert 'data-panel-toggle="target" checked' in report
    assert '<span class="label">Baseline</span>' not in report
    assert '<span class="label">Target</span>' in report
    assert '<span class="label">Error delta</span>' not in report
    assert '<div class="verdict">Target review</div>' in report
    assert '<span class="index-delta">1 errors</span>' in report