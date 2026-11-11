"""Local invariant tests; uses synthetic metadata and never opens audio or a network."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("select_subset.py")


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = [{"id": key, "text": f"词 {key}", "audio_path": f"zip!/{key}.wav",
                      "source_split": "train", "source_row": index, "parent_id": f"p-{key}"}
                     for index, key in enumerate(("a", "b", "c"))]
        self.scores = [dict(row, hypothesis="假设", n_ref=2, n_err=errors,
                            n_ins=0, n_del=0, n_sub=errors)
                       for row, errors in zip(self.rows, (2, 4, 4))]
        self.heldout = [{"id": "test-1", "text": "测试", "audio_path": "test.wav",
                         "source_split": "test", "parent_id": "test-parent"}]

    def write(self, name, rows):
        path = self.root / name
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                        encoding="utf-8")
        return str(path)

    def select(self, extra=(), scores=True, output="output", ok=True):
        source = self.write("source.jsonl", self.rows)
        heldout = self.write("heldout.jsonl", self.heldout)
        destination = self.root / output
        command = [sys.executable, str(SCRIPT), "--source", source,
                   "--expected-source-count", str(len(self.rows)), "--role", "train",
                   "--heldout", heldout, "--top-k", "2", "--output-dir", str(destination)]
        if scores:
            command += ["--rank", "errors", "--scores", self.write("scores.jsonl", self.scores),
                        "--scorer-id", "fixture-cer-v1"]
        else:
            command += ["--rank", "deterministic"]
        result = subprocess.run(command + list(extra), capture_output=True, text=True)
        self.assertEqual(result.returncode == 0, ok, result.stderr)
        if not ok:
            self.assertFalse(destination.exists(), "invalid input published an output")
            return result.stderr
        return {name: (destination / name).read_text(encoding="utf-8")
                for name in ("manifest.jsonl", "audit.jsonl", "summary.json")}

    def test_absolute_error_ranking_ties_and_original_references(self):
        result = self.select()
        rows = [json.loads(line) for line in result["manifest.jsonl"].splitlines()]
        self.assertEqual(rows, [self.rows[1], self.rows[2]])
        audit = [json.loads(line) for line in result["audit.jsonl"].splitlines()]
        self.assertEqual(audit[0]["evidence_type"], "model_confirmed")
        self.assertNotIn("hypothesis", rows[0])
        self.scores.reverse()
        repeated = self.select(output="repeated")
        self.assertEqual(result["manifest.jsonl"], repeated["manifest.jsonl"])

    def test_partial_and_extra_scores_rejected(self):
        removed = self.scores.pop()
        self.assertIn("exact source universe", self.select(ok=False))
        self.scores.extend([removed, dict(removed, id="extra")])
        self.assertIn("exact source universe", self.select(ok=False))

    def test_duplicate_ids_rejected(self):
        self.rows.append(dict(self.rows[0]))
        self.assertIn("duplicate id", self.select(ok=False))

    def test_reference_and_audio_selector_mismatch_rejected(self):
        for field in ("text", "audio_path"):
            with self.subTest(field=field):
                original = self.scores[0][field]
                self.scores[0][field] = "different"
                self.assertIn("differs from source", self.select(ok=False))
                self.scores[0][field] = original

    def test_edit_sum_rejected(self):
        self.scores[0]["n_ins"] = 1
        self.assertIn("edit counts disagree", self.select(ok=False))

    def test_heldout_parent_exclusion(self):
        self.heldout[0]["parent_id"] = "p-b"
        result = self.select()
        self.assertEqual([json.loads(r)["id"] for r in result["manifest.jsonl"].splitlines()],
                         ["c", "a"])
        summary = json.loads(result["summary.json"])
        self.assertEqual(summary["overlap_field_counts"]["parent_id"], 1)

    def test_heldout_audio_id_and_optional_text_exclusion(self):
        for field in ("audio_path", "id", "text"):
            with self.subTest(field=field):
                original = self.heldout[0][field]
                self.heldout[0][field] = self.rows[1][field]
                extra = ("--exclude-same-text",) if field == "text" else ()
                result = self.select(extra=extra, output=f"overlap-{field}")
                self.assertEqual(json.loads(result["summary.json"])["overlap_field_counts"][field], 1)
                self.heldout[0][field] = original

    def test_unexpected_source_count_rejected(self):
        self.assertIn("expected universe", self.select(
            extra=("--expected-source-count", "4"), ok=False))

    def test_wrong_train_split_rejected(self):
        self.rows[0]["source_split"] = "test"
        self.assertIn("incompatible", self.select(ok=False))

    def test_insufficient_eligible_rows_rejected(self):
        self.assertIn("refusing padding", self.select(extra=("--top-k", "4"), ok=False))

    def pattern(self, field="text", regex="词"):
        path = self.root / "patterns.json"
        path.write_text(json.dumps({"version": "fixture-v1", "mode": "any", "patterns": [
            {"id": "lexeme", "field": field, "regex": regex}]}), encoding="utf-8")
        return ("--patterns", str(path))

    def test_reference_retrieval_is_candidate_and_order_independent(self):
        extra = self.pattern()
        result = self.select(extra=extra, scores=False)
        audit = [json.loads(r) for r in result["audit.jsonl"].splitlines()]
        self.assertTrue(all(r["evidence_type"] == "reference_feature_candidate" for r in audit))
        self.rows.reverse()
        other = self.select(extra=extra, scores=False, output="permuted")
        self.assertEqual(result["manifest.jsonl"], other["manifest.jsonl"])

    def test_missing_pattern_field_fails(self):
        self.assertIn("missing/non-scalar", self.select(extra=self.pattern(field="unknown"), ok=False))

    def test_derived_feature_without_scores_not_marked_confirmed(self):
        for row in self.rows:
            row["acoustic_tag"] = "noisy"
        result = self.select(extra=self.pattern(field="acoustic_tag", regex="noisy"), scores=False)
        audit = [json.loads(r) for r in result["audit.jsonl"].splitlines()]
        self.assertTrue(all(r["evidence_type"] == "feature_candidate" for r in audit))

    def test_zero_error_row_not_labeled_confirmed(self):
        self.scores[0].update(n_err=0, n_sub=0)
        result = self.select(extra=("--min-errors", "0", "--top-k", "3"))
        audit = [json.loads(r) for r in result["audit.jsonl"].splitlines()]
        self.assertEqual(audit[-1]["evidence_type"], "scored_correct_candidate")


if __name__ == "__main__":
    unittest.main()
