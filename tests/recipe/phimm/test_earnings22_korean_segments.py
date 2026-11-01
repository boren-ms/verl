import json
import re
from collections import Counter
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not (Path.home() / "data/foreign_language_data/data").is_dir(),
    reason="local foreign-language audit data is unavailable",
)
def test_candidate_set_preserves_model_selection_and_original_references():
    data = Path.home() / "data/foreign_language_data/data/earnings22_korean_segments.jsonl"
    rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 70
    assert Counter(row["id"].split(":")[1] for row in rows) == {
        "5f91131019ea463c": 22,
        "c90eac9e8d8aece1": 37,
        "1152bfa19efb2628": 7,
        "ac9eb90653f86234": 4,
    }
    assert all(re.search(r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]", row["baseline_hypothesis"]) for row in rows)
    assert all(row["review_required"] and isinstance(row["text"], str) for row in rows)
    assert sum(not row["text"].strip() for row in rows) == 49
