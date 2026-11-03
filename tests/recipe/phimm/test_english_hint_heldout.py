import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

DATA = Path.home() / "data/foreign_language_data/data"


def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


@pytest.mark.skipif(not DATA.is_dir(), reason="local foreign-language audit data is unavailable")
def test_heldout_languages_and_training_texts_are_disjoint():
    training = [json.loads(line) for name in ("noisy_ls.jsonl", "mls_foreign.jsonl", "cjk_foreign.jsonl")
                for line in (DATA / name).read_text(encoding="utf-8").splitlines()]
    test = [
        json.loads(line)
        for line in (DATA / "english_hint_heldout/all.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(training) == 900 and len(test) == 110
    assert Counter(row["source_language"] for row in test) == dict.fromkeys(
        ("en", "de", "es", "fr", "it", "nl", "pl", "pt", "zh", "ja", "ko"), 10
    )
    assert not ({normalize(r["text"]) for r in training} & {normalize(r["text"]) for r in test})
    assert not ({r["id"] for r in training} & {r["id"] for r in test})
    assert all(r["source_corpus"] == "FLEURS" and r["source_split"] == "test" for r in test)
    assert all((DATA / Path(r["audio_path"]).relative_to("scripts/foreign_language_data/data")).is_file() for r in test)
    evidence = json.loads((DATA / "english_hint_heldout/evidence.json").read_text(encoding="utf-8"))
    assert evidence["training_snapshot_verification"]["all_three_training_jsonl_sha256_match_snapshot"]
    actual_sha = hashlib.sha256((DATA / "english_hint_heldout/all.jsonl").read_bytes()).hexdigest()
    assert actual_sha == evidence["all_jsonl_sha256"]
