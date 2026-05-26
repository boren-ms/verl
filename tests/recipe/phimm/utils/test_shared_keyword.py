import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

shared = importlib.import_module("recipe.phimm.utils.shared")
tn = importlib.import_module("recipe.phimm.utils.tn")


def test_has_missing_keyword_all_present_with_list_keywords():
    keywords = ["alice", "wonderland"]
    response = "hello alice in wonderland"

    assert shared.has_missing_keyword(keywords, response, norm_name="identity") is False


def test_has_missing_keyword_true_when_any_keyword_missing():
    keywords = ["alice", "bob"]
    response = "hello alice"

    assert shared.has_missing_keyword(keywords, response, norm_name="identity") is True


def test_has_missing_keyword_supports_serialized_keyword_field():
    keywords = '["alice", "bob"]'
    response = "alice and bob"

    assert shared.has_missing_keyword(keywords, response, norm_name="identity") is False


def test_has_missing_keyword_uses_default_tn_name_when_norm_not_provided(monkeypatch):
    monkeypatch.setattr(tn, "default_tn_name", lambda _lang: "lower")

    assert shared.has_missing_keyword(["abc"], "ABC", norm_name=None, lang="custom") is False


def test_has_missing_keyword_handles_dc_literal_with_identity_norm():
    keywords = ["D.C."]

    assert shared.has_missing_keyword(keywords, "I live in D.C.", norm_name="identity") is False
    assert shared.has_missing_keyword(keywords, "I live in DC", norm_name="identity") is True


def test_has_missing_keyword_handles_dc_with_punctuation_stripped_norm():
    keywords = ["D.C."]

    assert shared.has_missing_keyword(keywords, "I live in DC", norm_name="simple") is False
