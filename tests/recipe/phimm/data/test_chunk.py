import importlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_specs = importlib.import_module("recipe.phimm.data.chunk").load_specs


def test_load_specs_supports_spec_level_language(tmp_path):
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(
        json.dumps(
            {
                "language": "de",
                "data_sources": [
                    {
                        "manifest_file": "/datablob1/users/data/file_set_train.json",
                        "chunk_path": "/datablob1/users/data/chunks/",
                        "trans_path": "/datablob1/users/data/transcribe/",
                    }
                ],
            }
        )
    )

    specs = load_specs(str(spec_file))

    assert specs == [
        {
            "manifest_file": "az://orngwus2cresco/data/speech/users/data/file_set_train.json",
            "chunk_path": "az://orngwus2cresco/data/speech/users/data/chunks/",
            "trans_path": "az://orngwus2cresco/data/speech/users/data/transcribe/",
            "language": "de",
        }
    ]


def test_load_specs_preserves_data_source_language(tmp_path):
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(
        json.dumps(
            {
                "language": "en",
                "data_sources": [
                    {
                        "manifest_file": "/datablob1/users/data/file_set_train.json",
                        "chunk_path": "/datablob1/users/data/chunks/",
                        "trans_path": "/datablob1/users/data/transcribe/",
                        "language": "de",
                    }
                ],
            }
        )
    )

    specs = load_specs(str(spec_file))

    assert specs[0]["language"] == "de"
