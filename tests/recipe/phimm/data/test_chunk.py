import importlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

chunk_module = importlib.import_module("recipe.phimm.data.chunk")
load_specs = chunk_module.load_specs
parse_data = chunk_module.parse_data


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


def test_parse_data_decodes_pcm_s16le():
    samples = np.array([-32768, -16384, 0, 16384, 32767], dtype="<i2")

    audio, sample_rate = parse_data(
        samples.tobytes(),
        "audio",
        audio_encoding="pcm_s16le",
        audio_sample_rate=16000,
    )

    assert sample_rate == 16000
    np.testing.assert_allclose(audio, samples.astype(np.float32) / 32768.0)


def test_parse_data_rejects_malformed_pcm_s16le():
    with pytest.raises(ValueError, match="Invalid pcm_s16le payload size"):
        parse_data(
            b"\x00",
            "audio",
            audio_encoding="pcm_s16le",
            audio_sample_rate=16000,
        )


def test_parse_data_decodes_flac():
    expected = np.linspace(-0.5, 0.5, 160, dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, expected, 16000, format="FLAC")

    audio, sample_rate = parse_data(
        buffer.getvalue(),
        "audio",
        audio_sample_rate=16000,
    )

    assert sample_rate == 16000
    np.testing.assert_allclose(audio, expected, atol=4e-5)


def test_parse_data_rejects_flac_sample_rate_mismatch():
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(160, dtype=np.float32), 8000, format="FLAC")

    with pytest.raises(ValueError, match="does not match expected 16000"):
        parse_data(
            buffer.getvalue(),
            "audio",
            audio_sample_rate=16000,
        )
