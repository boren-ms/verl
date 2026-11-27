# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Validate binary record integrity and caption/audio pairing during preparation."""

import importlib.util
import json
import struct
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "prepare_long_chunk_jsonl", Path(__file__).parents[3] / "scripts/prepare_long_chunk_jsonl.py"
)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def sources(tmp_path, records):
    file_set = tmp_path / "file_set.json"
    file_set.write_text(json.dumps({"fileInfo": [{"name": "chunk_one", "count": len(records), "duration": 300}]}))
    with (tmp_path / "chunk_one.info").open("wb") as stream:
        stream.write(b"Info" + struct.pack("<I", 1))
        for index, record in enumerate(records):
            payload = json.dumps(record).encode()
            stream.write(struct.pack("<II", index, len(payload)) + payload)
    return file_set


def caption(text, duration=100, wer=0.1):
    return {
        "tn_transcription": text,
        "transcription_source": "display_human_caption_adjusted",
        "duration": duration,
        "normalized_WER": wer,
        "filename": "parent.wav",
    }


def test_filtered_rows_keep_original_audio_indices(tmp_path):
    file_set = sources(tmp_path, [caption(""), caption("human caption 中文"), caption("poor caption", wer=0.9)])
    output = tmp_path / "training.jsonl"
    result = prepare.build(file_set, tmp_path, "az://orange/data/ChunkFiles/", output, max_wer=0.25)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["text"] == "human caption 中文"
    assert rows[0]["audio_chunk"] == "az://orange/data/ChunkFiles/chunk_one.audio:3:1"
    assert rows[0]["audio_length"] == 100
    assert result["counts"] == {"input_rows": 3, "empty_reference": 1, "output_rows": 1, "wer_filtered": 1}
    assert result["audio_hours"] == 100 / 3600


def test_refuses_machine_caption_source_without_replacing_output(tmp_path):
    record = caption("machine text")
    record["transcription_source"] = "machine"
    file_set = sources(tmp_path, [record])
    output = tmp_path / "training.jsonl"
    output.write_text("previous output")
    with pytest.raises(ValueError, match="Unexpected human-caption source"):
        prepare.build(file_set, tmp_path, "az://orange/data", output)
    assert output.read_text() == "previous output"
    assert not output.with_suffix(".jsonl.partial").exists()


@pytest.mark.parametrize("corruption", ["truncated", "wrong_index", "trailing", "wrong_version"])
def test_rejects_corrupted_binary_chunks(tmp_path, corruption):
    file_set = sources(tmp_path, [caption("human text")])
    path = tmp_path / "chunk_one.info"
    data = bytearray(path.read_bytes())
    if corruption == "truncated":
        data = data[:-1]
    elif corruption == "wrong_index":
        data[8:12] = struct.pack("<I", 9)
    elif corruption == "trailing":
        data += b"unexpected record"
    else:
        data[4:8] = struct.pack("<I", 2)
    path.write_bytes(data)
    with pytest.raises(ValueError):
        prepare.build(file_set, tmp_path, "az://orange/data", tmp_path / "training.jsonl")


def test_deterministic_output_and_duration_filters(tmp_path):
    file_set = sources(tmp_path, [caption("short", 30), caption("valid", 100), caption("long", 2000)])
    args = (file_set, tmp_path, "az://orange/data", tmp_path / "training.jsonl")
    first = prepare.build(*args, min_duration=60, max_duration=1800)
    second = prepare.build(*args, min_duration=60, max_duration=1800)
    assert first == second
    assert first["counts"]["duration_filtered"] == 2


def test_rejects_duplicate_chunk_names(tmp_path):
    file_set = sources(tmp_path, [caption("human text")])
    index = json.loads(file_set.read_text())
    index["fileInfo"] *= 2
    file_set.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="Duplicate chunk names"):
        prepare.build(file_set, tmp_path, "az://orange/data", tmp_path / "training.jsonl")
