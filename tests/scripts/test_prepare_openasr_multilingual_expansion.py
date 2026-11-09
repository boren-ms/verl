import io
import json
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from scripts.prepare_openasr_multilingual_expansion import prepare


def snapshot(tmp_path, subtype="FLOAT"):
    local = tmp_path / "fleurs_vi"
    local.mkdir()
    wav = io.BytesIO()
    audio = np.linspace(-0.25, 0.25, 1600, dtype=np.float32)
    sf.write(wav, audio, 16000, format="WAV", subtype=subtype)
    audio, _ = sf.read(io.BytesIO(wav.getvalue()), dtype="float32")
    table = pa.Table.from_pylist([
        {"audio": {"bytes": wav.getvalue()}, "text": text, "id": "same-source-id"}
        for text in ["xin chào", "cảm ơn"]
    ])
    pq.write_table(table, local / "test.parquet")
    spec = {"result_name": "fleurs_vi", "hf_repo": "example/pinned",
            "source_revision": "pinned-revision", "parquet_file": "test.parquet",
            "examples": "2", "task": "openasr_ml", "metric": "wer"}
    return local, spec, audio


@pytest.mark.parametrize("subtype", ["FLOAT", "PCM_16"])
def test_preparation_preserves_rows_and_range_readable_audio(tmp_path, subtype):
    local, spec, audio = snapshot(tmp_path, subtype)
    proof = prepare(spec, "az://example/data/snapshot", tmp_path, prepare_only=True)
    rows = [json.loads(line) for line in (local / "data_zip.jsonl").read_text().splitlines()]
    assert [row["text"] for row in rows] == ["xin chào", "cảm ơn"]
    assert len({row["id"] for row in rows}) == 2
    assert proof["parents"] == 2
    assert all(row["include_in_average_wer"] is False for row in rows)
    with zipfile.ZipFile(local / "audio.zip") as archive, (local / "audio.zip").open("rb") as raw:
        for index, row in enumerate(rows):
            offset, size = map(int, row["audio_path"].rsplit("!", 1)[1].split(":"))
            raw.seek(offset)
            payload = raw.read(size)
            assert payload == archive.read(f"{index:06d}.wav")
            if subtype == "PCM_16":
                original = pq.read_table(local / "test.parquet").to_pylist()[index]["audio"]["bytes"]
                assert payload == original
            decoded, sr = sf.read(io.BytesIO(payload), dtype="float32")
            assert sr == 16000
            np.testing.assert_array_equal(decoded, audio)


def test_changed_snapshot_row_count_fails_before_output(tmp_path):
    local, spec, _ = snapshot(tmp_path)
    spec["examples"] = "3"
    with pytest.raises(ValueError, match="expected 3 rows, got 2"):
        prepare(spec, "az://example/data/snapshot", tmp_path, prepare_only=True)
    assert not (local / "audio.zip").exists()
