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
"""Probe the real ASR dataset and reference context using retained Qwen outputs.

The probe is QA only. A duration-rejected parent's real segments may be used to
test loader semantics; its rejection remains recorded and it is never relabelled
as an accepted training sample. No audio file or training job is created.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from align_long_human_captions import fingerprint, rows, save, write_row
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--orange-root", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = next(
        (row for row in rows(args.raw_manifest) if row["word_duration_filter"]["passed"] and len(row["segments"]) >= 2),
        None,
    )
    if source is None:
        source = next(
            row for row in rows(args.raw_manifest) if not row.get("alignment_error") and len(row["segments"]) >= 2
        )
    selected = source["segments"][:2]
    probe = {
        "id": source["id"] + ":loader_qa",
        "parent_id": source["id"],
        "parent_audio_path": source.get("audio_chunk") or source["audio_path"],
        "audio_path": [part[0] for part in selected],
        "text": " ".join(part[1] for part in selected),
        "segment_reference_texts": [part[1] for part in selected],
        "segments": selected,
        "segment_metadata": source["segment_metadata"][:2],
        "segment_indices": [0, 1],
        "audio_length": sum(part["end"] - part["start"] for part in source["segment_metadata"][:2]),
        "language": "English",
        "keywords": [],
        "transcription_source": source["transcription_source"],
        "word_duration_filter": source["word_duration_filter"],
        "alignment_flags": source.get("alignment_flags", []),
        "pairing_policy": "sliding_adjacent_stride_1_same_parent",
        "qa_only": True,
    }
    manifest = args.output_dir / "loader_probe.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        write_row(handle, probe)
    orange = args.orange_root.rstrip("/") + "/loader_probe.jsonl"
    subprocess.run(["bbb", "cp", str(manifest), orange], check=True)
    repo = Path(__file__).resolve().parents[1]
    conf_dir = repo / "plugins/phimm/src/recipe/phimm/config"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(
            config_name="data/train_data/long/hc_fy24q3_1_reference_context_training",
            overrides=[f"hydra.searchpath=[file://{repo / 'verl/trainer/config'}]"],
        )
    assert cfg.asr.context_aware.training_source == "reference"
    assert cfg.asr.context_aware.validation_source == "generated"
    assert cfg.asr.context_aware.max_history == 1
    source_conf = OmegaConf.to_container(
        OmegaConf.load(conf_dir / "data/train_data/long/hc_fy24q3_1_2segments_reference_context.yaml"), resolve=True
    )[0]
    source_conf["jsonl_paths"] = [orange]
    # Retain the real YAML's formatting and metadata fields; substitute only the QA manifest.
    cfg.data.train_data = [source_conf]
    cfg.data.val_data = [source_conf]
    OmegaConf.update(cfg, "data.mix", None, force_add=True)
    from recipe.phimm.agentic.agent_loop import ASRAgentLoop
    from recipe.phimm.agentic.context_prompt import context_source
    from recipe.phimm.agentic.dataset import ASRDataset
    from recipe.phimm.utils.audio import load_raw_audio

    dataset = ASRDataset(cfg.data.train_files, None, cfg.data)
    assert len(dataset) == 1
    row = dataset[0]
    segments = row["asr_segments"]
    assert len(segments) == 2
    assert [segment["audio_path"] for segment in segments] == probe["audio_path"]
    assert [segment["asr_reference_text"] for segment in segments] == probe["segment_reference_texts"]
    prompts = ASRAgentLoop._reference_prompts(ASRAgentLoop.__new__(ASRAgentLoop), cfg.asr.context_aware, segments)
    assert prompts == ["", "Previous context: " + probe["segment_reference_texts"][0]]
    assert context_source(cfg.asr.context_aware, training=True) == "reference"
    assert context_source(cfg.asr.context_aware, training=False) == "generated"
    parent, sr = load_raw_audio({"audio_chunk": probe["parent_audio_path"]})
    checks = []
    for segment, bounds in zip(segments, probe["segment_metadata"], strict=True):
        waveform, rate = load_raw_audio(segment)
        expected = parent[int(bounds["start"] * sr) : int(bounds["end"] * sr)]
        assert rate == sr == 16000 and np.array_equal(waveform, expected)
        checks.append(
            {
                "path": segment["audio_path"],
                "frames": len(waveform),
                "sha256": hashlib.sha256(waveform.tobytes()).hexdigest(),
            }
        )
    save(
        args.output_dir / "VERIFIED.json",
        {
            "status": "actual_ASRDataset_reference_context_Orange_selectors_verified",
            "qa_only": True,
            "parent_id": probe["parent_id"],
            "source_quality_filter": probe["word_duration_filter"],
            "raw_manifest_sha256": fingerprint(args.raw_manifest),
            "probe_manifest_sha256": fingerprint(manifest),
            "runner_sha256": fingerprint(Path(__file__)),
            "first_reference_context_empty": True,
            "second_reference_context_exactly_first_human_label": True,
            "training_source": "reference",
            "validation_source": "generated",
            "max_history": 1,
            "checks": checks,
            "training_started": False,
        },
    )
    print(json.dumps({"status": "verified", "dataset_rows": 1, "ordered_segments": 2, "checks": checks}))


if __name__ == "__main__":
    main()
