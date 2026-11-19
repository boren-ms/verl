# LS SpecAugment E4 comparison

Reference chat: `01a183af-ae1c-7680-97fd-aeca129e402f` (URGENT E4, OSS Noisy).
Candidate: `remax_urgent_specaug_mild_v0_e4`.
Compare against `remax_urgent_raw_v0_e4` and `remax_urgent_speedonly_v0_e4`.

## Data generation

`config/augment_mild.yaml` is the authoritative generation policy. Each source
FLAC is transformed once offline. Random widths and starts are independently
sampled for each audio; the per-audio seed derives from SHA256 of global seed,
sample ID and source path. Re-running or changing thread count preserves the
sample's policy. This is a fixed augmented dataset, not per-epoch redraws.
Time warp is enabled at W=20 as explicitly requested. Extremely short clips
may skip warping under the implementation's frame-length condition.

F=8 means an inclusive integer width range [0,8] linear STFT bins. T=10 is an
inclusive range [0,min(10,floor(0.02*num_frames))]. Both mask counts are one.
Warp picks a legal interior anchor and displacement in [-20,20] frames,
clipped so endpoints remain strictly ordered. All positions are randomized.
Each output row records its seed and actual warp/mask coordinates.

The builder reads **source_audio_path**, not the old augmented audio_path.
It preserves all 20,424 source IDs, labels and order. No speed, additive noise,
RIR, resampling, gain normalization or filtering is applied. Output is float
WAV. Hashes of uploaded WAV bytes are verified before the manifest is published.
A smoke build never publishes the final manifest.

Run on the remote data node from `/root/code/verl`:

```bash
python scripts/build_ls_specaugment_dataset.py \
  --source-manifest az://orngwus2cresco/data/boren/data/verl/gen_cached/ls_raw_rp_edge_nodigits_aug_audio/data.jsonl \
  --source-sha256 14e3ea1f46e5523de9d08c4d3de6bae5c28961acae5bc461aae8c50a31f89b81 \
  --config experiments/urgent_specaugment/config/augment_mild.yaml \
  --output-dir az://orngwus2cresco/data/boren/data/verl/gen_cached/ls_raw_rp_edge_nodigits_specaug_mild_w20_f8_t10_p002_v1_20261119 \
  --work-dir /root/specaugment_ls_mild_build --workers 24
```

## Training and provenance

`config/data_specaug_mild.yaml` describes the generated data. The training YAML
inherits a frozen, fully resolved copy of the raw baseline's effective recipe
(`config/reference_raw_e4.yaml`), and changes only training data and output
identity. Preserve 4 epochs, absolute final optimizer step 636, 2x8 GPUs,
train batch 128, rollout n=16, LR 1e-6 and LoRA rank/alpha 320/640.
The inherited URGENT delta validation remains unchanged.

**Use the pinned historical runtime, not current Main ASR defaults.** The
reference's runtime is based on Git `baf505939540b722eacad82926bc1933b6cc0526`,
plus its local ASR submitter overlay. Copy this experiment bundle and generation
scripts into an isolated clone of the reference source, preserving baseline code.
Runtime source and exact bpush snapshot hashes are recorded in pipeline state.
The standalone config directory avoids relying on moved/deleted recipe defaults.

Compose before launch and compare every effective value against the frozen
reference, allowing only data and output identity changes. Train only after
all rows pass the historical canonical dataset loader and remote capacity is
verified. Do not mix the legacy reward/task schema with the current Main ASR
trainer. The final export/evaluation/report pipeline is recorded in ignored
`tmp/run_openasr_exp/remax_urgent_specaug_mild_v0_e4/state.json`.
