# SpecAugment-style WAV augmentation

Run from the checkout root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy scipy soundfile
.venv/bin/python scripts/specaugment_wav.py input.wav output.wav --seed 42
```

This standalone CPU tool accepts mono or multichannel WAV files. It applies
time warping, frequency masking, and time masking, then saves an augmented WAV
with the same sample rate, sample count, and channel count. Channels share the
same augmentation policy. The source file cannot be overwritten; replacing an
existing output requires `--overwrite`. Parent directories are created as needed.
Output uses 32-bit floating-point WAV to preserve reconstructed samples without
clipping or gain normalization. Some players require integer PCM instead.

The original [SpecAugment paper](https://arxiv.org/abs/1904.08779) augments
filter-bank features consumed by an ASR model; it does not reconstruct audio.
This waveform adaptation instead uses invertible linear STFT magnitudes, a
piecewise-linear time warp, and inverse STFT with the original phase. It is an
approximation intended for offline WAV generation, not a reproduction of the
paper's log-mel transform or its sparse image warp. Re-extracting features from
the saved WAV need not yield exact zero masks because overlapping STFT windows
redistribute energy. Audition results before using aggressive settings for ASR.

Defaults are a 25 ms Hann window, 10 ms hop, up to 80 frames of time warping,
two frequency masks up to 27 **linear FFT bins** wide, and two time masks up to
100 frames wide. Each time mask is also capped at 20% of the frame count.
These are configurable starting points, not paper benchmark presets. Mask
widths are sampled uniformly including zero, and masks may overlap. Time
warping is skipped when there are too few frames for the requested warp range.
Short clips are zero padded for STFT and cropped back to their original length.

For masking only, or a gentler policy:

```bash
.venv/bin/python scripts/specaugment_wav.py input.wav output.wav \
  --seed 42 --time-warp-param 0 --freq-mask-param 12 \
  --time-mask-param 20 --num-freq-masks 1 --num-time-masks 1
```

Use `--help` for all options. Set a mask count or its width to zero to disable
that masking operation, and `--time-warp-param 0` to disable warping. Repeating
the same input, settings, and seed in the same dependency environment produces
the same output. Without a seed each invocation samples a fresh policy.

The NumPy API is available as
`scripts.specaugment_wav.augment_waveform(audio, sample_rate, config, seed=42)`
and the file API as `augment_wav(input_path, output_path, config, seed=42)`.
