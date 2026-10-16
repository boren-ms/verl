# Noise and RIR simulation

[`scripts/augment_audio.py`](../scripts/augment_audio.py) supports reproducible,
offline speech augmentation with recorded room impulse responses (RIRs) and
additive noise. This implements the noise/reverberation portion of an
URGENT-like simulation, not an exact reproduction of the challenge. It does not
simulate bandwidth limitation or clipping.

Dependencies: NumPy, SciPy, SoundFile, and blobfile. These are available in the
repository's audio/vLLM environment. Input audio, noise, and RIR directories
support local paths and blobfile URLs. Directories are searched recursively for
WAV and FLAC files.

## RIR sources used in URGENT 2024

### OpenSLR SLR28: real RIRs

- Dataset: <https://www.openslr.org/28/>
- Archive: <https://openslr.trmal.net/resources/28/rirs_noises.zip>
- Size: approximately 1.3 GB; 16 kHz, 16-bit audio.
- License listed by OpenSLR: Apache 2.0. Retain the archive's attribution notices.
- Real RIRs originate from RWCP, REVERB 2014, and Aachen AIR.

```bash
mkdir -p ~/data/rir
curl --fail --location --retry 3 \
  https://openslr.trmal.net/resources/28/rirs_noises.zip \
  -o ~/data/rir/rirs_noises.zip
unzip ~/data/rir/rirs_noises.zip -d ~/data/rir

# Select RIRs using the archive's list, not a filename suffix heuristic.
mkdir -p ~/data/rir/slr28_real_rirs
(
  cd ~/data/rir
  awk '{print $NF}' RIRS_NOISES/real_rirs_isotropic_noises/rir_list |
    while IFS= read -r rir; do
      cp -n "$rir" slr28_real_rirs/
    done
)
```

The archive also contains simulated RIRs; those are not the real RIR subset
used in the URGENT non-blind test set. Do not pass the entire archive directory
as `--rir_dir`, because it also contains ordinary noise recordings.

### MYRiAD v2: measured multi-array RIRs

- Original release, version 2.0: <https://zenodo.org/records/7389996>
- DOI: <https://doi.org/10.5281/zenodo.7389996>
- Economy archive, approximately 202 MB:
  <https://zenodo.org/api/records/7389996/files/MYRiAD_V2_econ.zip/content>
- Full archive, approximately 31.3 GB:
  <https://zenodo.org/api/records/7389996/files/MYRiAD_V2.zip/content>
- **The original Zenodo v2 record lists CC BY-NC-SA 4.0**, not CC BY 4.0.
  Follow the original release's terms; do not infer permission from a secondary
  dataset listing.
- Paper: <https://arxiv.org/abs/2301.13057>

```bash
curl --fail --location --retry 3 \
  https://zenodo.org/api/records/7389996/files/MYRiAD_V2_econ.zip/content \
  -o ~/data/rir/MYRiAD_V2_econ.zip
unzip ~/data/rir/MYRiAD_V2_econ.zip -d ~/data/rir/myriad_v2
```

Use the archive's documentation/access scripts to identify the impulse-response
recordings. Export RIRs to WAV/FLAC if necessary and collect **only RIRs** in a
dedicated directory. This CLI reads WAV/FLAC, not MATLAB containers. MYRiAD also
contains speech, music, and noise recordings: those must not be used as RIRs.
For multichannel RIR WAVs the simulator randomly selects one microphone channel
and records its zero-based index; it does not average channels.

The URGENT test set uses curated subsets of both databases, not necessarily all
RIRs from either archive. Source:
[Lessons Learned from the URGENT 2024 Speech Enhancement Challenge, Section 2.2](https://www.isca-archive.org/interspeech_2025/zhang25j_interspeech.pdf).

## Run a simulation

```bash
python scripts/augment_audio.py \
  --input_manifest /path/to/clean.jsonl \
  --manifest_audio_base /path/to/audio \
  --noise_dir /path/to/noise_wavs \
  --rir_dir ~/data/rir/slr28_real_rirs \
  --snr_min -5 --snr_max 20 \
  --speed_prob 0 \
  --seed 42 \
  --output_dir /path/to/simulated
```

Alternatively use `--input_dir /path/to/clean_wavs`, or
`--openasr_librispeech --openasr_split test.clean`, instead of `--input_manifest`.
Manifest paths use `audio_path`, `audio_file`, or `audio` (a string).
Additional fields such as `gts`, labels, and custom metadata are preserved;
`audio_path` is replaced with the generated file and `source_audio_path` records
the original. Set `--manifest_audio_base` when manifest paths are relative.

The output contains `audio/*.wav` and `data.jsonl` with SNR, noise/RIR paths,
RIR channel, peak delay in samples at the speech sample rate, speed factor, and
seed. Keep the input manifest and corpus files unchanged to reproduce a run.
The output directory should not be inside an input/noise/RIR directory.

### Signal conventions

1. Optional speed perturbation runs first. Existing speed/noise defaults remain
   unchanged; `--speed_prob 0` is important for the example above.
2. RIR and noise are resampled to the speech sampling rate using anti-aliased
   polyphase resampling. Speed perturbation retains its existing interpolation.
3. One unit-energy RIR is convolved with every speech channel using FFT
   convolution. The result is aligned to the largest RIR tap and cropped to the
   original speech length (the final reverberation tail is discarded). This is a
   practical fixed-length augmentation convention, not an estimate of the true
   direct-path arrival time. Reverberant RMS is matched to input RMS.
4. Noise is randomly cropped, or repeated if too short, and scaled to a uniformly
   sampled SNR. SNR means whole-utterance power of **reverberant speech / added
   noise**, not active-speech SNR or dry-speech SNR.
5. The mixture is attenuated only if its peak exceeds 0.99, preserving component
   SNR before WAV quantization. Output WAVs use SoundFile's default PCM-16.
6. Empty, nonfinite, or silent signals are errors, including a silent selected
   noise segment or RIR channel. They are not silently skipped.

Use `--rir_prob 0.5` to reverberate half the examples on average,
`--noise_prob 0` for RIR-only augmentation, or omit `--rir_dir` for the existing
noise/speed behavior. Noise is added after reverberation and is not separately
convolved with a noise-source RIR. This is a similar acoustic simulation, not a
claim of identical URGENT acoustic distributions, RT60s, or test-set content.
