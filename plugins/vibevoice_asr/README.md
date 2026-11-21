# VibeVoice-ASR offline model

This integration follows the Qwen audio package layout: `hf_vibevoice_asr`
registers the official offline model with Transformers, and
`vllm_vibevoice_asr` registers its multimodal vLLM implementation through the
`vibevoice_asr` general-plugin entry point. The upstream implementation is a
pinned dependency at `microsoft/VibeVoice@16fb2cb1217c9934a886e1948ffb06120caa2df5`.
It is not a copy of the streaming model and does not change the Qwen plugin.

Install the pinned upstream package and this package into a compatible GPU
environment. To preserve an existing environment, stage an upstream wheel and
register source with `uv pip install --no-deps` after checking its dependencies:

```bash
uv pip install --no-deps /path/to/vibevoice-1.0.0-py3-none-any.whl
uv pip install --no-deps -e plugins/vibevoice_asr -e plugins/phimm
```

The upstream HF model targets Transformers 4.x. Use a separate HF environment
when the vLLM environment requires Transformers 5; share the existing Torch
installation with `uv venv --system-site-packages`, then install Transformers
4.57.6, tokenizers 0.22.1 and huggingface-hub 0.36.0 into that environment.
The upstream package eagerly imports its TTS scheduler; diffusers 0.35.1 is
also required even for offline ASR. The opt-in vLLM adapter handles upstream
AutoModel registration/name collisions, the moved Qwen tokenizer module,
chat-template return types and modern multimodal parser/profiling interfaces.

Download `microsoft/VibeVoice-ASR` at revision
`d0c9efdb8d614685062c04425d91e01b6f37d944`. The original checkpoint does not ship
tokenizer files. Generate the offline tokenizer assets with the pinned
upstream `vllm_plugin/tools/generate_tokenizer_files.py` tool, passing
`--output /path/to/model`. Run that file directly to avoid importing the vLLM
plugin when preparing assets on an HF-only machine. Keep the resulting assets
beside the original checkpoint; do not use `--streaming`.

The two backends share the official audio processor, 24-kHz waveforms and
normalization. vLLM replaces one audio placeholder with the exact HF audio
span, and the runner checks the actual expanded prompt token IDs against HF.
Only greedy decoding is used. Audio is loaded through the existing native
recipe loader, including Azure, ZIP, indexed chunk and time-range references.
The default runtime bounds are 8,192 context tokens and 60 seconds per audio.
Longer audio requires increasing `--max-audio-seconds` and `--max-model-len`
with enough memory for encoder profiling and generation. Inputs exceeding the
configured audio bound raise an error instead of being clipped.
vLLM uses eager execution by default; `--no-enforce-eager` enables its optional
compilation and CUDA graph path, which requires separate runtime validation.

```bash
# HF smoke test
python -m vibevoice_asr.decode --backend hf --model /path/to/model \
  --limit 4 --output-dir /path/to/new-hf-smoke

# vLLM smoke test
VLLM_PLUGINS=vibevoice_asr python -m vibevoice_asr.decode \
  --backend vllm --model /path/to/model --limit 4 \
  --batch-size 4 --output-dir /path/to/new-vllm-smoke

# Complete native AMI dev evaluation, with no row limit
VLLM_PLUGINS=vibevoice_asr python -m vibevoice_asr.decode \
  --backend vllm --model /path/to/model --batch-size 16 \
  --output-dir /path/to/new-ami-dev-result
```

The default manifest is the existing `moe_ami_dev full_v1` validation set:
2,070 clips from 18 AMI IHM-mix recordings, approximately 8.919 clip hours.
This is a clip evaluation, not the paper's full-meeting benchmark. Reference
speaker annotations and reference text never enter the prompt.

Results contain the effective configuration, raw generations, parsed segments,
per-clip normalized WER counts and corpus-micro WER (`sum(n_err)/sum(n_ref)`).
Scoring reuses `recipe.phimm.reward.asr_eval.measure_openasr_en_wer`, including
its HF English normalizer and compound-aware alignment. Whole-content acoustic
tags such as `[Silence]` are excluded from spoken text and may omit a speaker.
The exact allowlist is the six official acoustic tags plus the `[Sigh]` event
observed in offline generations; arbitrary bracketed text is not removed.
Speech segments require speaker and timing fields. For a generation that hits
the token limit, the scorer may append only closing JSON delimiters, preserving
all emitted words, repetitions and partial words. Such outputs remain marked
as truncated, with delimiter recovery counted separately. Unrecoverable JSON
or invalid required fields make full-dataset WER unavailable; no samples are
silently dropped. `complete` means all selected rows were scored, while
`all_generations_finished` separately records whether any generation truncated.
An inverted time range on an excluded acoustic event does not prevent text
WER: its original times are preserved with `timing_valid=false`, and affected
rows are counted in `invalid_event_timing_rows`. Speech timing errors remain
failures.
Partial runs are explicitly marked. WER does not measure diarization quality;
this runner does not compute DER or speaker-attributed WER.

Saved generations can be rescored without loading the model or decoding audio.
This verifies exact sample coverage, references, audio paths and source hashes:

```bash
python -m vibevoice_asr.decode --backend vllm --model /path/to/model \
  --rescore-from /path/to/new-ami-dev-result \
  --output-dir /path/to/new-ami-dev-rescore
```

```bash
PYTHONPATH=plugins/vibevoice_asr/src python -m pytest -q plugins/vibevoice_asr/tests
```

CPU parsing tests do not establish GPU or runtime compatibility. Record the
actual GPU versions and successful smoke-test artifacts with each evaluation.

## Validated run: 2026-11-20

Both backends decoded real AMI audio on one H100 80 GB. HF used Transformers
4.57.6 in an isolated environment; vLLM used vLLM 0.24.0 and Transformers 5.7.0.
Both shared Torch 2.11.0+cu130 and BF16 weights. HF and vLLM each passed a
four-clip smoke test; vLLM's expanded prompt IDs matched the official HF
processor on every request. These small smoke tests establish functionality,
not an accuracy comparison between backends.

The full vLLM evaluation covered all **2,070 clips** of `moe_ami_dev full_v1`
(18 source recordings, approximately 8.919 clip hours). Corpus WER was
**24.7202% = 23,124 / 93,543**, with 7,013 insertions, 11,376 deletions and
4,735 substitutions under the repository's native English scorer. Reference
data was identified from chat `01a1f184-b30b-77d1-934d-c5525b904174` and its
`moe_en_full_ami_dev.yaml` validation config.

All 2,070 raw responses were retained unchanged and all rows scored. Two
generations reached the 4,096-token cap: one repeated speech and one repeated
`[Door]` events; delimiter-only
JSON closure retained every emitted word for WER. One excluded silence event
had inverted times (16.00 to 15.97 seconds), preserved and flagged. No retries,
repetition penalties or sample exclusions were applied. This is text WER on
the existing clip dataset; no DER, cpWER or tcpWER was measured.

Full generation job: `vibevoice-vllm-ami-dev-20261120-r1`. The original driver's
scoring failed because its in-memory parser required speakers on acoustic
events. CPU-only rescore `vibevoice-vllm-ami-dev-20261120-score-r3` succeeded
using the exact saved generations and corrected event handling. The rescore
checks full ID coverage, references, audio paths and source file hashes.

Artifacts (raw generations, final details/metrics, effective configurations,
Ray logs, coverage/count verification and provenance):

```text
az://orngwus2cresco/data/boren/data/verl/vibevoice_asr/results/20261120-offline-ami/
  ami-dev-vllm-20261120-score-r3/metrics.json
  ami-dev-vllm-20261120-score-r3/details.jsonl
  ami-dev-vllm-20261120-r1/details.jsonl
  verification.json
  reproducibility.json
```

## AMI dev DER comparison

The saved generations were scored with the native `measure_der` implementation
from `codex/dai-svad-diarization-eval`: zero collar, overlap included, optimal
one-to-one speaker matching independently per clip, and corpus aggregation by
reference speaker-seconds. Both models use the same reference speaker intervals
and clip evaluation bounds. Our model is 2611 R2 after AMI SFT, step 274.

| Matched cohort: 1,642 clips | DER | Miss | False alarm | Speaker confusion |
| --- | ---: | ---: | ---: | ---: |
| VibeVoice-ASR offline | **33.5906%** | 15.0078% | 9.5289% | 9.0539% |
| Our AMI-SFT step 274 | 52.0107% | 32.1994% | 11.2786% | 8.5327% |

Coverage is part of the result: our model has usable diarization on 1,644/2,070
clips; 426 outputs lack usable structured speaker/timing annotations. VibeVoice
has usable diarization on 2,068/2,070 clips; two outputs contain an
`[Unintelligible Speech]` segment without a speaker. The matched comparison
therefore uses 1,642 clips, with identical reference speaker-time denominators.
No missing predictions were replaced by invented speakers or timestamps.

VibeVoice DER on its 2,068 available clips is 31.7462%; this is a different
cohort and must not be compared directly to our available-only aggregate.
Excluding the one truncated VibeVoice output in the matched cohort gives
33.5896% versus 51.9972% on 1,641 clips. Native per-clip DER primitive counts
for our model were independently recomputed and exactly matched all saved
counts. Whole known non-speech events are excluded; unintelligible speech is
retained as speech. An output missing its unintelligible-speech speaker is
unavailable, rather than silently dropping that speech interval.

Detailed scores and reproducible calculation are retained in the task's local
`tmp/vibevoice_asr/der-comparison/` directory. These are clip-level results,
not a reproduction of the paper's full-recording benchmark.

## Paper reproduction status

The paper reports AMI-IHM DER 11.92% / WER 18.81% and AMI-SDM DER 13.43% /
WER 24.65%. The paper describes full-recording VibeVoice inference and cites
MeetEval. It does not specify enough detail to establish an exact match between
our zero-collar, overlap-included scoring and its DER protocol. The public
inference demo explicitly describes itself as a demo rather than a benchmark
evaluation script. Exact split/session lists, IHM audio mixing, reference
preparation, scoring options and decoding settings must be verified before
claiming reproduction. See also the upstream AMI reproduction discussion:
<https://github.com/microsoft/VibeVoice/issues/237>.

The current 60-second/8,192-token runner defaults were used only for the
existing short clips. Full meetings require explicitly increased audio and
context bounds, a sufficient output-token allowance, and separate memory and
runtime validation; concatenating independently decoded clips does not
reproduce full-recording inference.

A controlled full-meeting evaluation uses the standard **16-meeting
full-corpus-ASR test split** from `pyannote/AMI-diarization-setup`, pinned at
`67c2d539286e89f68952d5dcf83912bd9f01dfae`. Its words-only RTTM references
and published UEM files are used unchanged. Audio is the official AMI
`Mix-Headset.wav` for IHM and `Array1-01.wav` for SDM, approximately 9.062 hours
per condition. These split and mixing choices are explicit assumptions because
the authors have not confirmed them as the paper protocol.

For full meetings, use the official FFmpeg/librosa file-loading path, preserve
one full-context decode per recording, and explicitly configure the bounds:

```bash
python -m vibevoice_asr.decode \
  --backend hf --model /path/to/official-offline-model \
  --manifest /path/to/ihm.jsonl --output-dir /path/to/ihm-results \
  --audio-loader official --audio-workers 1 --batch-size 4 \
  --max-audio-seconds 3000 --max-new-tokens 32768 \
  --attention flash_attention_2

# Requires meeteval==0.4.3 and its md-eval-22.pl dependency.
python -m vibevoice_asr.der \
  --details /path/to/ihm-results/details.jsonl \
  --reference /path/to/references/test.rttm \
  --uem /path/to/references/test.uem --output-dir /path/to/ihm-der
```

The DER command scores both 0 s and 0.25 s collars, with overlap included and
speaker mapping per meeting. The aggregate divides summed error speaker-seconds
by summed reference speaker-seconds. It rejects missing/duplicate meetings,
truncated generations, unparsed outputs and speech lacking real speaker labels.
This prevents MeetEval's RTTM-ID intersection from silently shrinking coverage.
The 0.25 s collar is a sensitivity result, not a setting selected to match the
paper. The author confirmed Flash Attention in issue #237; exact kernel version
and inference backend remain undisclosed.

For explicit failure-inclusive scoring, `--reparse-raw
--allow-unassigned-speech --allow-truncated` can be added to the DER command.
These options preserve the saved raw response, list every truncated recording
and every speech interval without a speaker, and continue scoring the full
reference UEM. An unassigned speech event such as `[Speech]` emits no hypothesis
speaker activity and is therefore an abstention; no speaker identity is
invented and no reference speech is removed. Truncated outputs retain their
emitted intervals and incur errors for the missing remainder. The strict
behavior remains the default. A reparse writes a separate
`reparsed_details.jsonl`, leaving the original inference details unchanged.

MeetEval's `collar=0.25` passes `-c 0.25` to md-eval, excluding **±0.25 s**
around each reference boundary. In pyannote.metrics 4.2.0 the equivalent
argument is `collar=0.5`, because that package defines collar as the total width.
The real-meeting smoke test agreed between scorers at matched collar widths.


### Full-meeting results: 2026-11-20

The pinned official HF model decoded all 16 meetings in each condition with
FlashAttention-2, BF16, batch size 4, greedy decoding, seed 0 and a 32,768-token
output limit. Each condition contains approximately 9.062 hours. All full UEM
regions were scored, including failures and missing tails.

| Scoring setting | IHM DER | SDM DER |
| --- | ---: | ---: |
| Paper (exact protocol undisclosed) | 11.92% | 13.43% |
| Zero collar, overlap included | 28.4170% | 36.3033% |
| ±0.25 s collar, overlap included | 19.3916% | 27.7416% |
| Zero collar, overlap excluded | 17.5777% | 26.6550% |
| ±0.25 s collar, overlap excluded | 11.1487% | 20.2341% |

IHM is numerically close under one common protocol, while SDM remains 6.8041
percentage points above its paper result under that same protocol. This does
not establish exact paper reproduction: the authors have not confirmed the
split, IHM construction, reference preparation, collar or overlap policy.
All four settings are reported without selecting model parameters by test DER.

IHM `EN2002c` repeated near the end and truncated at the token limit, with a
maximum emitted end of 2921.51 s for 2972.256 s of audio. SDM `EN2002c` also
truncated (2950.22/2972.256 s); SDM `EN2002d` repeated much earlier
(1321.69/2209.899 s). These outputs remain in the full-reference scores.
All four settings in both conditions were independently checked with
pyannote.metrics 4.2.0: reference speaker-time, miss, false alarm and confusion
counts agree with MeetEval at matching collar widths.
There were 26 unassigned speech intervals in IHM and 29 in SDM, scored as
abstentions using the explicit failure-inclusive options above.

The original generation driver returned a parsing error for 13 IHM and 11 SDM
rows because the strict parser rejected speech events without speaker labels.
All 32 raw generations were saved. The separate DER reparse scores every
meeting, preserving raw responses and reporting missing speakers and
truncations. The overlap-excluded sensitivity uses exactly the same hypothesis
RTTM, reference and UEM; it can be reproduced with MeetEval directly:

```python
import meeteval.der

scores = meeteval.der.md_eval_22(
    "/path/to/references/test.rttm",
    "/path/to/ihm-der/hypothesis.rttm",
    uem="/path/to/references/test.uem",
    collar=0.25,
    regions="nooverlap",
)
print(sum(scores.values()))  # Corpus aggregate; not a mean of meeting DERs.
```

HF uses the checkpoint's 131,072-token context configuration; the runner's
`--max-model-len` option applies only to vLLM. HF's saved batched
`generated_tokens` counts include padding after EOS and must not be used as
emitted-token or throughput measurements. DER uses only emitted intervals.
The earlier vLLM full-meeting attempt was stopped after repetition; those
outputs are preserved separately and are not mixed with the HF evaluation.

Full raw results, scoring inputs/hashes, four-setting scores, independent
verification, configs, logs and reproduction report are retained at:

```text
az://orngwus2cresco/data/boren/data/verl/vibevoice_asr/paper-ami/
  results/full-ihm-hf-fa2-b4-r1/details.jsonl
  results/full-sdm-hf-fa2-b4-r1/details.jsonl
  scores/{ihm,sdm}/der.json
  scores/{ihm,sdm}/overlap-sensitivity.json
  metadata/reproduction-report.md
```

## Diarization RL integration

The experimental Main ReMax recipes are
`v2611_dev/vibevoice/remax_vibevoice_ami_ls` and
`v2611_dev/vibevoice/smoke_vibevoice_ami_ls`. They use the official offline
checkpoint as the seed, AMI **train-IHM** plus the existing noisy/RIR LS
manifest for training, and **DavidAI English internal-test** for evaluation.
AMI dev-SDM is no longer the validation source. The 16 paper reproduction
meetings remain excluded from training. DavidAI's 45-recording test subset
is evaluation only: do not use it as an optimization reward, tune the recipe
against it, or select checkpoints from its scores. Its disjoint 3-recording
smoke subset can be supplied for an ID-overlap audit.

Prepare manifests with the existing attached annotations:

```bash
python -m vibevoice_asr.prepare_rl_data \
  --ami-train /data/ami-train.jsonl --ami-dev /data/ami-dev.jsonl \
  --ls /data/ls.jsonl \
  --davidai-test /data/davidai-internal-test.jsonl \
  --davidai-smoke /data/davidai-validation-smoke.jsonl \
  --output-dir /data/vibevoice-rl
```

`--ami-dev` audits the AMI recording split only. Add `--limit-per-source 1`
and use a separate output directory for the single-update smoke recipe.
The preparation audits complete AMI recording identities before subsetting,
rejects train/dev/paper-test overlap, and checks supplied DavidAI smoke/test
IDs for overlap. It never uses transcripts or reference speaker labels to
construct the model prompt. LS has synthetic single-speaker activity over
the decoded waveform, rather than genuine speaker-turn labels; its WDER is
explicitly unavailable.

DavidAI's source manifest stores SVAD selectors for each recording. Preparation
restores their common full **mixed** waveform and preserves the complete
reference time axis. Each recording is decoded in one call so that independent
chunks cannot silently reset speaker IDs. The full recipe allows up to 901
seconds, an 8,192-token prompt and a 16,384-token validation response within a
24,576-token context. Training audio stays limited to 40 seconds. The smoke
recipe bounds evaluation at 180 seconds and 8,192 response tokens; its first
test recording is approximately 162 seconds. Longer inputs raise rather than
being truncated. These are explicit runtime limits, not guarantees of memory
capacity on every device. The full 8-GPU recipe requires separate validation.

The rollout agent uses the official 24-kHz processor and native JSON array
output with speaker IDs and timestamps **in seconds**. HF retains the expanded
audio prompt and tensors for logprob/backward; vLLM receives the corresponding
collapsed audio placeholder plus the exact same normalized waveform. The
upstream vLLM processor does not normalize raw audio itself. The HF adapter pads the native
acoustic mask across response tokens and freezes the acoustic/semantic encoders
and connectors. Decoder attention and MLP projections receive LoRA adapters.
The HF training recipe uses eager attention because Transformers 5 propagates
the root backend into the convolutional tokenizers, which reject FlashAttention
initialization. A compatibility initializer also preserves loaded parameters:
Transformers 5.7 tracks initialization per parameter, while upstream writes
through `.data`, which otherwise reinitializes the LM head and connectors after
loading. vLLM retains its own attention backend. Its engine-specific
config explicitly sets the audio frontend to BF16, matching the HF actor;
setting only the actor's dtype does not change vLLM's seed config.
Both backends use the acoustic distribution's mean in these RL recipes to
avoid fresh acoustic noise between rollout and policy logprob computation.
The paper reproduction used the original stochastic frontend; its scores are
not a baseline for this changed frontend without an additional evaluation.

The outcome reward is
`1 - 0.5 * min(DER, 1) - 0.5 * min(WER, 1)`. DER uses zero collar and includes
overlap; speaker assignment is permutation invariant. WER uses the existing
English normalizer. Only the optimization reward is clipped: pooled DER/WER
retain the original error and reference counts, including rates above 100%.
Malformed or incomplete native JSON receives zero reward and counts all
reference speech as missed; no samples disappear from the denominator.
ReMax subtracts the same prompt's greedy reward from sampled rewards. Existing
Main KL regularization and token importance correction remain enabled.

To submit the one-update integration check on an already verified, synced
single-GPU node:

```bash
export VIBEVOICE_RL_MODEL=/data/vibevoice-asr-model
export VIBEVOICE_RL_DATA_ROOT=/data/vibevoice-rl-smoke
python plugins/phimm/submit.py \
  --submission-id vibevoice-ami-ls-davidai-smoke-UNIQUE \
  --config-path config/v2611_dev/vibevoice \
  --config-name smoke_vibevoice_ami_ls
```

The submitter forwards these two explicit environment inputs to Ray. The smoke
uses two training rows (one AMI, one LS), two sampled rollouts each, rank-8 LoRA,
one optimizer update, validation before/after, and checkpoint saving at step 1.
Successful composition or an HF forward alone does not establish that the RL
pipeline works; check actual update, finite logprobs/gradients, validation and
complete checkpoint evidence. Verify loaded weights against the seed and inspect
HF/vLLM token logprob differences: finite loss alone can hide a reinitialized
head or unequal audio inputs. One DavidAI recording is a functional check,
not a full-test WER/DER or evidence of training improvement. Formal training
must be launched separately after reviewing the smoke results.
