---
name: remote-qwen35-audio-decode
description: 'Decode local bad-case audio with specified Qwen3.5-Audio checkpoints through Hugging Face or vLLM on a remote Brix GPU node. Use when: transcribing an audio list or folder with run_qwen35_audio_hf.py or run_qwen35_audio_vllm.py, evaluating az:// models on bad cases, uploading local audio before remote inference, or comparing HF and vLLM transcripts.'
argument-hint: 'Local audio path, model az:// path(s), hf or vllm backend, and optional Brix node'
---

# Remote Qwen3.5-Audio Decode

Decode audio stored on the local development machine by staging it through Orange Blob, downloading it to a remote Brix GPU node, and invoking the selected Qwen3.5-Audio backend with remote paths. Each model loads once for the complete audio list or folder. Return the verified batch log to the local source directory.

## Required Inputs

- Local audio file or directory, such as `~/data/bad_cases/time_format`
- One or more model paths, normally `az://.../qwen_hf/`
- Backend: `hf` or `vllm`; infer it from an explicit user request and otherwise default to `vllm` for compatibility
- Optional Brix node name

Ask only for required inputs that cannot be inferred from the request or active editor. Expand `~` locally before constructing remote paths.

## Fixed Paths

```text
HF decoder: /root/code/verl/plugins/qwen35_audio/scripts/run_qwen35_audio_hf.py
vLLM decoder: /root/code/verl/plugins/qwen35_audio/scripts/run_qwen35_audio_vllm.py
Remote repo: /root/code/verl
Blob staging root: az://orngwus2cresco/data/boren/data/verl/bad_cases/
Remote data root: /root/data/bad_cases/
```

## Procedure

### 1. Inventory the local inputs

1. Resolve the local path and confirm it exists.
2. Enumerate candidate `soundfile` inputs, normally `.wav`, `.flac`, `.mp3`, and `.ogg`. Do not include `.m4a`/AAC without converting it to a supported format first.
3. Exclude metadata such as `Zone.Identifier`, existing decode directories, and unrelated files.
4. Record the exact file count, relative paths, and byte sizes. Preserve relative paths during staging.

Stop and report clearly if no audio files are found.

### 2. Select and verify a remote node

If the user supplied a node, use it. Otherwise:

1. Run `brix pools --all` and select a Ready `verl-*` development node.
2. Check GPU state with `nvidia-smi`; use a node/GPU with enough free memory and no active workload.
3. Never terminate or interfere with an existing job to obtain a GPU.

Verify on the selected node:

- The selected decoder script exists.
- The Python environment imports `torch`, `transformers`, `numpy`, and `soundfile`; vLLM decoding additionally requires `vllm` and the local plugin.
- The specified model path is accessible with `bbb ls`.

Use `brix ssh`; do not use raw `ssh`, `scp`, or `brix tmux`.

### 3. Stage local audio before decoding

Derive a stable dataset name from the local directory basename. Use these corresponding paths:

```text
Blob:   az://orngwus2cresco/data/boren/data/verl/bad_cases/<dataset>/
Remote: /root/data/bad_cases/<dataset>/
```

Upload local files to Blob first:

```bash
bbb sync -x 'Zone\.Identifier$' <local-source>/ \
  az://orngwus2cresco/data/boren/data/verl/bad_cases/<dataset>/
```

Then download them on the selected node:

```bash
brix ssh <node> -- 'bash -l -c "mkdir -p /root/data/bad_cases/<dataset> && bbb sync az://orngwus2cresco/data/boren/data/verl/bad_cases/<dataset>/ /root/data/bad_cases/<dataset>/"'
```

Before inference, compare the remote audio count, names, and sizes with the local inventory. Decode using the remote audio paths only, never the local paths.

Probe every staged file with the same remote Python environment used by the decoder by calling `soundfile.read`. This catches corrupt files and image-dependent MP3 support before loading the model. Stop and report unreadable files rather than launching inference for them.

### 4. Decode with each specified model

Create a distinct result directory for every model and backend so results cannot overwrite each other. Derive a short, filesystem-safe model label from the path, including the checkpoint step when present, for example `step90`.

```text
Remote results: /root/data/bad_cases/<dataset>/decoded_<model-label>_<backend>/
Local results:  <local-source>/decoded_<model-label>_<backend>/
```

Create the remote result directory before starting any decoder process:

```bash
brix ssh <node> -- 'mkdir -p /root/data/bad_cases/<dataset>/decoded_<model-label>_<backend>'
```

Invoke the selected decoder once with the explicit model and remote audio folder. Both scripts also accept repeated `--audio <remote-file>` arguments when only a subset is needed.

For Hugging Face decoding:

```bash
brix ssh <node> -- 'bash -l -c "set -o pipefail; cd /root/code/verl && CUDA_VISIBLE_DEVICES=<gpu> python plugins/qwen35_audio/scripts/run_qwen35_audio_hf.py --model <model-path> --audio-folder <quoted-remote-audio-folder> 2>&1 | tee <quoted-remote-batch-log>"'
```

For vLLM decoding:

```bash
brix ssh <node> -- 'bash -l -c "set -o pipefail; cd /root/code/verl && CUDA_VISIBLE_DEVICES=<gpu> python plugins/qwen35_audio/scripts/run_qwen35_audio_vllm.py --model <model-path> --audio-folder <quoted-remote-audio-folder> 2>&1 | tee <quoted-remote-batch-log>"'
```

Requirements:

- Preserve the script's default instruction and generation settings unless the user overrides them.
- Quote every path that may contain spaces.
- Use `set -o pipefail` so `tee` does not hide decoder failures.
- Use one collision-resistant batch log per model/backend pair.
- Let each command finish. vLLM reports `load_seconds` and aggregate `generate_seconds`; HF reports `Loaded in` once and `Decoded in` for each audio.
- HF loads the model once and generates sequentially. vLLM loads once and submits all discovered audio requests together.
- Do not claim batch success unless the decoder exits zero, emits one complete `AUDIO_RESULT_START`/`AUDIO_RESULT_END` and `TRANSCRIPT_START`/`TRANSCRIPT_END` pair per input, and finishes with `BATCH_DONE count=<expected-count>`.

Do not start separate decoder processes per audio file. That reloads the model and defeats the batch interface. For a very large folder, discuss chunking before launching to bound vLLM request memory and batch-log size.

If evaluating multiple models, complete and verify one model's full result directory before starting the next.

### 5. Return results locally

Preferred path:

1. Sync the remote result directory to a model/backend-specific Blob result directory.
2. Sync Blob results into the local `decoded_<model-label>_<backend>/` directory.

If the remote node lacks upload credentials and logs are small text files, use this fallback for each known log:

```bash
mkdir -p <local-result-directory>
brix ssh <node> -- 'cat <quoted-remote-log-path>' > <quoted-local-log-path>
```

Do not use direct `cat` transfer for large or binary files. Never print SAS tokens or commands containing credentials.

### 6. Verify completion

For every model/backend run, verify:

1. A nonempty local batch log exists.
2. Remote and local SHA-256 hashes match.
3. `audio_count`, `AUDIO_RESULT_START`, `AUDIO_RESULT_END`, `TRANSCRIPT_START`, and `TRANSCRIPT_END` counts all equal the local audio inventory count.
4. The `model_source=` line exactly matches the requested model path.
5. `BATCH_DONE count=<expected-count>` is present and the process exited zero.

Extract and report each audio filename and the text between its transcript markers. If the model emits structured `<TXT>...</TXT>` output, report the inner text; otherwise report the complete marker-delimited output. When multiple models were evaluated, provide a compact comparison and explicitly identify changed versus identical transcripts. Include the selected node and local result directories in the final report.

## Failure Handling

- **Local path missing:** resolve `$HOME`, check the literal expanded path, and stop if it is still absent.
- **No idle Ready node:** report current node/GPU states; do not take over an active GPU.
- **Remote Python mismatch:** use the verified login-shell Python or pyenv shim that imports vLLM; do not install packages unless necessary and approved by the existing setup workflow.
- **Model staging failure:** preserve the log and report the failing `az://` path.
- **One audio fails:** keep the batch log, report the failed filename and error, and retry only after diagnosing the local cause. A missing `BATCH_DONE` means the run is incomplete.
- **Remote Blob upload unavailable:** use direct Brix text transfer only for small logs, then verify hashes.

## Completion Criteria

The task is complete only when every requested model/backend pair has a verified local batch log covering all discovered audio files, or each remaining failure is explicitly identified with its remote log and error.