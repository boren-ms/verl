# OpenLong hard 2-second validation

`openlong_hard_chunk_2s_verb_langhint_zip.yaml` includes the complete canonical
Earnings21 (44 recordings), Earnings22 (125), and CORAAL (271) test sets.
`v2611_dev/eval/eval_openlong_hard_chunk_2s_zip` is the evaluation entry point.

| Corpus | Recordings | Chunks |
| --- | ---: | ---: |
| Earnings21 | 44 | 70,696 |
| Earnings22 | 125 | 215,857 |
| CORAAL | 271 | 289,825 |
| Total | 440 | 576,378 |

Each recording is cut at 0, 2, 4, ... seconds without overlap, silence detection,
alignment, fades, or padding. The final shorter chunk is retained. Boundaries use
the original full recording duration, including tails omitted by the older
segmentation. Existing complete ZIP audio members are reused with new `#start:end`
selectors; no re-encoding or duplicate audio upload is needed.

Each JSONL row represents one parent and lists all its chunk selectors in
`audio_path`. The validation loader expands this list, generates a response for
each chunk, and scores the joined output against the exact original full-parent
`text`. These manifests contain no per-chunk transcript or reference history.
Verbatim English prompts and language hints match the canonical OpenLong config.

Published manifests and preparation evidence are under:

```text
az://orngwus2cresco/data/boren/data/Evaluation/OpenLong_hard_chunk_2s_v1/
  earnings21.jsonl.gz
  earnings22.jsonl.gz
  coraal.jsonl.gz
  val_data.yaml
  preparation.json
  validation_verified.json
```

Reproduce from the repository root in an environment with Blob access:

```bash
PYTHONPATH=plugins/phimm/src:. uv run --no-project python \
  -m recipe.phimm.tools.openlong_hard_chunk \
  --output-dir /path/to/new/OpenLong_hard_chunk_2s_v1 \
  --chunk-seconds 2 \
  --publish-root az://orngwus2cresco/data/boren/data/Evaluation/OpenLong_hard_chunk_2s_v1
```

The input parent manifests supply the original transcripts, durations, sample
rates, and ZIP member locations. Their previous segment boundaries, alignments,
and reference histories are discarded. The preparation audit records parent and
chunk counts, durations, full-reference hashes, and manifest hashes. Publishing
verifies uploaded bytes and refuses to overwrite different existing files.
