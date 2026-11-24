# OpenLong hard 4-second and 8-second validation

Both versions reuse the same complete recordings and ZIP audio members as the
verified 2-second release. Only `audio_path`, `n_segments`, and
`chunk_length_seconds` change in each parent record. No audio files are created.
Cuts are consecutive and non-overlapping; the shorter final chunk is retained.
Full-parent references and verbatim English language-hint prompts are preserved.

| Corpus | Recordings | 4-second chunks | 8-second chunks |
| --- | ---: | ---: | ---: |
| Earnings21 | 44 | 35,356 | 17,690 |
| Earnings22 | 125 | 107,965 | 54,013 |
| CORAAL | 271 | 144,984 | 72,555 |
| Total | 440 | 288,305 | 144,258 |

Validation data configs:

- `openlong_hard_chunk_4s_verb_langhint_zip.yaml`
- `openlong_hard_chunk_8s_verb_langhint_zip.yaml`

Evaluation entry points:

- `v2611_dev/eval/eval_openlong_hard_chunk_4s_zip`
- `v2611_dev/eval/eval_openlong_hard_chunk_8s_zip`

Each release contains `earnings21.jsonl.gz`, `earnings22.jsonl.gz`,
`coraal.jsonl.gz`, `val_data.yaml`, `preparation.json`, and
`validation_verified.json` under:

```text
az://orngwus2cresco/data/boren/data/Evaluation/OpenLong_hard_chunk_4s_v1/
az://orngwus2cresco/data/boren/data/Evaluation/OpenLong_hard_chunk_8s_v1/
```

Reproduce either version from the verified 2-second parent manifests:

```bash
PYTHONPATH=plugins/phimm/src:. uv run --no-project python \
  -m recipe.phimm.tools.openlong_hard_chunk \
  --input-root az://orngwus2cresco/data/boren/data/Evaluation/OpenLong_hard_chunk_2s_v1 \
  --output-dir /path/to/new/OpenLong_hard_chunk_4s_v1 \
  --chunk-seconds 4 \
  --publish-root az://orngwus2cresco/data/boren/data/Evaluation/OpenLong_hard_chunk_4s_v1
```

For the 8-second version, use `--chunk-seconds 8` and the corresponding output
directory and publication root. Each original recording is rechunked from zero
using its full duration, rather than merging previous chunk boundaries.
Publishing verifies the uploaded bytes and refuses to replace different files.
