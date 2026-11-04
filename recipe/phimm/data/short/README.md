# LibriSpeech short utterances

Qwen3-ForcedAligner-0.6B produces word timestamps for each original utterance.
A deterministic SHA-256 selection (seed 42) chooses 2–5 initial words; when the
original transcript is shorter, all words are retained. The FLAC clip starts at
the first aligned word and ends at the selected last word. These are estimated
boundaries, without a separate VAD or padding: leading silence is trimmed to the
first predicted word, but residual silence or clipped phones are possible. Listen
to the supplied review samples before relying on boundary quality.

Data root: `az://orngwus2cresco/data/boren/data/LibriSpeech/short_2to5_v1/`.
The 21,000 training candidates were selected deterministically from the LibriSpeech
training manifests; validation includes all 2,620 test-clean and 2,939 test-other
utterances. The initial text and boundary validation rejected none of the 26,559
rows. A subsequent stricter audit excluded 1,152 training rows with at least one
zero-duration word (including all eight clips shorter than 160 ms), leaving
19,848 training rows. Both the unfiltered and excluded manifests remain available.
There are five one-word test
utterances (two clean, three other), retained in full text as requested. The
remaining records have 2–5 words. The train and two validation YAMLs refer to
these separate manifests and audio files; the Base0 short recipe selects 19,848
short training rows and validates only on the short test sets. With batch size
128 and the trainer's `drop_last=True`, eight rows do not form a complete batch
in each epoch. No training was run.

The `aligned/` directory retains full-transcript alignment outputs. The two
`human_review_*_10.jsonl` files reference independently read-back-verified remote
audio and include original text, original path, crop times and word timestamps;
the validation review contains five examples from each test split, including two
flagged zero-duration alignments per split. All validation records were kept:
143/2,620 clean and 199/2,939 other contain a zero-duration word, and 45 clean
and 57 other contain a zero-duration last word. The validation manifests include
per-row `alignment_warnings`. These are quality warnings, not
verified transcription errors. `quality_audit.json` records these counts.
`summary.json`
records model revision, model weight hash, package versions, counts and seed.
`review_readback_sha256.json` records review audio hashes. The remote audio listing
was checked against every manifest entry; the three final JSONLs were read back
and hashed. The ASR dataset loader read all three final manifests and loaded one
audio from each.

The run used `qwen-asr==0.0.6`, `transformers==4.57.6` and the model weights
SHA-256 `47831d0e82f96b20e9034dba01a075ee06436654719f6a68289e49f1b65ce0e7`.
Qwen imports Japanese `nagisa` unconditionally; the isolated English-only
`english_shims/nagisa.py` explicitly raises if Japanese tokenization is attempted,
because DyNet is unavailable for this environment. English tokenization does not
use that module.

Reproduction: `prepare_alignment_inputs.py` selects input data; the run divided
training and test-other inputs into alternating, disjoint rows for each node.
`run_shard.sh` aligns one shard on eight GPUs and crops with 16 threads using
`build_parallel.py`. The execution used two independently reserved Ray jobs on
separate nodes and the existing `cache_eval_audio.py` to prefetch 24 audio files
per node concurrently. `finalize_shards.py` merges only completed shards, verifies
remote inventory and sample hashes, and publishes manifests; `audit_alignment.py`
publishes the filtered training manifest and targeted review samples. The temporary source
and raw shard outputs are not a substitute for the final manifests. Never submit
this workload to a shared pool without checking its owner and current state.
