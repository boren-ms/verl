# Mix4 reference-context data (ready for training)

Separate presets preserve the original mixture identity and each underlying datasource.

| Variant | Original utterances | Prepared parents | Audio segments | Status |
| --- | ---: | ---: | ---: | --- |
| short_orignal | 25,240 | 25,240 | 25,240 | Published exact singleton references |
| flshort_rm23x4 | 27,972 | 111,888 | 279,720 | Reuse existing six reference manifests |
| contig5 | 25,240 | 24,520 | 122,600 | Reuse existing five reference manifests |
| short_rs23 | 25,240 | 25,240 | 37,915 | Ready: published five manifests, all rows loader-verified |

The variants reuse recordings; counts are not additive unique utterances.
All presets retain the original Mix4 variant in `data_source`, and record the underlying source in `original_data_source`.
Training consumes only preceding segments' reference text (Text H4), never the current or future label. Singleton utterances have no preceding context.

Original manifests and preparation provenance are preserved locally at `tmp/mix4_reference_context/data/`. The new training configuration is `config/v2611_dev/earnings/prompt_ctx/remax_mix4_reference_text_h4_bs128_ppo512_n4_gspo.yaml` (ready snapshot: `tmp/mix4_reference_context/training.ready.yaml`). It preserves the current GSPO recipe, batch128/PPO512,2000 steps,32 GPUs,save100/test400 and reference-context OpenLong validation; the SC source is now prepared; complete Mix4 loader, publication and model/audio preflight proofs passed.

Random cuts can bisect words. Preserve the original cuts and assign words using full-parent Qwen timestamps and absolute word midpoints; retain boundary-crossing flags. No heuristic transcript splitting or audio recutting is used.

Existing training jobs, original experiment state and timers are owned by the original Chat and are untouched by this data preparation.

## Alignment preparation

Five independent JSONL inputs at `tmp/mix4_reference_context/data/alignment_input/` contain 12,675 complete source-audio/transcript pairs and the original cut selectors, with unique datasource/row identities. Exact labels for the other 12,565 one-part rows are preserved separately in `data/short_rs23_singletons/`. These staging artifacts are preserved as provenance. The complete SC release joins all 12,675 split parents with all 12,565 singleton parents and is published as five datasource-separated manifests.

Verification: `tmp/mix4_reference_context/verify_config_loader.py` composes the YAML, checks model/loss/optimizer/mixture preservation, and exercises the real ASR loader and reference context for all rows in each checked manifest. Its `--variant short_rs23` mode verifies SC independently, including the 25,000-parent epoch mixture budget.

Validation uses `openlong_reference_context_zip`: all three OpenLong datasets, complete parent groups, ZIP audio, per-segment reference labels, and previous-only Text H4. Stored automatic reference alignment retains its quality limitations; results with reference context must be labeled separately from generated-context results.

## SC release ready for training

The SC preset now uses five published manifests under `az://orngwus2cresco/data/boren/data/verl/mix4_reference_context_20261124/data/short_rs23/`. All 25,240 parents and 37,915 segments passed the production ASR loader, previous-only reference context and concatenated reward-target checks. Each published manifest was downloaded and its SHA-256 verified. The epoch mixture budget remains 25,000 SC parents.

Qwen aligns full parent audio/transcripts once; positive-duration words are assigned by midpoint to the unchanged original cuts. Original transcripts, removed zero-duration words and crossing flags remain in provenance. The four all-zero-duration cases are explicitly user-accepted and retained with empty segment references. Three user-selected transcript tails are removed only from training references; original text/audio/cuts are preserved. No alignment node was resumed.

Readiness evidence: `tmp/mix4_reference_context/sc_training_ready_proof.json`, `sc_loader_verification.json`, `sc_preparation_proof.json`, and `empty_reference_acceptance.json`. All four components now pass full production-loader and published manifest SHA checks: 186,888 physical parents / 465,475 segments, sampled into the unchanged 125,000-parent epoch mixture. Full proof: `tmp/mix4_reference_context/config_loader_verification.json`; model/audio proof: `tmp/mix4_reference_context/model_preflight/preflight.json`.
