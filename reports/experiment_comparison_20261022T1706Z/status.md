# Live experiment status

Observed: 2026-10-22T17:11:50.256294+00:00. Cluster `prod-westus2-cw-6`, namespace `boren`. Discovery includes `verl-*` and `verl2-*`.

Five training submissions are RUNNING by live Ray API and `ray job status`. `verl-n2h-i0` has no submission jobs. All discovered pods are Ready with zero container restarts; worker pod replacement can still occur without a restart count.

| Node | Experiment | Ray state | Completed console step | Latest validation step | W&B |
|---|---|---|---|---|---|
| verl-n2-i12 | Dev L0 | RUNNING | 504/720 (70.0%) | 500 | [0d485800242bfc5784c5bd4a88f8d102](https://msaip.wandb.io/genai/v2611/runs/0d485800242bfc5784c5bd4a88f8d102) |
| verl-n2-i13 | Hindi v1 | RUNNING | 660/4468 (14.8%) | 660 | [d9f3c96e5073da21a91680a234a6453f](https://msaip.wandb.io/genai/v2611/runs/d9f3c96e5073da21a91680a234a6453f) |
| verl-n2-i14 | Dev BCR64 | RUNNING | 429/1440 (29.8%) | 420 | [e51cc81ed31e918470dbcc8143a6b731](https://msaip.wandb.io/genai/v2611/runs/e51cc81ed31e918470dbcc8143a6b731) |
| verl-n2-i7 | Dev fix | RUNNING | 569/720 (79.0%) | 560 | [0ca34a537b05e74628264cac2ea86a75](https://msaip.wandb.io/genai/v2611/runs/0ca34a537b05e74628264cac2ea86a75) |
| verl2-n2-i0 | Dev BCR16 | RUNNING | 593/820 (72.3%) | 580 | [5dc3fdb4c014c9dc1570a5af6e0c1e6e](https://msaip.wandb.io/genai/v2611/runs/5dc3fdb4c014c9dc1570a5af6e0c1e6e) |

The table below is a status snapshot at each run's latest validation step, not a matched-step ranking. Lower p_err is better.

| Experiment | Val step | earning_chunk_verb | openml_bad | ls_clean | ls_other | monsoon_hi_in |
|---|---:|---:|---:|---:|---:|---:|
| Dev L0 | 500 | 4.3845% | 9.0570% | 1.1829% | 2.6816% | N/A |
| Hindi v1 | 660 | N/A | N/A | N/A | N/A | 8.9150% |
| Dev BCR64 | 420 | 4.5859% | 8.6130% | 1.1678% | 2.7081% | N/A |
| Dev fix | 560 | 4.5989% | 8.2060% | 1.1546% | 2.6949% | N/A |
| Dev BCR16 | 580 | 5.1770% | 8.8053% | 1.1641% | 2.7289% | N/A |

Current driver log scalar coverage:

| Experiment | Current submission | Driver scalar range | Driver mtime UTC |
|---|---|---|---|
| Dev L0 | `l0_recover_b9fea389_w3c436454_step350` | 350–504 (161 console rows) | 2026-10-22T17:11:37.981126+00:00 |
| Hindi v1 | `remax-hindi-base1-v1-20261022T152258Z-recovery-2` | 580–660 (85 console rows) | 2026-10-22T17:11:28.919295+00:00 |
| Dev BCR64 | `bcr64-recover390-20261022T1535Z` | 390–429 (47 console rows) | 2026-10-22T17:10:12.733953+00:00 |
| Dev fix | `raysubmit_wUz37YEmnzCbLxpH` | 450–569 (129 console rows) | 2026-10-22T17:11:45.867358+00:00 |
| Dev BCR16 | `raysubmit_UmggvqWDkrghRcrn` | 570–593 (27 console rows) | 2026-10-22T17:11:47.425979+00:00 |

Each scalar segment is bound to the exact current submission log and verified W&B identity. The logs were read with a 120 MB bound; all current logs were smaller than that bound, so their complete current-attempt scalar history was covered. Console metrics can extend beyond uploaded W&B history; keep this provenance visible when supplementing the charts. Duplicate step rows preserve log order and should use the last recorded value per metric and training step.

Pod creation timestamps and submission entrypoints are retained in `status.json`. The three earlier failed attempts on i7/i12/i14 reported worker node death / heartbeat loss; the later current attempts above are RUNNING.

