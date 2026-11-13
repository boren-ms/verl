# AISHELL-1 case: preparation example, not a demonstrated fix

Historical workspace: `/home/boren/code/verl.worktrees/aishell-refine`. Recheck current files before reuse:

- `docs/aishell_refine.md`
- `scripts/prepare_aishell_refine.py`
- `recipe/phimm/config/v2611_dev/aishell/base_aishell.yaml`
- `recipe/phimm/config/v2611_dev/aishell/remax_aishell_top20k.yaml`
- `recipe/phimm/config/v2611_dev/aishell/remax_aishell_top20k_2611v1a.yaml`
- `reports/aishell_refine_20261110/`

The verified preparation used source `az://highperf01eus2/data/am_data/publicdataset_process/AIShell-1/train_data/feature_extraction/`. `chunk_sanity_check/file_set.json` identified chunk files; `.info` records supplied identities and were checked against parent `train_filenames.txt`. Training contained 120,098 rows, 340 speakers and 301 chunks (150.852 hours), with zero ID overlap against the full 7,176-example test. That ID result alone is not an audio-content overlap proof.

An initial partial generation covering 40,960 rows was rejected as a basis for global Top-K. Complete 2611r2 shadow38000 generation on 2×8 GPUs produced six parts, with canonical grapheme CER 178,242/1,730,113 = 10.3023%. Top20,000 selection ranked absolute `n_err`, breaking ties by source row then ID, retaining original references. Its duration was 26.4474 hours. Two selected outputs hit the 768-token decode limit and were retained under the recorded policy.

The Top200 test errors formed a fixed, biased hard-case diagnostic, not representative validation. Full AISHELL test remained a separate evaluation. The case exposed lazy scorer imports; the environment pinned `opencc-python-reimplemented==0.1.7` and `wetext==0.2.1` and exercised scoring on every worker. These pins belong to that environment, not all future cases.

The case intentionally used character, language and format reward because word reward penalized spacing differences such as `你好世界` versus `你好 世界`. This decision requires review against each new recipe. An independent AISHELL base resolved a nested Hydra searchpath issue. The alternate recipe initialized from a step900 export while selection still came from shadow38000: those model identities must remain distinct in comparison reports.

At the verified preparation milestone, conversion, full generation, selection and configs were ready; training had not been submitted and no repair effect had been demonstrated. Do not copy historical paths, Top-K sizes, node assignments or status into a new case as current facts.
