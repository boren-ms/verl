# Offline LibriSpeech biasing

The shared `bias_base.yaml` directly inherits `../base.yaml` and explicitly sets the 2611r2 checkpoint, ReMax greedy-baseline advantages, GSPO loss, and clipping to match the Base3a GSPO template. Validation and checkpoint saving run every 100 steps. It contains no data configuration; the before/after entry YAMLs import their training and validation data.

Select the prompt position with `remax_bias_before` or
`remax_bias_after`:

```bash
python -m recipe.phimm.main_asr_remax \
  --config-path config/v2611_dev/biasing \
  --config-name remax_bias_before
```

The offline pool contains 268,760 positive and 280,393 pure-negative rows. Training samples all 268,760 positives and 26,876 negatives per epoch, shuffled together for a 10:1 ratio (295,636 rows). The shared bias base sets one epoch, giving 2,309 steps at batch size 128 with drop_last; the template's fixed 868-step limit is cleared. Each positive row covers 10–100% of its usable transcript hotwords; both training lists contain at most 1,000 unique candidates. JSONL contains no prompt; `add_task_info` constructs it at load time.

Training sources are defined separately in
`config/data/train_data/biasing/ls_pos.yaml` and `ls_neg.yaml`, with datasource
names `ls_pos` and `ls_neg`. Each ReMax entry defines `data.mix` sample budgets:
268,760 positive and 26,876 negative rows by default (10:1). Adjust the two `sample`
values to change the mix; the expected training-row count follows their sum.
Validation YAMLs live in `config/data/val_data/biasing`: `ls_before.yaml` and
`ls_after.yaml` each contain only test-clean/test-other at
100/500 (four datasets, 11,118 rows); `ls_none.yaml` contains only
the two 0 conditions (5,559 rows). Before/after training selects only test-other 100/500 from its corresponding
YAML, giving two datasets and 5,878 rows. Validation and checkpoint saving
both run every 100 steps.

Evaluate the unchanged 2611r2 checkpoint once with `eval_ls_bias`.
It inherits the evaluation worker template, disables LoRA and resuming, and
sets `val_only: true`. The three separate YAMLs in `data/val_data/biasing` include
LS0 with none and LS100/500 with both before_audio and after_audio: ten datasets,
27,795 rows. Source names include the prompt mode so results stay separate.

```bash
python -m recipe.phimm.main_asr_remax \
  --config-path config/v2611_dev/biasing \
  --config-name eval_ls_bias
```

All JSONL paths point to Orange under `az://orngwus2cresco/data/speech/users/boren/data/librispeech_biasing/`. Local copies remain available. Prompt length is 8,192 and actor/reference/rollout token budgets follow `data.max_model_length`.
