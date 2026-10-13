# Grouped training-data mixing

The [ASR dataset loader](../../../data/rl_dataset.py) supports `data.mix` in
addition to the existing concatenation and interleaving modes. Define the input
datasets in `data.train_data` as usual; `name` refers to each dataset's
`data_source` field, not its loader `dataset_name`. Multiple input datasets with
the same `data_source` are combined before applying a budget.

```yaml
data:
  shuffle: false
  train_batch_size: 64
  seed: 42
  mix:
    - shuffle: true
      datasource:
        - name: earning
          sample: 128
          shuffle: true
        - name: openml
          batch: 2
    - shuffle: false
      datasource:
        - name: earning
          epoch: 0.5
```

- Groups are concatenated in list order. Within a group, sources are concatenated
  in `datasource` order, then optionally shuffled together.
- A source's `shuffle: true` shuffles it **before** selecting rows. Group shuffle
  never changes which rows were selected. Both use `data.seed` (default `42`,
  including when null) and deterministic group/source offsets.
- Use at most one budget per source. All budgets accept finite non-negative
  integers or floats: `sample` counts rows; `batch` is multiplied by
  `data.train_batch_size`; `epoch` is multiplied by the source's row count.
  Round the resulting row count to the nearest integer using Python's `round()`
  (halfway values round to the nearest even integer). For example, `sample: 2.6`
  selects 3 rows and `batch: 0.5` with batch size 64 selects 32 rows.
  With no budget (or null budgets), use one full source pass.
- `range: [start, end]` selects a half-open subset in the original source order
  **before source shuffle and sampling**. Bounds must be integers satisfying
  `0 <= start <= end <= source length`. An absent/null range uses the full source.
  Default and `epoch` budgets use the selected subset's length. Repeated passes
  cycle only through the selected subset. A positive budget with an empty range
  raises an error.
  Set the same explicit `shuffle_seed` on repeated entries for a source to reuse
  its permutation when their ranges match. For example, `range: [0, 4]` and
  `range: [4, 10]` select disjoint subsets of a 10-row source, regardless of shuffle.
  Group shuffle still only reorders selected rows. Dataset-size-based budgets
  must be computed offline and configured as numbers.
- Selection takes the first rows in source order, cycling through complete
  passes when the budget exceeds the source size. Source shuffle changes that
  order once, not separately for each repeated pass.
- Each source entry starts from the beginning independently unless `range` is
  configured, including when the same name occurs again in later groups. Unmentioned sources are excluded;
  unknown or empty source names raise an error.
- Zero budgets omit a source. Entirely zero-sized groups are skipped with a
  warning; a mixture with no remaining rows raises an error.
- Set `data.shuffle: false`; global dataloader shuffle would destroy group
  ordering and is rejected. Custom samplers must also preserve dataset order.
  Groups are row blocks, not batch boundaries: the dataloader may span a boundary
  or drop an incomplete final batch. `batch` uses `train_batch_size`, even if
  `gen_batch_size` is configured differently.
- `data.mix` takes precedence over `use_interleave` and `interleave_ds`, and
  cannot be combined with legacy `data.data_source` budgets.
- Validation/generation (`is_train=False`) ignores training mixing and still
  concatenates the original input datasets. A missing/null `data.mix` preserves
  all existing training concatenation, interleaving, and legacy budget behavior.
