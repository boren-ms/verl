---
name: asr-train-config
description: 'Create ASR training configs in train/configs3 from a reference source YAML. Use when: creating train data configs from filter_entity or other pipeline output YAMLs, extracting dst_path or cache_path entries into train data configs, checking if cached mixes are available via bbb ls, commenting out unavailable training data paths. Triggers: "create train config", "make training config from", "check training data cached".'
argument-hint: 'Source YAML path and optional subset name, e.g. configs/filter_entity/en_hc_fy24_200k_entity.yaml'
---

# ASR Train Config Creator

Create training data configs (`train/configs3/data/*.yaml`) and corresponding train configs (`train/configs3/*.yaml`) from a reference source YAML, then validate that all cache paths are available.

## When to Use
- User has a pipeline output YAML (e.g. filter_entity config) with `dst_path` entries and wants to create training configs
- User wants to create a train/configs3 data config from a set of blob paths
- User wants to verify that training data mixes are cached before launching training
- User says "create train config", "make training config from ...", or "check training data"

## Inputs

| Input | Required | Default | Example |
|-------|----------|---------|---------|
| Source YAML | Yes | — | `data/speech/speech/scripts/configs/filter_entity/en_hc_fy24_200k_entity.yaml` |
| Config name | No | derived from source | `en_hc_200k_entity` |
| Subset filter | No | all entries | `FY24Q3-2` (single quarter) |

## File Layout

```
data/speech/speech/train/configs3/
├── <name>.yaml                    # Train config (references -> data/<data_name>.yaml)
└── data/
    └── <data_name>.yaml           # Data config (lists cache_path entries)
```

## Procedure

### Step 1 — Inspect the source YAML

Read the source YAML to understand its structure. Source configs typically have one of these formats:

**Filter-entity style** (list of `src_path`/`dst_path` pairs):
```yaml
- src_path: az://...
  dst_path: az://...
```

**Direct cache_path style** (already a list):
```yaml
cache_path:
  - az://...
```

Extract the **dst_path** values (filter-entity style) or **cache_path** values (direct style). These become the training data paths.

```bash
grep 'dst_path:' <source.yaml> | sed 's/.*dst_path: //'
```

### Step 2 — Create the data config

Create `data/speech/speech/train/configs3/data/<data_name>.yaml` with format:

```yaml
cache_path:
  - az://path1
  - az://path2
  ...
```

If user requests a subset (e.g. only `FY24Q3-2`), filter the paths to include only matching entries.

**Naming convention**: Derive from source. Examples:
- Source `en_hc_fy24_200k_entity.yaml` → data config `en_hc_fy24_200k_entity.yaml`
- Subset of above for FY24Q3-2 → `en_hc_fy24_200k_p2_entity.yaml`

### Step 3 — Create the train config

Create `data/speech/speech/train/configs3/<train_name>.yaml` referencing the data config:

```yaml
data_config:
  - data/<data_name>.yaml
```

**Naming convention**: Shorten the data name. Examples:
- Data `en_hc_fy24_200k_entity.yaml` → train `en_hc_200k_entity.yaml`
- Data `en_hc_fy24_200k_p2_entity.yaml` → train `en_hc_200k_p2_entity.yaml`

Look at existing train configs in `configs3/` for naming patterns to stay consistent.

### Step 4 — Validate cache availability

Check each `cache_path` entry for a `mix.json` file using `bbb ls`. A cached mix has `mix.json` present.

```bash
bbb ls <cache_path>/mix.json
```

For each path, check availability:
- **Available**: `bbb ls <path>/mix.json` exits with code 0
- **Unavailable**: `bbb ls <path>/mix.json` exits with non-zero code or returns empty

**Important**: `bbb` requires Azure credentials. If `bbb` is not available in the current terminal, tell the user to run the check from a terminal where `bbb` works.

### Step 5 — Comment out unavailable paths

In the data config, comment out any paths where `mix.json` was not found:

```yaml
cache_path:
  - az://available/path1
  # - az://unavailable/path2
  - az://available/path3
```

Report a summary:
```
Total: 15, Available: 13, Unavailable: 2
Commented out:
  - .../FY24Q4-3_.../Education
  - .../FY24Q4-3_.../People_Blogs
```

## Batch Check Script

For checking many paths at once, generate and run an inline script:

```bash
for path in $(grep '^ *- az://' <data_config.yaml> | sed 's/^ *- //'); do
  label=$(echo "$path" | rev | cut -d/ -f1-2 | rev)
  if bbb ls "$path/mix.json" &>/dev/null; then
    echo "OK: $label"
  else
    echo "MISSING: $label"
  fi
done
```

## Reference Configs

Existing train configs to use as patterns:
- `configs3/en_hc_200k_cer05.yaml` → `data/en_hc_fy24_200k_cer05.yaml`
- `configs3/en_hc_egs_cer05.yaml` → `data/en_hc_fy24_200k_p2_cer05.yaml`

## Notes
- Paths use `az://orngwus2cresco/...` blob storage format
- Quarter naming: `FY24Q3-2`, `FY24Q3-3`, `FY24Q4-1`, `FY24Q4-2`, `FY24Q4-3`
- Each quarter has 15 category subdirs: `Autos_Vehicles`, `Comedy`, `Education`, `Entertainment`, `Film_Animation`, `Gaming`, `Howto_Style`, `Music`, `News_Politics`, `Nonprofits_Activism`, `People_Blogs`, `Pets_Animals`, `Science_Technology`, `Sports`, `Travel_Events`
- The `p2` suffix conventionally refers to the `FY24Q3-2` quarter subset
