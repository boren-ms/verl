---
name: asr-check-train-data
description: "Check ASR training data YAML configs for mix.json availability and comment out unavailable paths. Use when: validating cache_path entries before training, checking if training data mixes exist, commenting out missing mixes in data configs, verifying train configs3/data YAMLs. Triggers: 'check training data', 'check mix paths', 'comment out unavailable', 'validate train config', 'check cache paths'."
argument-hint: "Training data YAML path, e.g. en_hc_fy24_200k_p2_entity.yaml or full path"
---

# ASR Check Train Data

Validate `cache_path` entries in ASR training data YAML configs by checking for `mix.json` availability, then comment out unavailable paths.

## When to Use

- Before launching training, to verify all data sources are cached
- After creating a new data config, to validate paths
- When a training run fails due to missing data
- When the user says "check training data", "check mix paths", "comment out unavailable"

## Inputs

| Input | Required | Description |
|-------|----------|-------------|
| YAML path | Yes | Path to a `train/configs3/data/*.yaml` file, or just the config name |

Config name resolution: if user provides just a name like `en_hc_fy24_200k_p2_entity`, resolve to `data/speech/speech/train/configs3/data/en_hc_fy24_200k_p2_entity.yaml`.

## Procedure

### Step 1 — Read the YAML

Read the target YAML file. It contains `cache_path` entries:

```yaml
cache_path:
  - az://orngwus2cresco/data/...
  # - az://orngwus2cresco/data/...  # mix.json not found
  - az://orngwus2cresco/data/...
```

Extract all **uncommented** `cache_path` entries (lines starting with `  - az://`). Skip lines starting with `  # -` as they are already commented out.

### Step 2 — Check mix.json availability in parallel

Check all paths simultaneously using background subprocesses with a timeout:

```bash
for path in <all_uncommented_paths>; do
  label=$(basename "$path")
  (timeout 15 bbb ls "${path}/mix.json" &>/dev/null && echo "OK $label" || echo "MISSING $label") &
done
wait
```

**Key details:**
- Use `timeout 15` to prevent hangs on unreachable paths
- Run all checks in parallel (`&` + `wait`) for speed
- Use `bbb ls <path>/mix.json` — exit code 0 means available

### Step 3 — Comment out unavailable paths

For each path reported as MISSING, comment it out in the YAML with a reason annotation:

```yaml
  # - az://orngwus2cresco/data/.../Entertainment  # mix.json not found
```

Use `multi_replace_string_in_file` to make all edits at once when multiple paths need commenting.

**Preserve already-commented paths**: Do not uncomment or modify lines that were already commented out.

### Step 4 — Report summary

Report the results:

```
Checked: 15 paths
Available: 13
Unavailable: 2 (commented out)
  - Entertainment  # mix.json not found
  - Education  # mix.json not found
```

## Notes

- Paths use `az://orngwus2cresco/...` blob storage format
- `bbb` requires Azure credentials — if not available, tell the user
- Common category subdirs: `Autos_Vehicles`, `Comedy`, `Education`, `Entertainment`, `Film_Animation`, `Gaming`, `Howto_Style`, `Music`, `News_Politics`, `Nonprofits_Activism`, `People_Blogs`, `Pets_Animals`, `Science_Technology`, `Sports`, `Travel_Events`
