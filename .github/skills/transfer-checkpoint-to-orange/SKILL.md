---
name: transfer-checkpoint-to-orange
description: Copy a checkpoint directory from an external Azure Blob account to Orange through the orngtransfer bridge, excluding optimizer states, mp_rank_00 model states, and vllm_export. Use for filtered checkpoint transfers; use transfer-to-orange for unfiltered transfers.
---

# Transfer a checkpoint to Orange

Copy the requested checkpoint directory using the shared `orngtransfer` bridge. Read and follow [transfer-to-orange](../transfer-to-orange/SKILL.md) for source SAS refresh, Green authentication, path mapping, single-file test, async azcopy jobs, and verification. This skill changes the selection rules below; do not perform a full unfiltered copy.

## Selection

Copy every source blob under the requested directory except:

- Files whose basename ends in `_optim_states.pt`.
- Files named exactly `mp_rank_00_model_states.pt`.
- Every file under a directory named `vllm_export`.

Keep other files and subdirectories, including `lora_merged_model_states.pt`, configuration, and evaluation output, unless the user specifies a narrower scope. Do not infer that a checkpoint without the excluded files supports training resumption. Do not delete excluded files already present in either destination; report their presence if relevant.

## Procedure

1. Derive the exact source account, container and relative path from the supplied URL, as in the base skill. Run its SAS helper and verify Green authentication. Never print SAS values; load the SAS from the matching cache entry independently in each shell.
2. List the source recursively and build a manifest of relative paths and sizes after applying the selection rules. Check that the source is stable enough to copy; if it changes during the transfer, refresh the manifest and reconcile before claiming completion. Inspect both destination paths before the test, choose an eligible, preferably small file, and perform the required two-stage test with `--overwrite=false`. Verify it at the final destination before the full copy; treat any existing or skipped test blob as unverified unless independently checked.
3. Apply the following flags to **both** recursive azcopy stages, including any dry run. `--exclude-pattern` operates on basenames; the regex excludes a `vllm_export` directory at any depth while retaining similarly named directories.

   ```bash
   --recursive --s2s-preserve-access-tier=false \
   --exclude-pattern='*_optim_states.pt;mp_rank_00_model_states.pt' \
   --exclude-regex='(^|/)vllm_export(/|$)'
   ```

   Use the base skill's wildcard source and destination mapping, `AZCOPY_AUTO_LOGIN_TYPE=AZCLI`, and the SAS only for the external source. Before copying, run the source-to-bridge command with `--dry-run` and check its selected paths against the manifest; do not proceed if an excluded path appears or an eligible path is missing.
4. Check for existing blobs in both destinations. Preserve unrelated data and avoid silently overwriting existing blobs: use `--overwrite=false`, and do not claim a skipped blob is identical solely because its size matches. Before stage 2, compare the eligible bridge paths against the current eligible source manifest. If the bridge has eligible stale extras, do not recursively copy them; restrict stage 2 to the intended manifest using a verified method, or stop and explain the conflict. Apply the filters even when the bridge contains excluded files from an earlier transfer.
5. Run the bridge-to-Orange stage, monitor both jobs, and use the base skill's bounded retry guidance for transient failures. Compare the eligible source and destination relative paths and sizes; report missing, mismatched, skipped, and extra files separately. Existing excluded or unrelated destination blobs are not evidence that the filters failed, and size equality alone is not a content hash check. Report the final Orange path and verification limits.
