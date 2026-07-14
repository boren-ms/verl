---
name: mix-source-jsonl
description: "Generate a source JSONL from a mixed ASR dataset manifest on a remote Brix node. Use when: flatten components, create source jsonl, extract mixed dataset components, generate mixed_22_100_source.jsonl, or process mix_lang JSONL on a remote node."
---

# Mixed Dataset Source JSONL

Create a JSONL manifest where each nested `components[]` object in a mixed ASR
dataset manifest is emitted as one standalone JSON line. Run the job on the
specified Brix node so the Azure manifest is streamed within the remote
environment rather than downloaded through the local workstation.

## Inputs

- `NODE`: Brix node name. Default: `verl-n1-i13`.
- `SOURCE_URI`: Azure URI for the mixed dataset JSONL, for example:
  `az://orngwus2cresco/data/boren/data/verl/mix_lang/mix_cv15_all/mixed_22_100.jsonl`.
- `DESTINATION_URI`: Azure URI for the flattened JSONL. By convention, append
  `_source` before `.jsonl`, for example:
  `az://orngwus2cresco/data/boren/data/verl/mix_lang/mix_cv15_all/mixed_22_100_source.jsonl`.

## Procedure

1. Confirm the target node is Ready:

   ```bash
   brix ls '<node-pattern>' 2>&1
   ```

2. Run this command from the local `verl` workspace, replacing the three
   variables with the requested values. It creates the target only after the
   complete flattened file has been written to a remote temporary file.

   ```bash
   brix ssh "$NODE" -- 'bash -l -c '"'"'
   set -euo pipefail
   source_uri="'"'"'$SOURCE_URI'"'"'"
   destination_uri="'"'"'$DESTINATION_URI'"'"'"
   output_file=$(mktemp /tmp/mix_source.XXXXXX.jsonl)
   trap '"'"'rm -f "$output_file"'"'"' EXIT

   bbb cat "$source_uri" | jq -c '"'"'.components[]'"'"' > "$output_file"
   bbb cp "$output_file" "$destination_uri"
   '"'"''
   ```

3. Validate remotely. The expected count is computed from each source row's
   actual component array length, so this works for pairs, triples, and larger
   mixes without assuming a fixed mix size.

   ```bash
   brix ssh "$NODE" -- 'bash -l -c '"'"'
   set -euo pipefail
   source_uri="'"'"'$SOURCE_URI'"'"'"
   destination_uri="'"'"'$DESTINATION_URI'"'"'"
   expected_count=$(bbb cat "$source_uri" | jq -s '"'"'map(.components | length) | add // 0'"'"')
   actual_count=$(bbb cat "$destination_uri" | wc -l)
   test "$actual_count" -eq "$expected_count"
   bbb cat "$destination_uri" | jq -e -c . >/dev/null
   printf "source_components=%s output_records=%s\\n" "$expected_count" "$actual_count"
   '"'"''
   ```

## Output Contract

Each output line is exactly one original component object, retaining fields
such as `language`, `text`, `audio_chunk`, and `duration`. Do not add the
parent mixed-sample fields (`id`, `audio_path`, `mix_type`, or `gap`) unless the
user explicitly asks to retain source-to-mixture provenance.

## Failure Handling

- If `brix ls` does not show the node as Ready, resume or ask the user to
  provide a ready node before transforming data.
- If `jq` reports invalid JSON, stop without treating the destination as
  validated; identify the malformed source row first.
- If validation counts differ, keep the destination for inspection but report
  the mismatch and do not claim the source JSONL is complete.