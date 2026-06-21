# Entity Tag Format & JSONL Schema

## Entity Tag Notation

Entities in the `Transcription` field are marked with XML-style tags:

### Basic entity (no type)
```
that's <NE> property coverage </NE> for your home
```

### Typed entity
```
check <NE:name> Facebook </NE> for more info
```

### General pattern
```
<NE>          ... </NE>         — generic named entity
<NE:type>     ... </NE:type>    — typed named entity (e.g., NE:name, NE:org)
```

Entities may contain one or more words. The text inside the tags is the entity span. Whitespace around the entity text (inside tags) is trimmed.

## JSONL Input Schema

Each line of the input JSONL must have at least:

| Field | Required | Description |
|-------|----------|-------------|
| `Transcription` | Yes | Reference text with `<NE>` / `<NE:type>` entity tags |
| `hyp` | Yes (or `--hyp-column` override) | Hypothesis transcription text (no tags) |

### Optional fields

| Field | Description |
|-------|-------------|
| `ref` | Clean reference text (no tags). If absent, derived by stripping tags from `Transcription` |
| `text` | Alternative clean reference text |
| `id` | Utterance identifier |
| `audio_file` | Path to audio file |
| `keywords` | List of expected entity keywords |
| `n_err` | Pre-computed error count |
| `n_ref` | Pre-computed reference word count |
| `wer` | Pre-computed WER |

### Example JSONL

```json
{"Transcription": "that's <NE> property coverage </NE> for your home", "hyp": "that's property coverages for your home", "id": "utt_001"}
{"Transcription": "check <NE:name> Facebook </NE> for info", "hyp": "check Facebook for info", "id": "utt_002"}
```

## Entity Extraction

The script extracts entities by parsing `<NE>` and `<NE:type>` tags from `Transcription`:
1. Find all `<NE...>...</NE>` spans
2. Record entity text, type, and word-level position in the reference
3. Strip tags to produce clean reference text
4. Align reference and hypothesis at word level
5. Compute errors only for positions that fall within entity spans
