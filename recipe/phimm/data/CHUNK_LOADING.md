# Chunk audio loading

The original `file_set_*.json` manifest describes chunks with `name`, `count`
and optional `duration`. It does not contain sample byte offsets. During
expansion, transcription/SFT payloads are read and samples normally retain an
ordinal reference such as `chunk.audio:2000:1375`.

The default `chunk_load_mode: indexed` builds a bounded, in-memory offset index
when a chunk is first read. It avoids rescanning preceding record headers on
every sample read. `sample` and `cached` remain explicit compatibility modes.

## Materialize byte references during manifest expansion

Set `index_audio: true` on a chunk dataset to scan record headers once during
manifest expansion and add an `audio_path` for each encoded audio payload:

```yaml
dataset_name: chunk
specs:
  - manifest_file: /data/chunks/file_set_train.json
    chunk_path: /data/chunks
    trans_path: /data/chunks
chunk_types: [audio, transcription]
chunk_shuffle: false
index_audio: true
```

Each resulting row contains both:

```json
{
  "audio_chunk": "chunk.audio:2000:1375",
  "audio_path": "chunk.audio!<byte_offset>:<byte_length>",
  "transcription": "original reference"
}
```

`audio_chunk` remains the stable sample identity. `audio_path` points directly
to the encoded WAV/FLAC payload, excluding the chunk record header. The audio
loader prefers this path and performs a local seek or a bounded remote range
read, without consulting the ordinal index. Both fields survive normal verl
formatting and can be saved in JSONL or Arrow metadata. Original audio and
reference files are unchanged. Time selectors such as `#0.1:0.5` still work.

For `audios` chunks, the byte locator selects the first audio of each record,
matching the existing native `load_raw_audio` behavior.

`index_audio` defaults to false. Building the first index needs access to the
audio chunk, not just the JSON manifest; Orange sources may need a full local
cache download. Eagerly indexing every chunk of a large corpus moves this cost
into dataset initialization. Use it when preparing reusable byte-indexed
metadata or when chunks are already available locally. Existing ordinal
references continue to use lazy indexed reads, so running jobs do not need a
full-corpus preparation pass or a restart.
