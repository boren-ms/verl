# Read audio from a chunk file

Use `load_chunk_sample` from
[`recipe/phimm/data/chunk.py`](../recipe/phimm/data/chunk.py) to read one
sample referenced by an `audio_path` or `audio_chunk` field.

## Reference format

```text
<chunk_file>:<sample_count>:<sample_index>
```

For example:

```json
{
  "audio_path": "am_data/en/human_caption_v2/FY23Q3Part3_AdjustBoundary_BiasLM/ChunkFiles/chunk_00002c1c-f582-11ed-beeb-000d3af595b0.audio:200:53"
}
```

Here, `200` is the number of samples in the chunk and `53` is the zero-based
sample index (the 54th sample). This is a chunk reference, not a standalone
audio filename. Pass the entire reference, including `:200:53`, to the loader.
Use an index in the range `0 <= sample_index < sample_count`.

## Load the sample and save a WAV

Run from the repository root in an environment with the project's dependencies
installed. Reading Azure paths also requires access to the storage account.

The example assumes `am_data/...` is relative to
`az://orngwus2cresco/data/speech/`. Change `audio_root` if your data lives
elsewhere; the loader does not automatically expand this relative prefix.

```python
import soundfile as sf

from recipe.phimm.data.chunk import load_chunk_sample

record = {
    "audio_path": (
        "am_data/en/human_caption_v2/"
        "FY23Q3Part3_AdjustBoundary_BiasLM/ChunkFiles/"
        "chunk_00002c1c-f582-11ed-beeb-000d3af595b0.audio:200:53"
    )
}

audio_root = "az://orngwus2cresco/data/speech/"
source = audio_root + record["audio_path"]
audio, sample_rate = load_chunk_sample(source)

print(f"Sample rate: {sample_rate} Hz")
print(f"Audio shape: {audio.shape}")
print(f"Duration: {len(audio) / sample_rate:.2f} seconds")
sf.write("sample_53.wav", audio, sample_rate)
```

For an `audio_chunk` field, use `record["audio_chunk"]` instead. If the field
already contains a full `az://...` reference, pass it directly without
prepending `audio_root`.

### Read a local chunk

Replace the storage URL with the actual local chunk filename, preserving the
count and index suffix:

```python
source = "/path/to/chunk_00002c1c-f582-11ed-beeb-000d3af595b0.audio:200:53"
audio, sample_rate = load_chunk_sample(source)
```

### Chunks containing multiple audio clips per sample

A `.audio` chunk returns one `(audio_array, sample_rate)` tuple.
A `.audios` chunk returns a list of those tuples:

```python
clips = load_chunk_sample("/path/to/chunk.audios:200:53")
for clip_index, (audio, sample_rate) in enumerate(clips):
    sf.write(f"sample_53_clip_{clip_index}.wav", audio, sample_rate)
```

## Read the entire chunk

Use `load_data_from_chunk` when you need every sample rather than a single
index. Here the filename must not include the `:count:index` suffix:

```python
from recipe.phimm.data.chunk import load_data_from_chunk

samples = load_data_from_chunk(
    chunk_path="/path/to/chunk.audio",
    chunk_type="audio",
    chunk_size=200,
)
audio, sample_rate = samples[53]
```

This loads and decodes all 200 samples into memory. Prefer `load_chunk_sample`
when inspecting only one sample.
