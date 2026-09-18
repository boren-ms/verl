# 2609 ASR Prompt and Output Format

This guide describes the ASR prompt, assistant prefix, output grammar, and
response parsing used when `version: 2609` is selected.

The implementation is defined in:

- [`recipe/phimm/data/prompts.py`](../../recipe/phimm/data/prompts.py) for
  prompts, prefixes, and expected outputs.
- [`recipe/phimm/data/dataset.py`](../../recipe/phimm/data/dataset.py) for
  dataset prompt assembly.
- [`recipe/phimm/reward/asr_response.py`](../../recipe/phimm/reward/asr_response.py)
  for response parsing and hypothesis extraction.

## Audio Token Placement

The 2609 format places the task instruction before `<audio>`.

For known-language prompts, the language hint follows the audio token:

```text
Transcribe the audio clip into text.<audio>
The language is Chinese.
```

For language-detection prompts, no language hint is included:

```text
Detect the language and transcribe the audio clip into text.<audio>
```

## Known-Language ASR

Known-language mode is selected when the assistant prefix is enabled by
`prefix_prob`.

### Prompt

```text
Transcribe the audio clip into text.<audio>
The language is Chinese.
```

### Assistant prefix

```text
<src=Chinese><tgt=Chinese>
```

The prefix ends with a newline.

### Generated completion

```text
<TXT>你好，世界。</TXT>
```

### Full assistant output

```text
<src=Chinese><tgt=Chinese>
<TXT>你好，世界。</TXT>
```

## Language-Detection ASR

When the prefix is disabled, the model detects the language and generates the
complete output block.

### Prompt

```text
Detect the language and transcribe the audio clip into text.<audio>
```

### Assistant prefix

The prefix is empty.

### Full assistant output

```text
<src=French><tgt=French>
<TXT>Bonjour le monde.</TXT>
```

## Multiple Known Languages

For a sample labeled with multiple languages, the prompt uses a plural
language hint:

```text
Transcribe the audio clip into text.<audio>
The languages are English and Spanish.
```

The assistant prefix contains the first language:

```text
<src=English><tgt=English>
```

The model generates additional language blocks when the transcription switches
language.

## Standard 2609 Output Grammar

Each language segment has this form:

```text
<src=SOURCE_LANGUAGE><tgt=TARGET_LANGUAGE>
<TXT>TRANSCRIPTION</TXT>
```

For ordinary ASR, source and target languages are normally identical:

```text
<src=English><tgt=English>
<TXT>Hello world.</TXT>
```

Source and target may differ when the sample explicitly provides a target
language:

```text
<src=Spanish><tgt=English>
<TXT>Hello, how are you?</TXT>
```

## Code-Switching Output

Code-switching output contains one block per language segment:

```text
<src=English><tgt=English>
<TXT>Hello, how are you?</TXT>
<src=Spanish><tgt=Spanish>
<TXT>Estoy bien, gracias.</TXT>
```

The parser returns one dictionary for each segment:

```python
[
    {"src": "English", "tgt": "English", "text": "Hello, how are you?"},
    {"src": "Spanish", "tgt": "Spanish", "text": "Estoy bien, gracias."},
]
```

## Lexical ASR

### Known-language prompt

```text
Transcribe the audio clip into text. Output must be in lexical format.<audio>
The language is English.
```

### Full assistant output

```text
<src=English><tgt=English>
<LEXICAL>
<TXT>twenty five dollars</TXT>
```

`parse_task_output` removes the leading `<LEXICAL>` marker and returns the
contents of `<TXT>`.

## Verbatim ASR

### Language-detection prompt

```text
Detect the language and transcribe the audio clip into text. Transcribe verbatim, including all filler words and disfluencies.<audio>
```

### Full assistant output

```text
<src=English><tgt=English>
<VERBATIM>
<TXT>um hello uh world</TXT>
```

`parse_task_output` removes the leading `<VERBATIM>` marker and returns the
contents of `<TXT>`.

## Nonspeech Output

The canonical nonspeech output is:

```text
<nonspeech>
```

The parser also accepts these wrapped forms:

```text
<TXT><nonspeech></TXT>
```

```text
<src=English><tgt=English>
<TXT><nonspeech></TXT>
```

Their parsed results are:

```python
[{"src": None, "tgt": None, "text": "<nonspeech>"}]
```

and:

```python
[{"src": "English", "tgt": "English", "text": "<nonspeech>"}]
```

The nonspeech reward checks the parsed `text` value directly.

## Plain Non-LID ASR

The non-language-aware `asr` task keeps a plain output:

```text
Prompt:
Transcribe the audio clip into text.<audio>

Output:
Hello world.
```

The structured `<src>`, `<tgt>`, and `<TXT>` grammar applies to
`lang_asr*` tasks.

## Parser Result Contract

`parse_task_output` always returns a list of dictionaries:

```python
{
    "src": str | None,
    "tgt": str | None,
    "text": str | None,
}
```

### Valid segment

Input:

```text
<src=English><tgt=English>
<TXT>Hello</TXT>
```

Result:

```python
[{"src": "English", "tgt": "English", "text": "Hello"}]
```

### Missing language fields

Input:

```text
<TXT>Hello</TXT>
```

Result:

```python
[{"src": None, "tgt": None, "text": "Hello"}]
```

### Invalid or missing text wrapper

Input:

```text
<src=English><tgt=English>
Hello
```

Result:

```python
[{"src": "English", "tgt": "English", "text": None}]
```

`check_fmt` returns `False` when any parsed segment has `text=None`.

### Completely invalid or empty response

Result:

```python
[{"src": None, "tgt": None, "text": None}]
```

## Configuration Example

```yaml
data:
  version: 2609
```

In a dataset source configuration:

```yaml
add_task_info:
  task: lang_asr
  version: 2609
  prefix_prob: 0.5
```

`prefix_prob` controls known-language mode:

- `1.0`: always include the assistant language prefix and put the language hint
  after `<audio>`.
- `0.0`: never include an assistant prefix; ask the model to detect the
  language.
- A value between `0.0` and `1.0`: randomly mix both modes.
