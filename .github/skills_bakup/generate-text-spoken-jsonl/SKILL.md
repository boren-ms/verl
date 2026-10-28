---
name: generate-text-spoken-jsonl
description: "Generate a dataset JSONL from user-provided content instructions, with a unique id, written/display-form text, semantically equivalent spoken words, and keywords found in the text. Use when: create text and spoken JSONL, generate written-versus-spoken examples, build text normalization data, create keyword-bearing synthetic text, or produce instruction-driven JSONL records."
argument-hint: "<generation-instructions> [count] [output-path]"
---

# Generate Text and Spoken JSONL

Create a UTF-8 JSONL dataset from the user's content instructions. Generate
natural, varied examples while keeping the written form, spoken realization,
and keyword annotations aligned.

## Output Contract

Write exactly one JSON object per nonempty line with these fields in this
order:

```json
{"id":"sample-000001","text":"The total is $12.50.","spoken":"the total is twelve dollars and fifty cents","keyword":["$12.50"]}
```

- `id`: Nonempty string that is unique within the output file. Use
  `<prefix>-<zero-padded-sequence>` unless the user specifies another format.
- `text`: The natural written/display form generated from the user's
  instructions. Preserve requested capitalization, punctuation, symbols,
  numerals, abbreviations, and formatting conventions.
- `spoken`: The words a speaker would naturally say when reading `text`
  aloud. Express verbalized symbols, numbers, dates, times, units, currencies,
  abbreviations, and other written forms as words. By default, use lowercase
  words without display punctuation. Do not add or omit semantic content.
- `keyword`: A JSON array of strings. Every item must occur verbatim as a
  nonempty substring of `text`. Preserve first-occurrence order and remove
  duplicates. Use `[]` when no keyword is required or identifiable.

Do not add fields unless the user explicitly requests them. Do not wrap the
output in a JSON array, Markdown fence, or explanatory prose.

## Inputs

Resolve these values from the request:

- Generation instructions: Topic, patterns, vocabulary, formatting rules,
  distributions, examples, exclusions, and diversity requirements.
- Record count: Number of JSONL rows.
- Output path: Destination ending in `.jsonl`.
- Language or locale: Controls both written conventions and natural spoken
  wording.
- ID prefix: Default to the output filename stem, normalized to lowercase
  ASCII letters, digits, and hyphens; use `sample` if no path is available.
- Keyword policy: User-supplied keywords, a requested keyword category, or
  inferred focal written spans.

Ask a concise clarification question before generation when the record count,
language, or output path cannot be inferred. If the keyword policy is omitted,
annotate only the focal written spans that instantiate the requested
phenomenon; do not broaden the annotation to general topical terms. Use `[]`
for rows without a focal span.

## Procedure

### 1. Convert the request into constraints

Write down the required count, locale, output path, ID scheme, content
patterns, keyword policy, and exclusions. Treat user examples as style and
schema guidance unless the user explicitly requires exact inclusion.

If requirements conflict, prioritize explicit constraints over examples and
ask before silently weakening a requirement.

### 2. Plan coverage before writing rows

Build a compact coverage plan across the requested dimensions. Balance
categories approximately unless exact proportions are provided. Vary names,
values, sentence structures, positions of keywords, and surrounding context.

For a small requested count, prefer broad coverage over repeated templates.
For a large count, generate in batches and track used `text` values to prevent
accidental duplicates.

### 3. Generate each record

For every row:

1. Create `text` in the requested written/display style.
2. Derive `spoken` from that exact `text`, using the requested locale.
3. Select keywords according to the resolved policy.
4. Verify each keyword occurs verbatim in `text`.
5. Assign the next unique ID only after the row is accepted.

Common written-to-spoken conversions include:

- `$12.50` -> `twelve dollars and fifty cents`
- `7:30 p.m.` -> `seven thirty p m`
- `Sep. 11, 2026` -> `september eleventh twenty twenty-six`
- `3.5 kg` -> `three point five kilograms`
- `1-800-555-0199` -> locale-appropriate digit-by-digit speech

These are examples, not universal rules. Follow the user's locale, domain, and
stated reading convention. Preserve ambiguity rather than inventing a meaning;
ask when one interpretation would materially change the dataset.

### 4. Write JSONL safely

Create parent directories when needed. Use a structured JSON serializer with
UTF-8 output so quotes, backslashes, and non-ASCII text are escaped correctly.
Do not construct JSON by concatenating strings.

Do not overwrite an existing output file unless the user explicitly requests
replacement. If extending an existing dataset, inspect its schema and IDs,
continue with collision-free IDs, and preserve its established field order and
locale conventions.

### 5. Validate the complete file

Validate all rows after writing:

- The number of nonempty lines equals the requested count.
- Every line parses independently as one JSON object.
- Every object has exactly `id`, `text`, `spoken`, and `keyword`, unless extra
  fields were explicitly requested.
- All IDs are nonempty strings and globally unique in the file.
- `text` and `spoken` are nonempty strings.
- `keyword` is an array of unique, nonempty strings.
- Every keyword occurs verbatim in its row's `text`.
- No two rows have identical `text` unless duplicates were requested.
- `spoken` faithfully verbalizes `text` under the chosen locale and does not
  leak written-only symbols that should have been spoken as words.
- The generated examples satisfy the requested categories, proportions, and
  exclusions.

Use a script or structured parser for schema, count, uniqueness, and keyword
checks. Manually inspect a representative sample, including edge cases, for
semantic alignment between `text` and `spoken`.

## Decision Rules

- If the user supplies a keyword list, generate only requested keywords unless
  told to add others; annotate each keyword actually present in that row.
- If matching is case-sensitive, preserve exact casing in `keyword`. Otherwise,
  still store the exact substring as it appears in `text`.
- If a requested keyword is absent from a generated row, either revise the row
  to include it or omit it from that row's list according to the requested
  coverage policy. Never annotate an absent string.
- If `text` is already naturally spoken prose, `spoken` may differ only in
  casing and punctuation removal.
- If one display form has multiple valid readings, follow the user-specified
  domain convention; otherwise choose one consistently and report the adopted
  convention.

## Completion Checklist

- The requested JSONL exists at the requested path and has the exact row count.
- Schema, types, IDs, JSON syntax, keyword containment, and text uniqueness
  pass automated validation.
- Written and spoken forms are semantically equivalent in the requested locale.
- Coverage and exclusions match the user's generation instructions.
- A representative sample and all unusual formatting cases were reviewed.