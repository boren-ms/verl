"""Keep shared fields on the row and only segment differences in its segments."""


def segment_fields(row, source):
    """Extract per-segment fields without copying shared transcripts or metadata."""
    fields = {
        key: value
        for key, value in source.items()
        if key != "extra_info" and (key == "audio" or key not in row or not (value is row[key] or value == row[key]))
    }
    shared_info = row.get("extra_info") or {}
    info = {
        key: value
        for key, value in (source.get("extra_info") or {}).items()
        if key not in shared_info or not (value is shared_info[key] or value == shared_info[key])
    }
    if info:
        fields["extra_info"] = info
    return fields


def segment_row(row, segment):
    """Combine shared row fields with one segment when decoding or inspecting it."""
    runtime_fields = {
        "asr_segments",
        "asr_validate",
        "raw_prompt",
        "index",
        "agent_name",
        "uid",
        "global_steps",
        "session_id",
        "trajectory",
        "trace",
        "reward_baselines",
        "segment_outputs",
        "previous_contexts",
    }
    return {
        **{key: value for key, value in row.items() if key not in runtime_fields},
        **segment,
        "extra_info": {**(row.get("extra_info") or {}), **(segment.get("extra_info") or {})},
    }


def validation_detail(row, texts, previous_contexts):
    """Keep shared fields directly in the detail record, alongside its segments."""
    return {
        **segment_row(row, {}),
        "asr_segments": row["asr_segments"],
        "segment_outputs": texts,
        "previous_contexts": previous_contexts,
    }
