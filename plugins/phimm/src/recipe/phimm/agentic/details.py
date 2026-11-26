"""Shared-row metadata for compact ASR segment records."""

RUNTIME_FIELDS = frozenset(
    "asr_segments asr_validate raw_prompt index agent_name uid global_steps session_id "
    "trajectory trace reward_baselines segment_outputs previous_contexts".split()
)


def expand_segments(row, segments=None):
    """Merge shared row metadata with each compact segment override."""
    shared = {key: value for key, value in row.items() if key not in RUNTIME_FIELDS}
    shared_info = row.get("extra_info") or {}
    items = row.get("asr_segments", [{}]) if segments is None else segments
    return [
        {**shared, **segment, "extra_info": {**shared_info, **(segment.get("extra_info") or {})}} for segment in items
    ]


def validation_detail(row, texts, previous_contexts):
    """Serialize metadata and segment outputs without waveform or runtime payloads."""
    return {
        **{key: value for key, value in row.items() if key not in RUNTIME_FIELDS and key != "audio"},
        "asr_segments": [{key: value for key, value in s.items() if key != "audio"} for s in row["asr_segments"]],
        "segment_outputs": texts,
        "previous_contexts": previous_contexts,
    }
