"""Lossless debug records for full recordings with many hard chunks."""


def pack_segment_rows(rows, *, validate):
    """Avoid repeating full-parent metadata through TransferQueue serialization."""
    if not validate or not rows:
        return rows
    detail = validation_detail(rows, [], rows[0].get("extra_info") or {}, [], validate=True)
    if "segment_rows_encoding" not in detail:
        return rows
    return {key: detail[key] for key in ("segment_rows_encoding", "parent_row", "parent_info", "segment_rows")}


def restore_segment_rows(value):
    """Restore exact original rows before generation, reference context or scoring."""
    if isinstance(value, list):
        return value
    if not isinstance(value, dict) or value.get("segment_rows_encoding") != "parent_row_plus_overrides_v1":
        raise ValueError("Unknown ASR segment row transport encoding")
    return [
        {**value["parent_row"], **row, "extra_info": {**value["parent_info"], **row["extra_info"]}}
        for row in value["segment_rows"]
    ]


def validation_detail(rows, texts, parent_info, previous_contexts, *, validate):
    """Store shared hard-chunk fields once, preserving exact reconstructed rows.

    The compact segment rows overlay ``parent_row``. Their ``extra_info`` and
    ``segment_info`` entries overlay ``parent_info``. Decoding and scoring use
    the original rows; this representation is only the persisted debug payload.
    """
    detail = {
        "parent_info": parent_info,
        "segment_outputs": texts,
        "segment_info": [row.get("extra_info") or {} for row in rows],
        "previous_contexts": previous_contexts,
    }
    if not validate:
        return detail
    detail["segment_rows"] = rows
    if (
        not rows
        or parent_info.get("chunking_method") != "hard_nonoverlapping"
        or not all(row.get("asr_full_reference") for row in rows)
        or any(set(row.get("extra_info") or {}) != set(parent_info) for row in rows)
    ):
        return detail

    # The dataset copies one parent into segment dictionaries. Identity checks
    # avoid repeatedly comparing shared full transcripts and selector lists.
    common = {key: value for key, value in rows[0].items() if key not in {"audio_path", "extra_info"}}
    for row in rows[1:]:
        for key in tuple(common):
            if key not in row or not (row[key] is common[key] or row[key] == common[key]):
                del common[key]

    info_cache = {}
    compact_rows, compact_infos = [], []
    for row in rows:
        info = row.get("extra_info") or {}
        # Preserve one original dictionary per distinct metadata object. The
        # cache is local and all source objects stay alive throughout this call.
        if id(info) not in info_cache:
            info_cache[id(info)] = {
                key: value
                for key, value in info.items()
                if not (value is parent_info[key] or value == parent_info[key])
            }
        overrides = info_cache[id(info)]
        segment = {key: value for key, value in row.items() if key not in common and key != "extra_info"}
        segment["extra_info"] = overrides
        compact_rows.append(segment)
        compact_infos.append(overrides)
    detail.update(
        segment_rows_encoding="parent_row_plus_overrides_v1",
        segment_info_encoding="parent_info_plus_overrides_v1",
        parent_row=common,
        segment_rows=compact_rows,
        segment_info=compact_infos,
    )
    return detail
