from types import SimpleNamespace

import pytest

from verl.workers.rollout.utils import get_max_position_embeddings


def test_native_decoder_context_limit_is_found_without_root_or_text_config():
    decoder = SimpleNamespace(max_position_embeddings=131072)
    config = SimpleNamespace(decoder_config=decoder, get_text_config=lambda: decoder)
    assert get_max_position_embeddings(config) == 131072


def test_explicit_root_context_limit_keeps_precedence():
    config = SimpleNamespace(
        max_position_embeddings=8192,
        get_text_config=lambda: SimpleNamespace(max_position_embeddings=131072),
    )
    assert get_max_position_embeddings(config) == 8192


def test_missing_context_limit_still_fails_closed():
    with pytest.raises(ValueError, match="max_position_embeddings not found"):
        get_max_position_embeddings(SimpleNamespace(get_text_config=lambda: SimpleNamespace()))
