"""Exercise the source frontend's view contract on collated ragged audio."""

import torch
from hf_qwen35_audio.flash_encoder import LightPatchEncoder

from verl.utils.model import extract_multi_modal_inputs


def test_collated_audio_frontend_matches_independent_contiguous_batch():
    short = torch.arange(8 * 80, dtype=torch.float32).reshape(1, 8, 80)
    long = torch.arange(16 * 80, dtype=torch.float32).reshape(1, 16, 80)
    batch = extract_multi_modal_inputs(
        [
            {"input_audio_embeds": short, "audio_attention_mask": torch.ones(1, 8)},
            {"input_audio_embeds": long, "audio_attention_mask": torch.ones(1, 16)},
        ]
    )
    # A multiple of the frontend window deliberately avoids its F.pad copy,
    # which can otherwise conceal an incompatible input stride.
    frontend = LightPatchEncoder(input_dim=80, window_size=8)
    actual, mask = frontend(batch["input_audio_embeds"], batch["audio_attention_mask"])
    padded = torch.stack([torch.nn.functional.pad(short[0], (0, 0, 0, 8)), long[0]])
    expected, expected_mask = frontend(padded, torch.tensor([[1] * 8 + [0] * 8, [1] * 16]))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(mask, expected_mask)
