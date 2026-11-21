"""Training adapters for the pinned official offline ASR model.

Keep the checkpoint's names and native prompt. Padded PPO forwards append
response tokens, whereas the processor's acoustic mask covers the prompt only.
"""

import torch.nn.functional as F
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor as OfficialProcessor


class VibeVoiceASRProcessor(OfficialProcessor):
    """AutoProcessor entry point; audio processing stays in the upstream class."""

    def save_pretrained(self, save_directory, **kwargs):
        # Upstream saves only preprocessor_config.json. A resumable offline
        # bundle also needs the checkpoint's generated tokenizer assets.
        super().save_pretrained(save_directory, **kwargs)
        return self.tokenizer.save_pretrained(save_directory, **kwargs)


class VibeVoiceASRForRL(VibeVoiceASRForConditionalGeneration):
    _no_split_modules = ["Qwen2DecoderLayer"]

    def tie_weights(self, *args, **kwargs):
        # Transformers 5 supplies recompute_mapping; upstream's fixed decoder
        # tie operation has no parameters. Preserve that operation on both APIs.
        return super().tie_weights()

    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "verl_deterministic_audio", False):
            # HF and rollout must condition on the same features. Upstream
            # otherwise draws fresh acoustic noise on every forward.
            self.model.acoustic_tokenizer.std_dist_type = "none"
        for component in self._frontends():
            component.requires_grad_(False)
            component.eval()

    def _frontends(self):
        return (
            self.model.acoustic_tokenizer,
            self.model.semantic_tokenizer,
            self.model.acoustic_connector,
            self.model.semantic_connector,
        )

    def train(self, mode=True):
        super().train(mode)
        for component in self._frontends():
            component.eval()
        return self

    def forward(self, input_ids=None, acoustic_input_mask=None, **kwargs):
        if acoustic_input_mask is not None:
            if input_ids is None or acoustic_input_mask.shape[0] != input_ids.shape[0]:
                raise ValueError("VibeVoice PPO requires padded input IDs with one acoustic mask per sample")
            extra = input_ids.shape[-1] - acoustic_input_mask.shape[-1]
            if extra < 0:
                raise ValueError("The acoustic mask cannot exceed the full token sequence")
            acoustic_input_mask = F.pad(acoustic_input_mask, (0, extra), value=False)
        return super().forward(input_ids=input_ids, acoustic_input_mask=acoustic_input_mask, **kwargs)
