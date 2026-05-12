"""HF-compatible Qwen3.5-Audio CausalLM for FSDP actor logprob computation.

Uses the bundled phyagi-derived Qwen3.5 text backbone (GatedDeltaNet hybrid
attention) together with vLLM's ConformerEncoder audio tower to compute
logprobs conditioned on audio.

Architecture
------------
- ``self.model``: ``Qwen3_5TextModel`` from the bundled model module.
  - ``self.model.embed_tokens``: standard token embedding table.
  - ``self.model.embed_tokens_extend``: ``Qwen3_5AudioEmbeddingHF`` that
    embeds tokens AND injects audio at ``<|audio_start|>`` positions.
  - ``self.model.layers``: alternating GatedDeltaNet + full-attention layers.
  - ``self.model.norm``: final RMSNorm.
- ``self.lm_head``: unembedding projection.

Weight key mapping (checkpoint → this model)
--------------------------------------------
Checkpoint keys are of the form ``model.*`` (text) and ``lm_head.*``.
The audio tower lives at ``model.embed_tokens_extend.*`` in the checkpoint
and maps directly to ``self.model.embed_tokens_extend.*`` here — no remap needed.

Forward
-------
Audio inputs come from ``multi_modal_inputs`` in dp_actor.py as:
- ``input_audio_embeds``: mel filterbank (B, T, 80) or list of (T_i, 80).
- ``audio_attention_mask``: optional mel mask (currently ignored).

The embedding adapter (``embed_tokens_extend``) handles text embedding and
audio injection at ``<|audio_start|>`` (248070) token positions.
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class Qwen3_5AudioForCausalLMHF(PreTrainedModel):
    """Qwen3.5-Audio (bundled backbone) for FSDP actor log-prob computation.

    Initialisation uses the bundled ``Qwen3_5TextModel`` which requires
    transformers >= 5.4.0.  The ``prepare_env`` pipeline ensures 5.7.0 is
    installed before any worker imports this class.
    """

    _keys_to_ignore_on_load_unexpected = [r"^mtp\."]
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        from .bundled_qwen3_5_model import Qwen3_5TextModel

        self.model = Qwen3_5TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    # ------------------------------------------------------------------
    # HF boilerplate
    # ------------------------------------------------------------------

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        # Audio kwargs (from multi_modal_inputs via dp_actor.py)
        input_audio_embeds=None,
        audio_attention_mask=None,
        **kwargs,
    ):
        # Pass input_ids + mel features to the text model.
        # embed_tokens_extend (Qwen3_5AudioEmbeddingHF) will:
        #   1. Embed text tokens via wte(input_ids)
        #   2. Process mel through the ConformerEncoder
        #   3. Inject at <|audio_start|> positions
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=input_audio_embeds,  # mel features (or None)
            use_cache=use_cache,
            audio_attention_mask=audio_attention_mask,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

