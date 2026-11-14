"""Check ASR checkpoint filtering against the installed vLLM loader."""

import pytest
import torch


@pytest.mark.parametrize("unexpected", [False, True])
def test_audio_loader_preserves_exclusions_and_rejects_other_unknown_weights(unexpected):
    from vllm_qwen35_audio.plugin import register

    register()
    from vllm.model_executor.models.utils import WeightsMapper

    from vllm_qwen35_audio.qwen3_5_audio import Qwen3_5AudioForCausalLM

    model = torch.nn.Module()
    model.language_model = torch.nn.Linear(2, 2, bias=False)
    model.hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.": "language_model."})
    expected = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    weights = [("model.weight", expected)]
    weights.extend((prefix + "weight", torch.zeros(1)) for prefix in ("mtp.", "lora_A.", "lora_B."))
    if unexpected:
        weights.append(("unexpected.weight", torch.zeros(1)))
        with pytest.raises(ValueError, match="unexpected"):
            Qwen3_5AudioForCausalLM.load_weights(model, iter(weights))
    else:
        loaded = Qwen3_5AudioForCausalLM.load_weights(model, iter(weights))
        assert loaded == {"language_model.weight"}
        torch.testing.assert_close(model.language_model.weight, expected)
