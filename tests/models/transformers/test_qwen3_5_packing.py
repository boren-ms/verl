"""Exercise sample isolation with the real Transformers Qwen3.5 decoder."""

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


native = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5", exc_type=ImportError)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402

from verl.models.transformers.qwen3_5 import patch_qwen3_5_packed_sequences  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _torch_kernels_only():
    # Installing optional CUDA kernels must not turn these CPU regressions into
    # CUDA-only tests or initialize a fused normalization on another device.
    return patch.multiple(
        native,
        causal_conv1d_fn=None,
        causal_conv1d_update=None,
        chunk_gated_delta_rule=None,
        fused_recurrent_gated_delta_rule=None,
        FusedRMSNormGated=None,
        is_fast_path_available=False,
    )


def _model(*, use_fast_kernels=False, **config_overrides):
    torch.manual_seed(7)
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        attention_dropout=0.0,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [1, 1, 2],
        },
    )
    for key, value in config_overrides.items():
        setattr(config, key, value)
    config._attn_implementation = "eager"
    with nullcontext() if use_fast_kernels else _torch_kernels_only():
        return native.Qwen3_5ForCausalLM(config).eval()


def _tokens(lengths):
    total = sum(lengths)
    tokens = (torch.arange(total) % 61 + 1).unsqueeze(0)
    positions = torch.cat([torch.arange(length) for length in lengths]).unsqueeze(0)
    return tokens, positions


def _separate_logits(model, tokens, lengths):
    return torch.cat(
        [model(part, use_cache=False).logits for part in torch.split(tokens, lengths, dim=1)],
        dim=1,
    )


def _assert_parameter_gradients_equal(actual, expected):
    expected_parameters = dict(expected.named_parameters())
    for name, parameter in actual.named_parameters():
        other = expected_parameters[name]
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, atol=2e-5, rtol=2e-4, msg=name)


def _preserve_monkey_patch_globals(monkeypatch):
    from transformers.integrations import flash_attention

    from verl.utils.import_utils import is_trl_available

    # Register existing values so pytest restores global hooks modified by the
    # real installation routine, including its optional TRL state_dict hook.
    monkeypatch.setattr(flash_attention, "_flash_attention_forward", flash_attention._flash_attention_forward)
    if is_trl_available():
        from trl import AutoModelForCausalLMWithValueHead

        monkeypatch.setattr(
            AutoModelForCausalLMWithValueHead, "state_dict", AutoModelForCausalLMWithValueHead.state_dict
        )


def test_native_unpatched_model_reproduces_cross_sample_state_leakage():
    model = _model()
    lengths = [5, 4]
    tokens, positions = _tokens(lengths)
    changed = tokens.clone()
    changed[:, : lengths[0]] += 17

    with torch.no_grad():
        separate = _separate_logits(model, tokens, lengths)
        packed = model(tokens, position_ids=positions, use_cache=False).logits
        changed_packed = model(changed, position_ids=positions, use_cache=False).logits

    assert (packed[:, lengths[0] :] - separate[:, lengths[0] :]).abs().max() > 1e-4
    assert (packed[:, lengths[0] :] - changed_packed[:, lengths[0] :]).abs().max() > 1e-4


@pytest.mark.parametrize("lengths", [[1, 1, 1], [5, 4], [63, 3], [65, 2]])
def test_packed_logits_match_independent_samples_including_full_attention(lengths):
    model = _model()
    tokens, positions = _tokens(lengths)
    with torch.no_grad():
        expected = _separate_logits(model, tokens, lengths)
        patch_qwen3_5_packed_sequences(model)
        actual = model(tokens, position_ids=positions, use_cache=False).logits

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_four_axis_positions_use_the_explicit_text_axis_for_boundaries():
    model = _model()
    lengths = [3, 5]
    tokens, positions = _tokens(lengths)
    four_axis = positions.unsqueeze(0).expand(4, -1, -1).clone()
    with torch.no_grad():
        expected = _separate_logits(model, tokens, lengths)
        patch_qwen3_5_packed_sequences(model)
        actual = model(tokens, position_ids=four_axis, use_cache=False).logits

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_resets_in_rotary_axes_do_not_split_a_single_text_sequence():
    original = _model()
    patched = _model()
    patch_qwen3_5_packed_sequences(patched)
    tokens, positions = _tokens([8])
    four_axis = positions.unsqueeze(0).expand(4, -1, -1).clone()
    four_axis[1:, :, 4:] -= 4
    with torch.no_grad():
        expected = original(tokens, position_ids=four_axis, use_cache=False).logits
        actual = patched(tokens, position_ids=four_axis, use_cache=False).logits

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_three_axis_rotary_positions_preserve_ordinary_inference():
    original = _model()
    patched = _model()
    patch_qwen3_5_packed_sequences(patched)
    tokens, positions = _tokens([8])
    three_axis = positions.unsqueeze(0).expand(3, -1, -1).clone()
    three_axis[1, :, 4:] -= 4
    with torch.no_grad():
        expected = original(tokens, position_ids=three_axis, use_cache=False).logits
        actual = patched(tokens, position_ids=three_axis, use_cache=False).logits

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_batched_three_axis_positions_with_internal_zeros_preserve_original_behavior():
    original = _model()
    patched = _model()
    patch_qwen3_5_packed_sequences(patched)
    tokens = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    positions = torch.arange(5).view(1, 1, -1).expand(3, 2, -1).clone()
    positions[:, 1, 3] = 0
    with torch.no_grad():
        expected = original(tokens, position_ids=positions, use_cache=False).logits
        actual = patched(tokens, position_ids=positions, use_cache=False).logits

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_three_axis_packing_without_a_text_axis_is_explicitly_rejected():
    model = _model()
    patch_qwen3_5_packed_sequences(model)
    tokens, positions = _tokens([3, 4])
    three_axis = positions.unsqueeze(0).expand(3, -1, -1)
    with pytest.raises(ValueError, match="text position IDs"):
        model(tokens, position_ids=three_axis, use_cache=False)


@pytest.mark.parametrize("positional_arguments", [False, True])
def test_redundant_all_ones_mask_preserves_packed_sample_boundaries(positional_arguments):
    model = _model()
    lengths = [3, 4]
    tokens, positions = _tokens(lengths)
    with torch.no_grad():
        expected = torch.cat(
            [model.model(part, use_cache=False).last_hidden_state for part in torch.split(tokens, lengths, dim=1)],
            dim=1,
        )
        patch_qwen3_5_packed_sequences(model)
        if positional_arguments:
            actual = model.model(tokens, torch.ones_like(tokens), positions, None, None, False).last_hidden_state
        else:
            actual = model.model(
                tokens, attention_mask=torch.ones_like(tokens), position_ids=positions, use_cache=False
            ).last_hidden_state

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_changing_previous_sample_cannot_change_later_sample_logits():
    model = _model()
    patch_qwen3_5_packed_sequences(model)
    lengths = [5, 4]
    tokens, positions = _tokens(lengths)
    changed = tokens.clone()
    changed[:, : lengths[0]] += 17
    with torch.no_grad():
        actual = model(tokens, position_ids=positions, use_cache=False).logits
        changed_actual = model(changed, position_ids=positions, use_cache=False).logits

    torch.testing.assert_close(actual[:, lengths[0] :], changed_actual[:, lengths[0] :], atol=0, rtol=0)


def test_later_sample_loss_has_zero_gradient_to_previous_sample_inputs():
    model = _model().train()
    patch_qwen3_5_packed_sequences(model)
    lengths = [5, 4]
    _, positions = _tokens(lengths)
    embeddings = torch.randn(1, sum(lengths), model.config.hidden_size, requires_grad=True)

    logits = model(inputs_embeds=embeddings, position_ids=positions, use_cache=False).logits
    logits[:, lengths[0] :].square().sum().backward()

    assert embeddings.grad is not None
    torch.testing.assert_close(
        embeddings.grad[:, : lengths[0]], torch.zeros_like(embeddings.grad[:, : lengths[0]]), atol=0, rtol=0
    )
    assert embeddings.grad[:, lengths[0] :].abs().max() > 0


def test_packed_parameter_gradients_match_sum_of_independent_sample_losses():
    separate = _model().train()
    packed = _model().train()
    patch_qwen3_5_packed_sequences(packed)
    lengths = [5, 4]
    tokens, positions = _tokens(lengths)

    _separate_logits(separate, tokens, lengths).square().sum().backward()
    packed(tokens, position_ids=positions, use_cache=False).logits.square().sum().backward()

    _assert_parameter_gradients_equal(packed, separate)


def test_grouped_value_heads_preserve_packed_outputs_and_gradients():
    separate = _model(linear_num_value_heads=4).train()
    packed = _model(linear_num_value_heads=4).train()
    patch_qwen3_5_packed_sequences(packed)
    lengths = [3, 5]
    tokens, positions = _tokens(lengths)
    expected_logits = _separate_logits(separate, tokens, lengths)
    actual_logits = packed(tokens, position_ids=positions, use_cache=False).logits
    expected_logits.square().sum().backward()
    actual_logits.square().sum().backward()

    torch.testing.assert_close(actual_logits, expected_logits, atol=2e-6, rtol=2e-5)
    _assert_parameter_gradients_equal(packed, separate)


def test_torch_fallback_calls_each_projection_once_for_all_packed_samples():
    model = _model()
    patch_qwen3_5_packed_sequences(model)
    linear_attention = model.model.layers[0].linear_attn
    calls = []
    handles = []

    def record(name):
        def hook(module, args, output):
            calls.append((name, args[0].shape))

        return hook

    projections = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]
    for name in projections:
        handles.append(getattr(linear_attention, name).register_forward_hook(record(name)))
    tokens, positions = _tokens([1, 5, 3])
    try:
        with torch.no_grad():
            model(tokens, position_ids=positions, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    assert [name for name, _ in calls] == projections
    assert calls[0][1][:2] == (3, 5)


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_gradient_checkpoint_recomputation_keeps_sample_boundaries(use_reentrant):
    expected = _model().train()
    actual = _model().train()
    patch_qwen3_5_packed_sequences(expected)
    patch_qwen3_5_packed_sequences(actual)
    actual.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})
    lengths = [5, 4]
    tokens, positions = _tokens(lengths)

    expected_logits = expected(tokens, position_ids=positions, use_cache=False).logits
    actual_logits = actual(tokens, position_ids=positions, use_cache=False).logits
    expected_logits.square().sum().backward()
    actual_logits.square().sum().backward()

    torch.testing.assert_close(actual_logits, expected_logits, atol=2e-6, rtol=2e-5)
    _assert_parameter_gradients_equal(actual, expected)


def test_padded_batch_preserves_original_model_behavior():
    original = _model()
    patched = _model()
    patch_qwen3_5_packed_sequences(patched)
    tokens = torch.tensor([[0, 0, 1, 2, 3], [4, 5, 6, 7, 8]])
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)
    with torch.no_grad():
        expected = original(tokens, attention_mask=mask, position_ids=positions, use_cache=False).logits
        actual = patched(tokens, attention_mask=mask, position_ids=positions, use_cache=False).logits

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_normal_single_sequence_cache_prefill_and_decode_preserve_original_behavior():
    original = _model()
    patched = _model()
    patch_qwen3_5_packed_sequences(patched)
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = original(tokens, use_cache=True)
        actual = patched(tokens, use_cache=True)
        torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
        expected_next = original(torch.tensor([[5]]), past_key_values=expected.past_key_values, use_cache=True)
        actual_next = patched(torch.tensor([[5]]), past_key_values=actual.past_key_values, use_cache=True)

    torch.testing.assert_close(actual_next.logits, expected_next.logits, atol=0, rtol=0)


def test_packed_cache_is_explicitly_rejected():
    model = _model()
    patch_qwen3_5_packed_sequences(model)
    tokens, positions = _tokens([3, 4])
    with pytest.raises((ValueError, RuntimeError, NotImplementedError), match="[Cc]ache|use_cache"):
        model(tokens, position_ids=positions, use_cache=True)


def test_sequence_parallel_packing_is_explicitly_rejected():
    model = _model()
    tokens, positions = _tokens([3, 4])
    with pytest.raises(
        (ValueError, RuntimeError, NotImplementedError), match="[Uu]lysses|[Ss]equence.parallel|sp_size"
    ):
        patch_qwen3_5_packed_sequences(model, ulysses_sp_size=2)
        model(tokens, position_ids=positions, use_cache=False)


def test_fast_kernel_calls_receive_convolution_and_recurrent_boundaries(monkeypatch):
    # These stubs verify argument plumbing only. Numerical isolation is checked
    # above using the real native Torch decoder, without simulated kernels.
    observed = {}

    def conv(*, x, weight, bias, activation, seq_idx):
        observed["seq_idx"] = seq_idx.clone()
        return x

    def delta(query, key, value, *, cu_seqlens, **kwargs):
        observed["cu_seqlens"] = cu_seqlens.clone()
        observed["initial_state"] = kwargs["initial_state"]
        observed["output_final_state"] = kwargs["output_final_state"]
        return torch.zeros_like(value), None

    monkeypatch.setattr(native, "causal_conv1d_fn", conv)
    monkeypatch.setattr(native, "chunk_gated_delta_rule", delta)
    monkeypatch.setattr(native, "FusedRMSNormGated", None)
    model = _model(use_fast_kernels=True)
    patch_qwen3_5_packed_sequences(model)
    tokens, positions = _tokens([2, 3, 1])
    with torch.no_grad():
        model(tokens, position_ids=positions, use_cache=False)

    torch.testing.assert_close(observed["seq_idx"], torch.tensor([[0, 0, 1, 1, 1, 2]], dtype=torch.int32))
    torch.testing.assert_close(observed["cu_seqlens"], torch.tensor([0, 2, 5, 6], dtype=torch.int32))
    assert observed["initial_state"] is None
    assert observed["output_final_state"] is False


def test_patch_installation_is_idempotent_and_preserves_saved_weights(tmp_path):
    model = _model()
    tokens, positions = _tokens([3, 4])
    original_keys = set(model.state_dict())
    patch_qwen3_5_packed_sequences(model)
    with torch.no_grad():
        expected = model(tokens, position_ids=positions, use_cache=False).logits
    patch_qwen3_5_packed_sequences(model)
    assert set(model.state_dict()) == original_keys
    with torch.no_grad():
        actual = model(tokens, position_ids=positions, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    model.save_pretrained(tmp_path)
    with _torch_kernels_only():
        restored = native.Qwen3_5ForCausalLM.from_pretrained(tmp_path, attn_implementation="eager").eval()
    assert set(restored.state_dict()) == original_keys
    patch_qwen3_5_packed_sequences(restored)
    with torch.no_grad():
        restored_logits = restored(tokens, position_ids=positions, use_cache=False).logits
    torch.testing.assert_close(restored_logits, expected, atol=0, rtol=0)


@pytest.mark.parametrize("use_remove_padding", [False, True])
@pytest.mark.parametrize("use_linear_attention_packing", [None, False, True], ids=["default", "disabled", "enabled"])
def test_apply_monkey_patch_installs_the_fix_only_when_requested(
    monkeypatch, use_remove_padding, use_linear_attention_packing
):
    from verl.models.transformers.monkey_patch import apply_monkey_patch

    _preserve_monkey_patch_globals(monkeypatch)
    model = _model()
    lengths = [3, 4]
    tokens, positions = _tokens(lengths)
    with torch.no_grad():
        independent = _separate_logits(model, tokens, lengths)
        unpatched = model(tokens, position_ids=positions, use_cache=False).logits
        packing_options = (
            {}
            if use_linear_attention_packing is None
            else {"use_linear_attention_packing": use_linear_attention_packing}
        )
        apply_monkey_patch(model, use_remove_padding=use_remove_padding, use_fused_kernels=False, **packing_options)
        actual = model(tokens, position_ids=positions, use_cache=False).logits

    assert getattr(model, "_verl_qwen3_5_packing_patched", False) is bool(use_linear_attention_packing)
    if use_linear_attention_packing:
        torch.testing.assert_close(actual, independent, atol=2e-6, rtol=2e-5)
    else:
        torch.testing.assert_close(actual, unpatched, atol=0, rtol=0)
        assert (actual[:, lengths[0] :] - independent[:, lengths[0] :]).abs().max() > 1e-4


@pytest.mark.parametrize("use_remove_padding", [False, True])
def test_sequence_parallel_size_does_not_enable_linear_attention_packing(monkeypatch, use_remove_padding):
    from verl.models.transformers.monkey_patch import apply_monkey_patch

    _preserve_monkey_patch_globals(monkeypatch)
    model = _model()
    tokens, positions = _tokens([3, 4])
    with torch.no_grad():
        expected = model(tokens, position_ids=positions, use_cache=False).logits
        apply_monkey_patch(model, ulysses_sp_size=2, use_remove_padding=use_remove_padding)
        actual = model(tokens, position_ids=positions, use_cache=False).logits

    assert not getattr(model, "_verl_qwen3_5_packing_patched", False)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_explicit_linear_attention_packing_rejects_unsupported_sequence_parallelism(monkeypatch):
    from verl.models.transformers.monkey_patch import apply_monkey_patch

    _preserve_monkey_patch_globals(monkeypatch)
    with pytest.raises(ValueError, match="Ulysses sequence parallelism"):
        apply_monkey_patch(_model(), ulysses_sp_size=2, use_linear_attention_packing=True)


def test_audio_model_uses_the_patched_native_decoder_through_normal_installation(monkeypatch):
    from verl.models.transformers.monkey_patch import apply_monkey_patch

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "plugins/qwen35_audio/src"))
    audio = pytest.importorskip("hf_qwen35_audio", exc_type=ImportError)
    _preserve_monkey_patch_globals(monkeypatch)
    kwargs = _model().config.to_dict()
    kwargs.pop("model_type", None)
    kwargs["audio_processor"] = {
        "name": "flash",
        "config": {
            "input_size": 8,
            "hidden_size": 16,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "rope_theta": 10000.0,
            "frontend_type": "temporal_conv",
            "frontend_window_size": 2,
        },
    }
    config = audio.Qwen3_5AudioConfig(**kwargs)
    config._attn_implementation = "eager"
    with _torch_kernels_only():
        model = audio.Qwen3_5AudioForCausalLM(config).eval()
    assert isinstance(model.model, native.Qwen3_5TextModel)
    lengths = [3, 4]
    tokens, positions = _tokens(lengths)
    with torch.no_grad():
        expected = _separate_logits(model, tokens, lengths)
        apply_monkey_patch(model, use_remove_padding=True, use_fused_kernels=False, use_linear_attention_packing=True)
        actual = model(tokens, position_ids=positions, use_cache=False).logits

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def _cuda_fast_model(**config_overrides):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("FLA regression requires an Ampere or newer GPU")
    if native.causal_conv1d_fn is None or native.chunk_gated_delta_rule is None:
        pytest.skip("Both FLA and causal-conv1d kernels are required")

    kwargs = dict(
        use_fast_kernels=True,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        dtype=torch.bfloat16,
    )
    kwargs.update(config_overrides)
    return _model(**kwargs).to(device="cuda", dtype=torch.bfloat16)


def _assert_cuda_parameter_gradients_equal(actual, expected):
    expected_parameters = dict(expected.named_parameters())
    for name, parameter in actual.named_parameters():
        other = expected_parameters[name]
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
            assert torch.isfinite(other.grad).all(), name
            # BF16 fused reductions and accumulation across independent calls
            # can differ in rounding. Scale the absolute tolerance to this
            # parameter's gradient so small projections are still checked.
            scale = other.grad.float().abs().max().item()
            torch.testing.assert_close(
                parameter.grad.float(),
                other.grad.float(),
                atol=max(1e-6, scale * 5e-3),
                rtol=5e-2,
                msg=name,
            )


@pytest.mark.parametrize(
    ("lengths", "value_heads"),
    [
        pytest.param([1, 1, 1], 2, id="adjacent-single-token-samples"),
        pytest.param([3, 5, 1], 4, id="short-convolution-segments-grouped-heads"),
        pytest.param([63, 1, 65], 4, id="fla-chunk-boundaries-with-single-token"),
        pytest.param([64, 65, 127], 4, id="multiple-fla-chunks"),
    ],
)
def test_cuda_fast_kernels_match_independent_samples(lengths, value_heads):
    model = _cuda_fast_model(linear_num_value_heads=value_heads)
    tokens, positions = _tokens(lengths)
    tokens, positions = tokens.cuda(), positions.cuda()
    with torch.no_grad():
        expected = _separate_logits(model, tokens, lengths)
        patch_qwen3_5_packed_sequences(model)
        actual = model(tokens, position_ids=positions, use_cache=False).logits
        changed = tokens.clone()
        changed[:, : lengths[0]] = (changed[:, : lengths[0]] + 17) % 63
        changed_actual = model(changed, position_ids=positions, use_cache=False).logits

    torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-2)
    torch.testing.assert_close(actual[:, lengths[0] :], changed_actual[:, lengths[0] :], atol=0, rtol=0)
    model.train()
    embeddings = model.get_input_embeddings()(tokens).detach().requires_grad_()
    logits = model(inputs_embeds=embeddings, position_ids=positions, use_cache=False).logits
    logits[:, lengths[0] :].float().square().sum().backward()
    assert torch.count_nonzero(embeddings.grad[:, : lengths[0]]) == 0
    assert torch.count_nonzero(embeddings.grad[:, lengths[0] :]) > 0
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


@pytest.mark.parametrize("lengths", [[3, 5, 1], [63, 1, 65]], ids=["short-segments", "fla-chunk-boundaries"])
def test_cuda_fast_kernel_parameter_gradients_match_independent_samples(lengths):
    expected = _cuda_fast_model().train()
    actual = _cuda_fast_model().train()
    patch_qwen3_5_packed_sequences(actual)
    tokens, positions = _tokens(lengths)
    tokens, positions = tokens.cuda(), positions.cuda()

    expected_logits = _separate_logits(expected, tokens, lengths)
    actual_logits = actual(tokens, position_ids=positions, use_cache=False).logits
    expected_logits.float().square().sum().backward()
    actual_logits.float().square().sum().backward()

    torch.testing.assert_close(actual_logits, expected_logits, atol=3e-3, rtol=3e-2)
    _assert_cuda_parameter_gradients_equal(actual, expected)


@pytest.mark.parametrize("use_reentrant", [False, True], ids=["non-reentrant", "reentrant"])
def test_cuda_gradient_checkpoint_recomputation_keeps_fast_kernel_boundaries(use_reentrant):
    expected = _cuda_fast_model().train()
    actual = _cuda_fast_model().train()
    patch_qwen3_5_packed_sequences(expected)
    patch_qwen3_5_packed_sequences(actual)
    actual.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})
    lengths = [63, 1, 65]
    tokens, positions = _tokens(lengths)
    tokens, positions = tokens.cuda(), positions.cuda()

    expected_logits = expected(tokens, position_ids=positions, use_cache=False).logits
    actual_logits = actual(tokens, position_ids=positions, use_cache=False).logits
    expected_logits.float().square().sum().backward()
    actual_logits.float().square().sum().backward()

    torch.testing.assert_close(actual_logits, expected_logits, atol=0, rtol=0)
    _assert_cuda_parameter_gradients_equal(actual, expected)


def test_cuda_real_fast_kernels_receive_all_sample_boundaries(monkeypatch):
    from verl.models.transformers.monkey_patch import apply_monkey_patch

    _preserve_monkey_patch_globals(monkeypatch)
    model = _cuda_fast_model()
    apply_monkey_patch(model, use_remove_padding=True, use_linear_attention_packing=True)
    linear_attention = model.model.layers[0].linear_attn
    original_conv = linear_attention.causal_conv1d_fn
    original_delta = linear_attention.chunk_gated_delta_rule
    observed = {}

    def conv(*args, **kwargs):
        observed["seq_idx"] = kwargs["seq_idx"].detach().cpu()
        return original_conv(*args, **kwargs)

    def delta(*args, **kwargs):
        observed["cu_seqlens"] = kwargs["cu_seqlens"].detach().cpu()
        return original_delta(*args, **kwargs)

    # Install transparent recorders after choosing the real fast-kernel path.
    # Both implementations still execute and their outputs are checked below.
    monkeypatch.setattr(linear_attention, "causal_conv1d_fn", conv)
    monkeypatch.setattr(linear_attention, "chunk_gated_delta_rule", delta)
    lengths = [1, 63, 1, 65]
    tokens, positions = _tokens(lengths)
    tokens, positions = tokens.cuda(), positions.cuda()
    with torch.no_grad():
        actual = model(tokens, position_ids=positions, use_cache=False).logits

    assert torch.isfinite(actual).all()
    expected_seq_idx = torch.repeat_interleave(torch.arange(len(lengths), dtype=torch.int32), torch.tensor(lengths))
    torch.testing.assert_close(observed["seq_idx"], expected_seq_idx.unsqueeze(0))
    expected_cu_seqlens = torch.tensor([0, 1, 64, 65, 130], dtype=torch.int32)
    torch.testing.assert_close(observed["cu_seqlens"], expected_cu_seqlens)
