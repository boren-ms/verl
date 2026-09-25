import ast
import re
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf


def _load_flatten_data_confs():
    module_path = Path(__file__).parents[3] / "recipe/phimm/data/rl_dataset.py"
    module = ast.parse(module_path.read_text())
    function = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_flatten_data_confs"
    )
    namespace = {"Sequence": Sequence}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(module_path), "exec"), namespace)
    return namespace["_flatten_data_confs"]


def _load_audio_retry_helper():
    module_path = Path(__file__).parents[3] / "recipe/phimm/data/rl_dataset.py"
    module = ast.parse(module_path.read_text())
    function = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_load_audio_with_retries"
    )
    namespace = {"logger": SimpleNamespace(warning=lambda *args, **kwargs: None)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(module_path), "exec"), namespace)
    return namespace["_load_audio_with_retries"]


def _load_think_helpers():
    module_path = Path(__file__).parents[3] / "recipe/phimm/data/rl_dataset.py"
    module = ast.parse(module_path.read_text())
    names = {
        "_apply_chat_template",
        "_validate_assistant_prefix",
    }
    functions = [
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "_THINK_PREFILL_RE": re.compile(
            r"(?:<think>\s*</think>|<think>)\s*$",
            re.IGNORECASE,
        ),
        "_THINK_TAG_RE": re.compile(r"</?think>", re.IGNORECASE),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(module_path), "exec"), namespace)
    return (
        namespace["_apply_chat_template"],
        namespace["_validate_assistant_prefix"],
    )


def test_flatten_data_confs_expands_nested_omegaconf_groups():
    flatten_data_confs = _load_flatten_data_confs()
    grouped = OmegaConf.create([
        [{"dataset_name": "jsonl", "cache_name": "first"}, {"dataset_name": "jsonl", "cache_name": "second"}],
        [{"dataset_name": "jsonl", "cache_name": "third"}],
    ])

    flattened = flatten_data_confs(grouped)

    assert [config.cache_name for config in flattened] == ["first", "second", "third"]


def test_flatten_data_confs_preserves_flat_and_scalar_inputs():
    flatten_data_confs = _load_flatten_data_confs()
    flat = OmegaConf.create([{"dataset_name": "jsonl"}])

    assert flatten_data_confs(flat) == list(flat)
    assert flatten_data_confs("data.jsonl") == ["data.jsonl"]


def test_load_audio_with_retries_skips_unreadable_training_sample():
    load_audio_with_retries = _load_audio_retry_helper()
    rows = [{"audio_path": "bad.wav"}, {"audio_path": "good.wav"}]

    def load_audio(row, max_dur):
        if row["audio_path"] == "bad.wav":
            raise OSError("unreadable")
        return max_dur, row["audio_path"]

    row, audio = load_audio_with_retries(rows, 0, 40, 1, load_audio, (OSError,))

    assert row == rows[1]
    assert audio == (40, "good.wav")


def test_load_audio_with_retries_spreads_fallbacks_across_dataset():
    load_audio_with_retries = _load_audio_retry_helper()
    rows = [{"audio_path": f"{index}.wav"} for index in range(20)]
    attempted = []

    def load_audio(row, max_dur):
        attempted.append(row["audio_path"])
        if row["audio_path"] == "0.wav":
            raise OSError("unreadable")
        return row["audio_path"]

    row, audio = load_audio_with_retries(rows, 0, 40, 1, load_audio, (OSError,))

    assert attempted == ["0.wav", "10.wav"]
    assert row == rows[10]
    assert audio == "10.wav"


def test_load_audio_with_retries_avoids_stringifying_broken_audio_error():
    load_audio_with_retries = _load_audio_retry_helper()

    class BrokenAudioError(Exception):
        def __str__(self):
            raise TypeError("broken exception formatting")

    def load_audio(row, max_dur):
        raise BrokenAudioError()

    try:
        load_audio_with_retries(
            [{"audio_path": "bad.wav"}],
            0,
            40,
            0,
            load_audio,
            (BrokenAudioError,),
        )
    except RuntimeError as exc:
        assert str(exc) == "Unable to load audio after 1 attempt(s): ['bad.wav']"
    else:
        raise AssertionError("expected contextual RuntimeError")


def test_assistant_prefix_rejects_think_tags():
    _, validate_assistant_prefix = _load_think_helpers()

    validate_assistant_prefix("Audio Language: English\n")
    for prefix in ("<think>", "<think>duplicate</think>"):
        try:
            validate_assistant_prefix(prefix)
        except ValueError as exc:
            assert "must not contain <think> tags" in str(exc)
        else:
            raise AssertionError("expected duplicate think prefix to be rejected")


def test_apply_chat_template_enables_thinking_and_strips_template_open_tag():
    apply_chat_template, _ = _load_think_helpers()

    class ChatTemplate:
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            self.kwargs = kwargs
            return "<|im_start|>assistant\n<think>\n"

    chat_template = ChatTemplate()
    messages = [{"role": "user", "content": "Think about rare words."}]

    raw_prompt = apply_chat_template(
        chat_template,
        messages,
        {"enable_thinking": True},
    )

    assert chat_template.messages == messages
    assert chat_template.kwargs == {
        "add_generation_prompt": True,
        "tokenize": False,
        "enable_thinking": True,
    }
    assert raw_prompt == "<|im_start|>assistant\n"


def test_apply_chat_template_strips_empty_think_block():
    apply_chat_template, _ = _load_think_helpers()

    class ChatTemplate:
        def apply_chat_template(self, messages, **kwargs):
            return "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    assert apply_chat_template(
        ChatTemplate(),
        [{"role": "user", "content": "Think about rare words."}],
        {"enable_thinking": True},
    ) == "<|im_start|>assistant\n"


def test_apply_chat_template_does_not_strip_think_instruction():
    apply_chat_template, _ = _load_think_helpers()
    prompt = (
        "List words inside <think> and </think>.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    class ChatTemplate:
        def apply_chat_template(self, messages, **kwargs):
            return prompt

    assert apply_chat_template(
        ChatTemplate(),
        [{"role": "user", "content": "Think about rare words."}],
        {"enable_thinking": True},
    ) == prompt