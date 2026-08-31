from recipe.phimm.data.prompts import get_task_output, get_task_prompt, resolve_task_language
from recipe.phimm.reward.asr_measure import _parse_task_output, check_fmt


def test_get_task_output_formats_mixed_components():
    output = get_task_output(
        task="lang_asr",
        lang="en_zh",
        text="hello 你好",
        components=[
            {"language": "en", "text": "hello"},
            {"language": "zh", "text": "你好"},
        ],
    )

    assert output == "<src=English><tgt=English>\nhello\n<src=Chinese><tgt=Chinese>\n你好"
    assert check_fmt(_parse_task_output(output))


def test_get_task_output_preserves_single_language_format():
    assert get_task_output(task="lang_asr", lang="en", text="hello") == (
        "<src=English><tgt=English>\nhello"
    )


def test_lang_asr_verb_uses_verbatim_prompt_and_standard_output_format():
    assert get_task_prompt(task="lang_asr_verb", rand=False) == (
        "Detect the language and transcribe the audio clip into text. "
        "Transcribe verbatim, including all filler words and disfluencies."
    )
    assert get_task_output(task="lang_asr_verb_en", lang="en", text="um hello") == (
        "<src=English><tgt=English>\num hello"
    )
    assert resolve_task_language(task="lang_asr_verb_en") == "English"