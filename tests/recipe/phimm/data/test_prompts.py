from recipe.phimm.data.prompts import get_task_output, get_task_prompt, resolve_task_language
from recipe.phimm.reward.mix_lang_measure import check_fmt


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

    assert output == (
        "Audio Language: English and Chinese\n"
        "<ASR><lang=English><TXT>hello</TXT>\n<lang=Chinese><TXT>你好</TXT></ASR>"
    )
    assert check_fmt(output)


def test_get_task_output_preserves_single_language_format():
    assert get_task_output(task="lang_asr", lang="en", text="hello") == (
        "Audio Language: English.\n<ASR><lang=English><TXT>hello</TXT></ASR>"
    )


def test_lang_asr_verb_uses_verbatim_prompt_and_output_tag():
    assert get_task_prompt(task="lang_asr_verb", rand=False) == (
        "Detect the language and transcribe the audio clip into text. "
        "Transcribe verbatim, including all filler words and disfluencies."
    )
    assert get_task_output(task="lang_asr_verb_en", lang="en", text="um hello") == (
        "Audio Language: English.\n"
        "<ASR_VERBATIM><lang=English><TXT>um hello</TXT></ASR_VERBATIM>"
    )
    assert resolve_task_language(task="lang_asr_verb_en") == "English"