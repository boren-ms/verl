from recipe.phimm.data.prompts import (
    get_task_output,
    get_task_prefix,
    get_task_prompt,
    resolve_task_language,
)
from recipe.phimm.reward.asr_measure import check_fmt
from recipe.phimm.reward.asr_response import parse_task_output


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
    assert check_fmt(parse_task_output(output))


def test_get_task_output_preserves_single_language_format():
    assert get_task_output(task="lang_asr", lang="en", text="hello") == (
        "<src=English><tgt=English>\nhello"
    )


def test_get_task_output_supports_2607_format():
    output = get_task_output(task="lang_asr", lang="en", text="hello", version=2607)

    assert output == "Audio Language: English.\n<ASR><lang=English><TXT>hello</TXT></ASR>"
    assert parse_task_output(output, version=2607) == (["English"], ["English"], ["hello"])


def test_task_prefix_and_output_support_2607_completion_format():
    prefix = get_task_prefix(task="lang_asr", lang="en", version=2607)
    output = get_task_output(task="lang_asr", lang="en", text="hello", version=2607)

    assert prefix == "Audio Language: English\n"
    assert output == "Audio Language: English.\n<ASR><lang=English><TXT>hello</TXT></ASR>"
    assert parse_task_output(output, version=2607) == (
        ["English"],
        ["English"],
        ["hello"],
    )


def test_get_task_output_supports_2607_mixed_components():
    output = get_task_output(
        task="lang_asr",
        lang="en_zh",
        text="hello 你好",
        components=[
            {"language": "en", "text": "hello"},
            {"language": "zh", "text": "你好"},
        ],
        version="2607",
    )

    assert output == (
        "Audio Language: English and Chinese.\n"
        "<ASR><lang=English><TXT>hello</TXT><lang=Chinese><TXT>你好</TXT></ASR>"
    )
    assert parse_task_output(output, version=2607) == (
        ["English", "Chinese"],
        ["English", "Chinese"],
        ["hello", "你好"],
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