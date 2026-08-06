from recipe.phimm.data.dataset import _is_bad_fmt, _is_bad_lang, add_task_info


class MinimalDataset:
    def map(self, function, **kwargs):
        self.example = function({"text": "bonjour"})
        return self


def test_add_task_info_enables_language_prefix_by_default():
    dataset = add_task_info(MinimalDataset(), task="lang_asr", language="French")

    assert dataset.example["prefix"] == "Audio Language: French\n"


def test_add_task_info_allows_language_prefix_opt_out():
    dataset = add_task_info(MinimalDataset(), task="lang_asr", language="French", prefix_prob=0.0)

    assert dataset.example["prefix"] == ""


def test_bad_format_uses_task_output_format():
    valid = "Audio Language: English.\n<ASR><lang=English><TXT>Hello</TXT></ASR>"

    assert not _is_bad_fmt({"raw_response": valid})
    assert _is_bad_fmt({"raw_response": "Hello"})


def test_bad_language_uses_task_output_languages():
    mixed = (
        "Audio Language: English and Chinese.\n"
        "<ASR><lang=English><TXT>Hello</TXT><lang=Chinese><TXT>ni hao</TXT></ASR>"
    )
    wrong = "Audio Language: French.\n<ASR><lang=French><TXT>Bonjour</TXT></ASR>"

    assert not _is_bad_lang({"raw_response": mixed, "language": "English_Chinese"})
    assert _is_bad_lang({"raw_response": wrong, "language": "English"})
    assert _is_bad_lang({"raw_response": "Hello", "language": "English"})


def test_nonspeech_is_not_bad_language():
    nonspeech = "Audio Language: English.\n<ASR><lang=English><TXT><nonspeech></TXT></ASR>"

    assert not _is_bad_lang({"raw_response": nonspeech, "language": "French"})