# %%
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Prompt for different tasks."""

import random

from recipe.phimm.utils.languages import get_language_name


def rand_prompt(prompts, rand=True):
    """Return a random prompt from the list."""
    return random.choice(prompts) if rand else prompts[0]


def _get_lang_asr_language(task, prefix, default=None):
    lid = task[len(prefix):].strip("_")
    return get_language_name(lid) if lid else default


def resolve_task_language(task, lang=None):
    task_language = _get_explicit_task_language(task)
    return get_language_name(task_language or lang or "Unknown")


def _get_explicit_task_language(task):
    if task.startswith("lang_asr_lex"):
        return _get_lang_asr_language(task, "lang_asr_lex")
    if task.startswith("lang_asr"):
        return _get_lang_asr_language(task, "lang_asr")
    return None


def _get_task_output_tag(task):
    if task.startswith("lang_asr_lex"):
        return "ASR_LEXICAL"
    elif task in ("asr", "rare_asr", "biasing") or task.startswith("lang_asr"):
        return "ASR"
    else:
        raise ValueError(f"Unknown task: {task}")


def _format_task_output(tag, lang, text, components=None):
    if components:
        languages = []
        segments = []
        for component in components:
            component_lang = get_language_name(component.get("language", "Unknown"))
            component_text = component.get("text", "")
            languages.append(component_lang)
            segments.append(f"<lang={component_lang}><TXT>{component_text}</TXT>")
        header_langs = " and ".join(languages)
        content = "\n".join(segments)
        return f"Audio Language: {header_langs}.\n<{tag}>{content}</{tag}>"

    lang = lang or "Unknown"
    return f"Audio Language: {lang}.\n<{tag}><lang={lang}><TXT>{text}</TXT></{tag}>"


def get_task_prompt(task="asr", rand=False):
    """Get the prompt for the specified task."""
    if task == "asr":
        prompt = rand_prompt(ASR_PROMPTS, rand=rand)
    elif task == "rare_asr":
        prompt = rand_prompt(ASR_PROMPTS, rand=rand)
        prompt = f"{prompt} Pay extra attention to rare words."
    elif task == "biasing":
        prompt = rand_prompt(BIASING_PROMPTS, rand=rand)
    elif task.startswith("lang_asr_lex"):
        prompt = rand_prompt(LANG_ASR_LEX_PROMPTS, rand=rand)
    elif task.startswith("lang_asr"):
        prompt = rand_prompt(LANG_ASR_PROMPTS, rand=rand)
    else:
        raise ValueError(f"Unknown task: {task}")
    return prompt


def get_task_prefix(task, lang, prob=1.0):
    """Get the assistant prefix for the specified task.

    Only tasks with an explicit language (``lang_asr*``) or an explicit ``lang``
    argument emit a non-empty prefix. A null/empty ``lang`` resolves to
    "Unknown" so the model detects the language itself.
    """
    if random.random() >= prob:
        return ""
    if task.startswith("lang_asr"):
        return f"Audio Language: {get_language_name(lang)}\n"
    
    raise ValueError(f"Unknown task: {task}")


def get_task_output(task="asr", lang="English", text="", components=None):
    """Get the expected output format for the specified task."""
    return _format_task_output(_get_task_output_tag(task), get_language_name(lang), text, components)
    


ASR_PROMPTS = [
    # from wavllm
    "Transcribe the audio clip into text.",
    "Listen to the given recording and provide a written transcription.",
    "Convert the spoken words in this audio file into a textual format.",
    "Please provide a text version of the speech in the provided audio.",
    "Create a written transcript of the conversation in the audio file.",
    "Transcribe the audio recording into text, capturing every spoken word.",
    "Listen carefully to the provided audio and produce a written transcript of the content.",
    "Analyze the speech in the given audio clip and provide a complete textual transcription.",
    "Transcribe the spoken language in this audio file into a written document.",
    "Based on the attached audio, generate a comprehensive text transcription of the spoken content.",
    # from GPT-4
    "Capture every word from the audio verbatim.",
    "Listen and transcribe the complete text from the audio.",
    "Record in writing what is spoken in the audio.",
    "Transcribe the spoken words from the audio with exact wording and punctuation.",
    "Recognize the transcription from audio.",
    "Identify the transcription from audio.",
    "Detect the transcription from audio.",
    "Extract the transcription from audio.",
    "Derive the transcription from audio.",
    "Interpret the transcription from audio.",
    "Convert the audio to text.",
    "Transcribe the audio.",
    "Convert the speech to text.",
    "Recognize speech from audio.",
    "Decode the transcription from audio.",
    "Capture the transcription from audio.",
    "Obtain the transcription from audio.",
    "Understand the transcription from audio.",
    "Register the transcription from audio.",
    "Pick out the transcription from audio.",
    "Extract text from audio.",
    "Decode speech from audio.",
    "Transcribe spoken words from audio.",
    "Analyze the transcription from audio.",
    "Comprehend the transcription from audio.",
    "Render the audio into text.",
    "Transcribe voice from audio.",
    "Ascertain the transcription from audio.",
    "Identify speech from audio.",
    "Retrieve the transcription from audio.",
    "Hear and transcribe from audio.",
    "Transcribe the audio content.",
    "Change the audio into text.",
    "Discern the transcription from audio.",
    "Recognize words from audio.",
    "Grasp the transcription from audio.",
    "Distill the transcription from audio.",
    "Scribe the audio.",
    "Detect speech from audio.",
    "Write the transcription from audio.",
    "Uncover the transcription from audio.",
    "Register speech from audio.",
    "Determine the transcription from audio.",
    "Perceive the transcription from audio.",
    "Decode audio into text.",
    "Enscribe the audio content.",
    "Make out the transcription from audio.",
    "Capture spoken words from audio.",
    "Interpret speech from audio.",
    "Record the transcription from audio.",
    "Write out the transcription from audio.",
    "Discern speech from audio.",
    "Render speech into text.",
    "Detect and transcribe from audio.",
    "Recognize the audio transcript.",
    "Convert spoken audio to text.",
    "Extract the speech transcript from audio.",
    "Transcribe the spoken audio.",
    "Identify and transcribe from audio.",
    "Decode the spoken audio.",
    "Convert the audio recording to text.",
    "Recognize text from audio.",
    "Transcribe the audio signal.",
    "Convert audio input to text.",
    "Interpret the audio data.",
    "Understand the spoken audio.",
    "Derive text from audio.",
    "Recognize and transcribe audio content.",
    "Decode the audio into transcription.",
    "Record the audio in text form.",
    "Transcribe the audible words.",
    "Convert the sound to text.",
    "Transcribe the audio file.",
    "Recognize the speech data.",
    "Extract the spoken content.",
    "Render the audio recording into text.",
    "Decode the spoken content.",
    "Transcribe the sound.",
    "Extract words from audio.",
    "Recognize speech signals from audio.",
    "Render the speech into transcription.",
    "Comprehend speech from audio.",
    "Write down the audio words.",
    "Extract and transcribe from audio.",
    "Transcribe and recognize the audio.",
    "Convert recorded speech to text.",
    "Interpret the speech in audio.",
    "Register the words from audio.",
    "Identify the spoken words.",
    "Understand and transcribe the audio.",
    "Capture and convert audio to text.",
    "Decode and transcribe the audio.",
    "Extract and write out the audio.",
    "Recognize and convert the audio.",
    "Analyze and transcribe the audio.",
    "Render and convert the audio.",
    "Identify and extract audio transcription.",
    "Write and decode the audio.",
    "Capture and decode spoken audio.",
    "Identify and render the audio to text",
]


BIASING_PROMPTS = [
    "Transcribe the audio clip into text. Pay extra attention to the following phrases/words.",
    "Convert the audio recording to text, focusing especially on the following phrases/words.",
    "Please transcribe the audio file and highlight the specified phrases/words.",
    "Listen to the audio and produce a transcript, ensuring you pay close attention to the following terms.",
    "Transcribe the given audio, making sure to accurately capture these particular phrases/words.",
    "Generate a text transcription of the audio, emphasizing the phrases/words listed below.",
    "Create a written transcript of the audio clip, with special attention to these phrases/words.",
    "Please convert the audio to text, carefully noting the following key phrases/words.",
    "Provide a transcript of the audio, ensuring you accurately transcribe the specified terms.",
    "Transcribe the audio and make sure the following phrases/words are captured precisely.",
    "Please turn the audio into text, focusing on accurately recording these phrases/words.",
    "Transform the audio into a written transcript, ensuring these phrases/words are transcribed accurately.",
    "Please listen to the audio and convert it into text, being especially mindful of the following phrases/words.",
    "Produce a text version of the audio, with particular attention paid to these terms.",
    "Turn the audio recording into text, making sure these specific phrases/words are included verbatim.",
    "Write out a transcript of the audio clip, ensuring accuracy for the following phrases/words.",
    "Please generate a textual transcript from the audio, with an emphasis on capturing these phrases/words.",
    "Convert the provided audio into a text document, highlighting these phrases/words when they occur.",
    "Transcribe the audio content, focusing on the precise transcription of the listed phrases/words.",
    "Create a written version of the audio, ensuring the following phrases/words are clearly marked.",
    "Please provide a transcript of the audio file, making sure to pay close attention to the accuracy of these terms.",
    "Please transcribe the audio, making sure to clearly capture the following key phrases/words.",
    "Render the audio as text, ensuring that these particular phrases/words are transcribed precisely.",
    "Write out the spoken content from the audio clip, giving special attention to these phrases/words.",
    "Convert the audio track into text, taking care to accurately include the listed phrases/words.",
    "Please produce a transcript of the audio, focusing on the correct capture of the following phrases/words.",
    "Transcribe what you hear from the audio, prioritizing these phrases/words for accuracy.",
    "Generate a transcript from the audio, making sure the following terms are not missed.",
    "Please return the audio as written text, with a focus on transcribing these specific phrases/words.",
    "Listen and transcribe the audio file, ensuring the accuracy of these highlighted phrases/words.",
    "Make a text transcription of the audio, being mindful to include the following phrases/words exactly as spoken.",
    "Please transcribe the provided audio, ensuring these phrases/words are transcribed exactly as spoken.",
    "Listen to the audio and create a detailed transcript, paying particular attention to the following phrases/words.",
    "Convert the audio content into written form, with special emphasis on these key phrases/words.",
    "Generate a full text transcript of the audio, making sure to carefully capture the specified phrases/words.",
    "Please write down everything spoken in the audio, taking extra care with these phrases/words.",
    "Produce an accurate transcription of the audio file, focusing on these important phrases/words.",
    "Please render the audio as a text transcript, with close attention to the following words/phrases.",
    "Transcribe the audio material, ensuring the terms below are included accurately.",
    "Provide a written account of the audio, making sure to highlight these phrases/words.",
    "Listen to and transcribe the audio, ensuring precision for the following listed phrases/words.",
]


LANG_ASR_PROMPTS = [
    "Detect the language and transcribe the audio clip into text.",
]


LANG_ASR_LEX_PROMPTS = [
    "Detect the language and transcribe the audio clip into text. Output must be in lexical format.",
]
