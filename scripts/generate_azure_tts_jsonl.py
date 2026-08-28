#!/usr/bin/env python3
"""Generate WAV files from the text and audio_path fields of a JSONL file."""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_ENDPOINT = "https://boren-8685-resource.cognitiveservices.azure.com/"
DEFAULT_VOICE = "en-US-Ava:DragonHDLatestNeural"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path, help="Local JSONL manifest")
    parser.add_argument(
        "--resource-id",
        default=os.environ.get("AZURE_SPEECH_RESOURCE_ID"),
        help="Speech resource ARM ID (or set AZURE_SPEECH_RESOURCE_ID)",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("AZURE_SPEECH_ENDPOINT", DEFAULT_ENDPOINT),
        help="Speech resource endpoint",
    )
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="Speech synthesis voice")
    parser.add_argument(
        "--random-voice",
        action="store_true",
        help="Randomly select an available voice for each row",
    )
    parser.add_argument(
        "--voice-locale",
        default="en-US",
        help="Locale used to fetch the random voice pool (default: en-US)",
    )
    parser.add_argument(
        "--voice-pool",
        nargs="+",
        help="Optional voice short names to use in random mode",
    )
    parser.add_argument("--seed", type=int, help="Seed for reproducible voice selection")
    parser.add_argument("--text-key", default="text", help="JSON field containing synthesis text")
    parser.add_argument("--audio-key", default="audio_path", help="JSON field containing output path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write WAV files into this directory instead of their manifest paths",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Maximum rows to synthesize (default: all rows)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing WAV files")
    parser.add_argument("--dry-run", action="store_true", help="Print planned outputs without Azure calls")
    return parser.parse_args()


def load_rows(input_jsonl: Path, max_rows: int | None) -> list[tuple[int, dict]]:
    rows = []
    with input_jsonl.open(encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            if not line.strip():
                continue
            rows.append((line_number, json.loads(line)))
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def resolve_audio_path(input_jsonl: Path, audio_path: str, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.expanduser().resolve() / Path(audio_path).name
    output_path = Path(audio_path).expanduser()
    if not output_path.is_absolute():
        output_path = input_jsonl.parent / output_path
    return output_path


def get_voice_pool(speechsdk, speech_config, locale: str, requested: list[str] | None) -> list[str]:
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    result = synthesizer.get_voices_async(locale).get()
    if result.reason != speechsdk.ResultReason.VoicesListRetrieved:
        raise RuntimeError(f"Failed to retrieve voices: {result.error_details}")

    available = {voice.short_name for voice in result.voices if voice.short_name}
    if requested:
        unavailable = sorted(set(requested) - available)
        if unavailable:
            raise ValueError(f"Unavailable voices: {', '.join(unavailable)}")
        return list(dict.fromkeys(requested))
    if not available:
        raise RuntimeError(f"No voices available for locale {locale!r}")
    return sorted(available)


def main() -> int:
    args = parse_args()
    input_jsonl = args.input_jsonl.expanduser().resolve()
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("--max-rows must be at least 1")
    if args.voice_pool and not args.random_voice:
        raise ValueError("--voice-pool requires --random-voice")

    rows = load_rows(input_jsonl, args.max_rows)
    jobs = []
    for line_number, row in rows:
        text = row.get(args.text_key)
        audio_path = row.get(args.audio_key)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Line {line_number}: missing non-empty {args.text_key!r}")
        if not isinstance(audio_path, str) or not audio_path.strip():
            raise ValueError(f"Line {line_number}: missing non-empty {args.audio_key!r}")
        jobs.append(
            (line_number, text, resolve_audio_path(input_jsonl, audio_path, args.output_dir))
        )

    if args.dry_run:
        randomizer = random.Random(args.seed)
        for line_number, text, output_path in jobs:
            if args.random_voice and args.voice_pool:
                voice = randomizer.choice(args.voice_pool)
            elif args.random_voice:
                voice = f"<random available {args.voice_locale} voice>"
            else:
                voice = args.voice
            print(f"line {line_number}: voice={voice!r}, {text!r} -> {output_path}")
        return 0

    try:
        import azure.cognitiveservices.speech as speechsdk
        from azure.identity import DefaultAzureCredential
    except ImportError as error:
        raise RuntimeError(
            "Install dependencies with: pip install azure-cognitiveservices-speech azure-identity"
        ) from error

    parsed_endpoint = urlparse(args.endpoint)
    if not parsed_endpoint.scheme or not parsed_endpoint.netloc:
        raise ValueError(f"Invalid endpoint: {args.endpoint!r}")
    base_endpoint = f"{parsed_endpoint.scheme}://{parsed_endpoint.netloc}"

    credential = DefaultAzureCredential()
    if args.resource_id:
        token = credential.get_token("https://cognitiveservices.azure.com/.default")
        speech_config = speechsdk.SpeechConfig(endpoint=base_endpoint)
        speech_config.authorization_token = f"aad#{args.resource_id}#{token.token}"
    else:
        speech_config = speechsdk.SpeechConfig(
            token_credential=credential,
            endpoint=base_endpoint,
        )
    if args.random_voice:
        voice_pool = get_voice_pool(
            speechsdk,
            speech_config,
            args.voice_locale,
            args.voice_pool,
        )
        print(f"Random voice pool: {len(voice_pool)} available {args.voice_locale} voices")
    else:
        voice_pool = [args.voice]
    randomizer = random.Random(args.seed)

    failures = 0
    for line_number, text, output_path in jobs:
        voice = randomizer.choice(voice_pool)
        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing file: {output_path}")
            continue

        speech_config.speech_synthesis_voice_name = voice
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio_config = speechsdk.audio.AudioOutputConfig(filename=str(output_path))
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=audio_config,
        )
        result = synthesizer.speak_text_async(text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(f"Generated line {line_number} with {voice}: {output_path}")
            continue

        failures += 1
        details = result.cancellation_details
        print(
            f"Failed line {line_number}: {details.reason}; {details.error_details}",
            file=sys.stderr,
        )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())