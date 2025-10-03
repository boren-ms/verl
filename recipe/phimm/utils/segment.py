# %%
import re
from pathlib import Path
import pandas as pd
import soundfile as sf
from tqdm import tqdm


def extract_segments(text, shift=True):
    """
    Extracts segments from text with [start end] markers.
    Returns a list of (start, end, segment_text) tuples.
    """
    pattern = re.compile(r"\[(\d+\.?\d*) (\d+\.?\d*)\]")
    matches = list(pattern.finditer(text))
    segments = []
    utt_s = float(matches[0].group(1)) if shift and matches else 0
    for i, match in enumerate(matches):
        start = float(match.group(1))
        end = float(match.group(2))
        text_start = match.end()
        text_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segment_text = text[text_start:text_end].strip()
        segments.append((start - utt_s, end - utt_s, segment_text))
    return segments


def segment_audio(text, audio_path, output_dir):
    """
    Segments the audio file at wav_path according to the timing information in text.
    Saves each segment as a separate wav file and writes segment info to a text file.
    """
    segments = extract_segments(text)
    data, fs = sf.read(audio_path)
    output_dir = Path(output_dir)
    audio_path = Path(audio_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_stem = audio_path.stem
    audio_ext = audio_path.suffix
    text_file = output_dir / f"{audio_stem}.txt"
    pairs = []
    # print(f"Segmenting {audio_stem}...")
    with open(text_file, "a", encoding="utf8") as f:
        for i, (start, end, seg_text) in enumerate(segments):
            seg_name = f"{audio_stem}_{i}{audio_ext}"
            # print(f"{seg_name}: {start:.2f} - {end:.2f} [{end-start:.2f}] | {seg_text}")
            print(f"{seg_name}\t{seg_text}", file=f)
            seg_path = output_dir / seg_name
            sf.write(seg_path, data[int(start * fs) : int(end * fs)], fs)
            pairs.append({"WavPath": str(seg_path), "Transcription": seg_text})
        # print(f"Saved {len(segments)} segments to {output_dir}")
        # print(f"Transcriptions saved to {text_file}")
    return pairs


def segment_dataset(src_jsonl, dst_jsonl=None, dst_dir=None):
    """
    Segments all audio files in a JSONL dataset according to their DisplayTranscription.
    Writes segmented audio and a new JSONL with segment info.
    """
    dst_dir = Path(dst_dir) if dst_dir else Path(src_jsonl).parent / "segments"
    dst_jsonl = Path(dst_jsonl) if dst_jsonl else Path(src_jsonl).with_name("segments.jsonl")
    df = pd.read_json(src_jsonl, lines=True)
    print(f"Processing : {src_jsonl}")
    print(f"Output directory: {dst_dir}")

    tqdm.pandas(desc="Segmenting audio")
    df["segments"] = df.progress_apply(lambda row: segment_audio(row["DisplayTranscription"], row["WavPath"], dst_dir), axis=1)

    df = df.explode("segments", ignore_index=True)
    sdf = pd.json_normalize(df["segments"])
    df.update(sdf)
    df.drop(columns=["segments", "DisplayTranscription"], inplace=True)
    df["UUID"] = df["WavPath"].apply(lambda x: str(Path(x).stem))

    # for idx, row in df.head().iterrows():
    #     wav_path = Path(row["WavPath"])
    #     print(f"[{idx}] {wav_path.name}: {row['Transcription']}")

    df.to_json(dst_jsonl, orient="records", lines=True)
    print(f"Segments: {len(df)} saved to {dst_jsonl}")
    return dst_jsonl


def segment_datasets(jsonl_files, forced=False):
    print(f"Processing {len(jsonl_files)} JSONL files.")
    for src_jsonl in jsonl_files:
        src_jsonl = Path(src_jsonl)
        if not src_jsonl.exists():
            print(f"Source JSONL file {src_jsonl} does not exist.")
            continue
        dst_jsonl = src_jsonl.with_name("segments.jsonl")
        if dst_jsonl.exists() and not forced:
            print(f"Skipping {dst_jsonl} (already exists).")
            continue
        segment_dataset(src_jsonl, dst_jsonl)
    print("All Done")


def segment_datasets_in_dir(folder, forced=False):
    """
    Processes all test.jsonl files under the given folder (recursively).
    """
    jsonl_files = list(Path(folder).rglob("test.jsonl"))
    print(f"Found {len(jsonl_files)} test.jsonl files in {folder}")
    segment_datasets(jsonl_files, forced=forced)


# %% Example usage:
if __name__ == "__main__":
    # To process a single file:
    # jsonl_path = Path.home() / "data/Evaluation/InhouseASR/EWER/en-US-entity-v3/Conversation_DomainSet_DTEST_Gaming_Entity_FY24Q1_en-US_DTEST/test.jsonl"
    # segment_dataset(jsonl_path)

    # To process all test.jsonl files under a folder:
    # folder = Path.home() / "data/Evaluation/InhouseASR/EWER/en-US-entity-v3"
    # segment_datasets_in_dir(folder)

    jsonl_files = [
        Path.home() / "data/Evaluation/InhouseASR/EWER/en-US-entity-v3/Conversation_DomainSet_DTEST_Gaming_Entity_FY24Q1_en-US_DTEST/test.jsonl",
        Path.home() / "data/Evaluation/InhouseASR/EWER/en-US-entity-v3/Conversation_DomainSet_DTEST_Insurance_Entity_FY24Q1_en-US_DTEST/test.jsonl",
        Path.home() / "data/Evaluation/InhouseASR/EWER/en-US-entity-v3/Conversation_DomainSet_DTEST_K12HigherEdu_Entity_FY24Q4_en-US_DTEST_OfflineDataCollection/test.jsonl",
        Path.home() / "data/Evaluation/InhouseASR/EWER/en-US-entity-v3/Conversation_DomainSet_DTEST_Retail_Entity_FY24Q4_en-US_DTEST_OfflineDataCollection/test.jsonl",
        Path.home() / "data/Evaluation/InhouseASR/EWER/en-US-entity-v3/Conversation_DomainSet_DTEST_ScienceTech_Entity_FY24Q1_en-US_DTEST/test.jsonl",
    ]

    segment_datasets(jsonl_files, forced=True)

# %%
