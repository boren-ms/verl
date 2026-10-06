#!/usr/bin/env python3

import argparse
import io
import json
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import soundfile as sf


ORANGE_SPEECH_PREFIX = "az://orngwus2cresco/data/speech/"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export selected ChunkFile audio to OGG and build a local HTML viewer."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--compression-level", type=float, default=0.8)
    return parser.parse_args()


def load_records(manifest):
    records = []
    jobs = OrderedDict()
    with manifest.open(encoding="utf-8") as stream:
        for record_index, line in enumerate(stream):
            if not line.strip():
                continue
            record = json.loads(line)
            chunk_path, chunk_count, chunk_index = record["audio_path"].rsplit(":", 2)
            chunk_count = int(chunk_count)
            chunk_index = int(chunk_index)
            if not 0 <= chunk_index < chunk_count:
                raise ValueError(
                    f"Invalid chunk index {chunk_index} for count {chunk_count}: "
                    f"{record['audio_path']}"
                )

            export_index = int(record.get("_export_index", record_index))
            local_name = f"{export_index:05d}_{Path(record['id']).stem}.ogg"
            record["local_audio"] = f"audio/{local_name}"
            record["_record_index"] = record_index
            records.append(record)

            job = jobs.setdefault(
                chunk_path,
                {"count": chunk_count, "records_by_index": {}},
            )
            if job["count"] != chunk_count:
                raise ValueError(f"Inconsistent counts for chunk {chunk_path}")
            if chunk_index in job["records_by_index"]:
                raise ValueError(f"Duplicate chunk sample: {record['audio_path']}")
            job["records_by_index"][chunk_index] = record
    return records, jobs


def write_jsonl(records, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            output = {key: value for key, value in record.items() if not key.startswith("_")}
            stream.write(json.dumps(output, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_html(records, path):
    browser_records = [
        {
            "id": record["id"],
            "audio": record["local_audio"],
            "transcription": record["transcription"],
            "names": record.get("names", []),
            "keywords": record.get("keywords", []),
        }
        for record in records
    ]
    payload = json.dumps(browser_records, ensure_ascii=False).replace("</", "<\\/")
    document = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>name_10k audio review</title>
  <style>
    :root { color-scheme: light dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #f5f6f8; color: #202124; }
    header { position: sticky; top: 0; z-index: 2; padding: 16px 24px;
      background: #fff; border-bottom: 1px solid #d8dce2; }
    h1 { margin: 0 0 12px; font-size: 22px; }
    .toolbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
    input, select, button { font: inherit; padding: 8px 10px; border: 1px solid #aeb4bd;
      border-radius: 6px; background: #fff; color: inherit; }
    input { flex: 1; min-width: 240px; }
    button:disabled { opacity: .45; }
    #status { color: #5f6368; min-width: 190px; }
    main { max-width: 1200px; margin: 18px auto; padding: 0 18px 32px; }
    article { display: grid; grid-template-columns: 280px 1fr; gap: 18px;
      margin-bottom: 12px; padding: 16px; background: #fff; border: 1px solid #dfe2e7;
      border-radius: 8px; box-shadow: 0 1px 2px rgb(0 0 0 / 5%); }
    audio { width: 100%; margin-top: 8px; }
    .id { overflow-wrap: anywhere; color: #5f6368; font-size: 13px; }
    .text { line-height: 1.5; white-space: pre-wrap; }
    mark.keyword { padding: 1px 2px; border-radius: 3px; background: #fff176;
      color: #202124; font-weight: 650; }
    .tags { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 10px; }
    .tag { padding: 2px 7px; border-radius: 999px; background: #e8f0fe;
      color: #174ea6; font-size: 12px; }
    nav { display: flex; justify-content: center; gap: 10px; align-items: center;
      margin: 20px 0; }
    @media (max-width: 720px) { article { grid-template-columns: 1fr; } }
    @media (prefers-color-scheme: dark) {
      body { background: #17191c; color: #e8eaed; }
      header, article { background: #23262b; border-color: #40454d; }
      input, select, button { background: #17191c; border-color: #5f6368; }
      #status, .id { color: #bdc1c6; }
      mark.keyword { background: #f9ab00; color: #202124; }
      .tag { background: #174ea6; color: #e8f0fe; }
    }
  </style>
</head>
<body>
  <header>
    <h1>name_10k audio review</h1>
    <div class="toolbar">
      <input id="search" type="search" placeholder="Search transcription, name, keyword, or ID">
      <label>Per page
        <select id="pageSize">
          <option>25</option><option selected>50</option><option>100</option><option>200</option>
        </select>
      </label>
      <span id="status"></span>
    </div>
  </header>
  <main>
    <div id="records"></div>
    <nav>
      <button id="previous" type="button">Previous</button>
      <span id="page"></span>
      <button id="next" type="button">Next</button>
    </nav>
  </main>
  <script>
    const allRecords = __DATA__;
    const search = document.querySelector("#search");
    const pageSize = document.querySelector("#pageSize");
    const container = document.querySelector("#records");
    const status = document.querySelector("#status");
    const pageLabel = document.querySelector("#page");
    const previous = document.querySelector("#previous");
    const next = document.querySelector("#next");
    let page = 0;
    let filtered = allRecords;

    function addText(parent, className, value) {
      const node = document.createElement("div");
      node.className = className;
      node.textContent = value;
      parent.appendChild(node);
    }

    function addHighlightedText(parent, value, keywords) {
      const node = document.createElement("div");
      node.className = "text";
      const uniqueKeywords = [...new Set(keywords.filter(Boolean))]
        .sort((left, right) => right.length - left.length);
      if (uniqueKeywords.length === 0) {
        node.textContent = value;
        parent.appendChild(node);
        return;
      }

      const escapedKeywords = uniqueKeywords.map(keyword =>
        keyword.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
      const matcher = new RegExp(`(${escapedKeywords.join("|")})`, "giu");
      let cursor = 0;
      for (const match of value.matchAll(matcher)) {
        node.append(document.createTextNode(value.slice(cursor, match.index)));
        const highlight = document.createElement("mark");
        highlight.className = "keyword";
        highlight.textContent = match[0];
        node.append(highlight);
        cursor = match.index + match[0].length;
      }
      node.append(document.createTextNode(value.slice(cursor)));
      parent.appendChild(node);
    }

    function render() {
      const size = Number(pageSize.value);
      const pageCount = Math.max(1, Math.ceil(filtered.length / size));
      page = Math.min(page, pageCount - 1);
      const start = page * size;
      const visible = filtered.slice(start, start + size);
      container.replaceChildren();
      for (const record of visible) {
        const article = document.createElement("article");
        const media = document.createElement("div");
        addText(media, "id", record.id);
        const audio = document.createElement("audio");
        audio.controls = true;
        audio.preload = "none";
        audio.src = record.audio;
        media.appendChild(audio);
        article.appendChild(media);

        const content = document.createElement("div");
        addHighlightedText(content, record.transcription, record.keywords);
        const tags = document.createElement("div");
        tags.className = "tags";
        for (const value of [...new Set([...record.names, ...record.keywords])]) {
          addText(tags, "tag", value);
        }
        content.appendChild(tags);
        article.appendChild(content);
        container.appendChild(article);
      }
      status.textContent = `${filtered.length.toLocaleString()} of ${allRecords.length.toLocaleString()} records`;
      pageLabel.textContent = `Page ${page + 1} of ${pageCount}`;
      previous.disabled = page === 0;
      next.disabled = page + 1 >= pageCount;
      window.scrollTo({top: 0, behavior: "instant"});
    }

    function applySearch() {
      const query = search.value.trim().toLocaleLowerCase();
      filtered = query
        ? allRecords.filter(record =>
            [record.id, record.transcription, ...record.names, ...record.keywords]
              .some(value => value.toLocaleLowerCase().includes(query)))
        : allRecords;
      page = 0;
      render();
    }

    search.addEventListener("input", applySearch);
    pageSize.addEventListener("change", () => { page = 0; render(); });
    previous.addEventListener("click", () => { page -= 1; render(); });
    next.addEventListener("click", () => { page += 1; render(); });
    render();
  </script>
</body>
</html>
""".replace("__DATA__", payload)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(path)


def download_batch(batch, cache_dir, concurrency):
    remote_paths = [ORANGE_SPEECH_PREFIX + chunk_path for chunk_path, _ in batch]
    command = [
        "bbb",
        "cp",
        "--concurrency",
        str(concurrency),
        *remote_paths,
        str(cache_dir) + "/",
    ]
    for attempt in range(1, 4):
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            return
        print(f"Download attempt {attempt}/3 failed with code {result.returncode}", flush=True)
        if attempt < 3:
            time.sleep(10 * attempt)
    raise RuntimeError(f"Failed to download batch after 3 attempts: {remote_paths}")


def extract_chunk(local_chunk, job, audio_dir, compression_level):
    targets = job["records_by_index"]
    written = 0
    with local_chunk.open("rb") as stream:
        target_type = stream.read(5).decode()
        if target_type.lower() != "audio":
            raise ValueError(f"Unexpected chunk type {target_type!r} in {local_chunk}")
        stream.read(4)
        for index in range(job["count"]):
            stored_index = int.from_bytes(stream.read(4), byteorder="little")
            if stored_index != index:
                raise ValueError(
                    f"Corrupted index in {local_chunk}: expected {index}, got {stored_index}"
                )
            data_size = int.from_bytes(stream.read(4), byteorder="little")
            record = targets.get(index)
            if record is None:
                stream.seek(data_size, io.SEEK_CUR)
                continue

            destination = audio_dir / Path(record["local_audio"]).name
            if destination.is_file() and destination.stat().st_size > 0:
                stream.seek(data_size, io.SEEK_CUR)
                continue

            audio_bytes = stream.read(data_size)
            if len(audio_bytes) != data_size:
                raise EOFError(
                    f"Expected {data_size} audio bytes for index {index} in {local_chunk}, "
                    f"got {len(audio_bytes)}"
                )
            audio, sample_rate = sf.read(io.BytesIO(audio_bytes), always_2d=False)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            sf.write(
                temporary,
                audio,
                sample_rate,
                format="OGG",
                subtype="VORBIS",
                compression_level=compression_level,
            )
            temporary.replace(destination)
            written += 1
    return written


def pending_jobs(jobs, output_dir):
    result = []
    for chunk_path, job in jobs.items():
        if any(
            not (output_dir / record["local_audio"]).is_file()
            for record in job["records_by_index"].values()
        ):
            result.append((chunk_path, job))
    return result


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if not 0 <= args.compression_level <= 1:
        raise ValueError("--compression-level must be between 0 and 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output_dir / "audio"
    cache_dir = args.output_dir / ".chunk_cache"
    audio_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    records, jobs = load_records(args.manifest)
    write_jsonl(records, args.output_dir / "data.jsonl")
    write_html(records, args.output_dir / "index.html")
    remaining = pending_jobs(jobs, args.output_dir)
    completed = len(jobs) - len(remaining)
    print(
        f"Loaded {len(records)} records across {len(jobs)} chunks; "
        f"{completed} chunks already complete",
        flush=True,
    )

    for offset in range(0, len(remaining), args.batch_size):
        batch = remaining[offset : offset + args.batch_size]
        missing = [
            (chunk_path, job)
            for chunk_path, job in batch
            if not (cache_dir / Path(chunk_path).name).is_file()
        ]
        if missing:
            download_batch(missing, cache_dir, args.concurrency)

        batch_written = 0
        for chunk_path, job in batch:
            local_chunk = cache_dir / Path(chunk_path).name
            if not local_chunk.is_file():
                raise FileNotFoundError(f"Downloaded chunk is missing: {local_chunk}")
            batch_written += extract_chunk(
                local_chunk,
                job,
                audio_dir,
                args.compression_level,
            )
            local_chunk.unlink()
        completed += len(batch)
        free_gib = shutil.disk_usage(args.output_dir).free / (1024**3)
        print(
            f"Completed chunks {completed}/{len(jobs)}; wrote {batch_written} clips; "
            f"{free_gib:.1f} GiB free",
            flush=True,
        )

    final_pending = pending_jobs(jobs, args.output_dir)
    if final_pending:
        raise RuntimeError(f"{len(final_pending)} chunks remain incomplete")
    write_jsonl(records, args.output_dir / "data.jsonl")
    write_html(records, args.output_dir / "index.html")
    print(
        f"Export complete: {len(records)} clips, "
        f"{args.output_dir / 'index.html'}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
