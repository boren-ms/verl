#!/usr/bin/env python3
"""Convert phi-fastllm ASR eval `.txt` results directly to the final layout
and enrich them with source-data metadata for downstream ``jsonl_dataset``.

``convert`` subcommand:

    <root>/eval_output/<dataset>/<seed>/generate_<name>.txt    # JSON array
    ->  <root>/jsonl_results/<dataset>/<name>.jsonl
        <root>/jsonl_results/<dataset>/<name>_1.jsonl, _2.jsonl  # extra seeds

``enrich`` subcommand: joins every ``<root>/jsonl_results/<ds>/*.jsonl`` record
against the source corpora under ``--src-root`` (default
``az://orngwus2cresco/data/speech/users/ruchaofan/Evaluation/InhouseASR_2605``)
on ``audio_id == UUID`` and rewrites each file with dataset-loader friendly
fields: ``id``, ``audio_path``, ``text``, ``language``, ``corpus``, ``hyp``,
``label``, ``keywords`` (entities extracted from ``Transcription``) plus the
raw source fields ``UUID``, ``WavPath``, ``Transcription``,
``DisplayTranscription``, ``CorpusName``, ``locale``.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import re
import sys

import blobfile as bf

DEFAULT_ROOT = (
    "az://orngwus2cresco/data/speech/projects/phi-fastllm-2605/amlt-results/"
    "fast-llm-2605-qwen3-5-9b-s2-st-example/90000"
)
DEFAULT_SRC_SUBDIR = "eval_output"
DEFAULT_DST_SUBDIR = "jsonl_results"
DEFAULT_SOURCE_ROOT = "az://orngwus2cresco/data/speech/users/ruchaofan/Evaluation/InhouseASR_2605"

AUX_SUFFIXES = ("_eer", "_ewer", "_disfluencytolerant_ter")


def is_main_result(name: str) -> bool:
    if not name.endswith(".txt") or not name.startswith("generate_"):
        return False
    stem = name[:-4]
    return not any(stem.endswith(s) for s in AUX_SUFFIXES)


def base_name(fname: str) -> str:
    """``generate_foo.txt`` -> ``foo``."""
    n = fname[:-4] if fname.endswith(".txt") else fname
    if n.startswith("generate_"):
        n = n[len("generate_") :]
    return n


def discover(src_root: str, workers: int) -> list[tuple[str, str, str, str]]:
    """Return [(dataset, base, seed, src_full_path), ...]."""
    datasets = sorted(d.rstrip("/") for d in bf.listdir(src_root))

    def list_seeds(ds: str) -> list[tuple[str, str]]:
        return [(ds, s.rstrip("/")) for s in bf.listdir(f"{src_root}/{ds}")]

    def list_files(ds_seed: tuple[str, str]) -> list[tuple[str, str, str, str]]:
        ds, seed = ds_seed
        base = f"{src_root}/{ds}/{seed}"
        try:
            entries = bf.listdir(base)
        except (NotADirectoryError, FileNotFoundError):
            return []
        return [(ds, base_name(n), seed, f"{base}/{n}") for n in entries if is_main_result(n)]

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        ds_seeds: list[tuple[str, str]] = []
        for seeds in ex.map(list_seeds, datasets):
            ds_seeds.extend(seeds)
        out: list[tuple[str, str, str, str]] = []
        for batch in ex.map(list_files, ds_seeds):
            out.extend(batch)
    return out


def plan_targets(
    items: list[tuple[str, str, str, str]], dst_root: str
) -> list[tuple[str, str]]:
    """Group by (dataset, base); assign ``_1``/``_2``/... to extra seeds."""
    groups: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for ds, base, seed, src in items:
        groups.setdefault((ds, base), []).append((seed, src))

    plan: list[tuple[str, str]] = []
    for (ds, base), members in groups.items():
        members.sort(key=lambda x: x[0])
        for i, (_seed, src) in enumerate(members):
            name = f"{base}.jsonl" if i == 0 else f"{base}_{i}.jsonl"
            plan.append((src, f"{dst_root}/{ds}/{name}"))
    return plan


def convert_one(src: str, dst: str, *, overwrite: bool) -> tuple[str, str, int, str]:
    if not overwrite and bf.exists(dst):
        return src, dst, -1, "skip-exists"
    try:
        with bf.BlobFile(src, "rb") as f:
            data = json.load(io.TextIOWrapper(f, encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return src, dst, 0, f"read-error: {e!r}"

    lines: list[str] = []
    for entry in data:
        gens = entry.get("generated_texts") or []
        aids = entry.get("audio_ids") or []
        labels = entry.get("label") or []
        n = max(len(gens), len(aids), len(labels))
        for i in range(n):
            aid = aids[i] if i < len(aids) else None
            if isinstance(aid, list) and len(aid) == 1:
                aid = aid[0]
            rec = {
                "audio_id": aid,
                "label": labels[i] if i < len(labels) else None,
                "hyp": gens[i] if i < len(gens) else None,
            }
            if i < len(gens) and isinstance(gens[i], list):
                rec["hyp"] = gens[i][0] if gens[i] else None
                rec["generated_texts"] = gens[i]
            lines.append(json.dumps(rec, ensure_ascii=False))

    payload = ("\n".join(lines) + "\n").encode("utf-8")
    with bf.BlobFile(dst, "wb") as f:
        f.write(payload)
    return src, dst, len(lines), "ok"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp_conv = sub.add_parser("convert", help=".txt JSON-array -> .jsonl in jsonl_results/")
    sp_conv.add_argument("--src-subdir", default=DEFAULT_SRC_SUBDIR)
    sp_conv.add_argument("--dst-subdir", default=DEFAULT_DST_SUBDIR)
    sp_conv.add_argument("--filter", default="", help="regex matched against `<dataset>/<base>`")

    sp_enr = sub.add_parser("enrich", help="join jsonl_results with source metadata by UUID")
    sp_enr.add_argument("--dst-subdir", default=DEFAULT_DST_SUBDIR)
    sp_enr.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    sp_enr.add_argument("--filter", default="", help="regex matched against `<dataset>/<file>`")
    sp_enr.add_argument(
        "--allow-missing", action="store_true",
        help="keep records whose audio_id has no source match (default: drop them)",
    )

    sp_fix = sub.add_parser(
        "fix_paths",
        help="rewrite audio_path/WavPath prefix and verify existence on blob",
    )
    sp_fix.add_argument("--dst-subdir", default=DEFAULT_DST_SUBDIR)
    sp_fix.add_argument("--filter", default="", help="regex matched against `<dataset>/<file>`")
    sp_fix.add_argument("--old-prefix", default="/datablob1/")
    sp_fix.add_argument("--new-prefix", default="az://orngwus2cresco/data/speech/")
    sp_fix.add_argument(
        "--no-check", action="store_true",
        help="skip blob-existence verification (faster, no listing).",
    )
    sp_fix.add_argument(
        "--drop-missing", action="store_true",
        help="drop records whose rewritten audio_path does not exist on blob",
    )

    args = ap.parse_args()

    if args.cmd == "convert":
        return run_convert(args)
    if args.cmd == "enrich":
        return run_enrich(args)
    if args.cmd == "fix_paths":
        return run_fix_paths(args)
    return 2


def run_convert(args) -> int:
    src_root = f"{args.root}/{args.src_subdir}"
    dst_root = f"{args.root}/{args.dst_subdir}"

    items = discover(src_root, args.workers)
    if args.filter:
        pat = re.compile(args.filter)
        items = [it for it in items if pat.search(f"{it[0]}/{it[1]}")]
    print(f"discovered {len(items)} source .txt files", file=sys.stderr)

    plan = plan_targets(items, dst_root)
    print(f"planned {len(plan)} writes", file=sys.stderr)

    src_marker = f"/{args.src_subdir}/"
    dst_marker = f"/{args.dst_subdir}/"

    if args.dry_run:
        for s, d in plan[:40]:
            print(f"{s.split(src_marker, 1)[-1]} -> {d.split(dst_marker, 1)[-1]}")
        if len(plan) > 40:
            print(f"... ({len(plan)} total)")
        return 0

    counts: dict[str, int] = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(convert_one, s, d, overwrite=args.overwrite) for s, d in plan]
        for fut in cf.as_completed(futs):
            _src, dst, n, status = fut.result()
            key = "ok" if status == "ok" else ("skip-exists" if status == "skip-exists" else "error")
            counts[key] = counts.get(key, 0) + 1
            print(f"[{status}] n={n} {dst.split(dst_marker, 1)[-1]}")
    print(f"done. {counts}", file=sys.stderr)
    return 0 if counts.get("error", 0) == 0 else 1


# ---------------------------------------------------------------------------
# Enrich: join jsonl_results with source corpora by UUID.
# ---------------------------------------------------------------------------

# Strip XML-like tags such as <CNOISE/>, <FILL/>, <NE>...</NE>, <ne:type>...</ne:type>.
_TAG_RE = re.compile(r"<[^>]+>")
# Extract named entity contents from <ne>...</ne> or <ne:type>...</ne:type> (case-insensitive).
_NE_RE = re.compile(r"<(ne(?::[^>\s]+)?)\s*>(.*?)</\1\s*>", re.IGNORECASE | re.DOTALL)
# Collapse internal whitespace.
_WS_RE = re.compile(r"\s+")


def _strip_tags(text: str) -> str:
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", text or "")).strip()


def _extract_entities(text: str) -> list[str]:
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for _tag, body in _NE_RE.findall(text):
        ent = _WS_RE.sub(" ", _TAG_RE.sub(" ", body)).strip()
        if ent and ent not in seen:
            seen.add(ent)
            out.append(ent)
    return out


def discover_source_corpora(source_root: str, workers: int) -> list[str]:
    """Return list of ``<top>/<corpus>`` dirs under ``source_root`` that contain ``test.jsonl``."""
    top_entries = [e for e in bf.listdir(source_root) if not e.endswith(".log") and e != "tsv_data"]

    def per_top(top: str) -> list[str]:
        base = f"{source_root}/{top}"
        out: list[str] = []
        try:
            for d in bf.listdir(base):
                try:
                    if "test.jsonl" in bf.listdir(f"{base}/{d}"):
                        out.append(f"{top}/{d}")
                except (NotADirectoryError, FileNotFoundError):
                    continue
        except (NotADirectoryError, FileNotFoundError):
            pass
        return out

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        result: list[str] = []
        for batch in ex.map(per_top, top_entries):
            result.extend(batch)
    return result


def _load_source_jsonl(path: str) -> list[dict]:
    with bf.BlobFile(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_uuid_index(source_root: str, workers: int) -> dict[str, dict]:
    corpora = discover_source_corpora(source_root, workers)
    print(f"[enrich] scanning {len(corpora)} source corpora", file=sys.stderr)
    index: dict[str, dict] = {}

    def load_one(rel: str) -> list[dict]:
        return _load_source_jsonl(f"{source_root}/{rel}/test.jsonl")

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for records in ex.map(load_one, corpora):
            for rec in records:
                uuid_val = rec.get("UUID")
                if uuid_val:
                    index[uuid_val] = rec
    print(f"[enrich] indexed {len(index)} source records", file=sys.stderr)
    return index


def list_result_jsonls(dst_root: str, workers: int) -> list[str]:
    datasets = [d for d in bf.listdir(dst_root)]

    def per_ds(ds: str) -> list[str]:
        base = f"{dst_root}/{ds}"
        try:
            return [f"{base}/{n}" for n in bf.listdir(base) if n.endswith(".jsonl")]
        except (NotADirectoryError, FileNotFoundError):
            return []

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        out: list[str] = []
        for batch in ex.map(per_ds, datasets):
            out.extend(batch)
    return out


def _enrich_record(rec: dict, src: dict | None, allow_missing: bool) -> dict | None:
    if src is None:
        if not allow_missing:
            return None
        out = dict(rec)
        out.setdefault("id", rec.get("audio_id"))
        return out

    transcription = src.get("Transcription", "") or ""
    out = {
        "id": src.get("UUID") or rec.get("audio_id"),
        "audio_path": src.get("WavPath"),
        "text": _strip_tags(transcription),
        "language": src.get("locale"),
        "corpus": src.get("CorpusName"),
        "hyp": rec.get("hyp"),
        "label": rec.get("label"),
        "keywords": _extract_entities(transcription),
        # raw source fields kept for downstream loaders (entity_dataset, etc.)
        "UUID": src.get("UUID"),
        "WavPath": src.get("WavPath"),
        "Transcription": transcription,
        "DisplayTranscription": src.get("DisplayTranscription"),
        "CorpusName": src.get("CorpusName"),
        "locale": src.get("locale"),
    }
    if "generated_texts" in rec:
        out["generated_texts"] = rec["generated_texts"]
    return out


def enrich_one(path: str, index: dict[str, dict], *, allow_missing: bool, overwrite: bool) -> tuple[str, int, int, str]:
    try:
        with bf.BlobFile(path, "r") as f:
            recs = [json.loads(line) for line in f if line.strip()]
    except Exception as e:  # noqa: BLE001
        return path, 0, 0, f"read-error: {e!r}"

    out_records: list[dict] = []
    missing = 0
    for r in recs:
        aid = r.get("audio_id") or r.get("id") or r.get("UUID")
        src = index.get(aid)
        if src is None:
            missing += 1
        merged = _enrich_record(r, src, allow_missing)
        if merged is not None:
            out_records.append(merged)

    payload = ("\n".join(json.dumps(r, ensure_ascii=False) for r in out_records) + "\n").encode("utf-8")
    with bf.BlobFile(path, "wb") as f:
        f.write(payload)
    return path, len(out_records), missing, "ok"


def run_enrich(args) -> int:
    dst_root = f"{args.root}/{args.dst_subdir}"
    files = list_result_jsonls(dst_root, args.workers)
    if args.filter:
        pat = re.compile(args.filter)
        files = [p for p in files if pat.search(p.split(f"/{args.dst_subdir}/", 1)[-1])]
    print(f"[enrich] {len(files)} result files to enrich", file=sys.stderr)
    if args.dry_run:
        for p in files[:40]:
            print(p.split(f"/{args.dst_subdir}/", 1)[-1])
        if len(files) > 40:
            print(f"... ({len(files)} total)")
        return 0

    index = build_uuid_index(args.source_root, args.workers)

    total_recs = 0
    total_missing = 0
    errors = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(enrich_one, p, index, allow_missing=args.allow_missing, overwrite=True) for p in files]
        for fut in cf.as_completed(futs):
            path, n, missing, status = fut.result()
            rel = path.split(f"/{args.dst_subdir}/", 1)[-1]
            if status != "ok":
                errors += 1
                print(f"[{status}] {rel}")
                continue
            total_recs += n
            total_missing += missing
            tag = "ok" if missing == 0 else f"ok ({missing} missing)"
            print(f"[{tag}] n={n} {rel}")
    print(
        f"[enrich] done. files={len(files)} recs={total_recs} missing={total_missing} errors={errors}",
        file=sys.stderr,
    )
    return 0 if errors == 0 else 1


# ---------------------------------------------------------------------------
# fix_paths: rewrite audio_path/WavPath prefix and verify existence on blob.
# ---------------------------------------------------------------------------


def _rewrite_path(path: str | None, old: str, new: str) -> str | None:
    if isinstance(path, str) and path.startswith(old):
        return new + path[len(old):]
    return path


class _DirExistsCache:
    """Thread-safe cache that lists directories once and answers exists(file)."""

    def __init__(self) -> None:
        import threading

        self._lock = threading.Lock()
        self._dirs: dict[str, set[str] | None] = {}

    def _load(self, d: str) -> set[str] | None:
        try:
            return set(bf.listdir(d))
        except (NotADirectoryError, FileNotFoundError):
            return None
        except Exception:  # noqa: BLE001
            return None

    def exists(self, path: str) -> bool:
        if "/" not in path:
            return False
        d, name = path.rsplit("/", 1)
        with self._lock:
            cached = self._dirs.get(d, ...)  # type: ignore[arg-type]
        if cached is ...:
            entries = self._load(d)
            with self._lock:
                self._dirs.setdefault(d, entries)
                cached = self._dirs[d]
        if cached is None:
            return False
        return name in cached


def fix_paths_one(
    path: str,
    *,
    old_prefix: str,
    new_prefix: str,
    cache: _DirExistsCache | None,
    drop_missing: bool,
) -> tuple[str, int, int, int, str]:
    """Returns (path, total, rewritten, missing, status)."""
    try:
        with bf.BlobFile(path, "r") as f:
            recs = [json.loads(line) for line in f if line.strip()]
    except Exception as e:  # noqa: BLE001
        return path, 0, 0, 0, f"read-error: {e!r}"

    rewritten = 0
    missing = 0
    out_records: list[dict] = []
    for r in recs:
        ap_old = r.get("audio_path")
        ap_new = _rewrite_path(ap_old, old_prefix, new_prefix)
        if ap_new is not None and ap_new != ap_old:
            r["audio_path"] = ap_new
            rewritten += 1
        wp_old = r.get("WavPath")
        wp_new = _rewrite_path(wp_old, old_prefix, new_prefix)
        if wp_new is not None and wp_new != wp_old:
            r["WavPath"] = wp_new

        target = r.get("audio_path")
        if cache is not None and isinstance(target, str) and target.startswith(new_prefix):
            if not cache.exists(target):
                missing += 1
                if drop_missing:
                    continue
        out_records.append(r)

    payload = ("\n".join(json.dumps(r, ensure_ascii=False) for r in out_records) + "\n").encode("utf-8")
    with bf.BlobFile(path, "wb") as f:
        f.write(payload)
    return path, len(out_records), rewritten, missing, "ok"


def run_fix_paths(args) -> int:
    dst_root = f"{args.root}/{args.dst_subdir}"
    files = list_result_jsonls(dst_root, args.workers)
    if args.filter:
        pat = re.compile(args.filter)
        files = [p for p in files if pat.search(p.split(f"/{args.dst_subdir}/", 1)[-1])]
    print(f"[fix_paths] {len(files)} result files", file=sys.stderr)
    if args.dry_run:
        for p in files[:40]:
            print(p.split(f"/{args.dst_subdir}/", 1)[-1])
        if len(files) > 40:
            print(f"... ({len(files)} total)")
        return 0

    cache = None if args.no_check else _DirExistsCache()

    total = 0
    total_rewritten = 0
    total_missing = 0
    errors = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                fix_paths_one, p,
                old_prefix=args.old_prefix,
                new_prefix=args.new_prefix,
                cache=cache,
                drop_missing=args.drop_missing,
            )
            for p in files
        ]
        for fut in cf.as_completed(futs):
            path, n, rw, miss, status = fut.result()
            rel = path.split(f"/{args.dst_subdir}/", 1)[-1]
            if status != "ok":
                errors += 1
                print(f"[{status}] {rel}")
                continue
            total += n
            total_rewritten += rw
            total_missing += miss
            tag = "ok" if miss == 0 else f"ok ({miss} missing)"
            print(f"[{tag}] n={n} rewritten={rw} {rel}")
    print(
        f"[fix_paths] done. files={len(files)} recs={total} rewritten={total_rewritten} "
        f"missing={total_missing} errors={errors}",
        file=sys.stderr,
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
