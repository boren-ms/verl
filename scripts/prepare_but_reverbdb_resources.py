#!/usr/bin/env python3
"""Select measured BUT ReverbDB RIRs and 100 licensed ambient-noise files.

Extract the official 2019 RIR-Only archive first. This writes inventories with
original member names and hashes, balanced noise slots, and source metadata.
Radio/music/secondary-source speaker setups are excluded from the noise pool.
"""

import argparse
import collections
import hashlib
import io
import json
import pathlib
import random

import numpy as np
import soundfile as sf


def main(args):
    root = pathlib.Path(args.work_dir)
    extracted = pathlib.Path(args.extracted_dir)
    rir_dir = root / "rirs_final"
    noise_dir = root / "noises_final"
    meta_dir = root / "metadata"
    for d in [rir_dir, noise_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)
    base = args.output_base.rstrip("/")
    source = "https://merlin.fit.vutbr.cz/ReverbDB/BUT_ReverbDB_rel_19_06_RIR-Only.tgz"
    rirs = []
    noises = collections.defaultdict(list)
    exclusions = set()

    def store(t, m, kind):
        data = t.extractfile(m).read()
        a, sr = sf.read(io.BytesIO(data), dtype="float32")
        if a.ndim != 1 or sr != 16000 or not np.isfinite(a).all() or np.mean(a * a) <= 0:
            raise ValueError(m.name)
        name = hashlib.sha256(m.name.encode()).hexdigest()[:16] + ".wav"
        local = (rir_dir if kind == "rirs" else noise_dir) / name
        local.write_bytes(data)
        return dict(
            audio_path=f"{base}/{kind}/{name}",
            filename=name,
            sha256=hashlib.sha256(data).hexdigest(),
            archive_member=m.name,
            source_archive=source,
            room=pathlib.PurePosixPath(m.name).parts[0],
            license="CC-BY-4.0",
            sample_rate=sr,
            duration=len(a) / sr,
        )

    class Entry:
        def __init__(self, p):
            self.name = str(p.relative_to(extracted))

    class FS:
        def extractfile(self, m):
            return (extracted / m.name).open("rb")

    t = FS()
    for p in sorted((extracted).rglob("*")):
        if not p.is_file():
            continue
        m = Entry(p)
        norm = m.name
        parts = pathlib.PurePosixPath(norm).parts
        if m.name.endswith(".txt"):
            data = p.read_bytes()
            text = data.decode("utf-8", errors="replace")
            target = meta_dir / norm
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            if "spk_meta" in m.name and any(w in text.lower() for w in ["radio", "music", "$envspk2"]):
                exclusions.add(str(pathlib.PurePosixPath(norm).parent))
        elif m.name.lower().endswith(".wav"):
            if "RIR" in parts:
                rirs.append(store(t, m, "rirs"))
            elif "silence" in parts:
                noises[parts[0]].append(norm)
    for room in list(noises):
        noises[room] = [n for n in noises[room] if not any(n.startswith(p + "/") for p in exclusions)]
        if not noises[room]:
            del noises[room]
    print(
        "rirs",
        len(rirs),
        "rooms",
        collections.Counter(r["room"] for r in rirs),
        "eligible noise files",
        {k: len(v) for k, v in noises.items()},
        flush=True,
    )
    selected = []
    rooms = sorted(noises)
    rng = random.Random(20261027)
    for room in rooms:
        pool = sorted(noises[room])
        rng.shuffle(pool)
        unique = {}
        for n in pool:
            unique.setdefault(n.rsplit("/silence/", 1)[0], n)
        selected.append(list(unique.values()))
    names = []
    for i in range(100):
        names.append(selected[i % len(rooms)][i // len(rooms)])
    found = {}
    for n in names:
        found[n] = store(t, Entry(extracted / n), "noises")
    chosen = [found[n] for n in names]
    for name, rows in [("rirs.jsonl", rirs), ("noises.jsonl", chosen)]:
        (root / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "selection.json").write_text(
        json.dumps(
            {
                "rir_count": len(rirs),
                "noise_count": len(chosen),
                "rir_rooms": dict(collections.Counter(r["room"] for r in rirs)),
                "noise_rooms": dict(collections.Counter(r["room"] for r in chosen)),
                "excluded_noise_setups": sorted(exclusions),
                "noise_seed": 20261027,
            },
            indent=2,
        )
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument(
        "--output-base",
        default="az://orngwus2cresco/data/boren/data/verl/gen_cached/ls_raw_rp_edge_nodigits_aug_audio_rir_but_v1/resources",
    )
    main(parser.parse_args())
