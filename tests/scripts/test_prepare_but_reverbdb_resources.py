import argparse
from collections import Counter
import hashlib
import json

import numpy as np
import soundfile as sf

from scripts.prepare_but_reverbdb_resources import main


def test_selects_only_rirs_and_excludes_radio_noise_setups(tmp_path):
    extracted = tmp_path / "extracted"
    for room in ("room_a", "room_b"):
        for index in range(61):
            setup = extracted / room / "MicID01" / f"SpkID{index:02d}_S"
            noise_dir = setup / "01" / "silence"
            noise_dir.mkdir(parents=True)
            sf.write(noise_dir / "ambient.wav", np.array([0.1, -0.1, 0.2]), 16000)
            if index == 60:
                (setup / "spk_meta.txt").write_text("radio playing in this setup")
            if index == 0:
                rir_dir = setup / "01" / "RIR"
                rir_dir.mkdir()
                sf.write(rir_dir / "impulse.wav", np.array([1.0, 0.5, 0.2]), 16000)
    work = tmp_path / "work"
    main(argparse.Namespace(extracted_dir=str(extracted), work_dir=str(work), output_base="az://test/data"))
    rirs = [json.loads(line) for line in (work / "rirs.jsonl").read_text().splitlines()]
    noises = [json.loads(line) for line in (work / "noises.jsonl").read_text().splitlines()]
    assert len(rirs) == 2
    assert len(noises) == 100
    assert Counter(row["room"] for row in noises) == {"room_a": 50, "room_b": 50}
    assert len({row["archive_member"] for row in noises}) == 100
    assert all("/RIR/" in row["archive_member"] for row in rirs)
    assert all("/silence/" in row["archive_member"] for row in noises)
    assert all("SpkID60_S" not in row["archive_member"] for row in noises)
    for row in [*rirs, *noises]:
        assert row["license"] == "CC-BY-4.0"
        original = extracted / row["archive_member"]
        assert row["sha256"] == hashlib.sha256(original.read_bytes()).hexdigest()
