"""Concurrent segment readers can evict cached audio without corrupting reads."""

import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
from recipe.phimm.utils.audio import sf_read


def test_concurrent_audio_cache_evictions_preserve_waveforms(tmp_path):
    paths = []
    for index in range(192):
        path = tmp_path / f"{index}.wav"
        sf.write(path, np.full(320, index / 256, dtype=np.float32), 16000, subtype="FLOAT")
        paths.append(str(path))
    order = list(range(192)) * 32
    random.Random(42).shuffle(order)

    def read(index):
        wave, rate = sf_read(paths[index])
        assert rate == 16000
        np.testing.assert_array_equal(wave, np.full(320, index / 256))

    sf_read.cache_clear()
    try:
        with ThreadPoolExecutor(max_workers=32) as executor:
            list(executor.map(read, order))
        assert len(sf_read.cache) <= 100
        for path in paths[:24]:
            cached, rate = sf_read(path)
            uncached, original_rate = sf_read.__wrapped__(path)
            assert rate == original_rate
            np.testing.assert_array_equal(cached, uncached)
    finally:
        sf_read.cache_clear()
