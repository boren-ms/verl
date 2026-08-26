import numpy as np
import soundfile as sf
from datasets import Dataset

from recipe.phimm.cache_dataset import materialize_audio_segments


def test_materialize_audio_segments_writes_concrete_wav(tmp_path):
    source_path = tmp_path / "source.wav"
    sf.write(source_path, np.zeros(16000, dtype=np.float32), 16000)
    dataset = Dataset.from_dict({
        "audio_path": [f"{source_path}#0%:50%", f"{source_path}#0%:25%", str(source_path)],
        "text": ["hello", "hi", "whole"],
    })

    result = materialize_audio_segments(dataset, str(tmp_path / "segments"))

    output_path = result[0]["audio_path"]
    assert "#" not in output_path
    assert sf.info(output_path).frames == 8000
    assert result[0]["text"] == "hello"
    assert sf.info(result[1]["audio_path"]).frames == 4000
    assert sf.info(result[2]["audio_path"]).frames == 16000