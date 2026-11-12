"""Exercise pipeline cleanup with a producer blocked on distributed loading."""

import threading

from test_main_asr_gen import _generation_batches
from test_main_asr_gen import generation_pipeline as generation_pipeline


class CloseableLoader:
    def __init__(self, block=False):
        self.closed = threading.Event()
        self.block = block

    def __iter__(self):
        yield from _generation_batches(1)
        if self.block:
            assert self.closed.wait(timeout=5), "Pipeline did not close the blocked loader"

    def close(self):
        self.closed.set()


def test_pipeline_closes_distributed_loader_after_success(generation_pipeline):
    loader = CloseableLoader()
    result = generation_pipeline.run(loader, total=1)
    assert result.error is None
    assert loader.closed.is_set()
    assert (generation_pipeline.output_dir / "part-000.jsonl").exists()


def test_decode_failure_unblocks_and_closes_distributed_producer(generation_pipeline):
    loader = CloseableLoader(block=True)

    def fail_decode(data):
        raise RuntimeError("decode failed during distributed loading")

    generation_pipeline.namespace["wg"].generate_sequences = fail_decode
    result = generation_pipeline.run(loader, total=2)
    assert isinstance(result.error, RuntimeError)
    assert str(result.error) == "decode failed during distributed loading"
    assert loader.closed.is_set()
    assert not list(generation_pipeline.output_dir.glob("part-*.jsonl"))
