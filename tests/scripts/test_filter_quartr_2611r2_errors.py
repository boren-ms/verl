"""Data-integrity gates tested with real canonical English scoring, no GPU."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from recipe.phimm.reward.asr_eval import measure_openasr_en_wer
from scripts.filter_quartr_2611r2_errors import SOURCES, freeze, sha


class FilterIntegrityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, self.previous)
        self.addCleanup(self.tmp.cleanup)
        self.report = self.root/'report'
        self.report.mkdir()
        self.generation = self.root/'generation'
        self.generation.mkdir()
        self.output = self.root/'output'
        inventory, self.generated = [], []
        config_root = Path('recipe/phimm/config/data/train_data')
        config_root.mkdir(parents=True)
        for source in SOURCES:
            raw = [dict(id=str(i), audio_path=f'{source}_{i}.wav', display='revenue grew',
                        transcript_status='preserve me') for i in range(3)]
            path = self.root/f'{source}.jsonl'
            content = ''.join(json.dumps(row)+'\n' for row in raw).encode()
            path.write_bytes(content)
            spec = dict(jsonl_paths=str(path), pre_process={})
            inventory.append(dict(uri=str(path), source=source, part=0, count=3,
                                  sha256=sha(content), spec=spec))
            (config_root/f'{source}_2611r2_errors.yaml').write_text(yaml.safe_dump([
                dict(jsonl_paths=str(self.output/'filtered'/f'{source}.jsonl'), post_process={})]))
            for row, hyp in zip(raw, ['Revenue grew.', 'revenue blue', 'a b c d e f g h']):
                score = measure_openasr_en_wer(hyp, row['display'])
                self.generated.append(dict(audio_path=row['audio_path'], text=row['display'],
                    response=hyp, audio_duration_seconds=2.0, response_at_limit=False, **score))
        (self.report/'source-inventory.json').write_text(json.dumps(inventory))
        self.args = argparse.Namespace(generation=str(self.generation), output=str(self.output),
                                      report_dir=str(self.report))

    @staticmethod
    def loader(**spec):
        rows = [json.loads(line) for line in Path(spec['jsonl_paths']).read_text().splitlines()]
        if 'post_process' in spec:
            return [dict(row, reward_model={'ground_truth': row['display']}) for row in rows]
        return [dict(row, text=row['display']) for row in rows]

    def run_filter(self):
        (self.generation/'part-000.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.generated))
        data_module = types.ModuleType('recipe.phimm.data.dataset')
        data_module.create_audio_dataset = self.loader
        audio_module = types.ModuleType('recipe.phimm.utils.audio')
        audio_module.load_audio = lambda *a: (np.ones(100), 16000)
        with patch.dict('sys.modules', {'recipe.phimm.data.dataset': data_module,
                                       'recipe.phimm.utils.audio': audio_module}):
            freeze(self.args)

    def test_exact_errors_only_preserving_labels_and_high_wer(self):
        self.run_filter()
        for source in SOURCES:
            rows = [json.loads(x) for x in (self.output/'filtered'/f'{source}.jsonl').read_text().splitlines()]
            self.assertEqual([x['id'] for x in rows], ['1', '2'])
            self.assertTrue(all(x['display'] == 'revenue grew' for x in rows))
            self.assertTrue(all(x['transcript_status'] == 'preserve me' for x in rows))
            self.assertTrue(all('response' not in x for x in rows))
        self.assertTrue((self.output/'COMPLETE.json').exists())

    def test_missing_generation_cannot_publish_subset(self):
        self.generated.pop()
        with self.assertRaisesRegex(ValueError, 'Incomplete generation'):
            self.run_filter()
        self.assertFalse(self.output.exists())

    def test_reference_mismatch_cannot_publish_subset(self):
        self.generated[0]['text'] = 'another human reference'
        with self.assertRaisesRegex(ValueError, 'reference differs'):
            self.run_filter()
        self.assertFalse(self.output.exists())

    def test_incorrect_stored_score_cannot_publish_subset(self):
        self.generated[0]['n_err'] = 1
        with self.assertRaisesRegex(ValueError, 'rescore mismatch'):
            self.run_filter()
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
