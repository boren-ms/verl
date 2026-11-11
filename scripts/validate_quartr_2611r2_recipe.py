"""Validate the frozen recipe with its real tokenizer, loader and reward, without training."""
import copy
import argparse
import json
from pathlib import Path
import tempfile

import blobfile as bf
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import hf_qwen35_audio  # registers the exact native processor

from recipe.phimm.data.rl_dataset import RLHFDataset, _flatten_data_confs
from recipe.phimm.data.dataset import create_audio_dataset
from recipe.phimm.reward.asr_vote import compute_score
from verl.utils import hf_tokenizer, hf_processor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-name', default='remax_earning_base4c_errors')
    parser.add_argument('--expected-rows', type=int, default=16235)
    parser.add_argument('--output-root', default='az://orngwus2cresco/data/boren/data/quartr_2611r2_errors_20261111')
    args = parser.parse_args()
    root = args.output_root.rstrip('/')
    OmegaConf.register_new_resolver('eval', lambda expr: eval(expr, {}, {}), replace=True)
    with initialize_config_dir(config_dir=str(Path('recipe/phimm/config/v2611_dev/earning').resolve()), version_base=None):
        cfg = compose(config_name=args.config_name)
    OmegaConf.resolve(cfg)
    data = cfg.data
    model = cfg.actor_rollout_ref.model.path.rstrip('/')
    # Tokenizer/processor tests need only metadata; no second model-weight download.
    with tempfile.TemporaryDirectory(prefix='quartr_recipe_tokenizer_') as local:
        for entry in bf.scandir(model):
            if entry.is_file and (entry.name.endswith('.json') or entry.name in ('merges.txt', 'vocab.txt', 'tokenizer.model')):
                bf.copy(entry.path, str(Path(local)/entry.name), overwrite=True)
        tokenizer = hf_tokenizer(local, trust_remote_code=False)
        processor = hf_processor(local, trust_remote_code=False, use_fast=True)
        ds = RLHFDataset(OmegaConf.to_container(data.train_data, resolve=True), tokenizer,
                        data, processor, is_train=True)
        assert len(ds) == args.expected_rows, len(ds)
        counts = {}
        first = {}
        for i, row in enumerate(ds.ds):
            source = row['data_source']
            counts[source] = counts.get(source, 0) + 1
            first.setdefault(source, i)
        samples = {}
        for source, index in first.items():
            row = ds[index]
            samples[source] = {'index': index, 'prompt_tokens': int(row['attention_mask'].sum()),
                               'reference': row['reward_model']['ground_truth'],
                               'audio_path': row['audio_path']}
        kwargs = OmegaConf.to_container(cfg.reward_functions.librispeech.reward_kwargs, resolve=True)
        reference = 'Revenue grew by ten percent.'
        exact = '<src=English><tgt=English>\n<VERBATIM>\nRevenue grew by ten percent.'
        reward = {name: compute_score(response, reference, **kwargs)
                  for name, response in {'correct': exact,
                      'incorrect': '<src=English><tgt=English>\n<VERBATIM>\nRevenue fell by fifty percent.',
                      'invalid': ''}.items()}
        paired_reward = {
            'correct_vs_incorrect': compute_score(exact, reference, **kwargs,
                extra_info={'greedy_reward': reward['incorrect']}),
            'incorrect_vs_correct': compute_score(
                '<src=English><tgt=English>\n<VERBATIM>\nRevenue fell by fifty percent.',
                reference, **kwargs, extra_info={'greedy_reward': reward['correct']}),
            'invalid_vs_correct': compute_score('', reference, **kwargs,
                extra_info={'greedy_reward': reward['correct']})}
        assert paired_reward['correct_vs_incorrect']['score'] > 0
        assert paired_reward['incorrect_vs_correct']['score'] < 0
        assert paired_reward['invalid_vs_correct']['score'] < 0
        # Delta-vote rewards may be neutral without a greedy baseline; preserve that behavior.
        import math
        assert all(math.isfinite(x['score']) for x in reward.values())
        train_pointers = {row.get('audio_path') for row in ds.ds}
        train_ids = {str(row.get('extra_info', {}).get('id')) for row in ds.ds}
        overlaps = []
        for spec in _flatten_data_confs(OmegaConf.to_container(data.val_data, resolve=True)):
            spec = copy.deepcopy(spec)
            val = create_audio_dataset(**{**spec, 'version': data.version})
            pointer_matches = [row['audio_path'] for row in val if row.get('audio_path') in train_pointers]
            id_matches = [str(row.get('extra_info', {}).get('id')) for row in val
                          if str(row.get('extra_info', {}).get('id')) in train_ids]
            overlaps.append({'manifest': spec.get('jsonl_paths'), 'validation_rows': len(val),
                             'exact_audio_pointer_overlap': len(pointer_matches),
                             'raw_id_overlap': len(id_matches), 'pointer_examples': pointer_matches[:10]})
        report = {'rows': len(ds), 'source_counts': counts, 'samples': samples,
                  'reward_checks': reward, 'paired_reward_checks': paired_reward,
                  'validation_overlap': overlaps,
                  'max_audio_dur': data.max_audio_dur, 'max_prompt_length': data.max_prompt_length,
                  'max_response_length': data.max_response_length, 'truncation': data.truncation,
                  'train_batch_size': data.train_batch_size, 'full_batches_per_epoch': len(ds)//data.train_batch_size,
                  'absolute_steps': cfg.trainer.total_training_steps, 'epoch_bound': cfg.trainer.total_epochs,
                  'limitations': ['Raw ID overlap is descriptive; source IDs can be reused.',
                                 'No audio-content, speaker or alias-equivalent parent leakage proof.',
                                 'No training run performed.']}
    with bf.BlobFile(root+'/recipe-validation.json', 'w') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
