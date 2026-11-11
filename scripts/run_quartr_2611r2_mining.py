"""Task entrypoint: complete generation, strict source joins, filtering and audit."""
import json
from pathlib import Path

import blobfile as bf
import hydra

from recipe.phimm.main_asr_gen import run_generation
from scripts.filter_quartr_2611r2_errors import freeze
from argparse import Namespace


@hydra.main(config_path='../recipe/phimm/config/v2611_dev/gen',
            config_name='gen_quartr_2611r2_errors', version_base=None)
def main(config):
    root = Path('tmp/experiment_reports/quartr_2611r2_errors_20261111')
    root.mkdir(parents=True, exist_ok=True)
    output = config.data.output_path.rsplit('/generation', 1)[0]
    with bf.BlobFile(output + '/source-inventory.json', 'r') as stream:
        inventory = json.load(stream)
    (root/'source-inventory.json').write_text(json.dumps(inventory, indent=2))
    run_generation(config)
    freeze(Namespace(generation=config.data.output_path, output=output, report_dir=str(root)))


if __name__ == '__main__':
    main()
