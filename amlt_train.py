#!/usr/bin/env python
"""submit a job to AMLT"""

import os
import subprocess
from pathlib import Path
import shortuuid
import fire
from omegaconf import OmegaConf


def uuid4():
    short_id = shortuuid.ShortUUID().random(length=4)
    return short_id


def amlt_run(conf_file, node=1, job_pfx="llm", sla_tier=None, tag="safn", prepare=False):
    """submit a job to AMLT"""
    # remove script/train from the path
    conf_file = Path(conf_file)
    assert conf_file.exists() or str(conf_file).startswith("dummy"), f"Config file {conf_file} does not exist"
    amlt_conf_dir = Path("amlt_conf")
    tmp_suf = f"_{tag}" if tag else ""
    conf = OmegaConf.load(amlt_conf_dir / f"amlt_train_temp{tmp_suf}.yaml")
    print("config file:", conf_file)
    file_stem = conf_file.stem
    job_name = "-".join([job_pfx, conf_file.stem, uuid4()])
    sku = conf.jobs[0].sku.split("x")[-1]

    if "dpo" in file_stem:
        tr_cmd = (
            f"python trl/scripts/dpo_bias.py --config {conf_file} --job_name {job_name} --output_dir $$AMLT_OUTPUT_DIR "
        )
    elif "grpo" in file_stem:
        tr_cmd = f"python trl/scripts/grpo_bias.py --config {conf_file} --job_name {job_name} --output_dir $$AMLT_OUTPUT_DIR "
    elif file_stem.startswith("dummy"):
        tr_cmd = "python dummy_train.py"
    else:
        raise ValueError(f"Unknown config file: {conf_file}")
    conf.jobs[0].name = job_name
    conf.jobs[0].sku = f"{node}x{sku}"
    for key in ["WANDB_API_KEY"]:
        if (val := os.getenv(key, None)) is not None:
            conf.jobs[0].command.insert(0, f'export {key}="{val}"')
    if sla_tier:
        conf.jobs[0].sla_tier = sla_tier
    conf.jobs[0].command[-1] = tr_cmd
    tmp_yaml = amlt_conf_dir / f"tmp_{uuid4()}.yaml"
    OmegaConf.save(conf, tmp_yaml, resolve=True)
    cmd = f'amlt run {tmp_yaml} {job_name} -y -d ""'
    if prepare:
        print(f"Prepare: {cmd}")
        return
    try:
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(e)
    finally:
        tmp_yaml.unlink()


if __name__ == "__main__":
    fire.Fire(amlt_run)
