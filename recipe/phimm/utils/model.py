"""Shared utility functions for bias training scripts."""

# %%
import re
import os
import json
import wandb
import blobfile as bf
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
from accelerate import PartialState
from contextlib import nullcontext
from trl.trainer.utils import add_adapter_func, rank_print
from trl.scripts.audio_dataset import create_audio_dataset
from trl.scripts.utils import get_config_path, cache_dir
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled


def get_speech_peft_model(model, lora_name):
    config = model.config
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=config.speech_lora["r"],
        lora_alpha=config.speech_lora["lora_alpha"],
        target_modules=config.speech_lora["layer"],
        lora_dropout=config.speech_lora["dp"],
        task_type="CAUSAL_LM",
    )
    get_peft_model(model.model, lora_config, adapter_name=lora_name)
    return model


def cache_chkp_dir(remote_dir, local_dir=None):
    """Sync a checkpoint directory from remote to local."""
    local_dir = cache_dir(remote_dir, local_dir)
    chat_template_file = Path(local_dir) / "chat_template.json"
    # remove chat template
    chat_template_file.unlink(missing_ok=True)
    return local_dir


def init_model(model_id=None, update_encoder=False, new_lora=None):
    """Initialize the model and processor."""
    model_id = model_id or "microsoft/Phi-4-multimodal-instruct"
    model_id = model_id.rstrip("/")  # Ensure no trailing slash
    model_id = cache_chkp_dir(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation="flash_attention_2",
    )
    model.set_lora_adapter("speech")
    model = add_adapter_func(model)
    if new_lora:
        # TODO: gather context before merging and unloading,
        # this not work yet.
        gather_if_zero3 = gather_context()
        rank_print("Gather context: ", gather_if_zero3)
        with gather_if_zero3(list(model.parameters())):
            rank_print("merge and unload model")
            model.merge_and_unload()  # merge lora and back to normal Linear
            rank_print("Prepare peft model with adapter:", new_lora)
            model = get_speech_peft_model(model, lora_name=new_lora)  # revert peft model
    if update_encoder:
        layers = [
            r"^model.embed_tokens_extend.audio_embed.encoder.encoders",
            r"^model.embed_tokens_extend.audio_embed.audio_projection.speech",
        ]
        model = train_modules(model, layers)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


class WandbHelper:
    """Helper class to manage wandb initialization and run info."""

    def __init__(self, run_name=None, project_name=None, work_dir=None, new_run=False):
        self.new_run = new_run
        self.run_name = run_name
        self.project_name = project_name
        work_dir = work_dir or Path.cwd()
        self.run_info_file = Path(work_dir) / "run_info.json"

    def _get_run_name(self):
        """Get the run name from environment variables or command line arguments."""
        if self.run_name:
            return self.run_name
        config_path = get_config_path()

        return config_path.stem if config_path else "default"

    def _wandb_info(self):
        """Get wandb information from environment variables."""
        run_name = self._get_run_name()
        project = self.project_name or os.environ.get("WANDB_PROJECT", "biasing")
        entity = os.environ.get("WANDB_ENTITY", "genai")

        print(
            f"Run name: {run_name}, Project: {project}, New run: {self.new_run}, Work dir: {self.run_info_file.parent}"
        )
        run_info = {} if self.new_run else self._load_info()
        print("WandB Run Info:", run_info)
        return {
            "entity": run_info.get("entity", entity),
            "project": run_info.get("project", project),
            "name": run_name,
            "id": run_info.get("run_id", None),
            "config": run_info.get("config", {}),
        }

    def _login(self):
        """Login to wandb."""
        key = os.environ.get("WANDB_API_KEY", "")
        host = os.environ.get("WANDB_ORGANIZATION", "https://msaip.wandb.io")
        wandb.login(host=host, key=key, relogin=True)

    def init(self, main_only=True):
        """Initialize wandb."""
        if not PartialState().is_main_process and main_only:
            return None
        self._login()
        run_info = self._wandb_info()
        run = wandb.init(**run_info, resume="allow")
        rank_print("wandb mode: ", run.settings.mode)  # Should be "offline"
        self.run_info_file = self._save_info(run)
        return run

    def _save_info(self, run):
        """Save run identifying information to a JSON file."""
        self.run_info_file.parent.mkdir(parents=True, exist_ok=True)
        info = {
            "entity": run.entity,
            "project": run.project,
            "run_id": run.id,
            "run_name": run.name,
            "run_url": run.url,
        }
        json.dump(info, self.run_info_file.open("w"), indent=2)
        rank_print(f"Run info saved to {self.run_info_file}")
        return self.run_info_file

    def _load_info(self):
        """Load run info from JSON and resume the run."""
        if not self.run_info_file.exists():
            rank_print(f"Run info file {self.run_info_file} does not exist.")
            return {}
        rank_print(f"Loading run info from {self.run_info_file}")
        info = json.load(self.run_info_file.open("r"))
        url = info.get("run_url", None)
        if not url:
            return info
        rank_print(f"Reuse run: {url}")
        parts = Path(url).parts
        if url.startswith("https://msaip.wandb.io/") and len(parts) > 4:
            rank_print("Run info from", url)
            info["run_id"] = info.get("run_id", parts[-1])
            info["project"] = info.get("project", parts[-3])
            info["entity"] = info.get("entity", parts[-4])
        return info


def numel(p):
    if is_deepspeed_zero3_enabled():
        return p.ds_numel if hasattr(p, "ds_numel") else p.numel()
    else:
        return p.numel()


def gather_context():
    """Gather context for vLLM updates."""
    if is_deepspeed_zero3_enabled():
        import deepspeed

        return deepspeed.zero.GatheredParameters
    else:
        return nullcontext


def train_modules(model, reg_exp, trainable=True):
    """Train the model with the given prefix, substring, and/or suffix.
    Only parameters matching all specified criteria will be set as trainable.
    """
    reg_exp = reg_exp if isinstance(reg_exp, (tuple, list)) else [reg_exp]
    for name, param in model.named_parameters():
        if any(re.search(exp, name) for exp in reg_exp):
            param.requires_grad = trainable
    return model


def print_modules(model, trainable=False, all_rank=False):
    """List trainable modules in the model and total trainable parameter size."""
    main_only = not all_rank
    rank_print("List modules in the model:", {model.__class__.__name__}, main=main_only)
    n_total = 0
    n_trainable = 0
    for name, param in model.named_parameters():
        n_total += numel(param)
        if param.requires_grad:
            n_trainable += numel(param)
            if trainable:
                rank_print(f"{name}: {human_readable(numel(param))} trainable", main=main_only)
    rank_print(f"Total trainable: {human_readable(n_trainable)}", main=main_only)
    rank_print(f"Total parameter: {human_readable(n_total)}", main=main_only)
    return n_total, n_trainable


def human_readable(num):
    """Convert a number to human readable format (K, M, G)."""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}G"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)


def is_valid_checkpoint(model_dir):
    """Check if the model path is valid."""
    if not model_dir:
        return False
    if not bf.exists(model_dir):
        # rank_print(f"Model path {model_dir} does not exist.")
        return False
    if not bf.isdir(model_dir):
        # rank_print(f"Model path {model_dir} is not a directory.")
        return False
    config_file = f"{model_dir}/config.json"
    if not bf.exists(config_file):
        # rank_print(f"Config file {config_file} does not exist in the model directory.")
        return False
    if not any(bf.glob(f"{model_dir}/*.safetensors")):
        # rank_print(f"No .safetensors files found in {model_dir}.")
        return False
    return True


def get_latest_valid_checkpoint(output_dir):
    """Get the latest valid checkpoint directory."""
    latest_chkp_dir = get_last_checkpoint(output_dir)
    if is_valid_checkpoint(latest_chkp_dir):
        return latest_chkp_dir
    return None
