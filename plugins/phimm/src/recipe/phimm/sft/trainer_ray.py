"""Base3a sampling and complete validation batches for native Ray SFT."""

from torchdata.stateful_dataloader import StatefulDataLoader

from verl.trainer.sft_trainer_ray import SFTTrainer
from verl.utils.checkpoint import OrchestrationMode
from verl.utils.dataset.dataset_utils import SFTTensorCollator

from .checkpoint import build_checkpoint_handler


class WorldSizeSFTCollator:
    """Repeat only the last few validation rows to allow even DP dispatch."""

    def __init__(self, pad_mode, world_size):
        self.collator = SFTTensorCollator(pad_mode)
        self.world_size = world_size

    def __call__(self, samples):
        extra = (-len(samples)) % self.world_size
        return self.collator([*samples, *([samples[-1]] * extra)])


class ASRSFTTrainerRay(SFTTrainer):
    """Train with Base3a mix order and include the validation remainder."""

    def _build_dataloader(self):
        world_size = self.config.trainer.nnodes * self.config.trainer.n_gpus_per_node
        if self.config.data.train_batch_size < world_size or self.config.data.train_batch_size % world_size:
            raise ValueError("data.train_batch_size must be a positive multiple of the SFT GPU count")
        super()._build_dataloader()
        self.train_sampler.shuffle = self.config.data.shuffle
        self.train_dataloader = StatefulDataLoader(
            self.train_dataset,
            batch_size=self.global_batch_size,
            sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config.data.num_workers,
            drop_last=True,
        )
        if not len(self.train_dataloader):
            raise ValueError("Training data must contain at least one global batch")
        if self.val_dataset is not None and len(self.val_dataset):
            self.val_dataloader = StatefulDataLoader(
                self.val_dataset,
                batch_size=self.global_batch_size,
                shuffle=False,
                collate_fn=WorldSizeSFTCollator(self.config.data.pad_mode, world_size),
                num_workers=self.config.data.num_workers,
                drop_last=False,
            )

    def _build_ckpt_handler(self):
        rank = self.config.model.lora_rank
        lora_meta = None
        if rank > 0:
            lora_meta = {
                "r": rank,
                "lora_alpha": self.config.model.lora_alpha,
                "task_type": "CAUSAL_LM",
            }
        self.ckpt_handler = build_checkpoint_handler(
            self, engine=self.training_client, mode=OrchestrationMode.RAY, lora_meta=lora_meta
        )
