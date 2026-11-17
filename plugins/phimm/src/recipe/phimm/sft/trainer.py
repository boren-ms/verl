"""ASR-specific data sampling around the native verl supervised trainer."""

from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.trainer.sft_trainer import SFTTrainer
from verl.utils.checkpoint import OrchestrationMode

from .checkpoint import build_checkpoint_handler


class ASRSFTTrainer(SFTTrainer):
    """Preserve Base3a mix order and retain the final validation batch."""

    def _build_ckpt_handler(self):
        self.ckpt_handler = build_checkpoint_handler(
            self, engine=self.engine, mode=OrchestrationMode.SPMD, lora_meta=self._get_lora_train_meta()
        )

    def _build_dataloader(self):
        dp_size = self.engine.get_data_parallel_size()
        if self.config.data.train_batch_size < dp_size or self.config.data.train_batch_size % dp_size:
            raise ValueError("data.train_batch_size must be a positive multiple of data parallel size")
        super()._build_dataloader()
        # The mix already shuffles each source and group using Base3a's seed.
        # Upstream SFT otherwise adds another unconditional sampler shuffle.
        self.train_sampler.shuffle = self.config.data.shuffle
        if not len(self.train_dataloader):
            raise ValueError("Training data must contain at least one global batch")
        if self.val_dataset is not None and len(self.val_dataset):
            self.val_sampler = DistributedSampler(
                self.val_dataset,
                shuffle=False,
                num_replicas=dp_size,
                rank=self.engine.get_data_parallel_rank(),
                drop_last=False,
            )
            self.val_dataloader = StatefulDataLoader(
                self.val_dataset,
                batch_size=self.train_batch_size_per_dp,
                sampler=self.val_sampler,
                collate_fn=self.collate_fn,
                num_workers=self.config.data.num_workers,
                pin_memory=False,
                drop_last=False,
            )
