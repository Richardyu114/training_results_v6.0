# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from typing import Any
from pprint import pformat

import lightning as pl
import torch
from warmup import run_training_warmup, reset_fp8_state

from mlperf_common.callbacks import LoggingCallback, MLPerfLogger
from omegaconf import DictConfig, OmegaConf
from flux_logging import get_rank_zero_logger

rank = int(os.environ.get("RANK", "0"))
logger = get_rank_zero_logger(__name__, rank)


def _get_gradient_accumulation_steps(cfg: DictConfig) -> int:
    world_size = int(cfg.trainer.num_nodes) * int(cfg.trainer.devices)
    model_parallel_size = (
        int(cfg.strategy.tensor_model_parallel_size)
        * int(cfg.strategy.pipeline_model_parallel_size)
        * int(cfg.strategy.context_parallel_size)
    )
    if world_size % model_parallel_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by model_parallel_size "
            f"({model_parallel_size})"
        )

    data_parallel_size = world_size // model_parallel_size
    global_batch_size = int(cfg.data.global_batch_size)
    micro_batch_size = int(cfg.data.micro_batch_size)
    per_step_batch = micro_batch_size * data_parallel_size
    if global_batch_size % per_step_batch != 0:
        raise ValueError(
            f"global_batch_size ({global_batch_size}) must be divisible by "
            f"micro_batch_size * data_parallel_size ({per_step_batch})"
        )
    return global_batch_size // per_step_batch


class CustomCallback(LoggingCallback):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.gbs = cfg.data.global_batch_size
        self.iter_time = time.time()
        self.enable_perf_debug = int(os.environ.get("ENABLE_PERF_DEBUG", "0"))
        self.enable_mem_profile = int(os.environ.get("ENABLE_MEMORY_PROFILE", "0"))
        if self.enable_mem_profile:
            torch.cuda.memory._record_memory_history(max_entries=100000)
        self.log_every_n_steps = cfg.trainer.log_every_n_steps

    def __deepcopy__(self, memo):
        return CustomCallback(self.cfg)

    def get_train_step_samples_count(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> int:
        return self.gbs

    def warmup(self, trainer, pl_module):
        if self.cfg.warmup.enabled:
            run_training_warmup(
                trainer,
                self.cfg.warmup.train_steps,
                self.cfg.warmup.validation_steps,
            )
            if (
                self.cfg.plugins.fp8 is not None
                and self.cfg.warmup.reset_fp8_stats_after_warmup
            ):
                reset_fp8_state(
                    pl_module,
                    reset_fp8_meta_tensors=self.cfg.plugins.fp8_recipe == "delayed",
                )

    def _flush_deferred_loss(self):
        """Log and clear any deferred loss queued by async GPU→CPU copy."""
        if hasattr(self, '_deferred_loss_cpu'):
            self.mllogger.event(
                key="tracked_stats",
                value={"train_loss": self._deferred_loss_cpu.item()},
                metadata=self._deferred_loss_metadata,
            )
            del self._deferred_loss_cpu, self._deferred_loss_metadata

    def on_train_end(self, trainer, pl_module):
        # Flush any deferred loss that was queued in the final on_train_batch_end
        self._flush_deferred_loss()

        # Release CUDA graphs while all ranks are still synchronized in the
        # training loop.  Previously this cleanup lived in teardown() behind a
        # dist.barrier(), but Lightning's teardown path after convergence
        # (should_stop=True) does not guarantee all ranks enter callback
        # teardown at the same time, causing the barrier to deadlock.
        # Moving the cleanup to on_train_end avoids the issue entirely —
        # matches the pattern used by llama31_405b/pytorch/callback_logging.py.
        module = trainer.model.module.module.module.module
        layers = module.double_blocks + module.single_blocks
        for layer in layers:
            if hasattr(layer, "cudagraph_manager"):
                layer.cudagraph_manager = None

        from megatron.core.full_cuda_graph import FullCudaGraphWrapper
        FullCudaGraphWrapper.cuda_graph = None

        super().on_train_end(trainer, pl_module)

    def on_train_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            logger.info(f"Optimized config:\n{OmegaConf.to_yaml(self.cfg)}")
            logger.info(f"\nMCore config:\n{pformat(trainer.model.config)}")
        super().on_train_start(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_rank == 0:
            if self.enable_perf_debug:
                torch.cuda.synchronize()
                print(f"iter {trainer.global_step} : {time.time() - self.iter_time}")
                torch.cuda.synchronize()
                self.iter_time = time.time()

            if trainer.global_step == 1 and self.enable_mem_profile:
                torch.cuda.memory._record_memory_history(max_entries=100000)

            if trainer.global_step == 3 and self.enable_mem_profile:
                torch.cuda.memory._dump_snapshot("/results/flux.pickle")
                print("memory profile written to /results/flux.pickle")

        # Log deferred loss from previous logging step (avoids CUDA sync on critical path)
        self._flush_deferred_loss()

        if trainer.global_step % self.log_every_n_steps == 0:
            # Async GPU→CPU copy: queue the transfer without blocking
            self._deferred_loss_cpu = outputs["loss"].detach().to("cpu", non_blocking=True)
            self._deferred_loss_metadata = {
                self.mllogger.constants.SAMPLES_COUNT: trainer.global_step
                * self.gbs,
                "lr": trainer.optimizers[0].param_groups[0]["lr"],
                "step": trainer.global_step,
            }


class MetricsLogger(MLPerfLogger):
    def __init__(
        self,
        cfg: DictConfig,
        model: pl.LightningModule,
    ):
        super().__init__(
            CustomCallback, model, "val_loss_sum", cfg.target_accuracy, "min", cfg
        )
        self.gbs = cfg.data.global_batch_size
        self.mbs = cfg.data.micro_batch_size
        self.cfg = cfg

    def __deepcopy__(self, memo):
        output = MetricsLogger(self.cfg, self.model)
        if self.trainer is not None:
            output.trainer = self.trainer
        return output

    def compute_hyperparams(
        self, params: dict[str, Any], *args, **kwargs
    ) -> dict[str, Any]:
        return {
            self.mllogger.constants.GLOBAL_BATCH_SIZE: self.gbs,
            self.mllogger.constants.TRAIN_SAMPLES: 1099776,  # TODO: can probably read from dataset index
            self.mllogger.constants.EVAL_SAMPLES: 29696,  # TODO: can probably read from dataset index
            "evaluation_frequency": self.cfg.trainer.val_check_interval,
            self.mllogger.constants.GRADIENT_ACCUMULATION_STEPS: (
                _get_gradient_accumulation_steps(self.cfg)
            ),
            self.mllogger.constants.OPT_NAME: self.mllogger.constants.ADAMW,
            self.mllogger.constants.OPT_LR_WARMUP_STEPS: (
                self.cfg.optim.lr_scheduler.warmup_steps
                if "lr_scheduler" in self.cfg.optim
                else 0
            ),
            self.mllogger.constants.OPT_ADAMW_BETA_1: self.cfg.optim.config.adam_beta1,
            self.mllogger.constants.OPT_ADAMW_BETA_2: self.cfg.optim.config.adam_beta2,
            self.mllogger.constants.OPT_ADAMW_EPSILON: self.cfg.optim.config.adam_eps,
            self.mllogger.constants.OPT_ADAMW_WEIGHT_DECAY: self.cfg.optim.config.weight_decay,
            self.mllogger.constants.OPT_BASE_LR: self.cfg.optim.config.lr,
            self.mllogger.constants.OPT_GRADIENT_CLIP_NORM: self.cfg.optim.config.clip_grad,
        }

    def compute_validation_metric(self, metrics: dict[str, float]) -> float:
        return metrics["val_loss_sum"] / metrics["val_loss_count"]
