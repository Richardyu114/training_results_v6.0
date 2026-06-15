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

"""
Custom LR scheduler matching TorchTitan's warmup behavior.

TorchTitan formula: lr = base_lr * (step + 1) / warmup_steps
NeMo formula:       lr = base_lr * (step + 1) / (warmup_steps + 1)

TorchTitan condition: step < warmup_steps
NeMo condition:       step <= warmup_steps
"""

from torch.optim.lr_scheduler import _LRScheduler
from nemo.lightning.pytorch.optim import LRSchedulerModule


class TorchTitanWarmupHoldScheduler(LRSchedulerModule):
    """
    Warmup then hold forever, matching TorchTitan's LR schedule.
    
    Args:
        warmup_steps: Number of warmup steps
    """

    def __init__(self, warmup_steps: int = 0):
        super().__init__()
        self.warmup_steps = warmup_steps

    def scheduler(self, model, optimizer):
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": _TorchTitanWarmupHold(optimizer, self.warmup_steps),
                "interval": "step",
                "frequency": 1,
            },
        }


class _TorchTitanWarmupHold(_LRScheduler):
    """Linear warmup then constant LR, matching TorchTitan."""

    def __init__(self, optimizer, warmup_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps and self.warmup_steps > 0:
            # TorchTitan: (step + 1) / warmup_steps
            mult = (step + 1) / self.warmup_steps
            return [lr * mult for lr in self.base_lrs]
        return self.base_lrs
