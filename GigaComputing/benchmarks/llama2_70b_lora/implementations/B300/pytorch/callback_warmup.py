# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
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

import logging
import time
from itertools import repeat
from pprint import pprint

import torch
from megatron.bridge.training import eval as eval_module
from megatron.bridge.training import train as train_module
from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.bridge.training.utils.train_utils import prepare_forward_step_func
from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.utils import get_model_config
from torch.utils.data import default_collate
from utils import get_rank

logger = logging.getLogger(__name__)


def create_synthetic_batch(num_microbatches):
    """Create synthetic input batch for warmup."""
    seq_length = 8192
    text = torch.ones(seq_length + 1, dtype=torch.int64) * 3545

    tokens = text[:-1]
    tokens[-1] = 2

    labels = text[1:]
    labels[-1] = 2

    loss_mask = torch.ones(seq_length, dtype=torch.int64)
    loss_mask[-1] = 0

    position_ids = torch.tensor([i for i in range(seq_length)], dtype=torch.int64)
    position_ids[-1] = 0

    single_data = {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
    }

    batch = default_collate([single_data] * num_microbatches)
    batch["token_count"] = [seq_length - 1 for _ in range(num_microbatches)]
    return batch


def reset_fp8_state(model):
    def reset_fp8(m):
        if hasattr(m, "fp8_initialized"):
            m.fp8_initialized = False
            if hasattr(m, "reset_fp8_meta_tensors"):
                m.reset_fp8_meta_tensors()

    for model_chunk in model:
        model_chunk.apply(reset_fp8)


class WarmupCallback(Callback):
    def __init__(self, cfg, forward_step_func=None):
        self.forward_step_func = forward_step_func
        self.train_steps = cfg.model.custom.warmup_train_steps
        self.val_steps = cfg.model.custom.warmup_validation_steps

    def on_data_init_start(self, context: CallbackContext):
        torch.cuda.synchronize()
        torch.distributed.barrier()
        if get_rank() == 0:
            pprint(context.model[0].config)
            logger.info("Starting warmup")

        context.state.warmup = True
        scheduler_state = context.scheduler.state_dict()
        saved_param_groups = []
        for group in context.optimizer.param_groups:
            saved_param_groups.append({"lr": group.get("lr", 0.0), "weight_decay": group.get("weight_decay", 0.0)})
            group["lr"] = 0.0
            group["weight_decay"] = 0.0

        pg_collection = get_pg_collection(context.model)
        model_config = get_model_config(context.model[0])

        forward_backward_func = get_forward_backward_func(
            pp_size=pg_collection.pp.size(),
        )

        if model_config.cuda_graph_impl == "local" and CudaGraphScope.full_iteration in model_config.cuda_graph_scope:
            forward_backward_func = FullCudaGraphWrapper(
                forward_backward_func,
                cuda_graph_warmup_steps=1,
            )

        wrapped_forward_step_func = prepare_forward_step_func(self.forward_step_func, context.state)

        pp_group = pg_collection.pp
        p2p_communicator = P2PCommunicator(pp_group=pp_group, config=model_config)

        num_microbatches = get_num_microbatches()
        data_iterator = repeat(create_synthetic_batch(num_microbatches))

        torch.distributed.barrier()
        start_time = time.time()
        for _ in range(self.train_steps):
            torch.cuda.synchronize()
            torch.distributed.barrier()
            train_module.train_step(
                forward_step_func=wrapped_forward_step_func,
                data_iterator=data_iterator,
                model=context.model,
                optimizer=context.optimizer,
                scheduler=context.scheduler,
                global_state=context.state,
                pg_collection=pg_collection,
                forward_backward_func=forward_backward_func,
                p2p_communicator=p2p_communicator,
            )
            torch.cuda.synchronize()
            torch.distributed.barrier()

        if hasattr(context.scheduler, "num_steps"):
            context.scheduler.num_steps = 0
        context.scheduler.load_state_dict(scheduler_state)

        for group, saved in zip(context.optimizer.param_groups, saved_param_groups):
            group.update(saved)

        if hasattr(context.optimizer, "optimizer") and hasattr(context.optimizer.optimizer, "state"):
            base_optim_state = context.optimizer.optimizer.state
            for _, state in base_optim_state.items():
                if "step" in state:
                    state["step"] = 0
                if "exp_avg" in state:
                    state["exp_avg"].zero_()
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].zero_()

        reset_fp8_state(context.model)

        if get_rank() == 0:
            logger.info(f"Training warmup time: {time.time() - start_time} seconds")

        if self.val_steps > 0:
            original_eval_iters = context.state.cfg.validation.eval_iters
            context.state.cfg.validation.eval_iters = self.val_steps
            eval_num_microbatches = context.state.cfg.validation.eval_global_batch_size // (
                context.state.cfg.validation.eval_micro_batch_size * context.state.cfg.data_parallel_size
            )
            logging.getLogger("megatron.core.timers").setLevel(logging.ERROR)
            data_iterator = repeat(create_synthetic_batch(eval_num_microbatches))
            torch.distributed.barrier()
            start_time = time.time()
            for _ in range(self.val_steps):
                torch.cuda.synchronize()
                torch.distributed.barrier()
                eval_module.evaluate(
                    context.state,
                    wrapped_forward_step_func,
                    data_iterator,
                    context.model,
                    None,
                    context.state.cfg,
                    False,
                    None,
                )
                torch.cuda.synchronize()
                torch.distributed.barrier()
            if get_rank() == 0:
                logger.info(f"Validation warmup time: {time.time() - start_time} seconds")
            context.state.train_state.consumed_valid_samples = 0
            context.state.cfg.validation.eval_iters = original_eval_iters

        context.state.warmup = False
        torch.distributed.barrier()
        if get_rank() == 0:
            logger.info("Finished warmup")


class NsysProfileCallback(Callback):
    def __init__(self, start_step, end_step, profile_ranks):
        self.start_step = start_step
        self.end_step = end_step
        self.profile_ranks = profile_ranks

    def on_train_step_end(self, context: CallbackContext):
        step = context.state.train_state.step
        if torch.distributed.get_rank() not in self.profile_ranks:
            return
        # train_state.step is 1-indexed; call Start at end of (start_step-1)
        # so nsys captures from start_step onwards
        if step == self.start_step - 1:
            torch.cuda.cudart().cudaProfilerStart()
        elif step == self.end_step:
            torch.cuda.cudart().cudaProfilerStop()
