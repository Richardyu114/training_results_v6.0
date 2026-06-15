# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
import traceback
from typing import List, Optional

import torch
from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.bridge.utils.common_utils import get_rank_safe, get_world_size_safe


class MemoryProfileCallback(Callback):
    """
    Callback that starts/stops the PyTorch CUDA memory profiler and writes
    snapshot pickle files. Supports step-based or phase-based (train_start/train_end)
    profiling. On OOM or any exception during training, cleanup_after_oom() is
    invoked so that memory snapshots and traceback are still written.

    Snapshots can be viewed at https://pytorch.org/memory_viz (drag-and-drop the
    .pickle file). If the visualizer shows nothing: (1) we now call
    torch.cuda.synchronize() before dumping so the snapshot is consistent;
    (2) ensure recording covered real training steps (start_step/end_step or
    train_start/train_end); (3) try generating HTML locally from the same
    PyTorch env: python -m torch.cuda._memory_viz trace_plot snapshot.pickle -o out.html
    then open out.html in a browser.

    Full-iteration CUDA graphs (FullCudaGraphWrapper):
    - Replay steps do almost no Python-side allocation; tensor lifetimes are tied to
      graph capture. Profiling ``start_step``/``end_step`` in the *middle* of training
      therefore shows little allocator activity.
    - Record allocations from **warmup/eager** iterations and from **graph capture**
      (the step where ``curr_iteration == cuda_graph_warmup_steps``): use
      ``start_step=None`` (begin at ``on_train_start``) and register this callback
      *before* ``WarmupCallback`` (see ``pretrain.py``). Optionally set ``end_step``
      to stop soon after capture so ``max_entries`` is not exhausted.
    - Validation graph capture uses the same idea: ensure recording is already on
      before the first eval that captures ``validation`` graph.
    - If training stops before ``end_step``, the snapshot is still written at ``on_train_end``.

    Env: ``MEMORY_PROFILE_START_STEP``, ``MEMORY_PROFILE_END_STEP`` (wired in ``model/671b.yaml``;
    set them in ``config_performance_profile.sh`` or the shell).
    """

    def __init__(
        self,
        enabled: bool = True,
        file_prefix: str = "memdump",
        output_dir: str = "/mem_dump",
        max_entries: int = 1000000,
        rank_0_only: bool = True,
        profile_ranks: Optional[List[int]] = None,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        force_oom_before_stop: bool = False,
    ):
        self.enabled = enabled
        self.file_prefix = file_prefix
        self.output_dir = output_dir
        self.max_entries = max_entries
        self.force_oom_before_stop = force_oom_before_stop
        self.start_step = start_step  # None = start at train_start
        self.end_step = end_step  # None = stop at train_end

        global_rank = get_rank_safe()
        world_size = get_world_size_safe()
        if profile_ranks is not None:
            ranks = profile_ranks
        else:
            ranks = [0] if rank_0_only else list(range(world_size))
        self.do_profile = enabled and (global_rank in ranks)
        self._rank = global_rank

        self._recording = False

    @staticmethod
    def _format_bytes(b: int) -> str:
        """Format byte count as human-readable string (e.g. 1.23 GiB)."""
        if b >= 1024**3:
            return f"{b / 1024**3:.2f} GiB"
        if b >= 1024**2:
            return f"{b / 1024**2:.2f} MiB"
        return f"{b} B"

    def _log_step_memory(self, step: int, label: str) -> None:
        """Print current GPU memory allocated (and reserved) for this rank."""
        try:
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            max_allocated = torch.cuda.max_memory_allocated()
            max_reserved = torch.cuda.max_memory_reserved()
            print_str = f"Rank {self._rank} step {step} {label}: \
            Allocated: {self._format_bytes(allocated)} \
            Reserved: {self._format_bytes(reserved)} \
            Max Allocated: {self._format_bytes(max_allocated)} \
            Max Reserved: {self._format_bytes(max_reserved)}"
            print(print_str, flush=True)
        except Exception:
            pass

    def _start_recording(self) -> None:
        if not self.do_profile or self._recording:
            return
        torch.cuda.memory._record_memory_history(max_entries=self.max_entries)
        self._recording = True

    def _stop_and_dump(self, suffix: str = "") -> None:
        if not self.do_profile or not self._recording:
            return
        try:
            if self.force_oom_before_stop:
                self._force_oom()
            os.makedirs(self.output_dir, exist_ok=True)
            filename = os.path.join(self.output_dir, f"{self.file_prefix}_{self._rank}{suffix}.pickle")
            # Synchronize so pending CUDA ops are reflected in the snapshot (required for a
            # valid timeline in the memory visualizer at https://pytorch.org/memory_viz).
            torch.cuda.synchronize()
            torch.cuda.memory._dump_snapshot(filename)
        finally:
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except Exception:
                pass
            self._recording = False

    def _force_oom(self) -> None:
        """Trigger an OOM for debugging (e.g. to capture allocator state)."""
        try:
            torch.cuda.empty_cache()
            _ = torch.empty(2**32, device="cuda", dtype=torch.uint8)
        except Exception:
            pass

    def cleanup_after_oom(self) -> None:
        """
        Write memory snapshot and traceback after OOM or any crash.
        Should be called from an external exception handler wrapping the training loop.
        """
        if not self.do_profile:
            return
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            memdump_filename = os.path.join(self.output_dir, f"{self.file_prefix}_failed_{self._rank}.pickle")
            traceback_filename = os.path.join(self.output_dir, f"{self.file_prefix}_failed_{self._rank}.traceback")
            with open(traceback_filename, "w") as f:
                traceback.print_exc(file=f)
            if self._recording:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.memory._dump_snapshot(memdump_filename)
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except Exception:
                pass
            self._recording = False
        except Exception:
            pass

    def on_train_start(self, context: CallbackContext) -> None:
        if self.start_step is None:
            self._start_recording()

    def on_train_end(self, context: CallbackContext) -> None:
        # Dump when: (1) profiling through full training (end_step unset), or (2) run ended
        # before MEMORY_PROFILE_END_STEP — otherwise we would never write a snapshot.
        if self.end_step is None or self._recording:
            self._stop_and_dump()

    def on_train_step_start(self, context: CallbackContext) -> None:
        step = context.state.train_state.step
        self._log_step_memory(step, "start")
        if self.start_step is not None and step == self.start_step:
            self._start_recording()

    def on_train_step_end(self, context: CallbackContext) -> None:
        step = context.state.train_state.step
        if self.end_step is not None and step == self.end_step:
            self._stop_and_dump()
        self._log_step_memory(step, "end")
