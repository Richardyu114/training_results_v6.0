import torch
import torch.distributed as dist
import os
import argparse
import time
import math
import statistics

from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Union, Sequence, Any


class Timer:
    def __init__(self, log_period):
        self._start = False
        self._start_event = list(torch.cuda.Event(enable_timing=True) for _ in range(log_period))
        self._end_event = list(torch.cuda.Event(enable_timing=True) for _ in range(log_period))
        for i in range(log_period):
            self._start_event[i].record()
            self._end_event[i].record()
        torch.cuda.synchronize()
        self._cur_idx = 0
        self._total_time = None

    def start(self):
        if self._start == False:
            self._start_event[self._cur_idx].record()
            self._start = True

    def end(self):
        self._end_event[self._cur_idx].record()
        self._cur_idx += 1
        self._start = False

    def accum(self):
        self._total_time = [self._start_event[idx].elapsed_time(self._end_event[idx]) for idx in range(self._cur_idx)]

    def reset(self):
        self._total_time = None
        self._cur_idx = 0

    def total_time(self):
        result = sum(self._total_time)
        return result


class distributed_test(ABC):
    def __init__(self, args: Sequence[Any]):
        # We use file store for initialization
        if dist.is_initialized():
            self.rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            if os.environ.get("SLURM_PROCID"):
                self.rank = (int)(os.environ["SLURM_PROCID"])
                world_size = (int)(os.environ["SLURM_NTASKS"])
            elif os.environ.get("OMPI_COMM_WORLD_RANK"):
                self.rank = (int)(os.environ["OMPI_COMM_WORLD_RANK"])
                world_size = (int)(os.environ["OMPI_COMM_WORLD_SIZE"])
            else:
                ValueError(f"Run w/ either srun or mpirun which are supported job schedulers for this test")
            # print(f"rank: {self.rank}, start init_process_group")
            if args.filename is not None:
                filename = args.filename
                store = dist.FileStore(file_name=filename, world_size=world_size)
                dist.init_process_group(
                    backend="nccl",
                    store=store,
                    rank=self.rank,
                    world_size=world_size,
                )
            else:
                dist.init_process_group(
                    backend="nccl",
                    rank=self.rank,
                    world_size=world_size,
                )
        # print(f"rank: {self.rank}, finished init_process_group")
        # dist.barrier()
        self.dp_group_size = world_size // (args.tp * args.pp)
        num_dp_groups = world_size // self.dp_group_size
        self.pp_stage = self.rank // (args.tp * self.dp_group_size)

        if self.rank == 0:
            print(f"DP group size: {self.dp_group_size}, number of DP groups: {num_dp_groups}")

        dp_base_rank_list = []
        dp_lists = defaultdict(list)
        self.TP_RANK = None
        self.PP_RANK = None
        self.DP_RANK = None
        self.DP_GROUPS = []
        for i in range(args.pp):
            for j in range(args.tp):
                dp_base_rank = j + i * self.dp_group_size * args.tp
                dp_base_rank_list.append(dp_base_rank)
                dp_rank_list = list(range(dp_base_rank, dp_base_rank + args.tp * self.dp_group_size, args.tp))
                group = dist.new_group(ranks=dp_rank_list)
                if self.rank == 0:
                    print(f"DP GROUP[{i * args.tp + j}]: {dp_rank_list}")
                self.DP_GROUPS.append(group)
                dp_lists[j].append(dp_rank_list)
                if self.rank in dp_rank_list:
                    self.TP_RANK = j
                    self.DP_RANK = (self.rank - dp_base_rank) // args.tp
                    self.PP_RANK = i
                    self.CUR_GROUP = group

        for i, each in enumerate(dp_lists.values()):
            cur_groups = list(zip(*each))
            for idx, group in enumerate(cur_groups):
                if self.rank == 0:
                    print(f"self.PP_GROUP: self.DP_RANK: {i}, self.TP_RANK: {idx}: {group}")
                created_group = dist.new_group(ranks=group)
                if self.rank in group:
                    CUR_PP_GROUP = created_group
                    self.PP_GROUP = group

        self.prev_pprank = self.PP_GROUP[self.PP_RANK + 1] if self.PP_RANK + 1 < args.pp else None
        self.next_pprank = self.PP_GROUP[self.PP_RANK - 1] if self.PP_RANK - 1 >= 0 else None

        # if args.verbose:
        #    print(f"self.rank: {self.rank}, prev_pprank: {self.prev_pprank}, next_pprank: {self.next_pprank}, {self.PP_GROUP}")

        if args.pp_group:
            self.pp_group = set(args.pp_group)
            dp_base_rank_list = list(
                rank for rank in dp_base_rank_list if rank // (args.tp * self.dp_group_size) in self.pp_group)

        self.DP_BASE_GROUP = dist.new_group(ranks=dp_base_rank_list)
        self.device = self.rank % torch.cuda.device_count()

        # Dummy variable to initiate SHARP on these communicators if it's enabled.
        if args.sharp:
            torch.cuda.set_device(self.device)
            if len(args.pp_group) == 0 or self.pp_stage in args.pp_group:
                dummy = torch.ones([1024], dtype=torch.float16, device=torch.device(self.device))
                dist.all_reduce(
                    dummy, group=self.CUR_GROUP
                )
                print("DP groups are enabled w/ SHARP")
            os.environ["NCCL_SHARP_DISABLE"] = "1"
        dist.barrier()

    @abstractmethod
    def run(self):
        raise NotImplementedError("This method should be implemented")

    @staticmethod
    def parse_size(dtype_size: int, nbytes: str) -> int:
        options = {'g': 1024 * 1024 * 1024,
                   'm': 1024 * 1024,
                   'k': 1024}
        unit = 1
        key = nbytes[-1].lower()
        if key in options:
            unit = options[key]
            value = int(nbytes[:-1])
        else:
            value = int(nbytes)
        count = unit * value // dtype_size

        if count < 0:
            ValueError(f"passed argument {nbytes} is not viable")
        return count
