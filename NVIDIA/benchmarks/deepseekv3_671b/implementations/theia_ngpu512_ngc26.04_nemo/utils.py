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
import logging

from functools import wraps

from megatron.bridge.utils.common_utils import is_last_rank


def get_rank():
    return int(os.getenv("SLURM_PROCID", 0))


def rank_last(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if is_last_rank():
            return func(*args, **kwargs)
    return wrapper


class RankZeroFilter(logging.Filter):
    def filter(self, record):
        return get_rank() == 0


def init_logging():
    root = logging.getLogger()
    root.addFilter(RankZeroFilter())

    l = logging.getLogger("megatron.core.dist_checkpointing.validation")
    l.setLevel(logging.DEBUG)
    h = logging.StreamHandler()
    h.setLevel(logging.DEBUG)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    l.addHandler(h)
    l.propagate = False  # prevent duplicate logs if root has handlers
