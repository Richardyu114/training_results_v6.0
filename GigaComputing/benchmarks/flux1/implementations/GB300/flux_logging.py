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

import logging
from nemo.utils import logging as nemo_logging
import builtins
import warnings


class RankZeroFilter(logging.Filter):
    def __init__(self, rank):
        self.rank = rank

    def filter(self, record):
        return self.rank == 0


def get_rank_zero_logger(name, rank):
    logger = logging.getLogger(name)
    logger.addFilter(RankZeroFilter(rank))
    return logger


def disable_print():
    def do_nothing(*args, **kwargs):
        pass

    builtins.print = do_nothing


def setup_logging():
    nemo_logging.setLevel(logging.ERROR)
    logging.getLogger("lightning").setLevel(logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("nv_one_logger.core.internal.safe_execution").setLevel(logging.ERROR)
    logging.getLogger("nv_one_logger.training_telemetry.api.training_telemetry_provider").setLevel(logging.ERROR)


    warnings.filterwarnings("ignore")
    warnings.filterwarnings(
        "ignore", message=".*Could not find the bitsandbytes CUDA binary.*"
    )
