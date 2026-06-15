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

import json
import math
import os
import re
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.core import parallel_state


class StatsLogCallback(Callback):
    """Statistics logging callback for the gpt_oss_20b training system."""

    def __init__(self, cfg, save_path=None):
        self.cfg = cfg
        self.logs = {
            "train_loss_batch": [],
            "val_loss_batch": [],
            "val_loss_epoch": [],
            "grad_stats": {},
            "activation_stats": {},
            "learning_rates": [],
            "weights_stats": {},
        }
        self.activation_hooks = []
        self.grad_hooks = []
        self.current_grad_stats = {}
        self.current_weights_stats = {}
        self.current_activation_stats = {}
        self.reduce_tp = os.environ.get("REDUCE_TP", "True").lower() in ("true", "1", "t")
        self.log_every_n_steps = int(os.environ.get("LOG_EVERY_N_BATCHES", "1"))
        self.current_batch_idx = 0
        self.run_n_iters = int(os.environ.get("RUN_N_ITERS", "0"))
        self.enabled = True

        self.save_path = save_path if save_path is not None else f"/results/stats_seed{os.getenv('SEED', '1')}.json"

        # Store model reference
        self.model = None
        self.use_vp = False
        self.layers_per_pipeline = None
        self.layers_per_vchunk = None
        self.pp_size = None
        self.vp_size = None

    @staticmethod
    def _compute_tensor_stats(tensor: torch.Tensor, per_batch: bool = True, batch_dim: int = 0) -> Dict[str, Any]:
        try:
            if per_batch:
                dims = list(range(tensor.ndim))
                dims.remove(batch_dim)
                return {
                    "mean": tensor.mean(dim=dims).tolist(),
                    "var": tensor.var(dim=dims).tolist(),
                    "min": tensor.amin(dim=dims).tolist(),
                    "max": tensor.amax(dim=dims).tolist(),
                    "norm": torch.norm(tensor, p=2, dim=dims).tolist(),
                }
            else:
                return {
                    "mean": tensor.mean().item(),
                    "var": tensor.var().item(),
                    "min": tensor.amin().item(),
                    "max": tensor.amax().item(),
                    "norm": torch.norm(tensor, p=2).item(),
                }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _get_empty_stats() -> Dict[str, list]:
        return {"mean": [], "var": [], "min": [], "max": [], "norm": []}

    def _gather_group_stats(self, stats: Dict[str, Any], group) -> Dict[str, Any]:
        group_size = dist.get_world_size(group=group)
        gathered_stats = [None for _ in range(group_size)]
        dist.all_gather_object(gathered_stats, stats, group=group)

        if isinstance(gathered_stats[0]["mean"], list):
            result = {key: [d[key][0] for d in gathered_stats] for key in gathered_stats[0]}
        else:
            result = {key: [d[key] for d in gathered_stats] for key in gathered_stats[0]}
        return result

    def _collect_pp_dicts(self, stats_dict: Dict[str, Any]) -> Dict[str, Any]:
        def alphanum_key(s: str):
            return [int(text) if text.isdigit() else text for text in re.split(r"(\d+)", s)]

        group = parallel_state.get_pipeline_model_parallel_group()
        group_size = dist.get_world_size(group=group)
        gathered_stats = [None for _ in range(group_size)]
        dist.all_gather_object(gathered_stats, stats_dict, group=group)
        result = {k: v for d in gathered_stats if isinstance(d, dict) for k, v in d.items()}
        stats_dict = dict(sorted(result.items(), key=lambda x: alphanum_key(x[0])))
        return stats_dict

    def _collect_tp_dicts(self, stats_dict: Dict[str, Any]) -> Dict[str, Any]:
        def alphanum_key(s: str):
            return [int(text) if text.isdigit() else text for text in re.split(r"(\d+)", s)]

        group = parallel_state.get_tensor_model_parallel_group()
        group_size = dist.get_world_size(group=group)
        gathered_stats = [None for _ in range(group_size)]
        dist.all_gather_object(gathered_stats, stats_dict, group=group)
        result = {k: v for d in gathered_stats if isinstance(d, dict) for k, v in d.items()}
        stats_dict = dict(sorted(result.items(), key=lambda x: alphanum_key(x[0])))
        return stats_dict

    def _collect_group_stats(
        self,
        stats: Dict[str, Any],
        use_tp_group: bool = False,
        use_dp_group: bool = True,
    ) -> Dict[str, float]:
        if not self.reduce_tp:
            group = parallel_state.get_context_parallel_group()
        elif use_tp_group:
            group = parallel_state.get_tensor_model_parallel_group()
        else:
            group = parallel_state.get_tensor_and_context_parallel_group()
        result = self._gather_group_stats(stats, group=group)
        result = self._aggregate_local_stats(result)

        if use_dp_group:
            dp_group = parallel_state.get_data_parallel_group()
            result = self._gather_group_stats(result, group=dp_group)
            result = self._aggregate_local_stats(result, avg_variance=True)

        return result

    def _reduce_stats_across_group(
        self, stats_dict: Dict[str, Any], group, avg_variance: bool = False
    ) -> Dict[str, Any]:
        group_size = dist.get_world_size(group=group)
        if group_size <= 1:
            return stats_dict

        gathered = [None for _ in range(group_size)]
        dist.all_gather_object(gathered, stats_dict, group=group)

        all_modules = set()
        for d in gathered:
            if isinstance(d, dict):
                all_modules.update(d.keys())

        result = {}
        for module_name in all_modules:
            rank_data = [d[module_name] for d in gathered if isinstance(d, dict) and module_name in d]
            if not rank_data:
                continue

            keys = list(rank_data[0].keys())
            n_entries = len(rank_data[0].get("mean", []))

            agg_stats = {k: [] for k in keys}
            for i in range(n_entries):
                per_rank_vals = {}
                for key in keys:
                    vals = [rd[key][i] for rd in rank_data if key in rd and i < len(rd[key])]
                    if vals and isinstance(vals[0], list):
                        per_rank_vals[key] = [v[0] for v in vals]
                    else:
                        per_rank_vals[key] = vals
                agg = self._aggregate_local_stats(per_rank_vals, avg_variance=avg_variance)
                for key in keys:
                    agg_stats[key].append(agg[key])

            result[module_name] = agg_stats
        return result

    def _aggregate_local_stats(self, stats: Dict[str, list], avg_variance: bool = False) -> Dict[str, float]:
        global_mean = np.mean(stats["mean"])
        global_min = np.min(stats["min"])
        global_max = np.max(stats["max"])

        if avg_variance:
            global_variance = np.mean(stats["var"])
            global_norm = np.mean(stats["norm"])
        else:
            means_sq = [m * m for m in stats["mean"]]
            alpha = [v + ms for (v, ms) in zip(stats["var"], means_sq)]
            avg_alpha = sum(alpha) / len(alpha)
            global_variance = avg_alpha - global_mean**2

            sum_sq = sum((norm**2) for norm in stats["norm"])
            global_norm = math.sqrt(sum_sq)

        return {
            "mean": global_mean,
            "var": global_variance,
            "min": global_min,
            "max": global_max,
            "norm": global_norm,
        }

    def record_stats(
        self,
        module_name: str,
        stats: Dict[str, Any],
        log_key: Optional[str],
        logs: Dict[str, Any],
    ) -> None:
        target_logs = logs[log_key] if log_key else logs
        if module_name not in target_logs:
            target_logs[module_name] = self._get_empty_stats()

        for stat_name, stat_value in stats.items():
            target_logs[module_name][stat_name].append(stat_value)

    def _get_tensor(self, x: Any) -> Optional[torch.Tensor]:
        if torch.is_tensor(x):
            return x
        elif isinstance(x, (list, tuple)):
            return self._get_tensor(x[0])
        return None

    def _activation_hook(self, name: str, module: torch.nn.Module, inp, out):
        if not module.training or not self.enabled:
            return
        if self.current_batch_idx % self.log_every_n_steps != 0:
            return

        act = self._get_tensor(out)
        if act is None:
            return

        stats = self._compute_tensor_stats(act)
        self.record_stats(name, stats, None, self.current_activation_stats)

    def _grad_hook(self, name: str, module: torch.nn.Module, grad_in, grad_out):
        if not self.enabled or self.current_batch_idx % self.log_every_n_steps != 0:
            return

        grad = self._get_tensor(grad_out)
        if grad is None:
            return

        stats = self._compute_tensor_stats(grad, batch_dim=1)
        self.record_stats(name, stats, None, self.current_grad_stats)

    def _log_weights_stats(self, model_chunk, chunk_idx=None) -> None:
        for name, param in model_chunk.named_parameters():
            if param.requires_grad:
                stats = self._compute_tensor_stats(param.data, per_batch=False)
                if parallel_state.get_tensor_model_parallel_world_size() > 1:
                    stats = self._collect_group_stats(stats, use_tp_group=True, use_dp_group=False)

                if self.use_vp:
                    name = self.set_layer_vp_chunk(name, chunk_idx)
                if not self.reduce_tp:
                    name = f"{name}_tp{parallel_state.get_tensor_model_parallel_rank()}"
                self.current_weights_stats[name] = stats

    def extract_loss(self, loss_dict: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if "lm loss" in loss_dict:
            loss = loss_dict["lm loss"]
        elif "loss" in loss_dict:
            loss = loss_dict["loss"]
        else:
            if loss_dict:
                loss = next(iter(loss_dict.values()))
            else:
                loss = None

        if loss is not None:
            loss = loss.cpu()

        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            group = parallel_state.get_pipeline_model_parallel_group()
            gathered_losses = [None for _ in range(dist.get_world_size(group=group))]
            dist.all_gather_object(gathered_losses, loss, group=group)
            gathered_losses = [t.mean() for t in gathered_losses if isinstance(t, torch.Tensor)]
            if gathered_losses:
                loss = torch.stack(gathered_losses).sum()
            else:
                loss = None
        return loss

    def set_layer_vp_chunk(self, name, vp_chunk):
        pattern = r"\.(\d+)\."
        match = re.search(pattern, name)

        if not match:
            return name

        layer_idx = int(match.group(1))
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        orig_layer_idx = pp_rank * self.layers_per_pipeline + vp_chunk * self.layers_per_vchunk + layer_idx
        orig_name = re.sub(rf"\.{layer_idx}\.", f".{orig_layer_idx}.", name, count=1)
        return orig_name

    def register_hooks(self, modules, chunk_idx=None):
        for name, module in modules.named_modules():
            if hasattr(module, "weight") or ".fused_attention" in name:
                if self.use_vp:
                    name = self.set_layer_vp_chunk(name, chunk_idx)
                if not self.reduce_tp:
                    name = f"{name}_tp{parallel_state.get_tensor_model_parallel_rank()}"
                self.activation_hooks.append(module.register_forward_hook(partial(self._activation_hook, name)))
                self.grad_hooks.append(module.register_full_backward_hook(partial(self._grad_hook, name)))

    def _save_logs(self):
        if parallel_state.get_data_parallel_rank() == 0:
            try:
                with open(self.save_path, "w") as f:
                    json.dump(self.logs, f, indent=4)
            except Exception as e:
                print(f"Error saving debugging info: {e}")

    def setup_model(self, model):
        self.model = model

        vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()
        self.use_vp = vp_size is not None and vp_size > 0

        if self.use_vp:
            if isinstance(model, list):
                config = model[0].config
                num_layers = config.num_layers
                if (
                    hasattr(config, "account_for_embedding_in_pipeline_split")
                    and config.account_for_embedding_in_pipeline_split
                ):
                    num_layers += 1
                if hasattr(config, "account_for_loss_in_pipeline_split") and config.account_for_loss_in_pipeline_split:
                    num_layers += 1

                self.layers_per_pipeline = num_layers // parallel_state.get_pipeline_model_parallel_world_size()
                self.layers_per_vchunk = (
                    self.layers_per_pipeline // parallel_state.get_virtual_pipeline_model_parallel_world_size()
                )
                self.pp_size = parallel_state.get_pipeline_model_parallel_world_size()
                self.vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

                for chunk_idx, model_chunk in enumerate(model):
                    self.register_hooks(model_chunk, chunk_idx)
                    self._log_weights_stats(model_chunk, chunk_idx)
            else:
                self.register_hooks(model)
                self._log_weights_stats(model)
        else:
            if isinstance(model, list):
                for model_chunk in model:
                    self.register_hooks(model_chunk)
                    self._log_weights_stats(model_chunk)
            else:
                self.register_hooks(model)
                self._log_weights_stats(model)

        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            self.current_weights_stats = self._collect_pp_dicts(self.current_weights_stats)
        if not self.reduce_tp:
            self.current_weights_stats = self._collect_tp_dicts(self.current_weights_stats)

        for module_name, stats in self.current_weights_stats.items():
            self.record_stats(module_name, stats, "weights_stats", self.logs)

    #
    # CALLBACK HOOKS (Callback / CallbackContext API)
    #

    def on_train_start(self, context: CallbackContext):
        if context.model is not None:
            self.setup_model(context.model)

    def on_train_end(self, context: CallbackContext):
        self._save_logs()

    def on_train_step_start(self, context: CallbackContext):
        self.current_grad_stats = {}
        self.current_weights_stats = {}
        self.current_activation_stats = {}

    def on_train_step_end(self, context: CallbackContext):
        self.current_batch_idx += 1

        # Disable hooks after run_n_iters if specified
        if self.run_n_iters > 0 and self.current_batch_idx > self.run_n_iters and self.enabled:
            self.enabled = False
            for hook in self.activation_hooks:
                hook.remove()
            for hook in self.grad_hooks:
                hook.remove()
            self.activation_hooks = []
            self.grad_hooks = []

        if not self.enabled or (self.current_batch_idx - 1) % self.log_every_n_steps != 0:
            return

        # Log learning rate
        lr = context.optimizer.param_groups[0]["lr"]
        self.logs["learning_rates"].append(lr)

        # Log loss
        loss_dict = context.state.train_state.loss_dict if hasattr(context.state.train_state, "loss_dict") else {}
        if loss_dict:
            loss = self.extract_loss(loss_dict)
            if loss is not None:
                self.logs["train_loss_batch"].append(loss.item())

        # Gather grad and activation stats across TP/CP and DP groups
        if parallel_state.get_tensor_and_context_parallel_world_size() > 1:
            if not self.reduce_tp:
                group = parallel_state.get_context_parallel_group()
            else:
                group = parallel_state.get_tensor_and_context_parallel_group()
            self.current_grad_stats = self._reduce_stats_across_group(self.current_grad_stats, group)
            self.current_activation_stats = self._reduce_stats_across_group(self.current_activation_stats, group)

            dp_group = parallel_state.get_data_parallel_group()
            self.current_grad_stats = self._reduce_stats_across_group(
                self.current_grad_stats, dp_group, avg_variance=True
            )
            self.current_activation_stats = self._reduce_stats_across_group(
                self.current_activation_stats, dp_group, avg_variance=True
            )

        # Collect across pipeline parallel
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            self.current_grad_stats = self._collect_pp_dicts(self.current_grad_stats)
            self.current_activation_stats = self._collect_pp_dicts(self.current_activation_stats)

        if not self.reduce_tp:
            self.current_grad_stats = self._collect_tp_dicts(self.current_grad_stats)
            self.current_activation_stats = self._collect_tp_dicts(self.current_activation_stats)

        # Record gradient stats
        for module_name, stats in self.current_grad_stats.items():
            stats_agg = self._aggregate_local_stats(stats, avg_variance=True)
            self.record_stats(module_name, stats_agg, "grad_stats", self.logs)

        # Record activation stats
        for module_name, stats in self.current_activation_stats.items():
            stats_agg = self._aggregate_local_stats(stats, avg_variance=True)
            self.record_stats(module_name, stats_agg, "activation_stats", self.logs)

        # Log weights stats
        if self.model is not None:
            if self.use_vp and isinstance(self.model, list):
                for chunk_idx, model_chunk in enumerate(self.model):
                    self._log_weights_stats(model_chunk, chunk_idx)
            elif isinstance(self.model, list):
                for model_chunk in self.model:
                    self._log_weights_stats(model_chunk)
            else:
                self._log_weights_stats(self.model)

            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                self.current_weights_stats = self._collect_pp_dicts(self.current_weights_stats)

            for module_name, stats in self.current_weights_stats.items():
                self.record_stats(module_name, stats, "weights_stats", self.logs)

        self._save_logs()

    def on_eval_start(self, context: CallbackContext):
        if not self.enabled:
            return
        self.logs["val_loss_batch"].append([])

    def on_eval_end(self, context: CallbackContext):
        if not self.enabled:
            return

        # Compute epoch validation loss
        if self.logs["val_loss_batch"] and self.logs["val_loss_batch"][-1]:
            losses = self.logs["val_loss_batch"][-1]
            epoch_loss = sum(losses) / len(losses)
            self.logs["val_loss_epoch"].append(epoch_loss)
