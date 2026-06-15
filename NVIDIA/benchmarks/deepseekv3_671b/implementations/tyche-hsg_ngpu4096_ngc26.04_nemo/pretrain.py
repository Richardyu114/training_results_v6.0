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

import hydra
import torch
from callback_logging import DeltaTimingCallback, MLPerfLoggingCallback, mllogger
from callback_warmup import WarmupCallback
from callback_memory_profiler import MemoryProfileCallback
from megatron.bridge.recipes.deepseek import deepseek_v3_pretrain_config
from megatron.bridge.recipes.deepseek.deepseek_v3 import (
    set_deepseek_v3_pipeline_model_parallel_layout,
)
from megatron.bridge.training.config import (
    CheckpointConfig,
    CommOverlapConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    DistributedInitConfig,
    GPTDatasetConfig,
    LoggerConfig,
    ProfilingConfig,
)
from megatron.bridge.training.flex_dispatcher_backend import (
    apply_flex_dispatcher_backend,
)
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from omegaconf import OmegaConf
from utils import get_rank, init_logging

OmegaConf.register_new_resolver("add", lambda x, y: x + y)
OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)
OmegaConf.register_new_resolver("ceil_div", lambda x, y: (x + y - 1) // y)
OmegaConf.register_new_resolver("floor_div", lambda x, y: x // y)
OmegaConf.register_new_resolver("div", lambda x, y: x / y)
OmegaConf.register_new_resolver("if", lambda x, y, z: y if x else z)
OmegaConf.register_new_resolver("lt", lambda x, y: x < y)
OmegaConf.register_new_resolver("eq", lambda x, y: x == y)
OmegaConf.register_new_resolver("neq", lambda x, y: x != y)
OmegaConf.register_new_resolver("or", lambda *args: any(args))
OmegaConf.register_new_resolver("min", lambda x, y: min(x, y))
OmegaConf.register_new_resolver("floor", lambda x: int(x // 1))


def get_data(config):
    # Parameters
    seq_length = config.model.encoder_seq_length
    seed = config.model.seed

    if config.data.mock_dataset:
        from megatron.bridge.training.config import MockGPTDatasetConfig
        return MockGPTDatasetConfig(
            seq_length=seq_length,
            dataloader_type=config.data.dataloader_type,
            random_seed=seed,
            num_workers=config.data.num_workers,
            path_to_cache=None,
            split="900,50,50",
            data_sharding=True,
            create_attention_mask=False,
            reset_position_ids=config.data.reset_position_ids,
            reset_attention_mask=config.data.reset_attention_mask,
            eod_mask_loss=config.data.eod_mask_loss,
        )

    val_test_path = "/preproc_data/c4-validation-91205-samples.en_text_document"

    is_8b_dataset = os.getenv("DATASET") == "8b"
    r = [6] if is_8b_dataset else [6, 7]
    train_datasets = [f"/preproc_data/c4-train.en_{idx}_text_document" for idx in r]
    train_datasets_weights = [50] * len(r)

    data_paths = [(train_datasets, train_datasets_weights), ([val_test_path], None), ([val_test_path], None)]

    return GPTDatasetConfig(
        dataloader_type=config.data.dataloader_type,
        blend_per_split=data_paths,
        sequence_length=seq_length,
        random_seed=seed,
        num_workers=config.data.num_workers,
        path_to_cache=config.data.path_to_cache,
        defer_npy_index_mmap=True,
        fast_cache_load=True,
        reset_position_ids=config.data.reset_position_ids,
        reset_attention_mask=config.data.reset_attention_mask,
        eod_mask_loss=config.data.eod_mask_loss,
    )


def log_hyperparams(config, mbridge_config: ConfigContainer):
    bmark = mllogger.constants.DEEPSEEKV3_671B
    if config.trainer.max_steps <= config.optim.sched.lr_warmup_iters:
        config.optim.sched.lr_warmup_iters = max(config.trainer.max_steps // 2, 1)
    mllogger.mlperf_submission_log(bmark)

    # Collects configs to be logged
    logging_configs = {
        # seeds
        mllogger.constants.SEED: config.model.seed,
        # HPs
        mllogger.constants.MAX_STEPS: config.trainer.max_steps,
        mllogger.constants.GLOBAL_BATCH_SIZE: config.model.global_batch_size,
        mllogger.constants.GRADIENT_ACCUMULATION_STEPS: (int(os.environ["MINIBS"]) / config.model.micro_batch_size),
        mllogger.constants.MAX_SEQUENCE_LENGTH: config.model.encoder_seq_length,
        mllogger.constants.EVAL_SAMPLES: config.trainer.limit_val_batches * config.trainer.eval_batch_size,
        mllogger.constants.TRAIN_SAMPLES: 1574207408,
        mllogger.constants.INIT_CHECKPOINT_STEP: 0,
        mllogger.constants.MOE_AUX_LOSS_COEFF: config.model.moe_aux_loss_coeff,
        # Optimizers
        mllogger.constants.OPT_NAME: mllogger.constants.ADAMW,
        mllogger.constants.OPT_BASE_LR: mbridge_config.optimizer.lr,
        mllogger.constants.OPT_ADAMW_BETA_1: mbridge_config.optimizer.adam_beta1,
        mllogger.constants.OPT_ADAMW_BETA_2: mbridge_config.optimizer.adam_beta2,
        mllogger.constants.OPT_ADAMW_EPSILON: mbridge_config.optimizer.adam_eps,
        mllogger.constants.OPT_ADAMW_WEIGHT_DECAY: mbridge_config.optimizer.weight_decay,
        mllogger.constants.OPT_GRADIENT_CLIP_NORM: mbridge_config.optimizer.clip_grad,
        # Schedulers
        mllogger.constants.OPT_END_LR: mbridge_config.optimizer.min_lr,
        mllogger.constants.OPT_LR_WARMUP_STEPS: mbridge_config.scheduler.lr_warmup_iters,
        mllogger.constants.OPT_LR_DECAY_STEPS: config.optim.sched.lr_decay_iters,
        mllogger.constants.OPT_LR_DECAY_SCHEDULE: "cosine with linear warmup",
        # custom
        "target_accuracy": config.custom.target_log_ppl,
    }

    for key, value in logging_configs.items():
        mllogger.event(key=key, value=value)

def get_mixed_precision(cfg):
    from megatron.bridge.training.mixed_precision import (
        bf16_mixed,
        bf16_with_mxfp8_mixed,
        bf16_with_nvfp4_mixed,
    )
    fp8 = cfg.model.fp8
    fp4 = cfg.model.fp4
    # Assert that fp8 and fp4 cannot both be true
    assert not (fp8 and fp4), "Both fp8 and fp4 cannot be True at the same time."

    if fp8:
        precision_config = bf16_with_mxfp8_mixed()
        precision_config.fp8_param_gather = cfg.optim.fp8_param_gather
        precision_config.reuse_grad_buf_for_mxfp8_param_ag = cfg.optim.reuse_grad_buf_for_mxfp8_param_ag
        precision_config.fp8_dot_product_attention = cfg.model.fp8_dot_product_attention
    elif fp4:
        precision_config = bf16_with_nvfp4_mixed()
        precision_config.fp8_dot_product_attention = cfg.model.fp8_dot_product_attention
        precision_config.fp4_param_gather = cfg.optim.fp4_param_gather
    else:
        precision_config = bf16_mixed()
        if cfg.model.fp8_dot_product_attention:
            raise ValueError("fp8_dot_product_attention (FP8_DPA) is only supported for mxfp8 and nvfp4.")
    return precision_config


def _parse_memory_profiler_step(value):
    """Normalize Hydra/oc.env values for MemoryProfileCallback start_step / end_step."""
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("", "null", "none", "~"):
            return None
        return int(s)
    return int(value)


def override_logger_config(hydra_config, logger_config: LoggerConfig) -> LoggerConfig:
    logger_config.log_interval = hydra_config.trainer.log_every_n_steps
    # Enable tensorboard logging
    if hydra_config.model.nsys_profile.use_pytorch_profiler:
        pytorch_profile_dir = os.path.join(hydra_config.exp_manager.explicit_log_dir, "pytorch_profile")
        logger_config.tensorboard_dir = pytorch_profile_dir
    return logger_config

def get_profiling_config(config):
    # Parse ranks_str from YAML (which reads PROFILE_RANKS env var) into list of integers
    pytorch_profile_dir = os.path.join(config.exp_manager.explicit_log_dir, "pytorch_profile")
    ranks_str = config.model.nsys_profile.ranks_str
    profile_ranks = [0]
    if ranks_str and ranks_str.strip():
        profile_ranks = [int(x.strip()) for x in ranks_str.split(",") if x.strip()]
    # Memory profiling is driven by MemoryProfileCallback, not MBridge. Do not set
    # record_memory_history=True in ProfilingConfig, or MBridge will also try to
    # save memory snapshots and can conflict with the callback.
    return ProfilingConfig(
        use_nsys_profiler=config.model.nsys_profile.enabled,
        use_pytorch_profiler=config.model.nsys_profile.use_pytorch_profiler,
        record_memory_history=False,
        memory_snapshot_path=os.path.join(pytorch_profile_dir, f"{config.model.nsys_profile.memory_profiler.file_prefix}.pickle"),
        profile_step_start=config.model.nsys_profile.start_step,
        profile_step_end=config.model.nsys_profile.end_step,
        profile_ranks=profile_ranks,
        record_shapes=config.model.nsys_profile.gen_shape,
        nvtx_ranges=config.model.nsys_profile.nvtx_ranges,
    )

def override_performance_configs(cfg):
    # (TODO:) Override for fast validation. Set this through yaml configs later.
    # disable ddp checks
    cfg.ddp.check_for_nan_in_grad = False
    cfg.ddp.check_for_large_grads = False
    cfg.rerun_state_machine.check_for_nan_in_loss = False
    # disable fp32 grad reduce
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.mixed_precision.grad_reduce_in_fp32 = False

    # enable overlap; disable overlap_param_gather for mxfp8 accuracy
    cfg.comm_overlap.overlap_grad_reduce = True

    # cfg.comm_overlap.overlap_param_gather = False
    # cfg.ddp.overlap_param_gather = False
    # cfg.optimizer.overlap_param_gather = False

    # cfg.optimizer.overlap_param_gather_with_optimizer_step = True
    # cfg.comm_overlap.overlap_param_gather_with_optimizer_step = True

    # reset manaual GC interval in training to 100
    cfg.train.manual_gc_interval = 100

    # (TODO:) Best configs from NeMo, verify with real dataset later.
    cfg.dataset.num_workers = 0
    cfg.dataset.pin_memory = False

    # keep it as True for now. But seems stale
    cfg.dist.enable_megatron_core_experimental = True


def get_tokenizer_config(cfg):
    if cfg.data.mock_dataset:
        return TokenizerConfig(
            tokenizer_type="NullTokenizer",
            vocab_size=cfg.data.mock_tokenizer_vocab_size,
        )

    return TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=cfg.model.tokenizer,
        hf_tokenizer_kwargs={"use_fast": True},
    )


def override_comm_overlap_config(hydra_config, comm_overlap_config) -> CommOverlapConfig:
    comm_overlap_config.overlap_grad_reduce = hydra_config.optim.overlap_grad_reduce
    comm_overlap_config.overlap_param_gather = hydra_config.optim.overlap_param_gather
    comm_overlap_config.overlap_param_gather_with_optimizer_step = hydra_config.optim.overlap_param_gather_with_optim_step
    comm_overlap_config.align_param_gather = hydra_config.optim.align_param_gather
    comm_overlap_config.bucket_size = hydra_config.optim.bucket_size
    comm_overlap_config.overlap_moe_expert_parallel_comm = hydra_config.model.overlap_moe_expert_parallel_comm
    comm_overlap_config.delay_wgrad_compute = hydra_config.model.delay_wgrad_compute
    return comm_overlap_config


def override_ddp_config(hydra_config, ddp_config) -> DistributedDataParallelConfig:
    ddp_config.bucket_size = hydra_config.optim.bucket_size
    # Overlap
    ddp_config.overlap_grad_reduce = hydra_config.optim.overlap_grad_reduce
    ddp_config.overlap_param_gather = hydra_config.optim.overlap_param_gather
    ddp_config.align_param_gather = hydra_config.optim.align_param_gather
    # Distributed optimizer
    ddp_config.use_distributed_optimizer = hydra_config.optim.use_distributed_optimizer
    ddp_config.num_distributed_optimizer_instances = hydra_config.optim.num_distributed_optimizer_instances
    ddp_config.data_parallel_sharding_strategy = "optim_grads_params"
    ddp_config.outer_dp_sharding_strategy = hydra_config.optim.outer_dp_sharding_strategy

    # Assert if outer_dp sharding strategy is "optim" and
    # num_distributed_optimizer_instances is 1.
    if (hydra_config.optim.outer_dp_sharding_strategy == "optim"
        and not hydra_config.optim.num_distributed_optimizer_instances > 1):
        raise ValueError(
            "Number of distributed optimizer instances must be greater than 1 "
            "when outer_dp_sharding_strategy is 'optim'."
        )

    # FP8
    ddp_config.fp8_param_gather = hydra_config.optim.fp8_param_gather
    ddp_config.reuse_grad_buf_for_mxfp8_param_ag = hydra_config.optim.reuse_grad_buf_for_mxfp8_param_ag
    # FSDP
    ddp_config.fsdp_double_buffer = hydra_config.optim.nccl_ub_dp
    ddp_config.use_megatron_fsdp = hydra_config.model.mcore_fsdp
    ddp_config.fsdp_all_gather_in_start_param_sync = hydra_config.model.fsdp_all_gather_in_start_param_sync
    # NCCL
    ddp_config.nccl_ub = hydra_config.optim.nccl_ub_dp
    ddp_config.fsdp_manual_registration = hydra_config.optim.fsdp_manual_registration

    # Fuse scaling and reduction for gradient averaging.
    ddp_config.average_in_collective = hydra_config.model.average_in_collective

    return ddp_config

def override_dist_config(hydra_config, dist_config) -> DistributedInitConfig:
    dist_config.nccl_communicator_config_path = hydra_config.model.nccl_communicator_config_path
    dist_config.use_tp_pp_dp_mapping = hydra_config.model.use_tp_pp_dp_mapping
    dist_config.use_sharp = hydra_config.model.sharp
    dist_config.use_gloo_process_groups = False
    return dist_config

def override_checkpoint_config(hydra_config, checkpoint_config) -> CheckpointConfig:
    # LOAD_CHECKPOINT is the single env var; it is read in both model.resume_from_checkpoint (671b.yaml)
    # and exp_manager.load_checkpoint (deepseekv3_671b.yaml). We decide the load path in one place below.
    def _normalize(path):
        if path in (None, "", "null"):
            return None
        return path

    checkpoint_config.save = None
    checkpoint_config.fully_parallel_load = hydra_config.model.dist_ckpt_parallel_load
    checkpoint_config.dist_ckpt_strictness = "log_all"
    checkpoint_config.load_optim = False
    checkpoint_config.ckpt_format = hydra_config.model.dist_ckpt_format

    if hydra_config.exp_manager.save_interval > 0:
        checkpoint_config.save = os.path.join(hydra_config.exp_manager.explicit_log_dir, "checkpoints")
        checkpoint_config.save_optim = hydra_config.exp_manager.save_optim
        checkpoint_config.save_rng = hydra_config.exp_manager.save_rng_state
        checkpoint_config.save_interval = hydra_config.exp_manager.save_interval
        checkpoint_config.most_recent_k = hydra_config.exp_manager.most_recent_k
        checkpoint_config.async_save = hydra_config.exp_manager.async_save
    else:
        checkpoint_config.save = None

    # Single source for load path: when base dir is mounted use exp_manager.load_checkpoint (or /checkpoints);
    # otherwise use model.resume_from_checkpoint for 671b. Both are set from LOAD_CHECKPOINT in the yamls.
    if hydra_config.exp_manager.load_checkpoints_dir:
        load_path = _normalize(getattr(hydra_config.exp_manager, "load_checkpoint", None))
        checkpoint_config.load = load_path if load_path else "/checkpoints"
        checkpoint_config.ckpt_step = hydra_config.exp_manager.load_checkpoint_step
        checkpoint_config.load_optim = hydra_config.exp_manager.load_optim
        checkpoint_config.load_rng = hydra_config.exp_manager.load_rng_state
        checkpoint_config.exit_on_missing_checkpoint = False
    else:
        resume_from = _normalize(hydra_config.model.resume_from_checkpoint)
        checkpoint_config.load = resume_from if hydra_config.model.model_size == "671b" else None

    return checkpoint_config

def create_config(cfg):
    config = deepseek_v3_pretrain_config()
    model_cfg = config.model

    config.mixed_precision = get_mixed_precision(cfg)
    model_cfg.moe_flex_dispatcher_backend = cfg.model.moe_flex_dispatcher_backend
    model_cfg.mtp_num_layers = cfg.model.mtp_num_layers
    model_cfg.mtp_loss_scaling_factor = cfg.model.mtp_loss_scaling_factor

    # --- Model Configuration ---
    config.ddp = override_ddp_config(hydra_config=cfg, ddp_config=config.ddp)
    config.comm_overlap = override_comm_overlap_config(hydra_config=cfg, comm_overlap_config=config.comm_overlap)
    config.dist = override_dist_config(hydra_config=cfg, dist_config=config.dist)
    config.checkpoint = override_checkpoint_config(hydra_config=cfg, checkpoint_config=config.checkpoint)
    config.checkpoint.load_main_params_from_ckpt = cfg.optim.fp8_param_gather or config.mixed_precision.fp4_param_gather

    # Apply architecture overrides from model variant config
    arch_overrides = OmegaConf.select(cfg.model, 'architecture_overrides')
    if arch_overrides:
        for param, value in arch_overrides.items():
            # Convert OmegaConf containers to native Python types for Megatron compatibility
            if OmegaConf.is_config(value):
                value = OmegaConf.to_container(value, resolve=True)
            setattr(model_cfg, param, value)



    if model_cfg.virtual_pipeline_model_parallel_size == "null" or model_cfg.virtual_pipeline_model_parallel_size == 0:
        model_cfg.virtual_pipeline_model_parallel_size = None

    # --- Parallelism Settings ---
    model_cfg.tensor_model_parallel_size = cfg.model.tensor_model_parallel_size
    model_cfg.pipeline_model_parallel_size = cfg.model.pipeline_model_parallel_size
    model_cfg.virtual_pipeline_model_parallel_size = cfg.model.virtual_pipeline_model_parallel_size
    model_cfg.context_parallel_size = cfg.model.context_parallel_size
    model_cfg.expert_model_parallel_size = cfg.model.expert_model_parallel_size
    model_cfg.expert_tensor_parallel_size = cfg.model.expert_tensor_parallel_size
    model_cfg.sequence_parallel = cfg.model.tensor_model_parallel_size > 1
    set_deepseek_v3_pipeline_model_parallel_layout(model_cfg, cfg.model.pipeline_layout)

    # --- Model Settings ---
    model_cfg.seq_length = cfg.model.encoder_seq_length
    model_cfg.seed = cfg.model.seed
    model_cfg.te_rng_tracker = cfg.model.use_rng_tracker

    # --- Fusions ---
    model_cfg.fused_residual_rmsnorm = cfg.model.fused_residual_rmsnorm
    model_cfg.use_te_activation_func = cfg.model.use_te_activation_func
    if cfg.model.use_te_activation_func:
        model_cfg.bias_activation_fusion = False

    # --- MoE Settings ---
    model_cfg.moe_token_dispatcher_type = cfg.model.moe_token_dispatcher_type
    model_cfg.moe_grouped_gemm = cfg.model.moe_grouped_gemm
    model_cfg.moe_single_grouped_weight = cfg.model.moe_single_grouped_weight
    model_cfg.moe_single_grouped_bias = cfg.model.moe_single_grouped_bias
    model_cfg.moe_permute_fusion = cfg.model.moe_permute_fusion
    model_cfg.moe_router_fusion = cfg.model.moe_router_fusion
    model_cfg.overlap_dispatch_backward_with_experts_wgrad = cfg.model.overlap_dispatch_backward_with_experts_wgrad
    model_cfg.moe_router_force_load_balancing = cfg.model.moe_router_force_load_balancing
    model_cfg.moe_router_force_biased = cfg.model.moe_router_force_biased
    if cfg.model.moe_expert_rank_capacity_factor is not None:
        model_cfg.moe_expert_rank_capacity_factor = cfg.model.moe_expert_rank_capacity_factor
    model_cfg.moe_paged_stash = cfg.model.moe_paged_stash
    if cfg.model.moe_paged_stash_buffer_size_factor_cuda is not None:
        model_cfg.moe_paged_stash_buffer_size_factor_cuda = float(
            cfg.model.moe_paged_stash_buffer_size_factor_cuda
        )
    if cfg.model.moe_paged_stash_buffer_size_factor_cpu is not None:
        model_cfg.moe_paged_stash_buffer_size_factor_cpu = float(
            cfg.model.moe_paged_stash_buffer_size_factor_cpu
        )
    model_cfg.log_moe_overload_factor = cfg.model.log_moe_overload_factor
    model_cfg.moe_aux_loss_coeff = cfg.model.moe_aux_loss_coeff
    model_cfg.apply_rope_fusion = cfg.model.apply_rope_fusion
    model_cfg.moe_router_dtype = cfg.model.moe_router_dtype
    model_cfg.moe_hybridep_num_sms = cfg.model.moe_hybridep_num_sms
    model_cfg.moe_hybridep_num_sms_preprocessing = cfg.model.moe_hybridep_num_sms_preprocessing
    model_cfg.high_priority_a2a_comm_stream = cfg.model.high_priority_a2a_comm_stream
    apply_flex_dispatcher_backend(model_cfg, cfg.model.moe_flex_dispatcher_backend)
    model_cfg.fp8_output_proj = cfg.model.fp8_output_proj

    # CPU activation offloading config
    model_cfg.cpu_offloading = cfg.model.cpu_offloading
    model_cfg.cpu_offloading_num_layers = cfg.model.cpu_offloading_num_layers
    model_cfg.cpu_offloading_weights = False

    # Fine-grained activation offloading config
    model_cfg.fine_grained_activation_offloading = cfg.model.fine_grained_activation_offloading
    model_cfg.offload_modules = cfg.model.offload_modules.split(",") if cfg.model.offload_modules else []
    model_cfg.min_offloaded_tensor_size = cfg.model.min_offloaded_tensor_size

    # Pinned CPU buffers are retained after offloading and reused for the
    # next iteration. It is useful for cuda graph capture.
    model_cfg.cpu_offloading_retain_pinned_cpu_buffers = cfg.model.cpu_offloading_retain_pinned_cpu_buffers

    # Gradient accumulation fusion
    model_cfg.gradient_accumulation_fusion = cfg.model.gradient_accumulation_fusion

    # FSDP config parameters
    model_cfg.init_model_with_meta_device = cfg.model.mcore_fsdp
    model_cfg.moe_router_topk = cfg.model.moe_router_topk
    model_cfg.moe_router_num_groups = cfg.model.moe_router_num_groups
    model_cfg.moe_router_group_topk = cfg.model.moe_router_group_topk
    model_cfg.moe_router_topk_scaling_factor = cfg.model.moe_router_topk_scaling_factor

    # Handle recompute modules.
    # Set recompute_granularity to None if no modules are specified.
    model_cfg.recompute_modules = cfg.model.recompute_modules.split(",") if cfg.model.recompute_modules else []
    if not model_cfg.recompute_modules:
        model_cfg.recompute_granularity = None

    # CUDA graph settings
    if cfg.model.cuda_graph_implementation:
        model_cfg.cuda_graph_impl = cfg.model.cuda_graph_implementation
    if cfg.model.cuda_graph_scope:
        model_cfg.cuda_graph_scope = cfg.model.cuda_graph_scope
        model_cfg.cuda_graph_warmup_steps = cfg.model.cuda_graph_warmup_steps
    if cfg.model.use_dynamic_comp_stream:
        model_cfg.use_dynamic_comp_stream = cfg.model.use_dynamic_comp_stream


    # Fusion optimizations
    model_cfg.use_transformer_engine_op_fuser = cfg.model.use_transformer_engine_op_fuser
    if model_cfg.use_transformer_engine_op_fuser:
        # CuTe DSL grouped GEMM + SwiGLU kernel expects block interleaved layout
        model_cfg.moe_mlp_glu_interleave_size = 32
    if (cfg.model.fp8 or cfg.model.fp4) and model_cfg.moe_flex_dispatcher_backend != "hybridep":
        # Fused MoE router + padding
        model_cfg.moe_router_padding_for_quantization = True

    # --- Training Configuration ---
    train_cfg = config.train
    train_cfg.global_batch_size = cfg.model.global_batch_size
    train_cfg.micro_batch_size = cfg.model.micro_batch_size
    train_cfg.train_iters = cfg.trainer.max_steps
    train_cfg.eval_interval = cfg.trainer.val_check_interval
    train_cfg.eval_iters = cfg.trainer.limit_val_batches

    # --- Validation Configuration ---
    validation_cfg = config.validation
    validation_cfg.eval_global_batch_size = cfg.trainer.eval_batch_size

    # --- Optimizer Configuration ---
    optimizer_cfg = config.optimizer
    optimizer_cfg.lr = cfg.optim.lr
    optimizer_cfg.min_lr = cfg.optim.min_lr
    optimizer_cfg.adam_eps = cfg.optim.adam_eps
    optimizer_cfg.use_precision_aware_optimizer = cfg.optim.use_precision_aware_optimizer
    if not cfg.optim.use_precision_aware_optimizer:
        optimizer_cfg.main_grads_dtype=torch.float32
        optimizer_cfg.exp_avg_dtype=torch.float32
        optimizer_cfg.exp_avg_sq_dtype=torch.float32
    else:
        config.ddp.megatron_fsdp_use_decoupled_grad = True
    optimizer_cfg.optimizer_cuda_graph = cfg.optim.optimizer_cuda_graph
    optimizer_cfg.on_device_clip_grad = cfg.optim.on_device_clip_grad

    # --- Scheduler Configuration ---
    scheduler_cfg = config.scheduler
    if cfg.trainer.max_steps <= cfg.optim.sched.lr_warmup_iters:
        scheduler_cfg.lr_warmup_iters = max(cfg.trainer.max_steps // 2, 1)
    else:
        scheduler_cfg.lr_warmup_iters = cfg.optim.sched.lr_warmup_iters
    scheduler_cfg.lr_decay_iters = cfg.optim.sched.lr_decay_iters

    # --- RNG Configuration ---
    rng_cfg = config.rng
    rng_cfg.seed = cfg.model.seed
    rng_cfg.te_rng_tracker = cfg.model.use_rng_tracker

    # --- Dataset Configuration ---
    config.dataset = get_data(cfg)

    # --- Tokenizer Configuration ---
    config.tokenizer = get_tokenizer_config(cfg)

    config.logger = override_logger_config(cfg, config.logger)

    # set up profiling config
    config.profiling = get_profiling_config(cfg)

    override_performance_configs(config)


    return config


@hydra.main(config_path="conf", config_name="deepseekv3_671b", version_base="1.2")
def main(cfg):
    init_logging()
    OmegaConf.resolve(cfg)
    config = create_config(cfg)

    if get_rank() == 0:
        log_hyperparams(cfg, config)
        mllogger.start(key=mllogger.constants.INIT_START)

    callbacks = [DeltaTimingCallback(cfg), MLPerfLoggingCallback(cfg)]
    if cfg.trainer.warmup_train_steps > 0:
        callbacks.insert(0, WarmupCallback(cfg, forward_step_func=forward_step))
    if cfg.model.nsys_profile.memory_profiler.enable:
        pytorch_profile_dir = os.path.join(cfg.exp_manager.explicit_log_dir, "pytorch_profile")
        ranks_str = cfg.model.nsys_profile.ranks_str or "0"
        profile_ranks = [int(x.strip()) for x in ranks_str.split(",") if x.strip()]
        mem_cfg = cfg.model.nsys_profile.memory_profiler
        mem_start = _parse_memory_profiler_step(mem_cfg.start_step)
        mem_end = _parse_memory_profiler_step(mem_cfg.end_step)
        memory_callback = MemoryProfileCallback(
            enabled=True,
            file_prefix=mem_cfg.file_prefix,
            output_dir=pytorch_profile_dir,
            max_entries=1000000,
            profile_ranks=profile_ranks,
            start_step=mem_start,
            end_step=mem_end,
            force_oom_before_stop=False,
        )
        # Register first so on_train_start runs before WarmupCallback: _record_memory_history must
        # be active during warmup / graph capture (full-iteration CUDA graph), not only after it.
        callbacks.insert(0, memory_callback)

    # GC Config
    config.train.manual_gc = True
    config.train.manual_gc_interval = 500
    config.train.manual_gc_eval = False

    # Memory management
    config.train.empty_unused_memory_level = 0
    config.train.train_sync_interval = None

    # Skip numeric checks
    config.train.check_optimizer_step_success = False
    config.train.skip_sync_grad_norm_across_mp = True
    config.train.exit_signal_handler = False

    # Skip logging & timers
    config.logger.skip_train_metrics_log = True
    config.logger.timing_log_level = -1


    try:
        pretrain(config, forward_step_func=forward_step, callbacks=callbacks)
    except Exception:
        for callback in callbacks:
            cleanup = getattr(callback, "cleanup_after_oom", None)
            if callable(cleanup):
                try:
                    callback.cleanup_after_oom()
                except Exception:
                    pass
        raise

if __name__ == "__main__":
    main()
