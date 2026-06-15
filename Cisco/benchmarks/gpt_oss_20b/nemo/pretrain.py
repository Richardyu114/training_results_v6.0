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
import utils


def _parse_cuda_graph_scope(value):
    if value is None:
        return None
    if isinstance(value, str):
        tokens = [t.strip() for t in value.replace(",", " ").split() if t.strip()]
        if len(tokens) > 1:
            return tokens
        return tokens[0] if tokens else None
    return value


from callback_warmup import WarmupCallback, create_mock_dataset_config
from callback_logging import (
    DeltaTimingCallback,
    MLPerfLoggingCallback,
    PeriodicCacheResetCallback,
    mllogger,
)
from megatron.bridge.models import GPTModelProvider
from megatron.bridge.training.config import (
    CheckpointConfig,
    CommOverlapConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    DistributedInitConfig,
    GPTDatasetConfig,
    LoggerConfig,
    MixedPrecisionConfig,
    OptimizerConfig,
    ProfilingConfig,
    RerunStateMachineConfig,
    RNGConfig,
    SchedulerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.core.config import set_experimental_flag
from megatron.core.fusions.fused_bias_geglu import quick_gelu
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from omegaconf import OmegaConf


if os.getenv("MCORE_ENABLE_EXPERIMENTAL", "True").lower() == "true":
    set_experimental_flag(True)


if os.getenv("PATCH_GEGLU_GRAPH_SAFE", "False").lower() == "true":
    from megatron.core.fusions import fused_bias_geglu as _fbg
    from megatron.core.transformer.moe import experts as _experts

    def _graph_safe_weighted_bias_quick_geglu_impl(
        input, bias, weights, fp8_input_store=False, linear_offset=0.0, clamp_value=None,
    ):
        ori_shape = input.shape
        assert len(ori_shape) in [2, 3]
        if clamp_value is not None:
            x_glu, x_linear = input.chunk(2, -1)
            input = torch.cat(
                (
                    x_glu.clamp(min=None, max=clamp_value),
                    x_linear.clamp(min=-clamp_value, max=clamp_value),
                ),
                -1,
            )
        input = input.view(-1, ori_shape[-1])
        if not isinstance(linear_offset, torch.Tensor):
            linear_offset = input.new_full((), linear_offset)
        if bias is not None:
            output = _fbg.WeightedBiasQuickGeGLUFunction.apply(
                input, bias, weights, fp8_input_store, linear_offset
            )
        else:
            output = _fbg.WeightedQuickGeGLUFunction.apply(
                input, weights, fp8_input_store, linear_offset
            )
        return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)

    _fbg.weighted_bias_quick_geglu_impl = _graph_safe_weighted_bias_quick_geglu_impl
    if hasattr(_experts, "weighted_bias_quick_geglu_impl"):
        _experts.weighted_bias_quick_geglu_impl = _graph_safe_weighted_bias_quick_geglu_impl


if os.getenv("PATCH_TE_BACKWARD_DW_BIAS_NONE", "False").lower() == "true":
    from transformer_engine.pytorch.module import base as _te_base
    from transformer_engine.pytorch.utils import get_nvtx_range_context as _te_nvtx
    from transformer_engine.pytorch.module.base import noop_cat as _te_noop_cat

    def _safe_backward_dw(self):
        if not self.need_backward_dw():
            return
        with _te_nvtx(f"_{self.__class__.__name__}_wgrad"):
            (wgrad, bgrad), _ = self.wgrad_store.pop()
            if not self.fuse_wgrad_accumulation:
                weight_tensor = _te_noop_cat(self._get_weight_tensors())
                weight_tensor.grad = wgrad.to(weight_tensor.dtype)
            if self.use_bias:
                bias_tensor = _te_noop_cat(
                    [getattr(self, name) for name in self.bias_names]
                )
                if bias_tensor.grad is None and bgrad is not None:
                    bias_tensor.grad = bgrad.to(bias_tensor.dtype)
            del wgrad
            del bgrad
            self._trigger_wgrad_accumulation_and_reduce_hooks()

    _te_base.TransformerEngineBaseModule.backward_dw = _safe_backward_dw


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
    if config.model.data.mock_dataset:
        return create_mock_dataset_config(config)

    r = [6]
    train_datasets = [f"/preproc_data/c4-train.en_{idx}_text_document" for idx in r]
    train_datasets_weights = [50] * len(r)
    val_test_path = "/preproc_data/c4-validation-91205-samples.en_text_document"

    data_paths = [(train_datasets, train_datasets_weights), ([val_test_path], None), ([val_test_path], None)]

    return GPTDatasetConfig(
        blend_per_split=data_paths,
        sequence_length=config.model.encoder_seq_length,
        random_seed=config.model.seed,
        dataloader_type="single",
        num_workers=config.model.data.num_workers,
        path_to_cache="/npy_index",
        defer_npy_index_mmap=True,
        fast_cache_load=True,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask=False,
    )


def get_model(config):
    tp = config.model.tensor_model_parallel_size
    ep = config.model.expert_model_parallel_size
    pp = config.model.pipeline_model_parallel_size
    pp_dtype = torch.bfloat16 if config.model.pipeline_model_parallel_size != 1 else None
    vp = config.model.virtual_pipeline_model_parallel_size
    cp = config.model.context_parallel_size
    sp = config.model.sequence_parallel

    etp = 1

    asym_pp_embed = config.model.account_for_embedding_in_pipeline_split
    asym_pp_loss = config.model.account_for_loss_in_pipeline_split

    if config.model.overwritten_attributes.num_layers is not None:
        num_layers = config.model.overwritten_attributes.num_layers
    else:
        num_layers = 24

    model_cfg = GPTModelProvider(
        num_layers=num_layers,
        num_moe_experts=32,
        hidden_size=2880,
        num_attention_heads=64,
        num_query_groups=8,
        ffn_hidden_size=2880,
        kv_channels=64,
        normalization="RMSNorm",
        gated_linear_unit=True,
        add_bias_linear=config.model.add_bias_linear,
        share_embeddings_and_output_weights=False,
        vocab_size=128256,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        bias_activation_fusion=True,
        bias_dropout_fusion=False,
        activation_func_clamp_value=7.0,
        glu_linear_offset=1.0,
        window_attn_skip_freq=2,
        rotary_base=150000,
        yarn_rotary_scaling_factor=32.0,
        yarn_original_max_position_embeddings=4096,
        yarn_beta_fast=32.0,
        yarn_beta_slow=1.0,
        yarn_correction_range_round_to_int=False,
        moe_router_topk=4,
        moe_token_dispatcher_type=config.model.moe_token_dispatcher_type,
        moe_ffn_hidden_size=2880,
        moe_router_load_balancing_type=config.model.moe_router_load_balancing_type,
        window_size=(127, 0),
        softmax_type="learnable",
        activation_func=quick_gelu,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vp,
        expert_tensor_parallel_size=etp,
        expert_model_parallel_size=ep,
        context_parallel_size=cp,
        sequence_parallel=sp,
        cp_comm_type=config.model.cp_comm_type,
        moe_grouped_gemm=config.model.moe_grouped_gemm,
        moe_router_pre_softmax=config.model.moe_router_pre_softmax,
        moe_flex_dispatcher_backend=config.model.moe_flex_dispatcher_backend,
        moe_permute_fusion=config.model.moe_permute_fusion,
        moe_deepep_num_sms=config.model.moe_deepep_num_sms,
        moe_hybridep_num_sms=config.model.moe_hybridep_num_sms,
        moe_hybridep_num_sms_preprocessing=config.model.moe_hybridep_num_sms_preprocessing,
        moe_permute_fusion_into_hybridep=config.model.moe_permute_fusion_into_hybridep,
        moe_router_fusion=config.model.moe_router_fusion,
        moe_router_force_load_balancing=config.model.moe_router_force_load_balancing,
        pipeline_dtype=pp_dtype,
        account_for_embedding_in_pipeline_split=asym_pp_embed,
        account_for_loss_in_pipeline_split=asym_pp_loss,
        seq_length=config.model.encoder_seq_length,
        init_model_with_meta_device=config.model.mcore_fsdp,
        position_embedding_type=config.model.position_embedding_type,
        cross_entropy_loss_fusion=config.model.cross_entropy_loss_fusion,
        cross_entropy_fusion_impl=config.model.cross_entropy_fusion_impl,
        cuda_graph_impl=config.model.overwritten_attributes.cuda_graph_implementation,
        cuda_graph_scope=_parse_cuda_graph_scope(config.model.overwritten_attributes.cuda_graph_scope),
        fused_single_qkv_rope=config.model.fused_single_qkv_rope,
        gradient_accumulation_fusion=config.model.gradient_accumulation_fusion,
        tp_only_amax_red=config.model.tp_only_amax_red,
        use_te_rng_tracker=config.model.use_te_rng_tracker,
        use_transformer_engine_op_fuser=config.model.use_transformer_engine_op_fuser,
        cuda_graph_warmup_steps=config.model.custom.cuda_graph_warmup_steps,
        cpu_offloading=config.model.cpu_offloading,
        cpu_offloading_num_layers=config.model.cpu_offloading_num_layers,
        cpu_offloading_weights=False,
    )
    if config.model.use_transformer_engine_op_fuser:
        model_cfg.moe_mlp_glu_interleave_size = 32
    model_cfg.moe_router_padding_for_quantization = True
    if os.getenv("MOE_ROUTER_PADDING_FOR_FP8", "False").lower() == "true":
        model_cfg.moe_router_padding_for_fp8 = True
    if config.model.moe_expert_rank_capacity_factor is not None:
        model_cfg.moe_expert_rank_capacity_factor = config.model.moe_expert_rank_capacity_factor
    if config.model.moe_expert_capacity_factor is not None:
        model_cfg.moe_expert_capacity_factor = float(
            config.model.moe_expert_capacity_factor
        )
    if config.model.moe_pad_expert_input_to_capacity:
        model_cfg.moe_pad_expert_input_to_capacity = True
    model_cfg.moe_paged_stash = config.model.moe_paged_stash
    if config.model.moe_paged_stash_buffer_size_factor_cuda is not None:
        model_cfg.moe_paged_stash_buffer_size_factor_cuda = float(
            config.model.moe_paged_stash_buffer_size_factor_cuda
        )
    model_cfg.moe_single_grouped_weight = config.model.moe_single_grouped_weight_bias
    model_cfg.moe_single_grouped_bias = config.model.moe_single_grouped_weight_bias

    if os.getenv("MOE_EP_A2A_OVERLAP", "False").lower() == "true":
        model_cfg.tp_comm_bulk_wgrad = False
    return model_cfg


def get_mixed_precision(config):
    fp8_type = None
    if config.model.fp8:
        fp8_type = "hybrid" if config.model.fp8_hybrid else "e4m3"
    fp8_margin = 0
    fp8_amax_history_len = config.model.fp8_amax_history_len
    fp8_amax_compute_algo = config.model.fp8_amax_compute_algo
    fp8_param_gather = config.model.optim.fp8_param_gather
    fp8_recipe = config.model.fp8_recipe
    reuse_grad_buf_for_mxfp8_param_ag = config.model.reuse_grad_buf_for_mxfp8_param_ag
    fp4_type = None
    if config.model.fp4:
        fp4_type = "e2m1"
    fp4_recipe = config.model.fp4_recipe

    first_last_layers_bf16 = config.model.first_last_layers_bf16
    num_layers_at_start_in_bf16 = config.model.num_layers_at_start_in_bf16
    num_layers_at_end_in_bf16 = config.model.num_layers_at_end_in_bf16

    if config.model.fp8 or config.model.fp4:
        mixed_precision = MixedPrecisionConfig(
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=False,
            fp8=fp8_type,
            fp8_margin=fp8_margin,
            fp8_recipe=fp8_recipe,
            fp8_amax_history_len=fp8_amax_history_len,
            fp8_amax_compute_algo=fp8_amax_compute_algo,
            fp8_param_gather=fp8_param_gather,
            reuse_grad_buf_for_mxfp8_param_ag=reuse_grad_buf_for_mxfp8_param_ag,
            fp8_dot_product_attention=config.model.fp8_dot_product_attention,
            fp4=fp4_type,
            fp4_recipe=fp4_recipe,
            first_last_layers_bf16=first_last_layers_bf16,
            num_layers_at_start_in_bf16=num_layers_at_start_in_bf16,
            num_layers_at_end_in_bf16=num_layers_at_end_in_bf16,
        )
    else:
        mixed_precision = MixedPrecisionConfig(
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=False,
        )

    return mixed_precision


def get_ddp_config(config):
    overlap_grad_reduce = config.model.optim.overlap_grad_reduce
    overlap_param_gather = config.model.optim.overlap_param_gather
    align_param_gather = config.model.optim.align_param_gather
    use_distributed_optimizer = config.model.optim.use_distributed_optimizer
    bucket_size = config.model.optim.bucket_size
    num_distributed_optimizer_instances = config.model.optim.num_distributed_optimizer_instances
    fp8_param_gather = config.model.optim.fp8_param_gather
    nccl_ub_dp = config.model.optim.nccl_ub_dp
    outer_dp_sharding_strategy = config.model.optim.outer_dp_sharding_strategy

    dp_sharding_strategy = os.getenv("DP_SHARDING_STRATEGY", "optim_grads_params")
    average_in_collective = os.getenv("AVERAGE_IN_COLLECTIVE", "0") == "1"

    return DistributedDataParallelConfig(
        check_for_nan_in_grad=False,
        grad_reduce_in_fp32=False,
        average_in_collective=average_in_collective,
        bucket_size=bucket_size,
        overlap_grad_reduce=overlap_grad_reduce,
        overlap_param_gather=overlap_param_gather,
        align_param_gather=align_param_gather,
        use_distributed_optimizer=use_distributed_optimizer,
        num_distributed_optimizer_instances=num_distributed_optimizer_instances,
        data_parallel_sharding_strategy=dp_sharding_strategy,
        outer_dp_sharding_strategy=outer_dp_sharding_strategy,
        fp8_param_gather=fp8_param_gather,
        keep_fp8_transpose_cache=False,
        fsdp_double_buffer=True,
        use_megatron_fsdp=config.model.mcore_fsdp,
        nccl_ub=nccl_ub_dp,
    )


def get_dist_config(config):
    nccl_communicator_config_path = config.model.nccl_communicator_config_path
    return DistributedInitConfig(
        nccl_communicator_config_path=nccl_communicator_config_path,
        use_tp_pp_dp_mapping=config.model.use_tp_pp_dp_mapping,
        use_sharp=config.model.sharp,
        enable_megatron_core_experimental=os.getenv(
            "MCORE_ENABLE_EXPERIMENTAL", "True"
        ).lower() == "true",
    )


def get_optimizer_config(config):
    return OptimizerConfig(
        optimizer="adam",
        lr=config.model.optim.lr,
        weight_decay=0.1,
        bf16=config.trainer.precision == "bf16",
        fp16=config.trainer.precision == "fp16",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=config.model.optim.adam_eps,
        use_distributed_optimizer=True,
        clip_grad=1.0,
        min_lr=config.model.optim.sched.min_lr,
    )


def get_scheduler_config(config):
    return SchedulerConfig(
        lr_decay_style="cosine",
        lr_decay_iters=int(os.environ.get("OPT_LR_DECAY_STEPS", 0)) - int(config.model.optim.sched.lr_warmup_iters),
        lr_warmup_iters=config.model.optim.sched.lr_warmup_iters,
        start_weight_decay=0.1,
        end_weight_decay=0.1,
    )


def get_train_config(config):
    return TrainingConfig(
        global_batch_size=config.model.global_batch_size,
        micro_batch_size=config.model.micro_batch_size,
        rampup_batch_size=None,
        train_iters=config.trainer.max_steps,
        empty_unused_memory_level=int(os.getenv("EMPTY_UNUSED_MEMORY_LEVEL", "0")),
    )


def get_validation_config(config):
    return ValidationConfig(
        eval_interval=config.trainer.val_check_interval,
        eval_iters=config.trainer.eval_iters,
    )


def get_tokenizer_config(config):
    return TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=config.model.tokenizer.model,
        hf_tokenizer_kwargs={"use_fast": True},
    )


def get_rerun_state_machine_config(config):
    return RerunStateMachineConfig(
        rerun_mode="disabled",
        check_for_nan_in_loss=False,
        check_for_spiky_loss=False,
    )


def get_logger_config(config):
    return LoggerConfig(log_interval=config.trainer.max_steps + 1)


def get_rng_config(config):
    return RNGConfig(seed=config.model.seed, te_rng_tracker=config.model.use_te_rng_tracker)


def get_comm_overlap_config(config):
    tp_comm_overlap = config.model.ub_tp_comm_overlap

    tp_comm_overlap_cfg = None
    if tp_comm_overlap:
        from megatron.bridge.training.comm_overlap import (
            BulkOverlapCfg,
            PipelineOverlapCfg,
            RingExchangeOverlapCfg,
            TransformerLayerTPOverlapCfg,
        )

        buffer_options = config.model.ub_tp_comm_overlap_cfg
        userbuffer_args = {}

        for key in [
            "qkv_dgrad",
            "qkv_wgrad",
            "fc1_dgrad",
            "fc1_wgrad",
            "qkv_fprop",
            "proj_dgrad",
            "fc1_fprop",
            "fc2_dgrad",
            "proj_fprop",
            "fc2_fprop",
        ]:
            if key not in buffer_options:
                userbuffer_args[key] = None
                continue
            attributes = buffer_options[key]
            if attributes is None:
                userbuffer_args[key] = None
                continue
            fp8_buf = False
            try:
                fp8_buf = bool(attributes.fp8_buf)
            except:
                pass
            if attributes.method == "pipeline":
                userbuffer_args[key] = PipelineOverlapCfg(
                    num_sm=attributes.num_sm,
                    cga_size=attributes.cga_size,
                    num_splits=attributes.num_splits,
                    set_sm_margin=bool(attributes.set_sm_margin),
                    fp8_buf=fp8_buf,
                )
            elif attributes.method == "bulk":
                userbuffer_args[key] = BulkOverlapCfg(
                    num_sm=attributes.num_sm,
                    cga_size=attributes.cga_size,
                    set_sm_margin=bool(attributes.set_sm_margin),
                )
            elif attributes.method == "ring_exchange":
                userbuffer_args[key] = RingExchangeOverlapCfg(
                    fp8_buf=fp8_buf,
                )
            else:
                assert False, f"method {attributes.method} is not defined."

        tp_comm_overlap_cfg = TransformerLayerTPOverlapCfg(**userbuffer_args)

    moe_ep_a2a = os.getenv("MOE_EP_A2A_OVERLAP", "False").lower() == "true"

    return CommOverlapConfig(
        tp_comm_overlap=tp_comm_overlap,
        tp_comm_overlap_cfg=tp_comm_overlap_cfg,
        overlap_p2p_comm=config.model.overlap_p2p_comm,
        batch_p2p_comm=config.model.batch_p2p_comm,
        overlap_grad_reduce=config.model.optim.overlap_grad_reduce,
        overlap_param_gather=config.model.optim.overlap_param_gather,
        overlap_param_gather_with_optimizer_step=config.model.optim.overlap_param_gather_with_optim_step,
        align_param_gather=config.model.optim.align_param_gather,
        bucket_size=config.model.optim.bucket_size,
        defer_embedding_wgrad_compute=config.model.defer_embedding_wgrad_compute,
        wgrad_deferral_limit=config.model.wgrad_deferral_limit,
        overlap_moe_expert_parallel_comm=moe_ep_a2a,
        delay_wgrad_compute=moe_ep_a2a,
    )


def get_profiling_config(config):
    if config.misc.memory_profiler.enable:
        return ProfilingConfig(
            use_pytorch_profiler=True,
            record_memory_history=True,
            memory_snapshot_path=f"{config.misc.memory_profiler.file_prefix}.pickle",
            profile_step_start=config.misc.memory_profiler.start_step,
            profile_step_end=config.misc.memory_profiler.end_step,
        )
    return ProfilingConfig(
        use_nsys_profiler=config.model.nsys_profile.enabled,
        profile_step_start=config.model.nsys_profile.start_step,
        profile_step_end=config.model.nsys_profile.end_step,
        profile_ranks=config.model.nsys_profile.ranks,
        record_shapes=config.model.nsys_profile.gen_shape,
        nvtx_ranges=config.model.nsys_profile.nvtx_ranges,
    )


def get_checkpoint_config(config):
    return CheckpointConfig(
        load=None,
        save=None,
        fully_parallel_load=config.model.dist_ckpt_parallel_load,
        dist_ckpt_strictness="log_all",
        load_optim=False,
        load_main_params_from_ckpt=config.model.optim.fp8_param_gather,
    )


def create_config(config):
    return ConfigContainer(
        comm_overlap=get_comm_overlap_config(config),
        checkpoint=get_checkpoint_config(config),
        dataset=get_data(config),
        ddp=get_ddp_config(config),
        dist=get_dist_config(config),
        logger=get_logger_config(config),
        mixed_precision=get_mixed_precision(config),
        model=get_model(config),
        optimizer=get_optimizer_config(config),
        rng=get_rng_config(config),
        profiling=get_profiling_config(config),
        rerun_state_machine=get_rerun_state_machine_config(config),
        scheduler=get_scheduler_config(config),
        tokenizer=get_tokenizer_config(config),
        train=get_train_config(config),
        validation=get_validation_config(config),
    )


def log_hyperparams(config):
    mllogger.mlperf_submission_log("gpt_oss_20b")
    if config.trainer.max_steps <= config.model.optim.sched.lr_warmup_iters:
        config.model.optim.sched.lr_warmup_iters = config.trainer.max_steps

    opt_lr_decay_steps = int(os.environ.get("OPT_LR_DECAY_STEPS", 0)) - int(config.model.optim.sched.lr_warmup_iters)

    logging_configs = {
        mllogger.constants.SEED: config.model.seed,
        mllogger.constants.GLOBAL_BATCH_SIZE: config.model.global_batch_size,
        mllogger.constants.GRADIENT_ACCUMULATION_STEPS: (int(os.environ["MINIBS"]) / config.model.micro_batch_size),
        mllogger.constants.MAX_SEQUENCE_LENGTH: config.model.encoder_seq_length,
        mllogger.constants.EVAL_SAMPLES: int(os.environ.get("VAL_SAMPLES", 0)),
        mllogger.constants.TRAIN_SAMPLES: 1574207408,
        mllogger.constants.INIT_CHECKPOINT_STEP: config.model.custom.init_global_step,
        mllogger.constants.OPT_NAME: mllogger.constants.ADAMW,
        mllogger.constants.OPT_BASE_LR: config.model.optim.lr,
        mllogger.constants.OPT_ADAMW_BETA_1: 0.9,
        mllogger.constants.OPT_ADAMW_BETA_2: 0.95,
        mllogger.constants.OPT_ADAMW_EPSILON: 1e-5,
        mllogger.constants.OPT_ADAMW_WEIGHT_DECAY: 0.1,
        mllogger.constants.OPT_GRADIENT_CLIP_NORM: 1.0,
        mllogger.constants.OPT_END_LR: config.model.optim.sched.min_lr,
        mllogger.constants.OPT_LR_WARMUP_STEPS: config.model.optim.sched.lr_warmup_iters,
        mllogger.constants.OPT_LR_DECAY_STEPS: opt_lr_decay_steps,
        mllogger.constants.MAX_STEPS: int(os.environ.get("MAX_STEPS", 0)),
        mllogger.constants.OPT_LR_DECAY_SCHEDULE: "cosine with linear warmup",
        "target_accuracy": config.custom.target_log_ppl,
    }

    for key, value in logging_configs.items():
        mllogger.event(key=key, value=value)


@hydra.main(config_path="conf", config_name="gpt_oss", version_base="1.2")
def main(cfg):
    OmegaConf.resolve(cfg)
    config_container = create_config(cfg)
    if utils.rank == 0:
        log_hyperparams(cfg)
    callbacks = [
        WarmupCallback(cfg, forward_step_func=forward_step),
        DeltaTimingCallback(cfg),
        MLPerfLoggingCallback(cfg),
        PeriodicCacheResetCallback(cfg),
    ]
    pretrain(config_container, forward_step_func=forward_step, callbacks=callbacks)


if __name__ == "__main__":
    if utils.rank == 0:
        mllogger.start(key=mllogger.constants.INIT_START)
    main()
