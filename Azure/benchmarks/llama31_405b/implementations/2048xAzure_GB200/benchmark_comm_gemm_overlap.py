# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# Steps to generate report for TP overlap times.
# 1. Use any recent Llama3.1 container and Optimized repo clone.
# 2. Use the following batch script from optimized/llama31_405b_llm/pytorch:
# ```RUN_ONLY_COMM_GEMM_OVLP=1 COMM_GEMM_OVERLAP_TEST=1 ../../mlperf_utils/launch-mlperf-benchmark.sh --job-name comm-overlap-llama31 --container gitlab-master.nvidia.com/dl/mlperf/optimized:llama31_405b_llm.pytorch.xxxxxxxx --config config_DGXB100_apilog_1x8x10xtp4pp1cp2.sh --runsub run.sub --npar 1 --nexp 1 --slurm-extra=--comment=lustreclient=nopcc --partition batch --account coreai_mlperf_training```

import os
import subprocess
import argparse
from collections import defaultdict
import yaml
import json

from run_layer_with_overlap import _te_layer_argtype, _train
import transformer_engine.pytorch as te

import hydra
from omegaconf.omegaconf import OmegaConf

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

from nemo.collections.llm.gpt.model.llama import Llama31Config405B

###config: Namespace(layer_type=<class 'transformer_engine.pytorch.module.linear.Linear'>, batch_size=1, seq_length=4096, num_heads=128, head_dim=128, in_features=None, out_features=None, tp=None, seed=1234, fp8=True, fp8_init=True, tcp_init=True, bind_to_device=True, bootstrap_backend='mpi', use_cuda_graphs='True', ub_cfg=None, ub_name=None, benchmark=False, debug=False)
class layer_overlap_args:
    def __init__(self, cfg, layer_type):
        base_llama_config = Llama31Config405B()
        H = base_llama_config.hidden_size
        te_layer_name = 'linear' if layer_type in ['proj', 'fc2'] else 'layernormlinear'
        self.layer_type = _te_layer_argtype(te_layer_name)
        self.batch_size = cfg.model.micro_batch_size
        self.seq_length = cfg.model.encoder_seq_length // cfg.model.context_parallel_size
        self.num_heads = base_llama_config.num_attention_heads
        self.head_dim = H // self.num_heads
        self.tp = cfg.model.tensor_model_parallel_size
        self.seed = cfg.model.seed
        self.fp8 = True
        self.fp8_init = True
        self.use_bf16_params = True
        self.tcp_init = True
        self.bind_to_device = True
        self.bootstrap_backend='mpi'
        self.use_cuda_graphs=True
        self.ub_cfg = f'/workspace/llm/conf/tp_overlap/{os.getenv("GPU_ARCH","h100")}tp{self.tp}mbs{self.batch_size}.yaml'
        self.ub_name = layer_type
        self.debug=False
        self.benchmark=True
        self.skip_verify=True
#       TE 2.0 changes to enable after merging https://github.com/NVIDIA/TransformerEngine/pull/1529
        self.fp8_init = False
        self.linear_parallel_mode = 'column' if te_layer_name == 'layernormlinear' else 'row'
        self.overlap_rs_dgrad = False
        self.benchmark_iter = int(os.getenv('BENCHMARK_ITER','100'))
        self.num_layers = 1
        self.quantization='fp8_delayed_scaling'

        GQ = base_llama_config.num_query_groups
        ffn_hidden_size = base_llama_config.ffn_hidden_size
        self.features = {
            'qkv': [None, H+H*2*GQ//self.num_heads],
            'proj': [H, None],
            'fc1': [None, ffn_hidden_size*2],
            'fc2': [ffn_hidden_size, None]
        }
        self.in_features = self.features[layer_type][0] if layer_type in self.features else None
        self.out_features = self.features[layer_type][1] if layer_type in self.features else None
#        print (f'{layer_type} in_features {self.in_features} out_features {self.out_features}')

class comm_gemm_test:
    def __init__(self, cfg, layer_type):
        self.cfg = cfg
        self.layer_type = layer_type

        #parsing
        self.fprop_starts = []
        self.fprop_ends = []
        self.dgrad_starts = []
        self.dgrad_ends = []
        self.wgrad_starts = []
        self.wgrad_ends = []
        self.gemm_keys = ['xmma', 'cutlass', 'nvjet']
        self.comm_keys = ['userbuffers']

    def run(self):
        assert self.layer_type in ['qkv', 'proj', 'fc1', 'fc2']
        opts = layer_overlap_args(self.cfg, self.layer_type)
        tp_conf = {}
        def get_tp_conf(ub_name):
            if ub_name in self.cfg.model.ub_tp_comm_overlap_cfg:
                tp_conf[ub_name] = {}
                for k, v in self.cfg.model.ub_tp_comm_overlap_cfg[ub_name].items():
                    tp_conf[ub_name][k] = v
#                print (f'!! get_tp_conf {ub_name} method {tp_conf[ub_name]["method"]}')

        get_tp_conf (f'{self.layer_type}_fprop')
        get_tp_conf (f'{self.layer_type}_dgrad')
        get_tp_conf (f'{self.layer_type}_wgrad')

        file=open("/workspace/tp.yaml","w")
        yaml.dump(tp_conf,file)
        file.close()
        opts.ub_cfg = '/workspace/tp.yaml'
        _train(opts)

    def parse_ag_fprop(self, opts, layer_type, gpu_trace_data, tp_conf):
        assert f'{layer_type}_fprop' in tp_conf and f'{layer_type}_dgrad' in tp_conf and f'{layer_type}_wgrad' in tp_conf

        i = len(gpu_trace_data) - 1
        def find_wgrad_dgrad_time(i):
            assert tp_conf[f'{layer_type}_dgrad']['method'] == 'bulk'
            assert tp_conf[f'{layer_type}_wgrad']['method'] == 'bulk'
            cur_bulk_overlap_layer_name = ''
            gemm_count = 0
            comm_count = 0
            start_time = float("inf")
            end_time = 0
            # wgrad/dgrad has no Bulk overlap
            while i > 0 and (gemm_count < 2 or comm_count < 2):
                if any(key in gpu_trace_data[i]['Name'] for key in self.gemm_keys):
                    gemm_count += 1

                if any(key in gpu_trace_data[i]['Name'] for key in self.comm_keys):
                    comm_count += 1

                    if '_ag<' in gpu_trace_data[i]['Name']:
                        cur_bulk_overlap_layer_name = 'dgrad'
                    elif '_rs<' in gpu_trace_data[i]['Name']:
                        cur_bulk_overlap_layer_name = 'wgrad'
                    else:
                        assert False

                if any(key in gpu_trace_data[i]['Name'] for key in self.gemm_keys) or any(key in gpu_trace_data[i]['Name'] for key in self.comm_keys):
                    if gpu_trace_data[i]['Start (ns)'] < start_time:
                        start_time = gpu_trace_data[i]['Start (ns)']
                    if (gpu_trace_data[i]['Start (ns)'] + gpu_trace_data[i]['Duration (ns)']) > end_time:
                        end_time = gpu_trace_data[i]['Start (ns)'] + gpu_trace_data[i]['Duration (ns)']
                    assert abs(gemm_count - comm_count) <= 1
                    if gemm_count == comm_count:
                        if cur_bulk_overlap_layer_name == 'wgrad':
                            self.wgrad_starts.insert(0, start_time)
                            self.wgrad_ends.insert(0, end_time)
                        elif cur_bulk_overlap_layer_name == 'dgrad':
                            self.dgrad_starts.insert(0, start_time)
                            self.dgrad_ends.insert(0, end_time)
                        else:
                            assert False
                        start_time = float("inf")
                        end_time = 0
                        cur_bulk_overlap_layer_name = ''

                i -= 1
            assert i == 0 or all((gemm_count == 2, comm_count == 2, len(self.wgrad_starts) == len(self.dgrad_starts), len(self.wgrad_starts) == len(self.fprop_starts) + 1))
            return i

        def find_fprop_time(i):
            if i == 0:
                return i
            # wgrad has no TP overlap
            num_comm_chunks_remaining = (opts.tp - 1) * 2
            for n in range(opts.tp):
                while i > 0 and not any(key in gpu_trace_data[i]['Name'] for key in self.gemm_keys):
                    if any(key in gpu_trace_data[i]['Name'] for key in self.comm_keys):
                        num_comm_chunks_remaining -= 1
#                        print (f'i {i} num_comm_chunks_remaining {num_comm_chunks_remaining} kernel {gpu_trace_data[i]["Name"]} start {gpu_trace_data[i]["Start (ns)"]}')
                        assert num_comm_chunks_remaining >= 0
                    i -= 1
                assert i > 0
                if n==0:
                    self.fprop_ends.insert(0, gpu_trace_data[i]['Start (ns)'] + gpu_trace_data[i]['Duration (ns)'])
                if n==opts.tp-1:
                    self.fprop_starts.insert(0, gpu_trace_data[i]['Start (ns)'])
                i -= 1
            while num_comm_chunks_remaining > 0:
                while i > 0 and not any(key in gpu_trace_data[i]['Name'] for key in self.comm_keys):
                    i -= 1
                assert i > 0
                self.fprop_starts[0] = gpu_trace_data[i]['Start (ns)']
                num_comm_chunks_remaining -= 1
                i -= 1
            assert num_comm_chunks_remaining == 0
            return i

        iter_n = 1
        while i > 0:
            #print (f'{iter_n} - {i}')
            i = find_wgrad_dgrad_time(i)
            i = find_fprop_time(i)
            iter_n += 1
        wgrad = [self.wgrad_ends[j]-self.wgrad_starts[j] for j in range(1,len(self.wgrad_starts))]
        dgrad = [self.dgrad_ends[j]-self.dgrad_starts[j] for j in range(1,len(self.dgrad_starts))]
        fprop = [self.fprop_ends[j]-self.fprop_starts[j] for j in range(1,len(self.fprop_starts))]
        avg_wgrad = sum(wgrad) / len(wgrad)
        avg_dgrad = sum(dgrad) / len(dgrad)
        avg_fprop = sum(fprop) / len(fprop)
#        print (f'Wgrad_time {wgrad} Avg: {avg_wgrad}')
#        print (f'Dgrad_time {dgrad} Avg: {avg_dgrad}')
#        print (f'Fprop_time {fprop} Avg: {avg_fprop}')
        return avg_fprop, avg_dgrad, avg_wgrad

    def parse_rs_fprop(self, opts, layer_type, gpu_trace_data, tp_conf):
        assert f'{layer_type}_fprop' in tp_conf and f'{layer_type}_dgrad' in tp_conf

        i = len(gpu_trace_data) - 1
        def find_wgrad_time(i):
            # wgrad has no TP overlap
            while i > 0 and not any(key in gpu_trace_data[i]['Name'] for key in self.gemm_keys):
                i -= 1
            if i > 0:
                self.wgrad_starts.insert(0, gpu_trace_data[i]['Start (ns)'])
                self.wgrad_ends.insert(0, self.wgrad_starts[0] + gpu_trace_data[i]['Duration (ns)'])
                i -= 1
            return i

        def find_dgrad_time(i):
            if i == 0:
                return i
            # wgrad has no TP overlap
            num_comm_chunks_remaining = (opts.tp - 1) * 2
            for n in range(opts.tp):
                while i > 0 and not any(key in gpu_trace_data[i]['Name'] for key in self.gemm_keys):
                    if any(key in gpu_trace_data[i]['Name'] for key in self.comm_keys):
                        num_comm_chunks_remaining -= 1
#                        print (f'i {i} num_comm_chunks_remaining {num_comm_chunks_remaining} kernel {gpu_trace_data[i]["Name"]} start {gpu_trace_data[i]["Start (ns)"]}')
                        assert num_comm_chunks_remaining >= 0
                    i -= 1
                assert i > 0
                if n==0:
                    self.dgrad_ends.insert(0, gpu_trace_data[i]['Start (ns)'] + gpu_trace_data[i]['Duration (ns)'])
                if n==opts.tp-1:
                    self.dgrad_starts.insert(0, gpu_trace_data[i]['Start (ns)'])
                i -= 1
            while num_comm_chunks_remaining > 0:
                while i > 0 and not any(key in gpu_trace_data[i]['Name'] for key in self.comm_keys):
                    i -= 1
                assert i > 0
                self.dgrad_starts[0] = gpu_trace_data[i]['Start (ns)']
                num_comm_chunks_remaining -= 1
                i -= 1
            assert num_comm_chunks_remaining == 0
            return i

        def find_fprop_time(i):
            if i == 0:
                return i
            # wgrad has no TP overlap
            tp_key = f'{layer_type}_fprop'
            if tp_conf[tp_key]['method'] == 'pipeline':
                assert 'num_splits' in tp_conf[tp_key], f'num_splits not found for {tp_key} PipelinedRS overlap'
                N = tp_conf[tp_key]['num_splits']
                if 'atomic_gemm' in tp_conf[tp_key] and tp_conf[tp_key]['atomic_gemm'] == 1:
                    N = 1
                end_keys = self.comm_keys
            else:
                N = opts.tp
                end_keys = ['reduce_bf16', 'reduce_fp8']
#            print (f'end_keys {end_keys}')
            while i > 0 and not any(key in gpu_trace_data[i]['Name'] for key in end_keys):
                i -= 1
            assert i > 0
#            print (f'fprop end ind {i}-{gpu_trace_data[i]["Start (ns)"]}')
            self.fprop_ends.insert(0, gpu_trace_data[i]['Start (ns)'] + gpu_trace_data[i]['Duration (ns)'])
            i -= 1
            for n in range(N):
                while i > 0 and not any(key in gpu_trace_data[i]['Name'] for key in self.gemm_keys):
                    i -= 1
                assert i > 0
                if n==N-1:
                    self.fprop_starts.insert(0, gpu_trace_data[i]['Start (ns)'])
#                    print (f'fprop start ind {i}-{gpu_trace_data[i]["Start (ns)"]}')
                i -= 1
            return i
        iter_n = 1
        while i > 0:
#            print (f'{iter_n} - {i}')
            i = find_wgrad_time(i)
            i = find_dgrad_time(i)
            i = find_fprop_time(i)
            iter_n += 1

        wgrad = [self.wgrad_ends[j]-self.wgrad_starts[j] for j in range(1,len(self.wgrad_starts))]
        dgrad = [self.dgrad_ends[j]-self.dgrad_starts[j] for j in range(1,len(self.dgrad_starts))]
        fprop = [self.fprop_ends[j]-self.fprop_starts[j] for j in range(1,len(self.fprop_starts))]
        avg_wgrad = sum(wgrad) / len(wgrad)
        avg_dgrad = sum(dgrad) / len(dgrad)
        avg_fprop = sum(fprop) / len(fprop)
        #print (f'Wgrad_time {wgrad} Avg: {avg_wgrad}')
        #print (f'Dgrad_time {dgrad} Avg: {avg_dgrad}')
        #print (f'Fprop_time {fprop} Avg: {avg_fprop}')
        return avg_fprop, avg_dgrad, avg_wgrad

    def parse_report(self):
        assert self.layer_type in ['qkv', 'proj', 'fc1', 'fc2', 'all']
        opts = layer_overlap_args(self.cfg, self.layer_type)
        if self.layer_type == 'all':
            layer_types = ['qkv', 'proj', 'fc1', 'fc2']
        else:
            layer_types = [self.layer_type]

        def get_tp_conf(ub_name):
            if ub_name in self.cfg.model.ub_tp_comm_overlap_cfg:
                tp_conf[ub_name] = {}
                for k, v in self.cfg.model.ub_tp_comm_overlap_cfg[ub_name].items():
                    tp_conf[ub_name][k] = v
#                print (f'!! get_tp_conf {ub_name} method {tp_conf[ub_name]["method"]}')

        trace_dir = os.getenv('CUDA_GPU_TRACE_DIR', '.')

        node_name = os.getenv('SLURMD_NODENAME', '')
        node_id = os.getenv('SLURM_NODEID', '0')
        proc_id = os.getenv('SLURM_PROCID', '0')
        for file in os.listdir(trace_dir):
            if file.startswith(f'{layer_types[0]}') and file.endswith(f'r{proc_id}_cuda_gpu_trace.json'):
                spl = file.split('_')
                if spl[1] != node_name or spl[2] != node_id:
                    #print (f'!! current node {node_name} remap to {spl[1]}')
                    node_name = spl[1]
                    node_id = spl[2]

        result = ['CGoverlap', f'r{proc_id}', node_name]
        layer_names = ['CGoverlap']
        shapes = {'CGoverlap':''}
        for i, layer_type in enumerate(layer_types):
            self.fprop_starts.clear()
            self.fprop_ends.clear()
            self.dgrad_starts.clear()
            self.dgrad_ends.clear()
            self.wgrad_starts.clear()
            self.wgrad_ends.clear()

            H = opts.head_dim * opts.num_heads
            input_size = opts.batch_size * opts.seq_length
            in_features = opts.features[layer_type][0] if layer_type in opts.features else None
            out_features = opts.features[layer_type][1] if layer_type in opts.features else None
            in_features = in_features if in_features is not None else H
            out_features = out_features if out_features is not None else H
            out_features = out_features // opts.tp if layer_type in ['qkv', 'fc1'] else out_features
            in_features = in_features // opts.tp if layer_type in ['proj', 'fc2'] else in_features

            tp_conf = {}
            get_tp_conf (f'{layer_type}_fprop')
            get_tp_conf (f'{layer_type}_dgrad')
            get_tp_conf (f'{layer_type}_wgrad')

            #print (f'TP_CONF {tp_conf}-{type(tp_conf)}\n{OmegaConf.to_yaml(tp_conf)}')
            report = f'{trace_dir}/{layer_type}_{node_name}_{node_id}_r{proc_id}_cuda_gpu_trace.json'
            # Reading JSON GPU report file
            with open(report, "r") as gpu_trace_file:
                gpu_trace_data = json.load(gpu_trace_file)

            # Reading TP conf yaml
            #with open('/workspace/tp.yaml', "r") as tp_file:
            #    tp_conf = yaml.safe_load(tp_file)
            for conf in tp_conf:
                assert 'aggregate' not in conf

            #print (f'JSON len:{len(gpu_trace_data)} {gpu_trace_data[0]} yaml {type(tp_conf)}-{tp_conf}')
            if layer_type in ['qkv', 'fc1']:
                fprop, dgrad, wgrad = self.parse_ag_fprop(opts, layer_type, gpu_trace_data, tp_conf)
            else:
                 fprop, dgrad, wgrad = self.parse_rs_fprop(opts, layer_type, gpu_trace_data, tp_conf)

            shapes[layer_type] = f'{out_features} x {input_size} x {in_features}'
            layer_names.append(f'{layer_type}_fprop')
            result.append (round(fprop/1000000,3))
            layer_names.append(f'{layer_type}_dgrad')
            result.append (round(dgrad/1000000,3))
            layer_names.append(f'{layer_type}_wgrad')
            result.append (round(wgrad/1000000,3))
            #print (f'!! {layer_type}_fprop {out_features} x {input_size} x {in_features} AvgTime {round(fprop/1000000,3)} ms')
            #print (f'!! {layer_type}_dgrad {in_features} x {input_size} x {out_features} AvgTime {round(dgrad/1000000,3)} ms')
            #print (f'!! {layer_type}_wgrad {in_features} x {out_features} x {input_size} AvgTime {round(wgrad/1000000,3)} ms')
        if proc_id == '0':
            print (layer_names)
            print (shapes)
        print (result)


@hydra.main(
    config_path="/workspace/llm/conf", config_name="llama31_config_custom", version_base="1.2"
)
def main(cfg):
#    print (f'\n{OmegaConf.to_yaml(cfg)}')

    layer_type = os.getenv('LAYER', '')
    action = os.getenv('ACTION', 'run')
    assert action in ['run', 'parse']

    if layer_type != '' and cfg.model.tensor_model_parallel_size > 1 and cfg.model.ub_tp_comm_overlap:
        test = comm_gemm_test(cfg, layer_type)
        if action == 'run':
            test.run()
        else:
            test.parse_report()
    else:
        print (f'Not running comm_gemm overlap tests due to config settings: ub_tp_comm_overlap {cfg.model.ub_tp_comm_overlap} or tensor_model_parallel_size {cfg.model.tensor_model_parallel_size} or LAYER not set')

if __name__ == "__main__":
    main()


