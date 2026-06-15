#!/usr/bin/env python
from contextlib import nullcontext
from distributed_test import *
from argument import parser, TORCH_DTYPES

global parser

# from argument import parser
TORCH_COLL = {'all_reduce': dist.all_reduce,
              'reduce_scatter': dist.reduce_scatter_tensor,
              'all_gather': dist.all_gather_into_tensor}

data_option = {'all_reduce': [1, '1'],
               'reduce_scatter': [2, '1/n', '1'],
               'all_gather': [2, '1', '1/n']}

TORCH_OP = {'sum': dist.ReduceOp.SUM,
            'max': dist.ReduceOp.MAX,
            'min': dist.ReduceOp.MIN,
            'avg': dist.ReduceOp.AVG,
            'product': dist.ReduceOp.PRODUCT}


class nccl_test(distributed_test):
    def __init__(self, args: Sequence[Any]):
        super().__init__(args)
        self.args = args

    def get_data(self, coll_name: str, tensor_size: int, target_dtype: str):
        cur_option = data_option[coll_name]
        data_list = []
        base_nelem = tensor_size // self.dp_group_size
        for idx in range(cur_option[0]):
            if cur_option[idx + 1] == '1':
                data_list.append(
                    torch.ones([base_nelem * self.dp_group_size], dtype=target_dtype, device=torch.device(self.device)))
            elif cur_option[idx + 1] == '1/n':
                data_list.append(torch.ones([base_nelem], dtype=target_dtype, device=torch.device(self.device)))
        return data_list

    def get_factor(self, coll_name: str) -> float:
        if coll_name == 'all_reduce':
            return 2 * (self.dp_group_size - 1) / self.dp_group_size
        elif coll_name in ['all_gather', 'reduce_scatter']:
            return (self.dp_group_size - 1) / self.dp_group_size
        else:
            raise ValueError(f'{coll_name} is not supported yet or non-valid collective name')

    def run(self):
        target_dtype = TORCH_DTYPES[args.dtype]
        elem_size = torch.tensor([], dtype=target_dtype).element_size()
        torch.cuda.set_device(self.device)

        mincount = distributed_test.parse_size(elem_size, args.B)
        maxcount = distributed_test.parse_size(elem_size, args.E)
        p2pcount = distributed_test.parse_size(elem_size, args.p2p_size)

        coll = TORCH_COLL[args.coll.lower()]
        operator = None
        if 'reduce' in args.coll.lower():
            operator = TORCH_OP[args.op.lower()]

        if args.I and args.F:
            ValueError(f"increment{args.I} and factor{args.F} cannot be used at the same time")

        if mincount > maxcount:
            ValueError(f"minbytes {args.B} should be smaller than {args.E}")

        if args.I:
            increment = distributed_test.parse_size(elem_size, args.I)
            input_args = list(count for count in range(mincount, maxcount, args.I))
        else:
            factor = args.F
            if maxcount // mincount >= factor and factor > 1:
                input_args = list(mincount * factor ** i for i in
                                  range(0, (int)(math.floor((maxcount // mincount) ** (1. / factor)) + 1)))
            else:
                input_args = [mincount]
        if self.rank == 0:
            print(
                f"{'DP ':<15} {'TP ':<15} {'PP ':<15} {'message size(B)':<15} {'count(elements)':<15} {'data type':<15} {'redop':<15} {'algo BW(GB/s)':<15} {'bus BW(GB/s)':<15}")

        dist.barrier()
        delay = 0
        torch.cuda.synchronize()
        first_call = False
        prefix_msg = f"self.rank {self.rank}, (D,T,P)=({self.DP_RANK, self.TP_RANK, self.PP_RANK})"
        for nelems in input_args:
            data = self.get_data(args.coll, nelems,
                                 target_dtype)  # torch.ones([nelems], dtype=target_dtype, device=torch.device(self.device))  # self.get_data(args.coll)
            kwargs = {'group': self.CUR_GROUP,
                      'async_op': True}
            if operator is not None:
                kwargs['op'] = operator
            if len(args.pp_group) == 0 or self.pp_stage in args.pp_group:
                # A dummy collective to runtime initialization
                if first_call is False:
                    coll(*data, **kwargs)  # op=operator, group=self.CUR_GROUP)
                    torch.cuda.synchronize()
                    first_call = True

                start = time.time()
                for _ in range(args.witer):
                    coll(*data, **kwargs)  # op=operator, group=self.CUR_GROUP)
                torch.cuda.synchronize()
                end = time.time()
                delay = ((end - start) / args.witer)

            avg_delay = torch.tensor([delay], dtype=TORCH_DTYPES['float'], device=torch.device(self.device))
            dist.all_reduce(avg_delay, op=dist.ReduceOp.AVG)
            delay = avg_delay.cpu().detach().numpy()[0] * (1 - args.overlap)
            if args.verbose and self.rank == 0:
                print(f"msg size: {nelems * elem_size}, delay: {delay}")
            total = 0
            step_time_list = []
            p2p_recv_data = torch.ones([p2pcount], dtype=target_dtype, device=torch.device(self.device))
            p2p_send_data = torch.ones([p2pcount], dtype=target_dtype, device=torch.device(self.device))

            timers = {}
            torch.cuda.synchronize()
            # This represents the cooldown phase of pipeline parallelism when args.pp > 1
            total_niter = niter = args.niter
            coll_iter = 1
            # to hide runtime overhead of pytorch
            if args.coll_only:
                coll_iter = 20 if niter > 20 else 1
                niter = niter // coll_iter
                total_niter = niter * coll_iter

            for event in ['p2p', 'coll', 'iter']:
                timers[event] = Timer(total_niter)

            comm_stream = torch.cuda.Stream()

            for i in range(niter):
                # Mimic the pipelining in 3D parallelism
                # The following does nothing because pytorch uses a stream pool for nccl kerenels regardless of this clause
                # Using individual async routines instead of batch with refernece to the following nvbug
                # https://nvbugswb.nvidia.com/NvBugs5/SWBug.aspx?bugid=3957578&cmtNo=
                with torch.cuda.stream(comm_stream) if args.single_stream else nullcontext():
                    timers['iter'].start()
                    timers['p2p'].start()
                    if not args.coll_only:
                        ops = []
                        torch.cuda.nvtx.range_push(prefix_msg + f'p2p[{i}]')
                        if self.prev_pprank is not None:
                            recv_op = dist.P2POp(torch.distributed.irecv, p2p_recv_data, self.prev_pprank, tag=i)
                            ops.append(recv_op)
                            # TODO: This runs on the host side so not accurately mimic the computation in pipelined-parallel DL apps.
                            # Need a custom kernel to make busy-waiting on GPU side
                            if delay > 0:
                                time.sleep(delay)  # waiting to simulate computation
                        if self.next_pprank is not None:
                            send_op = dist.P2POp(torch.distributed.isend, p2p_send_data, self.next_pprank, tag=i)
                            recv_first = (self.PP_RANK) % 2 == 0 if len(self.PP_GROUP) % 2 == 0 else (
                                                                                                                 self.PP_RANK + 1) % 2 == 0
                            ops.append(send_op) if recv_first else ops.insert(0, send_op)

                        if len(ops) > 0:
                            reqs = []
                            for op in ops:
                                reqs.append(op.op(op.tensor, op.peer, op.group))
                            #                            reqs = dist.batch_isend_irecv(ops)
                            for req in reqs:
                                req.wait()
                        torch.cuda.nvtx.range_pop()
                    timers['p2p'].end()

                    # Start collective
                    torch.cuda.nvtx.range_push(prefix_msg + f'{args.coll}[{i}]')
                    for j in range(coll_iter):
                        timers['coll'].start()
                        if not args.pp_only and (len(args.pp_group) == 0 or self.pp_stage in args.pp_group):
                            req_coll = coll(*data, **kwargs)
                            req_coll.wait()
                        timers['coll'].end()
                    torch.cuda.nvtx.range_pop()
                    timers['iter'].end()
                if not args.pp_only:
                    torch.cuda.synchronize()

            torch.cuda.synchronize()
            for each in timers.keys():
                timers[each].accum()
            total = timers['coll'].total_time() / 1e3  # sum(coll_total_time)/1e6 #
            step_time_list = timers['coll']._total_time  # coll_total_time #
            p2p_time_list = timers['p2p']._total_time  # p2p_total_time #
            dist.barrier()

            if dist.get_rank(self.CUR_GROUP) == 0:
                if len(args.pp_group) == 0 or self.pp_stage in args.pp_group:
                    baseBW = (((nelems // self.dp_group_size) * self.dp_group_size * elem_size) / (1e9)) / (
                                (total) / total_niter)
                    busBW = baseBW * self.get_factor(args.coll)  # (2*(self.dp_group_size - 1) / self.dp_group_size)
                    local_result = [dist.get_rank(self.DP_BASE_GROUP), self.TP_RANK, self.PP_RANK, baseBW, busBW]
                    d_local_result = torch.tensor(local_result, device=torch.device(self.device))
                    if dist.get_rank(self.DP_BASE_GROUP) == 0:
                        d_gathered_local_result = list(
                            torch.zeros(len(local_result), device=torch.device(self.device)) for _ in
                            range(dist.get_world_size(self.DP_BASE_GROUP)))
                    dist.gather(d_local_result,
                                gather_list=d_gathered_local_result if dist.get_rank(self.DP_BASE_GROUP) == 0 else None,
                                group=self.DP_BASE_GROUP)
                    torch.cuda.synchronize()
                    if dist.get_rank(self.DP_BASE_GROUP) == 0:
                        for each in d_gathered_local_result:
                            data = each.cpu().detach().numpy()
                            ranks = list(map(int, data[0:3]))
                            print(
                                f"{ranks[0]:<15d} {ranks[1]:<15d} {ranks[2]:<15d} {nelems * elem_size:<15} {nelems:<15} {args.dtype:<15} {args.op:<15} {each[-2]:<15.3f} {each[-1]:<15.3f}")
                        print("")
                        del d_gathered_local_result

                    if args.verbose:
                        d_step_list = torch.tensor(step_time_list, device=torch.device(self.device))
                        if dist.get_rank(self.DP_BASE_GROUP) == 0:
                            d_gathered_step_list = list(
                                torch.zeros(total_niter, device=torch.device(self.device)) for _ in
                                range(dist.get_world_size(self.DP_BASE_GROUP)))
                            if not args.coll_only:
                                d_gathered_p2p_list = list(
                                    torch.zeros(total_niter, device=torch.device(self.device)) for _ in
                                    range(dist.get_world_size(self.DP_BASE_GROUP)))
                        dist.gather(d_step_list, gather_list=d_gathered_step_list if dist.get_rank(
                            self.DP_BASE_GROUP) == 0 else None, group=self.DP_BASE_GROUP)
                        if not args.coll_only:
                            d_p2p_list = torch.tensor(p2p_time_list, device=torch.device(self.device))
                            dist.gather(d_p2p_list, gather_list=d_gathered_p2p_list if dist.get_rank(
                                self.DP_BASE_GROUP) == 0 else None, group=self.DP_BASE_GROUP)
                        torch.cuda.synchronize()
                        if dist.get_rank(self.DP_BASE_GROUP) == 0:
                            zipped_step_list = torch.stack(d_gathered_step_list, dim=1)
                            if not args.coll_only:
                                zipped_p2p_list = torch.stack(d_gathered_p2p_list, dim=1)
                            for step in range(len(zipped_step_list)):
                                step_times = zipped_step_list[step].tolist()
                                max_coll_idx = step_times.index(max(step_times))
                                min_coll_idx = step_times.index(min(step_times))
                                log_output = (
                                    f"step: {step}, coll and p2p(min,max,med) (ms): {min_coll_idx},{max_coll_idx}, "
                                    f"{min(step_times):<15}, {max(step_times):<15}, {statistics.median(step_times):<15}, ") if not args.pp_only else f"step: {step}, p2p(min,max,med) (ms):  "
                                if not args.coll_only:
                                    p2p_times = zipped_p2p_list[step].tolist()
                                    min_p2p_idx = p2p_times.index(min(p2p_times))
                                    max_p2p_idx = p2p_times.index(max(p2p_times))
                                    log_output += (f"{min_p2p_idx},{max_p2p_idx}, {min(p2p_times):<15},"
                                                   f"{max(p2p_times):<15}, {statistics.median(p2p_times):<15} ")
                                print(log_output)

                            del d_gathered_step_list
                            if not args.coll_only:
                                del d_gathered_p2p_list
                        del d_step_list
                    del d_local_result
            for each in timers.keys():
                timers[each].reset()
            del p2p_recv_data
            del p2p_send_data
            del data
            torch.cuda.empty_cache()
            dist.barrier()
            torch.cuda.synchronize()


if __name__ == '__main__':
    parser.add_argument(
        "-c",
        "--collectives",
        type=str,
        dest="coll",
        default='all_reduce',
        choices=TORCH_COLL.keys(),
        help="A collective to run",
    )

    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default='float',
        dest="dtype",
        choices=TORCH_DTYPES.keys(),
        help="data type for the collectives",
    )

    parser.add_argument(
        "-f",
        "--factor",
        type=int,
        dest="F",
        default=2,
        help="factor to increase message size",
    )

    parser.add_argument(
        "-o",
        "--op",
        type=str,
        dest="op",
        default='sum',
        choices=TORCH_OP.keys(),
        help="reduction operator for the collective",
    )
    parser.add_argument(
        "-i",
        "--inc",
        type=int,
        dest="I",
        help="increment to increase message size",
    )

    parser.add_argument(
        "-b",
        "--minbytes",
        type=str,
        dest="B",
        default="64M",
        help="the smallest message size for the test",
    )

    parser.add_argument(
        "-e",
        "--maxbytes",
        type=str,
        default="64M",
        dest="E",
        help="the biggest message size for the test",
    )

    parser.add_argument(
        "-s",
        "--p2pbytes",
        type=str,
        dest="p2p_size",
        default="12M",
        help="the message size for P2P",
    )

    parser.add_argument(
        '--pp-only',
        default=False,
        dest='pp_only',
        action='store_true',
        help="Measure p2p communication only",
    )

    parser.add_argument(
        '--coll-only',
        default=False,
        dest='coll_only',
        action='store_true',
        help="Measure collective communication only",
    )

    parser.add_argument(
        '--single-stream',
        default=False,
        dest='single_stream',
        action='store_true',
        help="Run all communication routines in a single stream",
    )

    args = parser.parse_args()
    test = nccl_test(args)
    test.run()
