import argparse
import torch
import torch.distributed as dist

TORCH_DTYPES = { 'char' : torch.int8,
                'float': torch.float32,
                'bfloat': torch.bfloat16,
                'int': torch.int32,
                'half': torch.float16,
                'double': torch.float64,
                'int64': torch.int64 }



parser = argparse.ArgumentParser()
parser.add_argument(
    "-n",
    "--niter",
    type=int,
    default=100,
    dest="niter",
    help="number of iterations to run",
)


parser.add_argument(
    "-F",
    "--filename",
    type=str,
    dest="filename",
    default=None,
    help="PyT file-based storage filename (Default: pyt_init.store)",
)

parser.add_argument(
    "-w",
    "--warmiter",
    type=int,
    dest="witer",
    default=10,
    help="number of warmup iterations",
)


parser.add_argument(
    "--sharp",
    dest="sharp",
    default=False,
    action='store_true',
    help="Enable sharp",
)


parser.add_argument(
    "-t",
    "--tensor",
    type=int,
    dest="tp",
    default=1,
    help="number of process groups for tensor model parallelism in 3D parallelism",
)

parser.add_argument(
    "-p",
    "--pipeline",
    type=int,
    dest="pp",
    default=1,
    help="number of process groups for pipeline model parallelism in 3D parallelism",
)

parser.add_argument(
    "--pp-group",
    type=int,
    nargs="+",
    dest="pp_group",
    default=[],
    help="list of pipeline parallel stages(int) of which corresponding DP groups will run the chosen collectives",
)

parser.add_argument(
    "-r"
    "--pp_overlap_ratio",
    type=float,
    dest="overlap",
    default=1,
    help="The ratio of delay over data-parallel all-reduce to run DP groups in pipeline stages at different timings. Default is 0 (DP runs w/o overlapping) ",
)

parser.add_argument(
    "-v",
    "--verbose",
    dest="verbose",
    action='store_true',
    help="increase the verbosity of the output",
)
'''
parser.add_argument(
    "--pattern",
    dest="pattern",
    default='compact',
    choices=['compact', 'strided'],
    help="pattern to place DP ranks - 'compact' or 'strided'",
)
'''
