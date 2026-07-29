import os

os.environ.setdefault("GPU_ARCHS", "gfx950")

from aiter.jit.core import get_args_of_build, build_module

# module_gemm_a4w4_asm (codegen -m f4gemm) is the FP4 GEMM asm kernel, which is CDNA4-only and
# unused by FP8 training on CDNA3 (gfx942). It is dropped here so it is not JIT-compiled at runtime.
MODULES = [
    "module_aiter_enum",
    "module_rope_general_fwd",
    "module_rope_general_bwd",
]

for name in MODULES:
    args = get_args_of_build(name)
    build_module(
        name,
        args["srcs"],
        args["flags_extra_cc"],
        args["flags_extra_hip"],
        args["blob_gen_cmd"],
        args["extra_include"],
        args["extra_ldflags"],
        args["verbose"],
        args["is_python_module"],
        args["is_standalone"],
        args["torch_exclude"],
        args.get("hipify", False),
    )
    print(f"  compiled {name}")

print("AITER pre-compile OK")
