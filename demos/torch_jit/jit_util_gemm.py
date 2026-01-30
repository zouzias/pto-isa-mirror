import ctypes
import torch

import os
from build_gemm import build_gemm_so

def torch_to_ctypes(t):
    return ctypes.c_void_p(t.data_ptr())

def jit_compile_gemm():
    lib_path = build_gemm_so()
    lib = ctypes.CDLL(lib_path)

    lib.call_kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.call_kernel.restype = None

    stream = torch.npu.current_stream()._as_parameter_

    def gemm(c, a, b, block_dim=24):
        lib.call_kernel(
            block_dim,
            stream,
            torch_to_ctypes(c),
            torch_to_ctypes(a),
            torch_to_ctypes(b),
            a.numel()
        )

    return gemm

