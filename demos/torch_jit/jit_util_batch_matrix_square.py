#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the
# terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance
# with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY,
# OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import ctypes
import os
import subprocess

import torch

ASCEND_TOOLKIT_HOME = os.environ["ASCEND_TOOLKIT_HOME"]
PTO_LIB_PATH = os.environ["PTO_LIB_PATH"]



def compile_cpp(src_path: str, verbose: bool = False, timeout: int = 120) -> str:

    # output .so next to src_path
    lib_path = os.path.join(os.path.dirname(src_path), "batch_matrix_square_jit.so")

    # NPU arch
    npu_arch = os.environ.get("NPU_ARCH", "dav-2201").strip()

    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        f"--npu-arch={npu_arch}",
        "-DMEMORY_BASE",
        "-O2",
        "-std=c++17",
        f"-I{PTO_LIB_PATH}/include"
    ]

    cmd = ["bisheng", *flags, src_path, "-o", lib_path]
    if verbose:
        print("compile command:\n", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True, timeout=timeout)
    except Exception as e:
        raise RuntimeError(f"Compile failed: {e}") from e

    if verbose:
        print(f"generated {lib_path}")
    return lib_path


def torch_to_ctypes(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def load_lib(lib_path: str, check_type: bool = True):
    lib_path = os.path.abspath(lib_path)
    lib = ctypes.CDLL(lib_path)

    if check_type:
        lib.call_kernel.argtypes = [
            ctypes.c_uint32,  # blockDim
            ctypes.c_void_p,  # stream
            ctypes.c_void_p,  # dst
            ctypes.c_void_p,  # src
            ctypes.c_int,     # matrixSize
        ]
        lib.call_kernel.restype = None

    default_stream_ptr = torch.npu.current_stream()._as_parameter_

    def batch_matrix_square(
        z: torch.Tensor,
        x: torch.Tensor,
        stream_ptr=default_stream_ptr,
    ):
        M = x.shape[-1]
        block_dim = x.shape[0]
        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(z),
            torch_to_ctypes(x),
            M
        )

    return batch_matrix_square


def jit_compile(src_path: str = "batch_matrix_square.cpp", verbose: bool = False, clean_up: bool = True):
    """
    Builds batch_matrix_square.cpp into batch_matrix_square_jit.so,
    loads call_kernel, and returns batch_matrix_square(z, x).
    """
    
    lib_path = compile_cpp(src_path, verbose=verbose)
    func = load_lib(lib_path)

    if clean_up:
        try:
            os.remove(lib_path)
        except OSError:
            pass

    return func
