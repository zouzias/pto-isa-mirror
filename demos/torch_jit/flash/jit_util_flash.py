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
BLOCK_DIM = 1  # S0 / CUBE_S0


def torch_to_ctypes(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def _npu_arch_flag() -> str:
    return os.environ.get("NPU_ARCH", "dav-2201").strip()


def compile_flash(kernel_cpp: str, verbose: bool = False, timeout: int = 300) -> str:
    """
    Compile a FlashAttention/TFA kernel cpp into a shared library with a call_kernel symbol.

    Output library is placed next to kernel_cpp, named: flash_jit.so
    """
    lib_path = os.path.join(os.path.dirname(kernel_cpp), "flash_jit.so")

    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        f"--npu-arch={_npu_arch_flag()}",
        "-O2",
        "-std=c++17",
        "-Wno-ignored-attributes",
        "-DMEMORY_BASE",
        f"-I{PTO_LIB_PATH}/include",
        f"-I{ASCEND_TOOLKIT_HOME}/include",
        f"-I{ASCEND_TOOLKIT_HOME}/aarch64-linux/pkg_inc/runtime",
        f"-I{ASCEND_TOOLKIT_HOME}/aarch64-linux/pkg_inc",
        f"-I{ASCEND_TOOLKIT_HOME}/aarch64-linux/pkg_inc/profiling"
    ]

    cmd = ["bisheng", *flags, kernel_cpp, "-o", lib_path]
    if verbose:
        print("compile command:\n", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True, timeout=timeout)
    except Exception as e:
        raise RuntimeError(f"Compile failed: {e}") from e

    if verbose:
        print(f"generated {lib_path}")
    return lib_path


def load_flash_lib(lib_path: str, check_type: bool = True):
    lib_path = os.path.abspath(lib_path)
    lib = ctypes.CDLL(lib_path)

    if check_type:
        lib.call_kernel.argtypes = [
            ctypes.c_uint32,  # blockDim
            ctypes.c_void_p,  # stream
            ctypes.c_void_p,  # q
            ctypes.c_void_p,  # k
            ctypes.c_void_p,  # v
            ctypes.c_void_p,  # o_out
            ctypes.c_void_p,  # qk_out fp32
            ctypes.c_void_p,  # p_out fp16
            ctypes.c_void_p,  # p_out fp32
            ctypes.c_void_p,  # pv_out tiles
            ctypes.c_void_p,  # global_sum
            ctypes.c_void_p,  # exp_max
            ctypes.c_void_p,  # o_parts
        ]
        lib.call_kernel.restype = None

    default_block_dim = BLOCK_DIM
    default_stream_ptr = torch.npu.current_stream()._as_parameter_

    def flash(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o_out: torch.Tensor,
        outDevice: torch.Tensor,
        xexpDevice: torch.Tensor,
        pOutFp32Device: torch.Tensor,
        out2Device: torch.Tensor,
        gSumDevice: torch.Tensor,
        expMaxDevice: torch.Tensor,
        oPartsDevice: torch.Tensor,
        block_dim: int = default_block_dim,
        stream_ptr=default_stream_ptr,
    ):

        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(q),
            torch_to_ctypes(k),
            torch_to_ctypes(v),
            torch_to_ctypes(o_out),
            torch_to_ctypes(outDevice),  # qk_out fp32
            torch_to_ctypes(xexpDevice),  # p_out fp16
            torch_to_ctypes(pOutFp32Device),  # p_out fp32
            torch_to_ctypes(out2Device),  # pv_out tiles
            torch_to_ctypes(gSumDevice),  # global_sum
            torch_to_ctypes(expMaxDevice),  # exp_max
            torch_to_ctypes(oPartsDevice),  # o snapshots
        )

    return flash


def jit_compile_flash(
    verbose: bool = False,
    clean_up: bool = True,
    kernel_cpp: str = "fa_kernel.cpp",
):
    """
    Builds the Flash/TFA kernel cpp into flash_jit.so,
    loads call_kernel, and returns flash(...) wrapper.

    Expected C++ export:
        extern "C" void call_kernel(uint32_t blockDim, void* stream, ...)
    """
    lib_path = compile_flash(kernel_cpp, verbose=verbose)
    func = load_flash_lib(lib_path)

    if clean_up:
        try:
            os.remove(lib_path)
        except OSError:
            pass

    return func
