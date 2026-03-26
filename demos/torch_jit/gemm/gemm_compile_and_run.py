# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import torch
import torch_npu

from jit_util_gemm import jit_compile_gemm
from util.device import get_test_device
from util.bench import do_bench

_DEVICE = get_test_device()
torch.npu.set_device(_DEVICE)


# GEMM FLOPs: 2 * m * n * k (each of m*n outputs does k muls + k adds)
def gemm_tflops(m: int, n: int, k: int, elapsed_ms: float) -> float:
    flops = 2 * m * n * k
    return (flops / 1e12) / (elapsed_ms / 1000.0)


def test_gemm(n_repeat=50):
    # NOTE: shapes are currently hard-coded as in gemm_kernel.cpp
    m = 6144
    n = 6144
    k = 6144

    dtype = torch.float16
    out_dtype = torch.float32

    a = torch.rand((m, k), device=_DEVICE, dtype=dtype)
    b = torch.rand((k, n), device=_DEVICE, dtype=dtype)
    c = torch.empty((m, n), device=_DEVICE, dtype=out_dtype)

    gemm_func = jit_compile_gemm(verbose=False)

    # Correctness check (single run)
    gemm_func(c, a, b)
    torch_npu.npu.synchronize()
    c_ref = torch.matmul(a, b)
    torch.testing.assert_close(c.to(torch.float16), c_ref, rtol=0.1, atol=1e-5)
    print("GEMM test pass!")

    # Benchmark: custom GEMM kernel (float16 input and float32 accumulator)
    # NOTE: `blockDim` in gemm_kernel.cpp is hard-coded to 24 (910B2)
    custom_ms = do_bench(
        lambda: gemm_func(c, a, b), benchmark_iters=n_repeat, unit="ms"
    )
    tflops_custom = gemm_tflops(m, n, k, custom_ms)

    # Benchmark: torch built-in matmul (float16 input and output)
    torch_ms = do_bench(lambda: torch.matmul(a, b), benchmark_iters=n_repeat, unit="ms")
    tflops_torch = gemm_tflops(m, n, k, torch_ms)

    print(
        f"Custom GEMM kernel: {custom_ms * n_repeat:.2f} ms / {n_repeat} runs "
        f"({custom_ms:.3f} ms/iter) -> {tflops_custom:.3f} TFLOPs"
    )

    print(
        f"Torch matmul (float16): {torch_ms * n_repeat:.2f} ms / {n_repeat} runs "
        f"({torch_ms:.3f} ms/iter) -> {tflops_torch:.3f} TFLOPs"
    )


if __name__ == "__main__":
    test_gemm()
