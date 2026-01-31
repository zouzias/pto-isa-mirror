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

import torch
import torch_npu

from jit_util_gemm import jit_compile_gemm


def test_gemm():
    device = "npu"
    dtype = torch.float16
    out_dtype = torch.float32

    m = 6144
    n = 6144
    k = 6144

    a = torch.rand((m, k), device=device, dtype=dtype)
    b = torch.rand((k, n), device=device, dtype=dtype)
    c = torch.empty((m, n), device=device, dtype=out_dtype)

    gemm_func = jit_compile_gemm(verbose=True)
    gemm_func(c, a, b)
    torch.npu.synchronize()

    c_ref = a.float() @ b.float()
    torch.testing.assert_close(c, c_ref, rtol=2e-1, atol=2e-1)

    print("GEMM test pass!")


if __name__ == "__main__":
    test_gemm()
