#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os
import numpy as np

np.random.seed(42)


def gen_golden_data(param):
    s0 = param.s0
    s1 = param.s1
    h = param.h

    # Q: (S0, H) in ND layout (row-major)
    # q_fp32 = (np.random.randn(s0, h).astype(np.float16) * 1.5).astype(np.float32)
    q_fp32 = np.eye(s0, h, dtype=np.float32)
    q_nd = q_fp32.astype(np.float16)
    
    # K: (H, S1) in ND layout (row-major), but stored as DN (transposed)
    k_fp32 = (np.random.randn(h, s1).astype(np.float16) * 1.5).astype(np.float32)
    block_rows = min(128, h)
    num_blocks = s1 // 128
    for block_idx in range(num_blocks):
        start_col = block_idx * 128
        end_col = start_col + 128
        k_fp32[:block_rows, start_col:end_col] = float(block_idx + 1)
    k_nd = k_fp32.astype(np.float16)
    
    # Compute QK = Q @ K: (S0, H) @ (H, S1) = (S0, S1)
    qk_nd = (q_nd.astype(np.float32).dot(k_nd.astype(np.float32))).astype(np.float32)
    
    # For DN layout, golden is transposed: K @ Q^T = (S1, S0)
    qk_dn = qk_nd.T.copy()
    
    # Store K as DN layout (transposed)
    k_t = k_nd.T.copy()
    q_t = q_nd.T.copy()

    q_nd.tofile('q.bin')
    q_t.tofile('q_t.bin')

    k_nd.tofile('k.bin')
    k_t.tofile('k_t.bin')
    qk_nd.tofile('golden_nd.bin')
    qk_dn.tofile('golden_dn.bin')  # DN golden is transpose of ND golden


class TQKParams:
    def __init__(self, name, s0, s1, h):
        self.name = name
        self.s0 = s0
        self.s1 = s1
        self.h = h


if __name__ == "__main__":
    case_params_list = [
        TQKParams("TQKTest.case512", 128, 512, 128),
        TQKParams("TQKTest.case128", 128, 128, 128),  # Simplest DN: S0=128, S1=128, H=128 (single block, single K tile)
        TQKParams("TQKTest.case256", 256, 64, 128),   # Cube_S0=256, Cube_S1=64, Cube_K=128
    ]

    for case in case_params_list:
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)
