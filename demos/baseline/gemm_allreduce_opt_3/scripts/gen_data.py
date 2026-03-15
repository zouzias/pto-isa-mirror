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

"""
Data generator for GEMM AllReduce demo (Data Parallel Mode).

Data Parallel scenario:
- Total input: A[M×K] × B[K×N] = C[M×N]
- Each rank gets: A_part[M×(K/r)] × B_part[(K/r)×N] = C_part[M×N]
- A matrix is split along K dimension (column-wise)
- B matrix is split along K dimension (row-wise)
- Each rank computes: C_part = A_part × B_part
- AllReduce sums: C_final = sum(C_part for all ranks)

This generates:
- input/x1_gm_rank{i}.bin: A_part matrix for rank i (M × K/r)
- input/x2_gm_rank{i}.bin: B_part matrix for rank i (K/r × N)
- output/golden.bin: Expected result = A × B (full matrix multiplication)
"""

import os
import argparse
import numpy as np


def gen_golden_data(param, nranks):
    """Generate input data and golden output for Data Parallel AllReduce demo."""
    src_type = param.atype
    dst_type = param.ctype

    m, k, n, is_atrans, is_btrans = param.m, param.k, param.n, param.is_atrans, param.is_btrans

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # Check if K is divisible by nranks
    if k % nranks != 0:
        raise ValueError(f"K dimension ({k}) must be divisible by number of ranks ({nranks})")
    
    k_per_rank = k // nranks
    print(f"Data Parallel Mode: K={k} split into {nranks} ranks, {k_per_rank} per rank")

    # Generate full A and B matrices for golden computation
    np.random.seed(42)  # Fixed seed for reproducibility
    a_full = np.random.randint(1, 5, [m, k]).astype(src_type)
    b_full = np.random.randint(1, 5, [k, n]).astype(src_type)
    
    # Compute golden result: C = A × B (full matrix multiplication)
    golden = np.matmul(a_full.astype(dst_type), b_full.astype(dst_type)).astype(dst_type)
    
    # Split A and B matrices along K dimension
    for rank in range(nranks):
        # A matrix: split along columns (K dimension)
        # Each rank gets columns [rank * k_per_rank : (rank + 1) * k_per_rank]
        a_start = rank * k_per_rank
        a_end = (rank + 1) * k_per_rank
        x1_gm = a_full[:, a_start:a_end]  # Shape: [m, k_per_rank]
        
        # B matrix: split along rows (K dimension)
        # Each rank gets rows [rank * k_per_rank : (rank + 1) * k_per_rank]
        b_start = rank * k_per_rank
        b_end = (rank + 1) * k_per_rank
        x2_gm = b_full[b_start:b_end, :]  # Shape: [k_per_rank, n]
        
        # Verify: A_part × B_part should contribute to the final result
        c_part = np.matmul(x1_gm.astype(dst_type), x2_gm.astype(dst_type)).astype(dst_type)
        
        # Save A_part matrix for this rank
        if is_atrans:
            x1_gm_save = x1_gm.transpose()
        else:
            x1_gm_save = x1_gm
        x1_gm_save.tofile(f"./input/x1_gm_rank{rank}.bin")
        print(f"Generated A_part for rank {rank}: shape={x1_gm.shape}, "
              f"columns [{a_start}:{a_end}], saved to input/x1_gm_rank{rank}.bin")
        
        # Save B_part matrix for this rank
        if is_btrans:
            x2_gm_save = x2_gm.transpose()
        else:
            x2_gm_save = x2_gm
        x2_gm_save.tofile(f"./input/x2_gm_rank{rank}.bin")
        print(f"Generated B_part for rank {rank}: shape={x2_gm.shape}, "
              f"rows [{b_start}:{b_end}], C_part sum={c_part.sum():.2f}, saved to input/x2_gm_rank{rank}.bin")

    # Save golden result
    golden.tofile("./output/golden.bin")
    print(f"Generated golden output: shape={golden.shape}, sum={golden.sum():.2f}, "
          f"saved to output/golden.bin")
    
    # Print some statistics
    print(f"\nData Parallel AllReduce configuration:")
    print(f"  nranks: {nranks}")
    print(f"  Total matrix dimensions: M={m}, K={k}, N={n}")
    print(f"  Per-rank dimensions: A_part[{m}×{k_per_rank}], B_part[{k_per_rank}×{n}], C_part[{m}×{n}]")
    print(f"  A type: {src_type}, B type: {src_type}, C type: {dst_type}")
    print(f"  Golden = A_full × B_full (full matrix multiplication)")


class GemmParams:
    def __init__(self, atype, btype, ctype, m, k, n, is_atrans=0, is_btrans=0):
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.m = m
        self.k = k
        self.n = n
        self.is_atrans = is_atrans
        self.is_btrans = is_btrans


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data for GEMM AllReduce demo")
    parser.add_argument("--nranks", type=int, default=2, help="Number of ranks")
    args = parser.parse_args()
    
    nranks = args.nranks
    if nranks <= 0:
        nranks = int(os.environ.get("NUM_RANKS", "2"))

    case_params_list = [
        GemmParams(np.float16, np.float16, np.float32, 16384, 16384, 4096, 0, 1)
    ]
    gen_golden_data(case_params_list[0], nranks)
