#!/usr/bin/python3
# coding=utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Data generator for AllGather + GEMM demo.
#
# Generates per-rank A slices (pe_*_a.bin), shared B matrix (pe_*_b.bin),
# and golden output (golden.bin) following the shmem examples convention.
#
# Usage:
#   python gen_data.py --n-ranks 2 --m 2048 --k 2048 --n 1024 --output-dir ./out
#   python gen_data.py --n-ranks 2 --m 2048 --k 2048 --n 1024 \
#                      --padded-m 2048 --padded-k 2048 --padded-n 1024 --output-dir ./out
#

import os
import sys
import argparse
import numpy as np

np.random.seed(19)


def gen_data(m, k, n, n_ranks, padded_m=None, padded_k=None, padded_n=None, output_dir="./out"):
    """
    Generate input and golden data for AllGather GEMM demo.

    Each rank processes padded_M/n_ranks rows of A matrix.
    B matrix is shared across ranks (stored transposed: Layout::DN).
    Golden output C is [orig_M, orig_N] = A_orig @ B_orig (unpadded, FP32).

    Output files follow the shmem examples naming convention:
      pe_<rank>_a.bin  - per-rank A slice (FP16)
      pe_<rank>_b.bin  - B matrix, same for all ranks (FP16, transposed)
      golden.bin       - golden output (FP32)
    """
    pm = padded_m if padded_m is not None else m
    pk = padded_k if padded_k is not None else k
    pn = padded_n if padded_n is not None else n

    os.makedirs(output_dir, exist_ok=True)

    src_type = np.float16
    dst_type = np.float32

    m_local_padded = pm // n_ranks

    a_orig = np.random.randint(1, 5, [m, k]).astype(src_type)
    a_global = np.zeros([pm, pk], dtype=src_type)
    a_global[:m, :k] = a_orig

    b_orig = np.random.randint(1, 5, [k, n]).astype(src_type)
    b_global = np.zeros([pk, pn], dtype=src_type)
    b_global[:k, :n] = b_orig

    golden = np.matmul(a_orig.astype(dst_type), b_orig.astype(dst_type)).astype(dst_type)

    b_transposed = b_global.transpose().astype(src_type)

    for rank in range(n_ranks):
        start_row = rank * m_local_padded
        end_row = (rank + 1) * m_local_padded
        a_rank = a_global[start_row:end_row, :].astype(src_type)

        a_file = os.path.join(output_dir, f"pe_{rank}_a.bin")
        a_rank.tofile(a_file)
        print(f"  - A rank{rank}: {a_rank.shape} -> {a_file}")

        b_file = os.path.join(output_dir, f"pe_{rank}_b.bin")
        b_transposed.tofile(b_file)

    golden_file = os.path.join(output_dir, "golden.bin")
    golden.tofile(golden_file)

    print(f"[INFO] Generated data: orig=({m},{k},{n}), padded=({pm},{pk},{pn}), n_ranks={n_ranks}")
    print(f"  - B (transposed, padded): {b_transposed.shape}")
    print(f"  - Golden C (orig): {golden.shape} -> {golden_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate data for AllGather GEMM demo")
    parser.add_argument("--n-ranks", type=int, default=2, help="Number of ranks")
    parser.add_argument("--m", type=int, required=True, help="M dimension (original)")
    parser.add_argument("--k", type=int, required=True, help="K dimension (original)")
    parser.add_argument("--n", type=int, required=True, help="N dimension (original)")
    parser.add_argument("--padded-m", type=int, default=None, help="Padded M dimension")
    parser.add_argument("--padded-k", type=int, default=None, help="Padded K dimension")
    parser.add_argument("--padded-n", type=int, default=None, help="Padded N dimension")
    parser.add_argument("--output-dir", type=str, default="./out", help="Output directory")

    args = parser.parse_args()
    gen_data(args.m, args.k, args.n, args.n_ranks,
             args.padded_m, args.padded_k, args.padded_n, args.output_dir)


if __name__ == "__main__":
    main()
