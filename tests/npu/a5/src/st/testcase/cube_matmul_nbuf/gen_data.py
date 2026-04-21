#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0
# --------------------------------------------------------------------------------

"""
Generate test data for cube_matmul_nbuf benchmark suite.

GM Memory Layout — Normal Row-Major (standard [M,K] x [K,N] matmul layout)
===========================================================================
A single set of A/B/golden files is shared by all 5 configs (K16 and K32).

  A_gm.bin  — [M, K_total] = [32, 1024]  fp16  row-major
               element A[r, c] at byte offset  (r * K_total + c) * 2
               Kernel tile k access: ptr = src0 + k * K_TILE  (fp16 elements)
               GlobalTensor stride on dim-K: GM_K = K_total = 1024

  B_gm.bin  — [K_total, N] = [1024, 256]  fp16  row-major
               element B[r, c] at byte offset  (r * N + c) * 2
               Kernel tile k access: ptr = src1 + k * K_TILE * N  (fp16 elements)
               GlobalTensor stride on dim-K: GM_K = K_total = 1024 (outer dim only)

  golden.bin — [M, N] = [32, 256]  fp32
               C = A.astype(fp32) @ B.astype(fp32)

This layout is identical to the standard GEMM convention and matches:
  - cube_matmul_4buf (same split-k tiling, but that uses tile-major layout)
  - GlobalTensor<inType, Shape<1,1,1,M,K_TILE>, Stride<...,GM_K,1>>
    with ptr = src0 + k * K_TILE  addresses A[:,k*K_TILE:(k+1)*K_TILE] correctly.
"""

import numpy as np
import os
import argparse

M, K_TOTAL, N = 32, 1024, 256

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default="CubeMatmulNBufTest.golden")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
rng = np.random.default_rng(args.seed)

# Normal row-major layout
A = rng.uniform(-1, 1, (M, K_TOTAL)).astype(np.float16)    # [32, 1024]
B = rng.uniform(-1, 1, (K_TOTAL, N)).astype(np.float16)    # [1024, 256]
C = A.astype(np.float32) @ B.astype(np.float32)            # [32, 256]

A.flatten().tofile(os.path.join(args.output_dir, "A_gm.bin"))
B.flatten().tofile(os.path.join(args.output_dir, "B_gm.bin"))
C.flatten().astype(np.float32).tofile(os.path.join(args.output_dir, "golden.bin"))

print("Generated data (normal row-major layout):")
print(f"  A_gm:   {A.shape}   ({A.nbytes} bytes)  fp16  [M={M}, K={K_TOTAL}]")
print(f"  B_gm:   {B.shape}  ({B.nbytes} bytes)  fp16  [K={K_TOTAL}, N={N}]")
print(f"  golden: {C.shape}  ({C.nbytes} bytes)  fp32  [M={M}, N={N}]")
print()
print(f"  Kernel tile access:")
print(f"    A tile k : src0 + k * K_TILE    (stride GM_K={K_TOTAL} in fp16 elems)")
print(f"    B tile k : src1 + k * K_TILE*N  (stride GM_N={N} in fp16 elems)")
print()
print(f"  A[0, 0:4] = {A[0, 0:4]}")
print(f"  B[0, 0:4] = {B[0, 0:4]}")
print(f"  C[0, 0:4] = {C[0, :4]}")
