#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# --------------------------------------------------------------------------------

"""
Generate test data for cube_matmul_4buf kernel.

Memory layout matches the SPLIT_K pattern in tmatmul reference:
- A stored as K_ITERS tiles of (M, K_TILE) = [32, 16], each contiguous
  Total A: K_ITERS * M * K_TILE = 64 * 32 * 16 = 32768 elements (65536 bytes)
- B stored as K_ITERS tiles of (K_TILE, N) = [16, 256], each contiguous
  Total B: K_ITERS * K_TILE * N = 64 * 16 * 256 = 262144 elements (524288 bytes)
- C: (M, N) = [32, 256], contiguous

The kernel accesses:
  A_tile[i] = A + i * M * K_TILE
  B_tile[i] = B + i * K_TILE * N
"""

import os
import numpy as np

np.random.seed(42)

# Matrix dimensions
M = 32
K_TILE = 16
N = 256
K_ITERS = 64
K_TOTAL = K_ITERS * K_TILE  # 1024


def gen_golden_data(case_name):
    """Generate input data and golden output for the test case."""
    
    # Generate K_ITERS tiles of A and B
    # A_tiles: shape (K_ITERS, M, K_TILE)
    # B_tiles: shape (K_ITERS, K_TILE, N)
    A_tiles = np.random.uniform(-1.0, 1.0, [K_ITERS, M, K_TILE]).astype(np.float16)
    B_tiles = np.random.uniform(-1.0, 1.0, [K_ITERS, K_TILE, N]).astype(np.float16)
    
    # Compute golden output: C = sum(A_tiles[i] @ B_tiles[i])
    C = np.zeros([M, N], dtype=np.float32)
    for i in range(K_ITERS):
        A_f32 = A_tiles[i].astype(np.float32)
        B_f32 = B_tiles[i].astype(np.float32)
        C += np.matmul(A_f32, B_f32)
    
    # Flatten tiles to match kernel memory layout
    # A: stored as [tile0, tile1, ..., tile63] where each tile is M*K_TILE elements
    A_gm = A_tiles.reshape(-1)  # Flatten to (K_ITERS * M * K_TILE,)
    B_gm = B_tiles.reshape(-1)  # Flatten to (K_ITERS * K_TILE * N,)
    
    # Write binary files
    A_gm.tofile("./A_gm.bin")
    B_gm.tofile("./B_gm.bin")
    C.tofile("./golden.bin")
    
    print(f"Generated data for {case_name}:")
    print(f"  A_tiles: {A_tiles.shape} {A_tiles.dtype}")
    print(f"  B_tiles: {B_tiles.shape} {B_tiles.dtype}")
    print(f"  A_gm: {A_gm.shape} ({A_gm.nbytes} bytes)")
    print(f"  B_gm: {B_gm.shape} ({B_gm.nbytes} bytes)")
    print(f"  C: {C.shape} {C.dtype}, range [{C.min():.4f}, {C.max():.4f}]")
    
    # Verify layout by checking first tile
    print(f"\n  A_tile[0][0,0:4] = {A_tiles[0, 0, 0:4]}")
    print(f"  B_tile[0][0,0:4] = {B_tiles[0, 0, 0:4]}")
    print(f"  C[0,0:4] = {C[0, 0:4]}")


if __name__ == "__main__":
    case_name_list = [
        "CubeMatmul4BufTest.case_f16_32x1024_1024x256",
    ]
    
    for case_name in case_name_list:
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name)
        os.chdir(original_dir)
    
    print("\nDone! Generated test data for all cases.")
