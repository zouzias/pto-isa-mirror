#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Data generator for AllGather + GEMM demo.
#
# The golden data is identical to gemm_allgather because:
#   gemm_allgather:  C = [A_0@B; A_1@B; ...; A_{n-1}@B] = A_full @ B
#   allgather_gemm:  C = A_full @ B   (A_full = AllGather(A_0, A_1, ..., A_{n-1}))
#
# So both demos use the same input files and golden output.
# This script is a copy of gemm_allgather_multi/scripts/gen_data.py.
# --------------------------------------------------------------------------------

import os
import sys
import numpy as np
np.random.seed(19)


def gen_golden_data(m, k, n, size_name, n_ranks=2):
    """
    Generate input and golden data for AllGather GEMM demo (multi-card version).
    
    Each card processes G_M/n_ranks rows of A matrix.
    B matrix is shared across cards (stored transposed: Layout::DN).
    Output C is [G_M, G_N] = A_full @ B.
    """
    input_dir = f"input/{size_name}"
    output_dir = f"output/{size_name}"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if all required files already exist
    required_files = [f"{input_dir}/a_rank{i}.bin" for i in range(n_ranks)]
    required_files.append(f"{input_dir}/b.bin")
    required_files.append(f"{output_dir}/golden.bin")
    
    if all(os.path.exists(f) for f in required_files):
        print(f"[INFO] Data for {size_name} (n_ranks={n_ranks}) already exists, skipping generation")
        return
    
    src_type = np.float16
    dst_type = np.float32
    
    m_local = m // n_ranks
    
    # Generate global A matrix [M, K]
    a_global = np.random.randint(1, 5, [m, k]).astype(src_type)
    
    # Generate global B matrix [K, N]
    b_global = np.random.randint(1, 5, [k, n]).astype(src_type)
    
    # Compute golden output: C = A @ B
    golden = np.matmul(a_global.astype(dst_type), b_global.astype(dst_type)).astype(dst_type)
    
    # Split A matrix for each rank
    for rank in range(n_ranks):
        start_row = rank * m_local
        end_row = (rank + 1) * m_local if rank < n_ranks - 1 else m
        a_rank = a_global[start_row:end_row, :].astype(src_type)
        a_rank_file = f"{input_dir}/a_rank{rank}.bin"
        a_rank.tofile(a_rank_file)
        print(f"  - A rank{rank}: {a_rank.shape} -> {a_rank_file}")
    
    # B matrix needs to be transposed (Layout::DN)
    b_transposed = b_global.transpose().astype(src_type)
    b_file = f"{input_dir}/b.bin"
    b_transposed.tofile(b_file)
    
    # Write golden output
    golden_file = f"{output_dir}/golden.bin"
    golden.tofile(golden_file)
    
    print(f"[INFO] Generated data for {size_name}: M={m}, K={k}, N={n}, n_ranks={n_ranks}")
    print(f"  - B (transposed): {b_transposed.shape} -> {b_file}")
    print(f"  - Golden C: {golden.shape} -> {golden_file}")


def main():
    size_configs = {
        "small": (512, 512, 256),
        "medium": (1024, 1024, 512),
        "large": (2048, 2048, 1024),
        "xlarge": (4096, 4096, 2048),
        "xxlarge": (8192, 8192, 4096),
        "kv_7b_2k": (2048, 4096, 4096),
        "kv_7b_4k": (4096, 4096, 4096),
        "kv_7b_8k": (8192, 4096, 4096),
        "kv_7b_16k": (16384, 4096, 4096),
        "kv_7b_32k": (32768, 4096, 4096),
        "kv_7b_64k": (65536, 4096, 4096),
        "kv_7b_128k": (131072, 4096, 4096),
        "kv_7b_256k": (262144, 4096, 4096),
        "kv_13b_4k": (4096, 5120, 5120),
        "kv_13b_8k": (8192, 5120, 5120),
        "kv_13b_16k": (16384, 5120, 5120),
        "kv_13b_32k": (32768, 5120, 5120),
        "kv_13b_64k": (65536, 5120, 5120),
        "kv_13b_128k": (131072, 5120, 5120),
        "kv_70b_2k": (2048, 8192, 4096),
        "kv_70b_4k": (4096, 8192, 8192),
        "kv_70b_8k": (8192, 8192, 8192),
        "kv_70b_16k": (16384, 8192, 8192),
        "kv_70b_32k": (32768, 8192, 8192),
        "kv_70b_64k": (65536, 8192, 8192),
        "kv_70b_128k": (131072, 8192, 8192),
        "mla_prefill": (4096, 7168, 512),
        "mla_decode": (128, 7168, 512),
    }
    
    n_ranks = 2
    requested_size = None
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--n-ranks" and i + 1 < len(sys.argv):
            n_ranks = int(sys.argv[i + 1])
            i += 2
        elif requested_size is None:
            requested_size = sys.argv[i]
            i += 1
        else:
            i += 1
    
    if requested_size:
        if requested_size in size_configs:
            m, k, n = size_configs[requested_size]
            gen_golden_data(m, k, n, requested_size, n_ranks)
            print(f"\n[INFO] Data generation for {requested_size} (n_ranks={n_ranks}) completed!")
        else:
            print(f"[ERROR] Unknown size: {requested_size}")
            print(f"Available sizes: {', '.join(size_configs.keys())}")
            sys.exit(1)
    else:
        for size_name, (m, k, n) in size_configs.items():
            gen_golden_data(m, k, n, size_name, n_ranks)
        print(f"\n[INFO] All data generation completed! (n_ranks={n_ranks})")


if __name__ == "__main__":
    main()
