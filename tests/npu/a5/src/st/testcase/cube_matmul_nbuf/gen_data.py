"""
gen_data.py for cube_matmul_nbuf benchmark suite.

Generates A_gm.bin, B_gm.bin, golden.bin for M=32, K=1024, N=256 fp16->fp32 matmul.
Output goes to CubeMatmulNBufTest.golden/ (same golden reused across all buffer configs).
"""
import numpy as np
import os
import argparse

M, K, N = 32, 1024, 256

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default="CubeMatmulNBufTest.golden")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
rng = np.random.default_rng(args.seed)

A = rng.uniform(-1, 1, (M, K)).astype(np.float16)
B = rng.uniform(-1, 1, (K, N)).astype(np.float16)
C = A.astype(np.float32) @ B.astype(np.float32)

A.flatten().tofile(os.path.join(args.output_dir, "A_gm.bin"))
B.flatten().tofile(os.path.join(args.output_dir, "B_gm.bin"))
C.flatten().astype(np.float32).tofile(os.path.join(args.output_dir, "golden.bin"))

print(f"Generated cube_matmul_nbuf golden data in {args.output_dir}/")
print(f"  A: {A.shape} fp16  ({A.nbytes} bytes)")
print(f"  B: {B.shape} fp16  ({B.nbytes} bytes)")
print(f"  C: {C.shape} fp32  ({C.nbytes} bytes)")
print(f"  C[0,:4] = {C[0,:4]}")
