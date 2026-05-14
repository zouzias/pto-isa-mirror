#!/usr/bin/python3
# coding=utf-8
# expert_ffn - gen_data.py
#
# Generates input and golden data for the expert FFN kernel.
# Kernel computes per-token:  output[t] = relu(x[t] * w1[e]) * w2[e]
# where * is element-wise (Hadamard product) and e is the expert for token t.
#
# Tokens are randomly assigned to experts, then sorted by expert assignment
# (packed-token convention, same as moe_segmented_gemm_relu).
# expert_count[e] and expert_start[e] describe each expert's slice.
#
# All operands are float32. W1 and W2 are per-expert feature scale vectors of
# shape (kE, kD). Values in [-2, 2] give a healthy mix of pre-ReLU negatives
# and positives.
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_x_packed.bin     kT * kD  float32  (tokens sorted by expert)
#   ./input/input_expert_count.bin kE       int32
#   ./input/input_expert_start.bin kE       int32
#   ./input/input_w1.bin           kE * kD  float32
#   ./input/input_w2.bin           kE * kD  float32
#   ./output/golden_output.bin     kT * kD  float32

import os
import numpy as np

np.random.seed(42)

kT = 256  # num_tokens
kD = 64   # feature dimension
kE = 32   # num_experts


def gen_golden_data():
    # Random float32 inputs in [-2, 2]. ReLU clips ~50% of intermediate values.
    x  = np.random.uniform(-2.0, 2.0, size=(kT, kD)).astype(np.float32)
    w1 = np.random.uniform(-2.0, 2.0, size=(kE, kD)).astype(np.float32)
    w2 = np.random.uniform(-2.0, 2.0, size=(kE, kD)).astype(np.float32)

    # Assign each token to one expert (approximately uniform).
    expert_id = np.random.randint(0, kE, size=kT).astype(np.int32)
    expert_count = np.bincount(expert_id, minlength=kE).astype(np.int32)
    expert_start = np.zeros(kE, dtype=np.int32)
    expert_start[1:] = np.cumsum(expert_count[:-1])

    # Sort tokens by expert assignment (stable preserves original order within).
    order    = np.argsort(expert_id, kind='stable')
    x_packed = x[order]  # (kT, kD) sorted by expert

    # Golden computation: element-wise gated activation per expert batch.
    output = np.zeros((kT, kD), dtype=np.float32)
    for e in range(kE):
        s = int(expert_start[e])
        c = int(expert_count[e])
        if c == 0:
            continue
        x_batch  = x_packed[s:s + c]           # (c, kD)
        h        = x_batch * w1[e]              # broadcast: (c, kD)
        relu_h   = np.maximum(h, 0.0)
        output[s:s + c] = relu_h * w2[e]       # broadcast: (c, kD)

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    x_packed.tofile("./input/input_x_packed.bin")
    expert_count.tofile("./input/input_expert_count.bin")
    expert_start.tofile("./input/input_expert_start.bin")
    w1.tofile("./input/input_w1.bin")
    w2.tofile("./input/input_w2.bin")
    output.tofile("./output/golden_output.bin")

    n_total   = output.size
    n_clipped = int((output == 0.0).sum())
    n_positive = int((output > 0.0).sum())
    print(f"[gen_data] kT={kT}  kD={kD}  kE={kE}")
    print(f"[gen_data] expert_count = {expert_count.tolist()}")
    print(f"[gen_data] expert_start = {expert_start.tolist()}")
    print(f"[gen_data] output zero (ReLU-clipped or zero-gated): {n_clipped}/{n_total} "
          f"({100.0 * n_clipped / n_total:.1f}%)")
    print(f"[gen_data] output positive: {n_positive}/{n_total} "
          f"({100.0 * n_positive / n_total:.1f}%)")
    print(f"[gen_data] output[0, :8] = {output[0, :8].tolist()}")


if __name__ == "__main__":
    gen_golden_data()
