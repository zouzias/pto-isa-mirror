#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# moe_top1_permute - gen_data.py (Milestone 1b).
#
# Produces inputs (tokens, expert_id) and reference goldens for the four
# outputs the device kernel emits (packed_tokens, expert_count, expert_start,
# token_to_packed). The golden uses the SAME packing rule the kernel
# implements: group tokens by expert id; within each expert preserve the
# original token order; emit a slot-by-slot mapping.
#
# Output files (all raw little-endian, contiguous, no header):
#   ./input/input_tokens.bin             (kT * kH float32)
#   ./input/input_expert_id.bin          (kT      int32)
#   ./output/golden_packed_tokens.bin    (kT * kH float32)
#   ./output/golden_expert_count.bin     (kE      int32)
#   ./output/golden_expert_start.bin     (kE      int32)
#   ./output/golden_token_to_packed.bin  (kT      int32)
# --------------------------------------------------------------------------------

import os
import numpy as np

np.random.seed(19)


def gen_golden_data(kT, kH, kE):
    tokens    = np.random.randint(-9, 10, size=(kT, kH)).astype(np.float32)
    expert_id = np.random.randint(0, kE,  size=(kT,)).astype(np.int32)

    # Histogram and prefix-sum, mirroring the kernel's passes 1 and 2 exactly.
    expert_count = np.bincount(expert_id, minlength=kE).astype(np.int32)
    expert_start = np.zeros(kE, dtype=np.int32)
    expert_start[1:] = np.cumsum(expert_count[:-1])

    # Pack: stable argsort groups tokens by expert in ascending expert id and
    # preserves the original within-expert token order — same rule the kernel
    # produces via counter[e]++ in pass 3.
    order = np.argsort(expert_id, kind="stable")
    packed_tokens = tokens[order]

    # token_to_packed[t] = the packed_pos at which token t lands.
    token_to_packed = np.empty(kT, dtype=np.int32)
    for p, t in enumerate(order):
        token_to_packed[int(t)] = int(p)

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    tokens.tofile("./input/input_tokens.bin")
    expert_id.tofile("./input/input_expert_id.bin")
    packed_tokens.tofile("./output/golden_packed_tokens.bin")
    expert_count.tofile("./output/golden_expert_count.bin")
    expert_start.tofile("./output/golden_expert_start.bin")
    token_to_packed.tofile("./output/golden_token_to_packed.bin")


if __name__ == "__main__":
    kT = 256
    kH = 64
    kE = 4
    gen_golden_data(kT, kH, kE)
