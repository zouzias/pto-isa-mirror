#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# moe_top1_gather_precomp - gen_data.py (Milestone 1a).
# Host-precomputed permutation; the kernel just executes the gather.
#
# Output files (all raw little-endian, contiguous, no header):
#   ./input/input_tokens.bin            (kT * kH float32)
#   ./input/input_packed_to_token.bin   (kT      uint32)
#   ./output/golden_packed_tokens.bin   (kT * kH float32)
#
# Token values are random integers in [-9, 9] cast to float32 so the gather
# result is bit-exact under IEEE-754; main.cpp uses ResultCmp tolerance 0.001f.
# --------------------------------------------------------------------------------

import os
import numpy as np

np.random.seed(19)


def gen_golden_data(kT, kH, num_experts):
    tokens = np.random.randint(-9, 10, size=(kT, kH)).astype(np.float32)
    expert_id = np.random.randint(0, num_experts, size=(kT,)).astype(np.int32)

    # Stable argsort groups tokens by expert and preserves original within-expert
    # order. This is the SAME permutation the future M1b kernel will produce on
    # the device, so the two milestones can share golden tooling later.
    order = np.argsort(expert_id, kind="stable")
    packed_to_token = order.astype(np.uint32)

    golden_packed_tokens = tokens[packed_to_token]

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    tokens.tofile("./input/input_tokens.bin")
    packed_to_token.tofile("./input/input_packed_to_token.bin")
    golden_packed_tokens.tofile("./output/golden_packed_tokens.bin")


if __name__ == "__main__":
    kT = 256
    kH = 64
    num_experts = 4
    gen_golden_data(kT, kH, num_experts)
