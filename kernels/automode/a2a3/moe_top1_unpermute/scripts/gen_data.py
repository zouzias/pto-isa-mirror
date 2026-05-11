#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# moe_top1_unpermute - gen_data.py
#
# Reverse token movement after the forward top-1 permute.
# We do not need to re-run the device permute here; we replay it in Python so
# the unpermute kernel can be exercised standalone with deterministic inputs.
#
# Pipeline (Python-only):
#
#   tokens [T, H]
#   expert_id [T]
#       ↓ (same recipe as moe_top1_permute: argsort stable by expert_id)
#   packed_tokens [T, H]
#   token_to_packed [T]
#       ↓ (fake "expert FFN" output for v1 — adds 1.0 to every element)
#   packed_output = packed_tokens + 1.0
#       ↓ (golden unpermute: output[t, :] = packed_output[token_to_packed[t], :])
#   output [T, H]
#
# Output files (all raw little-endian, contiguous, no header):
#   ./input/input_packed_output.bin     (kT * kH float32)
#   ./input/input_token_to_packed.bin   (kT      int32)
#   ./output/golden_output.bin          (kT * kH float32)
#
# Debug prints: expert_count, expert_start, first 32 expert_id,
# first 32 token_to_packed, first 4 rows of packed_output / golden_output.
# --------------------------------------------------------------------------------

import os
import numpy as np

np.random.seed(19)


def gen_golden_data(kT, kH, kE):
    # ---- Same deterministic inputs as moe_top1_permute/scripts/gen_data.py ----
    tokens    = np.random.randint(-9, 10, size=(kT, kH)).astype(np.float32)
    expert_id = np.random.randint(0, kE,  size=(kT,)).astype(np.int32)

    # ---- Replay the forward permute (same recipe the device kernel uses) -----
    expert_count = np.bincount(expert_id, minlength=kE).astype(np.int32)
    expert_start = np.zeros(kE, dtype=np.int32)
    expert_start[1:] = np.cumsum(expert_count[:-1])

    order = np.argsort(expert_id, kind="stable")
    packed_tokens = tokens[order]

    token_to_packed = np.empty(kT, dtype=np.int32)
    for p, t in enumerate(order):
        token_to_packed[int(t)] = int(p)

    # ---- Fake expert output (v1: +1.0). Real FFN lands in M3/M4/M5. ---------
    packed_output = packed_tokens + 1.0

    # ---- Golden unpermute (the kernel's reference) --------------------------
    output = np.empty_like(tokens)
    for t in range(kT):
        output[t, :] = packed_output[int(token_to_packed[t]), :]

    # Equivalently:
    #   output_check = packed_output[token_to_packed]
    # assert np.array_equal(output, output_check)

    # ---- Save -------------------------------------------------------------------
    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)
    packed_output.astype(np.float32).tofile("./input/input_packed_output.bin")
    token_to_packed.tofile("./input/input_token_to_packed.bin")
    output.astype(np.float32).tofile("./output/golden_output.bin")

    # ---- Debug prints -----------------------------------------------------------
    print("[gen_data] expert_count       =", expert_count.tolist())
    print("[gen_data] expert_start       =", expert_start.tolist())
    print("[gen_data] expert_id[:32]     =", expert_id[:32].tolist())
    print("[gen_data] token_to_packed[:32]=", token_to_packed[:32].tolist())
    print("[gen_data] packed_output[0, :8] =", packed_output[0, :8].tolist())
    print("[gen_data] golden_output[0, :8] =", output[0, :8].tolist())
    print("[gen_data] golden_output[T-1, :8] =", output[-1, :8].tolist())


if __name__ == "__main__":
    kT = 256
    kH = 64
    kE = 4
    gen_golden_data(kT, kH, kE)
