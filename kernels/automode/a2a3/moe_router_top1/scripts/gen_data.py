#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# moe_router_top1 - gen_data.py
#
# Generates inputs and golden for the missing-router stage of the top-1 MoE
# pipeline:
#
#   logits  = X @ W_router                       (FP32 acc; FP16 X, FP16 W)
#   expert_id[t] = argmax_e logits[t, e]         (uint32)
#
# Tested shape (matches main.cpp / kernel.cpp constants):
#   T = 256
#   H = 64
#   E = 16        <-- CHOSEN to match cube blockAlign of 16 for FP16; avoids
#                     the validN<N=16 assumption (A2). MoE downstream code
#                     accepts E=16 just as readily as the previously-used
#                     E=4 (the segmented-FFN milestones use E=4 only because
#                     that was the shape exercised — the kernel template
#                     widens to any compile-time E).
#   kTileM = 128  microtile (same as moe_segmented_ffn_top1).
#
# Datatype contract (mirrors moe_segmented_ffn_top1 where overlapping):
#   X         : float16        [T, H]
#   W_router  : float16        [H, E]      (ColMajor-friendly in the cube tile)
#   logits    : float32        [T, E]      cube FP32 accumulator -> ND GM
#   expert_id : uint32         [T]
#
# Why FP16 for X / W_router:
#   Identical to §A16 / §A18; FP16×FP16 -> FP32 acc is the proven cube combo.
#
# Why uint32 for expert_id:
#   Matches the metadata dtype of moe_top1_permute / moe_top1_unpermute. The
#   permute kernel reads expert_id as int32; uint32 is bit-compatible for the
#   small positive range we care about (E <= 32 << 2**31).
#
# Output files (all raw little-endian, contiguous, no header):
#   ./input/input_X.bin                  (T * H float16)
#   ./input/input_W_router.bin           (H * E float16)
#   ./output/golden_logits.bin           (T * E float32; debug)
#   ./output/golden_expert_id.bin        (T     uint32 ; primary)
#   ./output/t.txt                       (single int line; T)
# --------------------------------------------------------------------------------

import os
import numpy as np

np.random.seed(29)


def gen_golden_data(kT, kH, kE):
    # Small integer-valued FP16 inputs keep GEMM exact. Same range
    # discipline as §A18 (FP16-exact integer regime, |x| <= 2048).
    X        = np.random.randint(-4, 5, size=(kT, kH)).astype(np.float16)
    W_router = np.random.randint(-4, 5, size=(kH, kE)).astype(np.float16)

    logits = (X.astype(np.float32) @ W_router.astype(np.float32)).astype(np.float32)
    expert_id = np.argmax(logits, axis=1).astype(np.uint32)

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)
    X.tofile("./input/input_X.bin")
    W_router.tofile("./input/input_W_router.bin")
    logits.tofile("./output/golden_logits.bin")
    expert_id.tofile("./output/golden_expert_id.bin")
    with open("./output/t.txt", "w") as f:
        f.write(f"{kT}\n")

    print(f"[gen_data] kT = {kT}  kH = {kH}  kE = {kE}")
    print(f"[gen_data] logits range = [{float(logits.min()):.1f}, {float(logits.max()):.1f}]")
    print( "[gen_data] expert_id[:16] =", expert_id[:16].tolist())
    hist = np.bincount(expert_id, minlength=kE)
    print(f"[gen_data] expert_id histogram = {hist.tolist()}")


if __name__ == "__main__":
    kT = 256
    kH = 64
    kE = 16
    gen_golden_data(kT, kH, kE)
