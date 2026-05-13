#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# moe_top1_full - gen_data.py
#
# Generates inputs and golden for the integrated top-1 MoE forward path:
#
#   logits        = X @ W_router                              (FP32)
#   expert_id[t]  = argmax_e logits[t, e]                     (uint32)
#   hidden[t]     = relu( X[t]      @ W1[expert_id[t]] )      (FP32 ->FP16 acc/scratch)
#   Y     [t]     =       hidden[t] @ W2[expert_id[t]]        (FP32)
#
# Tested shape (matches main.cpp / kernel.cpp constants):
#   T = 256
#   H = 64        hidden dim (and N dim of GEMM2 by design choice F=H)
#   F = 64        FFN intermediate dim
#   E = 16        experts (cube-aligned)
#   kTileM = 128
#
# Datatype contract (mirrors §A18 where overlapping):
#   X         : float16   [T, H]
#   W_router  : float16   [H, E]
#   W1        : float16   [E, H, F]
#   W2        : float16   [E, F, H]      (kF=H by design; kept consistent with §A18)
#   logits    : float32   [T, E]         debug
#   expert_id : uint32    [T]            debug
#   Y         : float32   [T, H]         primary
#
# Worst-case T_PADDED:
#   T_PADDED_MAX = T + kE * kTileM       coarse upper bound (used by host
#                                        scratch allocation; see README).
#
# Output files (all raw little-endian, contiguous, no header):
#   ./input/input_X.bin                  (T * H float16)
#   ./input/input_W_router.bin           (H * E float16)
#   ./input/input_W1.bin                 (kE * H * F float16)
#   ./input/input_W2.bin                 (kE * F * H float16)
#   ./output/golden_logits.bin           (T * E float32; debug)
#   ./output/golden_expert_id.bin        (T     uint32 ; debug)
#   ./output/golden_Y.bin                (T * H float32; primary)
#   ./output/t.txt                       (single int line; T)
#   ./output/t_padded_max.txt            (single int line; upper-bound scratch size)
# --------------------------------------------------------------------------------

import os
import numpy as np

np.random.seed(31)


def gen_golden_data(kT, kH, kF, kE, kTileM):
    # FP16-exact integer regime. Same range discipline as §A18 (|x| <= 4 keeps
    # GEMM products well inside the FP16-exact-integer boundary at this kH).
    X        = np.random.randint(-4, 5, size=(kT, kH)     ).astype(np.float16)
    W_router = np.random.randint(-4, 5, size=(kH, kE)     ).astype(np.float16)
    W1       = np.random.randint(-4, 5, size=(kE, kH, kF) ).astype(np.float16)
    W2       = np.random.randint(-4, 5, size=(kE, kF, kH) ).astype(np.float16)

    # Stage 1: router GEMM.
    logits      = (X.astype(np.float32) @ W_router.astype(np.float32)).astype(np.float32)
    expert_id   = np.argmax(logits, axis=1).astype(np.uint32)

    # Stage 2+: per-token FFN through the routed expert.
    Y = np.zeros((kT, kH), dtype=np.float32)
    for t in range(kT):
        e = int(expert_id[t])
        x_t    = X[t, :].astype(np.float32)               # [H]
        w1_e   = W1[e, :, :].astype(np.float32)           # [H, F]
        w2_e   = W2[e, :, :].astype(np.float32)           # [F, H]
        h_t_f32 = x_t @ w1_e                              # [F]   FP32
        # Match the §A18 fused TSTORE: ReLU on FP32 acc, then FP32 -> FP16
        # cast for scratch. Re-cast back to FP32 for GEMM2.
        h_t_fp16 = np.maximum(h_t_f32, 0.0).astype(np.float16)
        h_t      = h_t_fp16.astype(np.float32)
        Y[t, :]  = h_t @ w2_e

    # Worst-case T_PADDED upper bound: every expert gets at least one token
    # padded to kTileM, plus its real count rounded up. Coarse but safe.
    T_PADDED_MAX = int(kT + kE * kTileM)

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)
    X.tofile("./input/input_X.bin")
    W_router.tofile("./input/input_W_router.bin")
    W1.tofile("./input/input_W1.bin")
    W2.tofile("./input/input_W2.bin")
    logits.tofile("./output/golden_logits.bin")
    expert_id.tofile("./output/golden_expert_id.bin")
    Y.tofile("./output/golden_Y.bin")
    with open("./output/t.txt", "w") as f:
        f.write(f"{kT}\n")
    with open("./output/t_padded_max.txt", "w") as f:
        f.write(f"{T_PADDED_MAX}\n")

    print(f"[gen_data] kT = {kT}  kH = {kH}  kF = {kF}  kE = {kE}  kTileM = {kTileM}")
    print(f"[gen_data] T_PADDED_MAX = {T_PADDED_MAX} (host scratch upper bound)")
    print( "[gen_data] expert_id[:16]      =", expert_id[:16].tolist())
    print(f"[gen_data] logits range = [{float(logits.min()):.1f}, {float(logits.max()):.1f}]")
    print(f"[gen_data] Y      range = [{float(Y.min()):.1f},     {float(Y.max()):.1f}]")
    hist = np.bincount(expert_id, minlength=kE)
    print(f"[gen_data] expert_id histogram = {hist.tolist()}")


if __name__ == "__main__":
    kT     = 256
    kH     = 64
    kF     = 64
    kE     = 16
    kTileM = 128
    gen_golden_data(kT, kH, kF, kE, kTileM)
