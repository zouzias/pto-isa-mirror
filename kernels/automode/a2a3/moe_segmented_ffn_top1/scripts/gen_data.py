#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# moe_segmented_ffn_top1 - gen_data.py
#
# Full per-expert FFN: GEMM1 -> ReLU -> GEMM2. Reuses §A15/§A16/§A17 layout
# (host-padded expert segments, stable argsort grouping) and adds a second
# weight matrix w2 plus the final-layer golden.
#
# Pipeline (Python-only):
#
#   packed_tokens [T_PADDED, H]   FP16
#   w1            [kE, H, F]      FP16
#   w2            [kE, F, O]      FP16
#     ↓ per expert e:
#       hidden   = packed_tokens[s:s+c]   @ w1[e]    (FP32)
#       hidden_r = max(hidden, 0)                    (FP32, post-ReLU)
#       y        = hidden_r @ w2[e]                  (FP32)
#       packed_output[s:s+c] = y
#
# Datatypes (canonical A3 auto-mode cube combo for both GEMMs):
#   inputs/weights : float16
#   accumulator    : float32
#   final output   : float32
#
# Numerical correctness:
#   Inputs in [-3, 4] (cast exact to FP16). One product fits in FP16 (max |3*3|=9
#   < 65504), inner products of length K=64 fit in FP32 (max 64*9 = 576).
#   ReLU(hidden) is FP32 -> FP16 cast: integer-valued FP32 in [-576, 576] cast
#   to FP16 IS exact (FP16 covers integers up to 2^11 = 2048 exactly).
#   Second GEMM products are |F=64 max-val FP16 * FP16 max-val| -> FP32 sum.
#   With ReLU clipping, average effective F is ~F/2 = 32; max FP16 magnitude
#   after layer 1 is ~576; product max |576 * 4| = 2304 (representable exactly
#   in FP16, but the FP32 accumulator dominates). Sum over 64 such products
#   is at most ~64 * 2304 = 147456 — within FP32 integer-exact range
#   (2^24 = 16777216).
#   So both GEMMs are bit-exact in this distribution.
#
# Output files (all raw little-endian, contiguous, no header):
#   ./input/input_packed_tokens.bin       (T_PADDED * H float16)
#   ./input/input_expert_count.bin        (kE        int32 )  PADDED counts
#   ./input/input_expert_start.bin        (kE        int32 )  PADDED starts
#   ./input/input_w1.bin                  (kE * H * F float16)
#   ./input/input_w2.bin                  (kE * F * O float16)
#   ./output/golden_packed_output.bin     (T_PADDED * O float32; final FFN out)
#   ./output/golden_hidden_relu.bin       (T_PADDED * F float32; post-ReLU; debug)
#   ./output/t_padded.txt                 (single int line; consumed by main.cpp)
#   ./output/expert_count_real.bin        (kE        int32 )  debug only
# --------------------------------------------------------------------------------

import math
import os
import numpy as np

np.random.seed(19)


def gen_golden_data(kT, kH, kF, kO, kE, kTileM):
    # ---- Small symmetric integer inputs (cast to float16 is exact) ---------
    tokens = np.random.randint(-3, 5, size=(kT, kH)).astype(np.float16)
    expert_id = np.random.choice(
        np.arange(kE, dtype=np.int32),
        size=kT,
        p=np.array([0.55, 0.20, 0.15, 0.10]),
    ).astype(np.int32)
    w1 = np.random.randint(-3, 5, size=(kE, kH, kF)).astype(np.float16)
    w2 = np.random.randint(-3, 5, size=(kE, kF, kO)).astype(np.float16)

    # ---- Real (unpadded) histogram and stable grouping --------------------
    expert_count_real = np.bincount(expert_id, minlength=kE).astype(np.int32)
    order = np.argsort(expert_id, kind="stable")

    # ---- Padded layout ----------------------------------------------------
    expert_count_padded = np.array(
        [int(math.ceil(int(c) / kTileM) * kTileM) for c in expert_count_real],
        dtype=np.int32,
    )
    expert_start_padded = np.zeros(kE, dtype=np.int32)
    expert_start_padded[1:] = np.cumsum(expert_count_padded[:-1])
    T_padded = int(expert_count_padded.sum())

    packed_tokens = np.zeros((T_padded, kH), dtype=np.float16)
    cursor = 0
    for e in range(kE):
        rc = int(expert_count_real[e])
        ps = int(expert_start_padded[e])
        sel = order[cursor : cursor + rc]
        packed_tokens[ps : ps + rc, :] = tokens[sel, :]
        cursor += rc

    # ---- Per-expert FFN (FP32 accumulator over FP16 inputs) ---------------
    hidden_relu = np.zeros((T_padded, kF), dtype=np.float32)
    golden_packed_output = np.zeros((T_padded, kO), dtype=np.float32)
    for e in range(kE):
        s = int(expert_start_padded[e])
        c = int(expert_count_padded[e])
        A  = packed_tokens[s : s + c, :].astype(np.float32)       # [c, H]
        W1 = w1[e, :, :].astype(np.float32)                       # [H, F]
        H1 = A @ W1                                               # [c, F]
        H1 = np.maximum(H1, 0.0)
        # IMPORTANT — to match the device kernel's behaviour, the post-ReLU
        # hidden state is DOWN-CAST to float16 on its way to GM (the
        # `TSTORE<..., ReluPreMode::NormalRelu>` overload's accDataType=float,
        # dstDataType=half lowering). Then GEMM2 loads it as FP16 inputs.
        H1_fp16 = H1.astype(np.float16)
        H1_back = H1_fp16.astype(np.float32)                      # [c, F]
        W2 = w2[e, :, :].astype(np.float32)                       # [F, O]
        Y  = H1_back @ W2                                         # [c, O]
        hidden_relu[s : s + c, :]            = H1_back
        golden_packed_output[s : s + c, :]   = Y

    # ---- Save -------------------------------------------------------------
    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)
    packed_tokens.tofile("./input/input_packed_tokens.bin")
    expert_count_padded.tofile("./input/input_expert_count.bin")
    expert_start_padded.tofile("./input/input_expert_start.bin")
    w1.tofile("./input/input_w1.bin")
    w2.tofile("./input/input_w2.bin")
    golden_packed_output.tofile("./output/golden_packed_output.bin")
    hidden_relu.tofile("./output/golden_hidden_relu.bin")
    with open("./output/t_padded.txt", "w") as f:
        f.write(f"{T_padded}\n")
    expert_count_real.tofile("./output/expert_count_real.bin")

    # ---- Debug ------------------------------------------------------------
    n_total       = int(hidden_relu.size)
    pre_relu_neg  = 0
    for e in range(kE):
        s = int(expert_start_padded[e]); c = int(expert_count_padded[e])
        A = packed_tokens[s : s + c, :].astype(np.float32)
        W1 = w1[e, :, :].astype(np.float32)
        pre_relu_neg += int((A @ W1 < 0).sum())
    print("[gen_data] dtype packed_tokens = float16")
    print("[gen_data] dtype w1            = float16")
    print("[gen_data] dtype w2            = float16")
    print("[gen_data] dtype packed_output = float32 (final FFN output)")
    print("[gen_data] dtype hidden_relu   = float32 (post-ReLU; on-device this is FP16 in scratch GM)")
    print(f"[gen_data] kT       = {kT}")
    print(f"[gen_data] kH       = {kH}")
    print(f"[gen_data] kF       = {kF}")
    print(f"[gen_data] kO       = {kO}")
    print(f"[gen_data] kE       = {kE}")
    print(f"[gen_data] kTileM   = {kTileM}")
    print(f"[gen_data] T_padded = {T_padded}")
    print("[gen_data] expert_count_real   =", expert_count_real.tolist())
    print("[gen_data] expert_count_padded =", expert_count_padded.tolist())
    print("[gen_data] expert_start_padded =", expert_start_padded.tolist())
    inner_iters = (expert_count_padded // kTileM).tolist()
    print(f"[gen_data] inner m0 iters per expert = {inner_iters} "
          f"(sum: {sum(inner_iters)} (GEMM1 + ReLU + GEMM2) tile-iters)")
    print(f"[gen_data] PRE-ReLU hidden negatives (clipped to 0 by ReLU): "
          f"{pre_relu_neg}/{n_total} ({100.0 * pre_relu_neg / n_total:.1f}%)")
    print("[gen_data] golden_packed_output[0, :8]  =", golden_packed_output[0, :8].tolist())
    print("[gen_data] hidden_relu[0, :8]           =", hidden_relu[0, :8].tolist())
    print("[gen_data] golden_packed_output[T-1, :8] =", golden_packed_output[-1, :8].tolist())


if __name__ == "__main__":
    kT     = 256
    kH     = 64
    kF     = 64
    kO     = 64
    kE     = 4
    kTileM = 128
    gen_golden_data(kT, kH, kF, kO, kE, kTileM)
