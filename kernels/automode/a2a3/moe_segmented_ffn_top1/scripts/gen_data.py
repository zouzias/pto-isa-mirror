#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# moe_segmented_ffn_top1 - gen_data.py
#
# Builds inputs and golden for the first full top-1 segmented FFN milestone:
#
#   per expert e:
#     A1 = packed_tokens[start:start+padded_count, :H]      float16
#     B1 = w1[e]                                           float16
#     Acc1 = A1 @ B1                                       float32 (fp32 accumulator)
#     scratch[start:start+padded_count, :F] = relu(Acc1)   float16  (fused TSTORE)
#     A2 = scratch[start:start+padded_count, :F]           float16
#     B2 = w2[e]                                           float16
#     Acc2 = A2 @ B2                                       float32
#     packed_output[start:start+padded_count, :H] = Acc2   float32
#
# Datatype contract:
#
#   packed_tokens   : float16
#   w1, w2          : float16
#   scratch         : float16  (post-ReLU; FP32 acc -> FP16 GM + ReLU in one TSTORE)
#   packed_output   : float32  (cube FP32 accumulator on GEMM2)
#   expert_count    : int32    (PADDED counts; multiples of kTileM)
#   expert_start    : int32    (PADDED starts; prefix sum)
#
# Tail policy: identical to §A15 / §A16 / §A17 — host pads each expert
# segment length to a multiple of kTileM. Padded rows are zero. The padded
# rows produce zero GEMM1 output (zero in, zero out), which ReLU passes
# through as zero, which GEMM2 then multiplies by w2 to produce zero
# output. So padded rows trivially remain zero across the entire FFN.
#
# Numerical-stability note: with `tokens, w1, w2 ∈ {-4..4} ⊂ float16`, the
# GEMM1 inner-product magnitude is bounded by 64 * 16 = 1024; that fits
# exactly in float32 (and is exact when cast to float16, |1024| << 65504).
# After ReLU + downcast, scratch values lie in [0, 1024]. GEMM2 then sums
# 64 products, each bounded by 1024 * 4 = 4096, with sum bound 64 * 4096 =
# 262144 — still within float32 range. The float16 representation of
# intermediate scratch values is exact when they are integer-valued and
# their absolute value is <= 2048 (the FP16 mantissa precision boundary).
# To avoid FP16 rounding in scratch, gen_data widens the value range
# *modestly* — the same [-4, 4] range used by §A17 is reused, which keeps
# GEMM1 outputs comfortably inside the FP16-exact integer regime for this
# tested shape (kH = 64). For larger kH this assumption may not hold.
#
# Output files (all raw little-endian, contiguous, no header):
#   ./input/input_packed_tokens.bin       (T_PADDED * kH float16)
#   ./input/input_expert_count.bin        (kE          int32 )  PADDED counts
#   ./input/input_expert_start.bin        (kE          int32 )  PADDED starts
#   ./input/input_w1.bin                  (kE * kH * kF float16)
#   ./input/input_w2.bin                  (kE * kF * kH float16)
#   ./output/golden_packed_output.bin     (T_PADDED * kH float32; full FFN)
#   ./output/golden_scratch.bin           (T_PADDED * kF float16; post-ReLU; debug)
#   ./output/golden_gemm1_output.bin      (T_PADDED * kF float32; PRE-ReLU GEMM1; debug)
#   ./output/t_padded.txt                 (single int line; consumed by main.cpp)
#   ./output/expert_count_real.bin        (kE          int32 )  debug only
# --------------------------------------------------------------------------------

import math
import os
import numpy as np

np.random.seed(19)


def gen_golden_data(kT, kH, kF, kE, kTileM):
    # ---- Small symmetric integer inputs (cast to float16 is exact). --------
    # [-4, 4] matches §A17 gemm_relu so both ReLU-clipped and pass-through
    # values appear in scratch after GEMM1. With kH=64 the pre-ReLU GEMM1
    # output stays inside |x| <= 1024 (= 64 * 4 * 4), well inside the FP16
    # exact-integer regime (<= 2048). So scratch is bit-exact even though
    # it is FP16. The same range bounds GEMM2's FP32 output safely.
    tokens = np.random.randint(-4, 5, size=(kT, kH)).astype(np.float16)
    expert_id = np.random.choice(
        np.arange(kE, dtype=np.int32),
        size=kT,
        p=np.array([0.55, 0.20, 0.15, 0.10]),
    ).astype(np.int32)
    w1 = np.random.randint(-4, 5, size=(kE, kH, kF)).astype(np.float16)
    w2 = np.random.randint(-4, 5, size=(kE, kF, kH)).astype(np.float16)

    # ---- real (unpadded) histogram and stable grouping ---------------------
    expert_count_real = np.bincount(expert_id, minlength=kE).astype(np.int32)
    order = np.argsort(expert_id, kind="stable")            # [T] in expert order

    # ---- padded layout -----------------------------------------------------
    expert_count_padded = np.array(
        [int(math.ceil(int(c) / kTileM) * kTileM) for c in expert_count_real],
        dtype=np.int32,
    )
    expert_start_padded = np.zeros(kE, dtype=np.int32)
    expert_start_padded[1:] = np.cumsum(expert_count_padded[:-1])
    T_padded = int(expert_count_padded.sum())

    # ---- packed_tokens layout with zero-padded tails per expert ------------
    packed_tokens = np.zeros((T_padded, kH), dtype=np.float16)
    cursor = 0
    for e in range(kE):
        rc = int(expert_count_real[e])
        ps = int(expert_start_padded[e])
        sel = order[cursor : cursor + rc]
        packed_tokens[ps : ps + rc, :] = tokens[sel, :]
        cursor += rc

    # ---- Golden FFN: GEMM1 -> ReLU -> GEMM2 --------------------------------
    # Mirrors the kernel exactly: FP32 accumulator over FP16 inputs, then
    # FP16 cast for scratch, then FP32 accumulator over FP16 inputs again.
    gemm1_output_fp32 = np.zeros((T_padded, kF), dtype=np.float32)
    scratch_fp16      = np.zeros((T_padded, kF), dtype=np.float16)
    golden_output     = np.zeros((T_padded, kH), dtype=np.float32)

    for e in range(kE):
        s = int(expert_start_padded[e])
        c = int(expert_count_padded[e])
        A1 = packed_tokens[s : s + c, :].astype(np.float32)        # [c, H]
        B1 = w1[e, :, :].astype(np.float32)                        # [H, F]
        gemm1_output_fp32[s : s + c, :] = A1 @ B1                  # [c, F]

        # Fused step in the kernel: ReLU + FP32 -> FP16 in a single TSTORE.
        scratch_fp16[s : s + c, :] = np.maximum(
            gemm1_output_fp32[s : s + c, :], 0.0
        ).astype(np.float16)

        # GEMM2: scratch (FP16) @ w2[e] (FP16) -> FP32 packed_output.
        A2 = scratch_fp16[s : s + c, :].astype(np.float32)         # [c, F]
        B2 = w2[e, :, :].astype(np.float32)                        # [F, H]
        golden_output[s : s + c, :] = A2 @ B2                      # [c, H]

    # Padded rows: zero tokens -> zero GEMM1 -> zero scratch -> zero output.
    assert np.all(scratch_fp16 >= 0.0), "scratch must be nonnegative"

    # ---- save --------------------------------------------------------------
    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)
    packed_tokens.tofile("./input/input_packed_tokens.bin")
    expert_count_padded.tofile("./input/input_expert_count.bin")
    expert_start_padded.tofile("./input/input_expert_start.bin")
    w1.tofile("./input/input_w1.bin")
    w2.tofile("./input/input_w2.bin")
    golden_output.tofile("./output/golden_packed_output.bin")
    scratch_fp16.tofile("./output/golden_scratch.bin")          # debug-only
    gemm1_output_fp32.tofile("./output/golden_gemm1_output.bin")  # debug-only
    with open("./output/t_padded.txt", "w") as f:
        f.write(f"{T_padded}\n")
    expert_count_real.tofile("./output/expert_count_real.bin")

    # ---- debug -------------------------------------------------------------
    n_total       = int(gemm1_output_fp32.size)
    n_clipped     = int((gemm1_output_fp32 < 0.0).sum())
    n_passthr     = int((gemm1_output_fp32 > 0.0).sum())
    n_exact_zero  = int((gemm1_output_fp32 == 0.0).sum())
    print("[gen_data] dtype packed_tokens = float16")
    print("[gen_data] dtype w1, w2        = float16")
    print("[gen_data] dtype scratch       = float16 (post-ReLU, FP32 acc -> FP16 GM in one TSTORE)")
    print("[gen_data] dtype packed_output = float32")
    print(f"[gen_data] kT       = {kT}")
    print(f"[gen_data] kH       = {kH}")
    print(f"[gen_data] kF       = {kF}")
    print(f"[gen_data] kE       = {kE}")
    print(f"[gen_data] kTileM   = {kTileM}")
    print(f"[gen_data] T_padded = {T_padded}")
    print("[gen_data] expert_count_real   =", expert_count_real.tolist())
    print("[gen_data] expert_count_padded =", expert_count_padded.tolist())
    print("[gen_data] expert_start_padded =", expert_start_padded.tolist())
    inner_iters = (expert_count_padded // kTileM).tolist()
    print(f"[gen_data] inner m0 iters per expert = {inner_iters} "
          f"(sum: {sum(inner_iters)} cube tiles per GEMM step)")
    print(f"[gen_data] ReLU stats over PRE-ReLU GEMM1 output:")
    print(f"           total elements                  = {n_total}")
    print(f"           negative (will be clipped to 0) = {n_clipped} "
          f"({100.0 * n_clipped / n_total:.1f}%)")
    print(f"           positive (passes through)       = {n_passthr} "
          f"({100.0 * n_passthr / n_total:.1f}%)")
    print(f"           exactly zero (incl. padded)     = {n_exact_zero}")
    g1_min = float(gemm1_output_fp32.min())
    g1_max = float(gemm1_output_fp32.max())
    print(f"[gen_data] PRE-ReLU GEMM1 range: [{g1_min:.1f}, {g1_max:.1f}] "
          f"(must fit in FP16 exactly when nonnegative)")
    g2_min = float(golden_output.min())
    g2_max = float(golden_output.max())
    print(f"[gen_data] golden_output  range: [{g2_min:.1f}, {g2_max:.1f}]")
    print("[gen_data] scratch_fp16[0, :8]      =", scratch_fp16[0, :8].tolist())
    print("[gen_data] golden_output[0, :8]     =", golden_output[0, :8].tolist())


if __name__ == "__main__":
    kT     = 256
    kH     = 64
    kF     = 64
    kE     = 4
    kTileM = 128
    gen_golden_data(kT, kH, kF, kE, kTileM)
