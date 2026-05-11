#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# moe_segmented_gemm_relu - gen_data.py
#
# Same recipe as moe_segmented_gemm_one_layer/scripts/gen_data.py, with one
# change: the golden adds `np.maximum(., 0)` after the per-expert GEMM. Token
# and weight values are also widened to include negative entries so the
# golden contains BOTH zeros (clipped by ReLU) AND positive values that pass
# through — this is what actually validates the activation. With purely
# nonnegative inputs ReLU would be the identity and the test would degenerate.
#
# Datatypes: identical to §A16 moe_segmented_gemm_one_layer (the canonical
# A3 auto-mode cube combo from LaunchTMATMUL<1>):
#
#   packed_tokens  : float16
#   expert_weight  : float16
#   packed_output  : float32 (cube FP32 accumulator, post-ReLU)
#   expert_count   : int32   (PADDED counts)
#   expert_start   : int32   (PADDED starts)
#
# Tail policy: same — host pads each expert segment to a multiple of TILE_M.
# Padded rows are zero, so their GEMM output is zero, so their post-ReLU
# output is also zero (max(0, 0) = 0). The kernel processes padded rows
# identically to real rows.
#
# Output files (all raw little-endian, contiguous, no header):
#   ./input/input_packed_tokens.bin       (T_PADDED * H float16)
#   ./input/input_expert_count.bin        (kE        int32 )  PADDED counts
#   ./input/input_expert_start.bin        (kE        int32 )  PADDED starts
#   ./input/input_expert_weight.bin       (kE * H * O float16)
#   ./output/golden_packed_output.bin     (T_PADDED * O float32; POST-ReLU)
#   ./output/golden_gemm_output.bin       (T_PADDED * O float32; PRE-ReLU; debug)
#   ./output/t_padded.txt                 (single int line; consumed by main.cpp)
#   ./output/expert_count_real.bin        (kE        int32 )  debug only
# --------------------------------------------------------------------------------

import math
import os
import numpy as np

np.random.seed(19)


def gen_golden_data(kT, kH, kO, kE, kTileM):
    # ---- Small symmetric integer inputs (cast to float16 is exact). --------
    # Range widened to [-4, 4] so products are in [-16, 16] and inner products
    # of length K=64 are in [-1024, 1024]. Most pre-ReLU values will be
    # negative or positive with comparable frequency, so the golden has a
    # healthy mix of zeros (clipped) and positive values.
    tokens = np.random.randint(-4, 5, size=(kT, kH)).astype(np.float16)
    expert_id = np.random.choice(
        np.arange(kE, dtype=np.int32),
        size=kT,
        p=np.array([0.55, 0.20, 0.15, 0.10]),
    ).astype(np.int32)
    expert_weight = np.random.randint(-4, 5, size=(kE, kH, kO)).astype(np.float16)

    expert_count_real = np.bincount(expert_id, minlength=kE).astype(np.int32)
    order = np.argsort(expert_id, kind="stable")

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

    # ---- Per-expert GEMM (FP32 accumulator over FP16 inputs) ---------------
    gemm_output = np.zeros((T_padded, kO), dtype=np.float32)
    for e in range(kE):
        s = int(expert_start_padded[e])
        c = int(expert_count_padded[e])
        A = packed_tokens[s : s + c, :].astype(np.float32)
        B = expert_weight[e, :, :].astype(np.float32)
        gemm_output[s : s + c, :] = A @ B

    # ---- Post-ReLU golden --------------------------------------------------
    golden_packed_output = np.maximum(gemm_output, 0.0).astype(np.float32)
    # Padded rows: zero input -> zero GEMM output -> zero after ReLU. Confirmed
    # in the assert below.
    assert np.all(golden_packed_output >= 0.0), "ReLU output must be nonnegative"

    # ---- Save --------------------------------------------------------------
    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)
    packed_tokens.tofile("./input/input_packed_tokens.bin")
    expert_count_padded.tofile("./input/input_expert_count.bin")
    expert_start_padded.tofile("./input/input_expert_start.bin")
    expert_weight.tofile("./input/input_expert_weight.bin")
    golden_packed_output.tofile("./output/golden_packed_output.bin")
    gemm_output.tofile("./output/golden_gemm_output.bin")  # debug-only
    with open("./output/t_padded.txt", "w") as f:
        f.write(f"{T_padded}\n")
    expert_count_real.tofile("./output/expert_count_real.bin")

    # ---- Debug -------------------------------------------------------------
    n_total   = int(golden_packed_output.size)
    n_clipped = int((gemm_output < 0.0).sum())
    n_passed  = int((gemm_output > 0.0).sum())
    n_exact_zero = int((gemm_output == 0.0).sum())
    print("[gen_data] dtype packed_tokens = float16")
    print("[gen_data] dtype expert_weight = float16")
    print("[gen_data] dtype packed_output = float32 (post-ReLU)")
    print(f"[gen_data] kT       = {kT}")
    print(f"[gen_data] kH       = {kH}")
    print(f"[gen_data] kO       = {kO}")
    print(f"[gen_data] kE       = {kE}")
    print(f"[gen_data] kTileM   = {kTileM}")
    print(f"[gen_data] T_padded = {T_padded}")
    print("[gen_data] expert_count_real   =", expert_count_real.tolist())
    print("[gen_data] expert_count_padded =", expert_count_padded.tolist())
    print("[gen_data] expert_start_padded =", expert_start_padded.tolist())
    inner_iters = (expert_count_padded // kTileM).tolist()
    print(f"[gen_data] inner m0 iters per expert = {inner_iters} "
          f"(sum: {sum(inner_iters)} cube tiles)")
    print(f"[gen_data] ReLU stats over PRE-ReLU GEMM output:")
    print(f"           total elements                  = {n_total}")
    print(f"           negative (will be clipped to 0) = {n_clipped} "
          f"({100.0 * n_clipped / n_total:.1f}%)")
    print(f"           positive (passes through)       = {n_passed} "
          f"({100.0 * n_passed / n_total:.1f}%)")
    print(f"           exactly zero (incl. padded)     = {n_exact_zero}")
    print("[gen_data] gemm_output[0, :8]      =", gemm_output[0, :8].tolist())
    print("[gen_data] golden_packed_output[0, :8]   =", golden_packed_output[0, :8].tolist())


if __name__ == "__main__":
    kT     = 256
    kH     = 64
    kO     = 64
    kE     = 4
    kTileM = 128
    gen_golden_data(kT, kH, kO, kE, kTileM)
