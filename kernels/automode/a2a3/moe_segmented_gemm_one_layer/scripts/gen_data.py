#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# moe_segmented_gemm_one_layer - gen_data.py
#
# Builds inputs and golden for the first cube/GEMM milestone inside the
# working MoE expert-segment loop. Reuses the host-padded segment layout from
# moe_segmented_identity, but replaces the elementwise op with one expert-
# specific GEMM tile:
#
#   for e in range(num_experts):
#       s = expert_start_padded[e]
#       c = expert_count_padded[e]
#       packed_output[s:s+c, :O] =
#           packed_tokens[s:s+c, :H].astype(fp32) @ expert_weight[e].astype(fp32)
#
# Datatypes (mirroring tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp
# LaunchTMATMUL tilingKey=1 — the canonical A3 auto-mode-eligible cube combo):
#
#   packed_tokens  : float16
#   expert_weight  : float16
#   packed_output  : float32 (cube FP32 accumulator)
#   expert_count   : int32   (PADDED counts)
#   expert_start   : int32   (PADDED starts)
#
# Tail policy: identical to moe_segmented_identity — host pads each expert
# segment length to a multiple of TILE_M. Padded rows are zero, so their GEMM
# output is also zero (no bias). The kernel processes padded rows identically
# to real rows; no SetValidRow / partial-tile stores.
#
# Numerical correctness: token / weight values are random integers in [-4, 4]
# cast to float16. Products fit comfortably in float16 (|9*9|=81 vs FP16 max
# ~65504), and inner products of length K=64 produce sums bounded by
# 64 * 16 = 1024 — fits exactly in float32. So the golden equals the device
# output bit-exact under the cube's FP16xFP16 -> FP32 accumulator semantics.
#
# Output files (all raw little-endian, contiguous, no header):
#   ./input/input_packed_tokens.bin       (T_PADDED * H float16)
#   ./input/input_expert_count.bin        (kE        int32 )  PADDED counts
#   ./input/input_expert_start.bin        (kE        int32 )  PADDED starts
#   ./input/input_expert_weight.bin       (kE * H * O float16)
#   ./output/golden_packed_output.bin     (T_PADDED * O float32)
#   ./output/t_padded.txt                 (single int line; consumed by main.cpp)
#   ./output/expert_count_real.bin        (kE        int32 )  debug only
# --------------------------------------------------------------------------------

import math
import os
import numpy as np

np.random.seed(19)


def gen_golden_data(kT, kH, kO, kE, kTileM):
    # ---- deterministic small-integer inputs (cast to float16 is exact) -----
    tokens = np.random.randint(-4, 5, size=(kT, kH)).astype(np.float16)
    # Skewed distribution so at least one expert exceeds TILE_M, mirroring
    # moe_segmented_identity.
    expert_id = np.random.choice(
        np.arange(kE, dtype=np.int32),
        size=kT,
        p=np.array([0.55, 0.20, 0.15, 0.10]),
    ).astype(np.int32)
    # Per-expert weights, also small integers cast to float16.
    expert_weight = np.random.randint(-4, 5, size=(kE, kH, kO)).astype(np.float16)

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
    cursor_in_order = 0
    for e in range(kE):
        rc = int(expert_count_real[e])
        ps = int(expert_start_padded[e])
        sel = order[cursor_in_order : cursor_in_order + rc]
        packed_tokens[ps : ps + rc, :] = tokens[sel, :]
        cursor_in_order += rc

    # ---- golden GEMM per expert (FP32 accumulator over FP16 inputs) --------
    golden_packed_output = np.zeros((T_padded, kO), dtype=np.float32)
    for e in range(kE):
        s = int(expert_start_padded[e])
        c = int(expert_count_padded[e])
        A = packed_tokens[s : s + c, :].astype(np.float32)        # [c, H]
        B = expert_weight[e, :, :].astype(np.float32)             # [H, O]
        golden_packed_output[s : s + c, :] = A @ B                # [c, O]
    # Padded input rows are zero, so their GEMM output is exactly zero. The
    # kernel processes them identically and should produce 0 too.

    # ---- save --------------------------------------------------------------
    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)
    packed_tokens.tofile("./input/input_packed_tokens.bin")
    expert_count_padded.tofile("./input/input_expert_count.bin")
    expert_start_padded.tofile("./input/input_expert_start.bin")
    expert_weight.tofile("./input/input_expert_weight.bin")
    golden_packed_output.tofile("./output/golden_packed_output.bin")
    with open("./output/t_padded.txt", "w") as f:
        f.write(f"{T_padded}\n")
    expert_count_real.tofile("./output/expert_count_real.bin")

    # ---- debug -------------------------------------------------------------
    print("[gen_data] dtype packed_tokens = float16")
    print("[gen_data] dtype expert_weight = float16")
    print("[gen_data] dtype packed_output = float32 (cube FP32 accumulator)")
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
    print(f"[gen_data] inner m0 iters per expert = {inner_iters}  "
          f"(sum: {sum(inner_iters)} cube tiles)")
    print("[gen_data] expert_id[:32]      =", expert_id[:32].tolist())
    print("[gen_data] packed_tokens[0, :8]      =", packed_tokens[0, :8].tolist())
    print("[gen_data] expert_weight[0, 0, :8]   =", expert_weight[0, 0, :8].tolist())
    print("[gen_data] golden_packed_output[0, :8]   =", golden_packed_output[0, :8].tolist())
    print("[gen_data] golden_packed_output[T-1, :8] =", golden_packed_output[-1, :8].tolist())


if __name__ == "__main__":
    kT     = 256
    kH     = 64
    kO     = 64
    kE     = 4
    kTileM = 128
    gen_golden_data(kT, kH, kO, kE, kTileM)
