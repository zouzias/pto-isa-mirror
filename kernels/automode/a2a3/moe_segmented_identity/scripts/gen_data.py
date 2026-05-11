#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# moe_segmented_identity - gen_data.py
#
# Builds inputs and golden for the first segmented-loop A3 auto-mode kernel.
# The kernel walks dynamic expert segments described by (expert_start[e],
# expert_count[e]) and runs a fixed TILE_M = 128 microtile inside each
# segment. v1 uses host-side padding so every segment length is a multiple
# of TILE_M — the kernel therefore never needs SetValidRow / partial tile
# stores; padded rows are processed identically to real rows.
#
# Pipeline (Python-only):
#
#   tokens [T, H], expert_id [T]           (deterministic; see kT/kH/kE below)
#     ↓ stable argsort by expert_id
#   expert_count_real[e], packed order
#     ↓ pad each segment length to a multiple of TILE_M
#   expert_count_padded[e], expert_start_padded[e], T_PADDED
#     ↓ lay out packed_tokens with real rows first per expert,
#       padded rows initialised to 0.0
#   packed_tokens [T_PADDED, H]
#     ↓ golden_packed_output = packed_tokens + 1.0   (for ALL rows, padded too)
#
# Output files (all raw little-endian, contiguous, no header):
#   ./input/input_packed_tokens.bin       (T_PADDED * H float32)
#   ./input/input_expert_count.bin        (kE        int32)   PADDED counts
#   ./input/input_expert_start.bin        (kE        int32)   PADDED starts
#   ./output/golden_packed_output.bin     (T_PADDED * H float32)
#   ./output/t_padded.txt                 (single int line; consumed by main.cpp)
#   ./output/expert_count_real.bin        (kE        int32)   REAL counts
#                                          (debug only; consumed by
#                                           scripts/compare_outputs.py to
#                                           label padded vs real rows)
#
# Distribution choice: we use a deliberately skewed expert_id so at least one
# expert ends up with padded_count > TILE_M (i.e., the inner m0 loop runs
# more than once). This exercises the nested loop, not just the outer.
# --------------------------------------------------------------------------------

import math
import os
import numpy as np

np.random.seed(19)


def gen_golden_data(kT, kH, kE, kTileM):
    # ---- deterministic tokens (integer-valued FP32 for exactness) -----------
    tokens = np.random.randint(-9, 10, size=(kT, kH)).astype(np.float32)

    # ---- skewed expert_id so expert 0 gets > TILE_M tokens ------------------
    # Uniform [0, kE) over 256 tokens averages ~64/expert; we'd never exercise
    # the inner loop. Bias to roughly [55%, 20%, 15%, 10%] of T = [140, 51,
    # 38, 27] expected. Seeded; verify in the debug print.
    expert_id = np.random.choice(
        np.arange(kE, dtype=np.int32),
        size=kT,
        p=np.array([0.55, 0.20, 0.15, 0.10]),
    ).astype(np.int32)

    # ---- real (unpadded) histogram and stable grouping ----------------------
    expert_count_real = np.bincount(expert_id, minlength=kE).astype(np.int32)
    order = np.argsort(expert_id, kind="stable")            # [T] in expert order

    # ---- padded layout ------------------------------------------------------
    expert_count_padded = np.array(
        [int(math.ceil(int(c) / kTileM) * kTileM) for c in expert_count_real],
        dtype=np.int32,
    )
    expert_start_padded = np.zeros(kE, dtype=np.int32)
    expert_start_padded[1:] = np.cumsum(expert_count_padded[:-1])
    T_padded = int(expert_count_padded.sum())

    # ---- packed_tokens layout with zero-padded tails per expert -------------
    packed_tokens = np.zeros((T_padded, kH), dtype=np.float32)
    real_start_in_order = 0
    for e in range(kE):
        rc = int(expert_count_real[e])
        ps = int(expert_start_padded[e])
        # rows the kernel will see at [ps .. ps+rc) come from `tokens[order[real_start_in_order .. +rc]]`,
        # which (by stable argsort) preserves the original within-expert order.
        sel = order[real_start_in_order : real_start_in_order + rc]
        packed_tokens[ps : ps + rc, :] = tokens[sel, :]
        real_start_in_order += rc
        # rows at [ps+rc .. ps+padded_count[e]) stay zero (numpy zeros default).

    # ---- golden: identity-+1 over ALL padded rows (padded rows become 1.0) --
    golden_packed_output = packed_tokens + 1.0

    # ---- save ---------------------------------------------------------------
    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)
    packed_tokens.astype(np.float32).tofile("./input/input_packed_tokens.bin")
    expert_count_padded.tofile("./input/input_expert_count.bin")
    expert_start_padded.tofile("./input/input_expert_start.bin")
    golden_packed_output.astype(np.float32).tofile("./output/golden_packed_output.bin")
    # main.cpp reads this so it can size the GM buffers (T_PADDED depends on
    # the seeded distribution and is not a compile-time constant).
    with open("./output/t_padded.txt", "w") as f:
        f.write(f"{T_padded}\n")
    # Debug-only: compare_outputs.py uses this to label padded vs real rows
    # in the first-mismatch report.
    expert_count_real.tofile("./output/expert_count_real.bin")

    # ---- debug --------------------------------------------------------------
    print("[gen_data] kT          =", kT)
    print("[gen_data] kH          =", kH)
    print("[gen_data] kE          =", kE)
    print("[gen_data] kTileM      =", kTileM)
    print("[gen_data] expert_count_real   =", expert_count_real.tolist())
    print("[gen_data] expert_count_padded =", expert_count_padded.tolist())
    print("[gen_data] expert_start_padded =", expert_start_padded.tolist())
    print("[gen_data] T_padded            =", T_padded)
    # Per-expert inner loop iters = padded_count / TILE_M.
    iters = (expert_count_padded // kTileM).tolist()
    print("[gen_data] inner m0 iters per expert =", iters,
          "(sum:", sum(iters), "tiles)")
    print("[gen_data] expert_id[:32]      =", expert_id[:32].tolist())
    print("[gen_data] packed_tokens[0, :8] =", packed_tokens[0, :8].tolist())
    # Show the boundary between real and padded rows for expert 0.
    rc0 = int(expert_count_real[0])
    if rc0 < int(expert_count_padded[0]):
        print(f"[gen_data] last real row of expert 0 (row {rc0 - 1}, :8) ="
              f" {packed_tokens[rc0 - 1, :8].tolist()}")
        print(f"[gen_data] first padded row of expert 0 (row {rc0}, :8) ="
              f" {packed_tokens[rc0, :8].tolist()}  (should be all 0.0)")


if __name__ == "__main__":
    kT     = 256
    kH     = 64
    kE     = 4
    kTileM = 128
    gen_golden_data(kT, kH, kE, kTileM)
