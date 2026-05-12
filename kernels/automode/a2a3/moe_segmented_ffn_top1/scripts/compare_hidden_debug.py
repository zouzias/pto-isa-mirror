#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# moe_segmented_ffn_top1 - compare_hidden_debug.py
#
# Compare the device-written FP16 hidden state (produced by the debug
# executable `./moe_segmented_ffn_top1_debug`) against the Python golden
# `./output/golden_hidden_fp16.bin`. Purpose: validate Assumption A.combined
# (the combined-mode TSTORE FP32 Acc -> FP16 GM + ReLU in one call).
#
# If this PASSes  -> A.combined is fine. The main FFN failure (zero output)
#                    is on the GEMM2 / tile-reuse (A.reuse) side. Next step:
#                    declare separate Mat/Left/Right/Acc tiles for GEMM2.
# If this FAILs   -> A.combined is broken. Fall back to F1 (separate
#                    FP32-with-ReLU store + FP32->FP16 cast kernel) or F2
#                    (FP32->FP16 no-ReLU store + separate vector ReLU).
#
# Run from build/ after gen_data.py + moe_segmented_ffn_top1_debug:
#   python ../scripts/compare_hidden_debug.py
#
# Reads:
#   ./output/golden_hidden_fp16.bin     (T_PADDED * F float16)
#   ./output/debug_hidden_fp16.bin      (T_PADDED * F float16; written by the
#                                        debug executable)
#   ./output/t_padded.txt
#   ./input/input_expert_count.bin      (PADDED counts)
#   ./input/input_expert_start.bin      (PADDED starts)
#   ./output/expert_count_real.bin      (REAL counts; padded-vs-real labels)
# --------------------------------------------------------------------------------

import os
import sys
import numpy as np


def main():
    kF     = 64
    kE     = 4
    kTileM = 128

    if not os.path.exists("./output/t_padded.txt"):
        print("[compare_hidden] missing ./output/t_padded.txt (run scripts/gen_data.py first)")
        sys.exit(2)
    with open("./output/t_padded.txt") as f:
        T_padded = int(f.read().strip())

    gold_path = "./output/golden_hidden_fp16.bin"
    got_path  = "./output/debug_hidden_fp16.bin"
    for p in (gold_path, got_path):
        if not os.path.exists(p):
            print(f"[compare_hidden] missing {p}")
            print("                  (did you build and run ./moe_segmented_ffn_top1_debug ?)")
            sys.exit(2)

    gold = np.fromfile(gold_path, dtype=np.float16).reshape(T_padded, kF)
    got  = np.fromfile(got_path,  dtype=np.float16).reshape(T_padded, kF)

    expert_count_padded = np.fromfile("./input/input_expert_count.bin", dtype=np.int32)
    expert_start_padded = np.fromfile("./input/input_expert_start.bin", dtype=np.int32)
    real_path = "./output/expert_count_real.bin"
    have_real = os.path.exists(real_path)
    expert_count_real = (np.fromfile(real_path, dtype=np.int32)
                         if have_real else np.zeros(kE, dtype=np.int32))

    # FP16 deltas evaluated in FP32 to avoid float16 nan/inf issues in stats.
    diff = got.astype(np.float32) - gold.astype(np.float32)
    abs_diff = np.abs(diff)
    max_abs = float(abs_diff.max())

    n_total = int(gold.size)
    n_gold_zero    = int((gold == 0).sum())
    n_gold_nonzero = n_total - n_gold_zero
    n_got_zero     = int((got == 0).sum())
    n_got_nonzero  = n_total - n_got_zero

    row_eq = np.all(diff == 0, axis=1)
    n_match_rows    = int(row_eq.sum())
    n_mismatch_rows = T_padded - n_match_rows

    print(f"[compare_hidden] shape:                  ({T_padded}, {kF}) float16")
    print(f"[compare_hidden] max abs diff (in FP32): {max_abs:.6g}")
    print(f"[compare_hidden] matching rows:          {n_match_rows}/{T_padded}")
    print(f"[compare_hidden] mismatching rows:       {n_mismatch_rows}/{T_padded}")
    print(f"[compare_hidden] golden:  zero={n_gold_zero} nonzero={n_gold_nonzero} (total={n_total})")
    print(f"[compare_hidden] device:  zero={n_got_zero} nonzero={n_got_nonzero}")
    if n_got_zero == n_total:
        print(f"[compare_hidden] !! DEVICE OUTPUT IS ALL ZEROS — TSTORE never wrote anything,")
        print(f"                  OR the kernel ran but produced 0 (e.g., GEMM1 produced 0,")
        print(f"                  or the combined-mode TSTORE template-arg form is unsupported)")
    elif n_gold_zero > 0 and n_got_nonzero == 0:
        print(f"[compare_hidden] !! Device wrote 0 wherever golden has nonzero -> ReLU may be inverted")
        print(f"                  or the combined-mode TSTORE clipped EVERYTHING to zero.")

    if n_mismatch_rows == 0 and max_abs == 0.0:
        print("[compare_hidden] PASS (bit-exact) -> Assumption A.combined is OK at this shape.")
        print("                                     Failure of moe_segmented_ffn_top1 is on")
        print("                                     the GEMM2 / tile-reuse (A.reuse) side.")
        sys.exit(0)

    # FP16 ulp tolerance is harsher than FP32. For our integer-valued
    # distribution, golden_hidden_fp16 is an exact integer in [0, 576] and
    # cannot have rounding. Any nonzero diff is structural.
    if max_abs == 0.0:
        # All zero diffs but row_eq says mismatch? Shouldn't happen; guard.
        print("[compare_hidden] PASS (no nonzero diffs)")
        sys.exit(0)

    flat_gold = gold.reshape(-1).astype(np.float32)
    flat_got  = got.reshape(-1).astype(np.float32)
    flat_neq  = flat_gold != flat_got
    first_flat = int(np.argmax(flat_neq))
    first_row  = first_flat // kF
    first_col  = first_flat %  kF

    def segment_of(row):
        for e in range(kE):
            s = int(expert_start_padded[e])
            c = int(expert_count_padded[e])
            if s <= row < s + c:
                return e, s, c
        return None, None, None

    e0, s0, c0 = segment_of(first_row)

    print()
    print(f"[compare_hidden] FAIL  -> Assumption A.combined LIKELY BROKEN at this shape.")
    print(f"[compare_hidden] ---- first mismatch ----")
    print(f"  flat_index        = {first_flat}  (0x{first_flat:x})")
    print(f"  row               = {first_row}  (flat // kF={kF})")
    print(f"  col               = {first_col}  (flat %  kF={kF})")
    print(f"  golden (post-ReLU FP16) = {float(gold[first_row, first_col])}")
    print(f"  got    (device)         = {float(got [first_row, first_col])}")
    print(f"  abs_diff          = {float(abs_diff[first_row, first_col])}")
    if e0 is not None:
        offset_in_seg = first_row - s0
        tile_idx      = offset_in_seg // kTileM
        n_inner       = c0 // kTileM
        real_c        = int(expert_count_real[e0]) if have_real else None
        is_padded     = None if not have_real else (offset_in_seg >= real_c)
        print(f"  expert            = {e0}")
        print(f"  expert_start[e]   = {s0}  (PADDED)")
        print(f"  expert_count[e]   = {c0}  (PADDED, multiple of {kTileM})")
        if have_real:
            print(f"  expert_count_real = {real_c}  (REAL)")
            print(f"  is_padded_row     = {is_padded}")
        print(f"  offset_in_segment = {offset_in_seg}")
        print(f"  tile_idx          = {tile_idx} / {n_inner}")
    else:
        print(f"  segment           = OUT-OF-RANGE")

    mismatch_idx_rows = np.where(~row_eq)[0]
    show = mismatch_idx_rows[: min(5, mismatch_idx_rows.size)]
    print()
    print(f"[compare_hidden] ---- first {show.size} mismatching rows ----")
    for row in show:
        e, s, c = segment_of(int(row))
        seg_lbl = (f"e={e} s={s} c={c} off={int(row) - s} tile#{(int(row) - s)//kTileM}"
                   if e is not None else "out-of-range")
        if have_real and e is not None:
            offs = int(row) - s
            seg_lbl += f" {'PAD' if offs >= int(expert_count_real[e]) else 'REAL'}"
        print(f"  row {row} ({seg_lbl}):")
        print(f"    golden[:8] = {[float(x) for x in gold[row, :8]]}")
        print(f"    got[:8]    = {[float(x) for x in got [row, :8]]}")

    print()
    print(f"[compare_hidden] Suggested next action: try Fallback F1 (FP32 hidden + cast)")
    print(f"                  or F2 (FP32->FP16 no-ReLU store + separate vec ReLU).")
    sys.exit(1)


if __name__ == "__main__":
    main()
