#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# moe_segmented_ffn_top1 - compare_outputs.py
#
# Same first-mismatch breakdown as the §A17 GEMM-relu compare script, with
# an additional check: the FFN has TWO stages, and a failure here could mean:
#   (a) GEMM1 is wrong  -> hidden state mismatch
#   (b) ReLU didn't fire / fired wrong -> hidden zero/nonzero pattern wrong
#   (c) FP32->FP16 down-cast of hidden state is lossy / wrong
#   (d) GEMM2 is wrong  -> final output mismatch despite correct hidden
#
# This script compares packed_output (final FFN output) vs the Python golden,
# and uses golden_hidden_relu.bin (FP32 post-ReLU hidden) to give context
# about how much pre-GEMM2 information was already in the hidden buffer.
#
# Run after `bash run.sh -r npu -v Ascend910B*`:
#   python ./scripts/compare_outputs.py
#
# Reads:
#   ./output/golden_packed_output.bin    (T_PADDED * kO float32)
#   ./output/output_packed_output.bin    (T_PADDED * kO float32)
#   ./output/golden_hidden_relu.bin      (T_PADDED * kF float32; debug)
#   ./output/t_padded.txt
#   ./input/input_expert_count.bin       (PADDED counts)
#   ./input/input_expert_start.bin       (PADDED starts)
#   ./output/expert_count_real.bin       (REAL counts; padded-vs-real labels)
# --------------------------------------------------------------------------------

import os
import sys
import numpy as np


def main():
    kO     = 64
    kF     = 64
    kE     = 4
    kTileM = 128

    if not os.path.exists("./output/t_padded.txt"):
        print("[compare] missing ./output/t_padded.txt (run scripts/gen_data.py first)")
        sys.exit(2)
    with open("./output/t_padded.txt") as f:
        T_padded = int(f.read().strip())

    gold_path = "./output/golden_packed_output.bin"
    got_path  = "./output/output_packed_output.bin"
    hidden_path = "./output/golden_hidden_relu.bin"
    for p in (gold_path, got_path):
        if not os.path.exists(p):
            print(f"[compare] missing {p}")
            sys.exit(2)

    gold = np.fromfile(gold_path, dtype=np.float32).reshape(T_padded, kO)
    got  = np.fromfile(got_path,  dtype=np.float32).reshape(T_padded, kO)

    have_hidden = os.path.exists(hidden_path)
    hidden = (np.fromfile(hidden_path, dtype=np.float32).reshape(T_padded, kF)
              if have_hidden else None)

    expert_count_padded = np.fromfile("./input/input_expert_count.bin", dtype=np.int32)
    expert_start_padded = np.fromfile("./input/input_expert_start.bin", dtype=np.int32)

    real_path = "./output/expert_count_real.bin"
    if os.path.exists(real_path):
        expert_count_real = np.fromfile(real_path, dtype=np.int32)
        have_real = True
    else:
        expert_count_real = np.zeros(kE, dtype=np.int32)
        have_real = False

    diff = got - gold
    abs_diff = np.abs(diff)
    max_abs = float(abs_diff.max())

    row_eq = np.all(diff == 0, axis=1)
    n_match = int(row_eq.sum())
    n_mismatch = T_padded - n_match

    print(f"[compare] shape:        ({T_padded}, {kO}) float32 (final FFN output)")
    print(f"[compare] max abs diff: {max_abs:.6g}")
    print(f"[compare] matching rows:    {n_match}/{T_padded}")
    print(f"[compare] mismatching rows: {n_mismatch}/{T_padded}")

    if n_mismatch == 0 and max_abs == 0.0:
        print("[compare] PASS (bit-exact)")
        sys.exit(0)

    if np.allclose(gold, got, atol=1e-2, rtol=0.0):
        print("[compare] PASS (within atol=1e-2)")
        sys.exit(0)

    flat_gold = gold.reshape(-1)
    flat_got  = got.reshape(-1)
    flat_neq  = flat_gold != flat_got
    if not flat_neq.any():
        print("[compare] FAIL — no flat-index mismatch found despite tolerance check failing.")
        sys.exit(1)

    first_flat = int(np.argmax(flat_neq))
    first_row  = first_flat // kO
    first_col  = first_flat %  kO

    def segment_of(row):
        for e in range(kE):
            s = int(expert_start_padded[e])
            c = int(expert_count_padded[e])
            if s <= row < s + c:
                return e, s, c
        return None, None, None

    e0, s0, c0 = segment_of(first_row)

    print()
    print(f"[compare] FAIL  total mismatching rows: {n_mismatch}/{T_padded}")
    print(f"[compare] ---- first mismatch ----")
    print(f"  flat_index               = {first_flat}  (0x{first_flat:x})")
    print(f"  row                      = {first_row}  (flat // kO={kO})")
    print(f"  col                      = {first_col}  (flat %  kO={kO})")
    print(f"  golden (final)           = {float(gold[first_row, first_col])}")
    print(f"  got    (device)          = {float(got [first_row, first_col])}")
    print(f"  abs_diff                 = {float(abs_diff[first_row, first_col])}")

    # NEW: row-wise verdicts and a coarse failure classification.
    row_gold_all_zero = bool(np.all(gold[first_row, :] == 0.0))
    row_got_all_zero  = bool(np.all(got [first_row, :] == 0.0))
    print(f"  golden row all zero?     = {row_gold_all_zero}")
    print(f"  device row all zero?     = {row_got_all_zero}")
    print(f"  final-output expected    = "
          f"{'NONZERO (real token result)' if not row_gold_all_zero else 'ZERO (padded row or fully-clipped)'}")
    print(f"  final-output got         = "
          f"{'NONZERO' if not row_got_all_zero else 'ZERO'}")

    if e0 is None:
        print(f"  segment                  = OUT-OF-RANGE")
    else:
        offset_in_seg = first_row - s0
        tile_m0       = (offset_in_seg // kTileM) * kTileM
        tile_idx      = offset_in_seg // kTileM
        n_inner_iters = c0 // kTileM
        real_c        = int(expert_count_real[e0]) if have_real else None
        is_padded     = None if not have_real else (offset_in_seg >= real_c)
        print(f"  expert                   = {e0}")
        print(f"  expert_start[e]          = {s0}  (PADDED)")
        print(f"  expert_count[e]          = {c0}  (PADDED, multiple of {kTileM})")
        if have_real:
            print(f"  expert_count_real        = {real_c}  (REAL)")
            print(f"  is_padded_row            = {is_padded}")
        print(f"  offset_in_segment        = {offset_in_seg}")
        print(f"  tile_m0                  = {tile_m0}")
        print(f"  tile_idx                 = {tile_idx}  (0-based inner-loop iter)")
        print(f"  inner iters total        = {n_inner_iters}")

    # NEW: pull the post-ReLU FP32 hidden golden value at this (row, ...)
    # and report a few representative cols so the user can compare against
    # the debug device hidden if it is available.
    if have_hidden:
        col_show = [first_col, 0, 1, 2, 7, 31, 63]
        col_show = [c for c in col_show if 0 <= c < kF]
        print(f"  hidden_relu_fp32 (golden, post-ReLU) representative cols:")
        for c in col_show:
            print(f"    [{first_row}, {c}] = {float(hidden[first_row, c])}")

    if have_hidden:
        n_nonzero_hidden_row = int(np.count_nonzero(hidden[first_row, :]))
        n_zero_hidden_row    = kF - n_nonzero_hidden_row
        print()
        print(f"[compare] ---- hidden_relu (POST-ReLU, golden) context for row {first_row} ----")
        print(f"  hidden_relu[row, :8] = {hidden[first_row, :8].tolist()}")
        print(f"  hidden_relu[row, :] nonzero count = {n_nonzero_hidden_row}/{kF} "
              f"(zeros from ReLU = {n_zero_hidden_row})")
        if n_nonzero_hidden_row == 0:
            print(f"  NOTE: hidden_relu row is all zero -> final output should be all zero too.")
            print(f"         If got[row, :] != 0 here, the device left dirty data from a prior expert.")
        else:
            print(f"  NOTE: hidden_relu has nonzero values -> if device row is all zero,")
            print(f"         GEMM2 did not run (or read all-zero from hidden_scratch).")

    # NEW: optional cross-check with the debug-hidden output if it exists.
    # Lets the user see in ONE script whether (a) device hidden matches golden
    # hidden (A.combined OK) AND (b) device final matches golden final.
    debug_hidden_path = "./output/debug_hidden_fp16.bin"
    if os.path.exists(debug_hidden_path):
        try:
            dbg = np.fromfile(debug_hidden_path, dtype=np.float16).reshape(T_padded, kF)
            gold_h16_path = "./output/golden_hidden_fp16.bin"
            if os.path.exists(gold_h16_path):
                gh16 = np.fromfile(gold_h16_path, dtype=np.float16).reshape(T_padded, kF)
                dbg_diff = dbg.astype(np.float32) - gh16.astype(np.float32)
                dbg_max = float(np.abs(dbg_diff).max())
                dbg_match_rows = int(np.all(dbg_diff == 0, axis=1).sum())
                dbg_n_zero = int((dbg == 0).sum())
                print()
                print(f"[compare] ---- ALSO loaded debug_hidden_fp16.bin (GEMM1-only debug binary) ----")
                print(f"  debug hidden shape       = {dbg.shape}")
                print(f"  debug hidden max abs diff vs golden_hidden_fp16 = {dbg_max:.6g}")
                print(f"  debug hidden matching rows                       = {dbg_match_rows}/{T_padded}")
                print(f"  debug hidden zero elements                       = {dbg_n_zero}/{int(dbg.size)}")
                if dbg_n_zero == int(dbg.size):
                    print(f"  >>> Debug hidden is ALL ZEROS -> Assumption A.combined LIKELY BROKEN.")
                elif dbg_max == 0.0:
                    print(f"  >>> Debug hidden matches golden -> Assumption A.combined OK.")
                    print(f"      Main FFN failure is likely A.reuse (tile reuse across GEMM1/GEMM2).")
        except Exception as ex:
            print(f"[compare] note: could not parse debug_hidden_fp16.bin: {ex}")

    mismatch_idx_rows = np.where(~row_eq)[0]
    show = mismatch_idx_rows[: min(5, mismatch_idx_rows.size)]
    print()
    print(f"[compare] ---- first {show.size} mismatching rows ----")
    for row in show:
        e, s, c = segment_of(int(row))
        seg_lbl = (f"e={e} s={s} c={c} off={int(row) - s} tile#{(int(row) - s)//kTileM}"
                   if e is not None else "out-of-range")
        if have_real and e is not None:
            offs = int(row) - s
            seg_lbl += f" {'PAD' if offs >= int(expert_count_real[e]) else 'REAL'}"
        print(f"  row {row} ({seg_lbl}):")
        print(f"    golden[:8]   = {gold[row, :8].tolist()}")
        print(f"    got[:8]      = {got[row, :8].tolist()}")
        print(f"    abs_diff[:8] = {abs_diff[row, :8].tolist()}")
        if have_hidden:
            print(f"    hidden[:8]   = {hidden[row, :8].tolist()}")

    print()
    print(f"[compare] ---- mismatch counts per expert segment ----")
    for e in range(kE):
        s = int(expert_start_padded[e]); c = int(expert_count_padded[e])
        n_bad = int((~row_eq[s : s + c]).sum())
        n_tiles = c // kTileM
        per_tile = []
        for t in range(n_tiles):
            ts = s + t * kTileM; te = ts + kTileM
            per_tile.append(int((~row_eq[ts:te]).sum()))
        print(f"  expert {e}: {n_bad}/{c} bad rows, per-tile = {per_tile}")

    sys.exit(1)


if __name__ == "__main__":
    main()
