#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# moe_segmented_ffn_top1 - compare_outputs.py
#
# Stage-isolation comparator. Runs TWO comparisons in order:
#
#   1. SCRATCH  : output_scratch.bin        (FP16, [T_PADDED, F=64])
#                 vs golden_scratch.bin
#                 — verifies GEMM1 + ReLU + (FP32 acc -> FP16 GM in ND-layout TSTORE).
#                 If this is wrong, GEMM2's input is wrong, so the final output
#                 will also be wrong — debug GEMM1 first.
#
#   2. OUTPUT   : output_packed_output.bin  (FP32, [T_PADDED, H=64])
#                 vs golden_packed_output.bin
#                 — verifies GEMM2. Only meaningful once scratch matches.
#
# For each mismatch the script reports row/col decoding using:
#     row = flat_index // 64
#     col = flat_index %  64
# (NOT row = flat_index — that was the confusion that triggered this script.)
#
# The host driver pre-fills scratchDev with 0x7B bytes and outputDev with 0x5A
# bytes BEFORE the kernel runs. This script detects those poison patterns and
# reports them as "kernel never wrote this buffer" — distinguishing that from
# "kernel wrote zeros".
#
# Reads:
#   ./output/golden_packed_output.bin     (T_PADDED * kH float32; final FFN)
#   ./output/output_packed_output.bin     (T_PADDED * kH float32; device)
#   ./output/golden_scratch.bin           (T_PADDED * kF float16; debug)
#   ./output/output_scratch.bin           (T_PADDED * kF float16; device)   NEW
#   ./output/golden_gemm1_output.bin      (T_PADDED * kF float32; PRE-ReLU; debug)
#   ./output/t_padded.txt
#   ./input/input_expert_count.bin        (PADDED counts)
#   ./input/input_expert_start.bin        (PADDED starts)
#   ./output/expert_count_real.bin        (REAL counts; for padded-vs-real labels)
# --------------------------------------------------------------------------------

import os
import sys
import numpy as np

# Must match main.cpp:
#   constexpr uint8_t kPoisonOutput  = 0x5A;  // packed_output FP32
#   constexpr uint8_t kPoisonScratch = 0x7B;  // scratch       FP16
POISON_SCRATCH_BYTE = 0x7B
POISON_OUTPUT_BYTE  = 0x5A

# Hardcoded by the kernel namespace (kH = kF = 64). If those change, update.
kH     = 64
kF     = 64
kE     = 4
kTileM = 128


def load_padded_meta(T_padded):
    expert_count_padded = np.fromfile("./input/input_expert_count.bin", dtype=np.int32)
    expert_start_padded = np.fromfile("./input/input_expert_start.bin", dtype=np.int32)
    real_path = "./output/expert_count_real.bin"
    if os.path.exists(real_path):
        expert_count_real = np.fromfile(real_path, dtype=np.int32)
        have_real = True
    else:
        expert_count_real = np.zeros(kE, dtype=np.int32)
        have_real = False
    return expert_count_padded, expert_start_padded, expert_count_real, have_real


def segment_of(row, expert_start_padded, expert_count_padded):
    for e in range(kE):
        s = int(expert_start_padded[e])
        c = int(expert_count_padded[e])
        if s <= row < s + c:
            return e, s, c
    return None, None, None


def looks_like_poison(raw_bytes, poison_byte):
    """True iff every byte in raw_bytes equals poison_byte (kernel never wrote)."""
    if raw_bytes.size == 0:
        return False
    return bool(np.all(raw_bytes == poison_byte))


def fraction_poison(raw_bytes, poison_byte):
    if raw_bytes.size == 0:
        return 0.0
    return float(np.mean(raw_bytes == poison_byte))


def compare_buffer(name, actual_path, golden_path,
                   logical_cols, dtype, poison_byte,
                   expert_count_padded, expert_start_padded,
                   expert_count_real, have_real, T_padded,
                   atol=0.0):
    """Generic mismatch reporter. Returns (status, summary_dict) where status is
    one of 'pass', 'poison', 'all_zero', 'mismatch', 'missing'.
    """
    print()
    print("=" * 76)
    print(f"[compare:{name}] {actual_path}  vs  {golden_path}")
    print("=" * 76)

    if not os.path.exists(actual_path):
        print(f"  MISSING actual file: {actual_path}")
        return "missing", {}
    if not os.path.exists(golden_path):
        print(f"  MISSING golden file: {golden_path}")
        return "missing", {}

    raw_actual = np.fromfile(actual_path, dtype=np.uint8)
    raw_golden = np.fromfile(golden_path, dtype=np.uint8)

    actual_total = np.fromfile(actual_path, dtype=dtype)
    golden_total = np.fromfile(golden_path, dtype=dtype)
    if actual_total.size != T_padded * logical_cols:
        print(f"  WARN: actual element count {actual_total.size} != "
              f"T_padded * cols = {T_padded * logical_cols}")
    if golden_total.size != T_padded * logical_cols:
        print(f"  WARN: golden element count {golden_total.size} != "
              f"T_padded * cols = {T_padded * logical_cols}")

    actual = actual_total[: T_padded * logical_cols].reshape(T_padded, logical_cols)
    golden = golden_total[: T_padded * logical_cols].reshape(T_padded, logical_cols)

    # Promote to float32 for diff math (FP16 -> FP32 widens losslessly).
    actual_f = actual.astype(np.float32)
    golden_f = golden.astype(np.float32)

    # Poison check on the raw bytes.
    if looks_like_poison(raw_actual, poison_byte):
        print(f"  [POISON] every byte of actual is 0x{poison_byte:02X}.")
        print(f"           kernel never wrote {name}. This is fatal for {name};")
        print(f"           do not interpret values as numbers.")
        print(f"  nonzeros in actual (numeric): {int(np.count_nonzero(actual_f))} / {actual_f.size}")
        print(f"  nonzeros in golden (numeric): {int(np.count_nonzero(golden_f))} / {golden_f.size}")
        return "poison", {"poison": True}

    frac_poison_actual = fraction_poison(raw_actual, poison_byte)
    if frac_poison_actual > 0.5:
        print(f"  [PARTIAL POISON] {100.0 * frac_poison_actual:.1f}% of bytes in actual "
              f"still equal 0x{poison_byte:02X}.")
        print(f"  kernel likely wrote SOME rows but not all (suspect inner-loop bound, "
              f"e.g. expert_count[e] == 0 for some e or fewer iters than expected).")

    # Summary stats (numeric).
    nz_actual = int(np.count_nonzero(actual_f))
    nz_golden = int(np.count_nonzero(golden_f))
    print(f"  shape        : ({T_padded}, {logical_cols}) {np.dtype(dtype).name}")
    print(f"  nonzeros actual: {nz_actual:>8} / {actual_f.size}   "
          f"({100.0 * nz_actual / actual_f.size:.1f}%)")
    print(f"  nonzeros golden: {nz_golden:>8} / {golden_f.size}   "
          f"({100.0 * nz_golden / golden_f.size:.1f}%)")
    print(f"  row 0 actual[:16] = {actual_f[0, :16].tolist()}")
    print(f"  row 0 golden[:16] = {golden_f[0, :16].tolist()}")
    a_min = float(actual_f.min()) if actual_f.size else float("nan")
    a_max = float(actual_f.max()) if actual_f.size else float("nan")
    g_min = float(golden_f.min()) if golden_f.size else float("nan")
    g_max = float(golden_f.max()) if golden_f.size else float("nan")
    print(f"  range actual : [{a_min}, {a_max}]")
    print(f"  range golden : [{g_min}, {g_max}]")

    diff = actual_f - golden_f
    abs_diff = np.abs(diff)
    max_abs = float(abs_diff.max()) if abs_diff.size else 0.0
    print(f"  max abs diff : {max_abs:.6g}")

    if max_abs == 0.0:
        print(f"  PASS (bit-exact)")
        return "pass", {"max_abs": 0.0}
    if atol > 0.0 and np.allclose(golden_f, actual_f, atol=atol, rtol=0.0):
        print(f"  PASS (within atol={atol})")
        return "pass", {"max_abs": max_abs}

    if nz_actual == 0 and nz_golden > 0:
        print(f"  [ALL-ZERO actual] but golden has nonzero values.")
        print(f"  Diagnosis: kernel cleared the buffer (or auto-allocator left it 0)")
        print(f"  but did not produce computed values. For SCRATCH this almost always")
        print(f"  means the GEMM1 TSTORE with fused ReLU+downcast in ND layout did")
        print(f"  not actually emit data — the new assumption fired.")

    # First mismatch: report flat index + row/col decoding.
    flat_actual = actual_f.reshape(-1)
    flat_golden = golden_f.reshape(-1)
    flat_neq    = flat_actual != flat_golden
    first_flat  = int(np.argmax(flat_neq)) if flat_neq.any() else -1
    if first_flat >= 0:
        first_row = first_flat // logical_cols
        first_col = first_flat %  logical_cols
        print(f"  ---- first mismatch ----")
        print(f"    flat_index  = {first_flat}  (0x{first_flat:x})")
        print(f"    row         = flat // {logical_cols} = {first_row}")
        print(f"    col         = flat %  {logical_cols} = {first_col}")
        print(f"    actual      = {float(actual_f[first_row, first_col])}")
        print(f"    golden      = {float(golden_f[first_row, first_col])}")
        print(f"    abs_diff    = {float(abs_diff[first_row, first_col])}")
        print(f"    actual row {first_row} [:16] = {actual_f[first_row, :16].tolist()}")
        print(f"    golden row {first_row} [:16] = {golden_f[first_row, :16].tolist()}")

        e0, s0, c0 = segment_of(first_row, expert_start_padded, expert_count_padded)
        if e0 is not None:
            offset_in_seg = first_row - s0
            tile_m0       = (offset_in_seg // kTileM) * kTileM
            tile_idx      = offset_in_seg // kTileM
            n_inner_iters = c0 // kTileM
            real_c        = int(expert_count_real[e0]) if have_real else None
            is_padded     = None if not have_real else (offset_in_seg >= real_c)
            print(f"    expert      = {e0}")
            print(f"    expert_start[e] = {s0} (PADDED)")
            print(f"    expert_count[e] = {c0} (PADDED, multiple of {kTileM})")
            if have_real:
                print(f"    expert_count_real = {real_c}")
            print(f"    offset_in_segment = {offset_in_seg}")
            print(f"    tile_m0     = {tile_m0}")
            print(f"    tile_idx    = {tile_idx}  (0-based inner-loop iter)")
            print(f"    inner iters = {n_inner_iters}")
            if have_real:
                print(f"    is_padded_row = {is_padded}")
        else:
            print(f"    segment     = OUT-OF-RANGE")

    # First 5 mismatching rows.
    row_eq = np.all(diff == 0, axis=1)
    mismatch_rows = np.where(~row_eq)[0]
    show = mismatch_rows[: min(5, mismatch_rows.size)]
    if show.size:
        print(f"  ---- first {show.size} mismatching rows ----")
        for row in show:
            e, s, c = segment_of(int(row), expert_start_padded, expert_count_padded)
            seg_lbl = (f"e={e} s={s} c={c} off={int(row) - s} "
                       f"tile#{(int(row) - s)//kTileM}"
                       if e is not None else "out-of-range")
            if have_real and e is not None:
                offs = int(row) - s
                seg_lbl += f" {'PAD' if offs >= int(expert_count_real[e]) else 'REAL'}"
            print(f"    row {row} ({seg_lbl}):")
            print(f"      actual[:16] = {actual_f[row, :16].tolist()}")
            print(f"      golden[:16] = {golden_f[row, :16].tolist()}")
            print(f"      diff[:16]   = {(actual_f[row, :16] - golden_f[row, :16]).tolist()}")

    return "mismatch", {"max_abs": max_abs,
                        "nz_actual": nz_actual,
                        "nz_golden": nz_golden,
                        "first_flat": first_flat}


def main():
    if not os.path.exists("./output/t_padded.txt"):
        print("[compare] missing ./output/t_padded.txt (run scripts/gen_data.py first)")
        sys.exit(2)
    with open("./output/t_padded.txt") as f:
        T_padded = int(f.read().strip())

    expert_count_padded, expert_start_padded, expert_count_real, have_real = \
        load_padded_meta(T_padded)

    print(f"[compare] T_padded = {T_padded}")
    print(f"[compare] expert_count_padded = {expert_count_padded.tolist()}")
    print(f"[compare] expert_start_padded = {expert_start_padded.tolist()}")
    if have_real:
        print(f"[compare] expert_count_real   = {expert_count_real.tolist()}")

    # ----- STAGE 1: SCRATCH (GEMM1 + ReLU + fused FP32->FP16 TSTORE) ---------
    scratch_status, _ = compare_buffer(
        name="SCRATCH",
        actual_path="./output/output_scratch.bin",
        golden_path="./output/golden_scratch.bin",
        logical_cols=kF,
        dtype=np.float16,
        poison_byte=POISON_SCRATCH_BYTE,
        expert_count_padded=expert_count_padded,
        expert_start_padded=expert_start_padded,
        expert_count_real=expert_count_real,
        have_real=have_real,
        T_padded=T_padded,
        # FP16 scratch in this test is bit-exact under the value range used by
        # gen_data.py (integer values <= 2048). Use atol=0 -> require exact.
        atol=0.0,
    )

    # ----- STAGE 2: OUTPUT (GEMM2) ------------------------------------------
    output_status, _ = compare_buffer(
        name="OUTPUT",
        actual_path="./output/output_packed_output.bin",
        golden_path="./output/golden_packed_output.bin",
        logical_cols=kH,
        dtype=np.float32,
        poison_byte=POISON_OUTPUT_BYTE,
        expert_count_padded=expert_count_padded,
        expert_start_padded=expert_start_padded,
        expert_count_real=expert_count_real,
        have_real=have_real,
        T_padded=T_padded,
        atol=1e-3,
    )

    # ----- Stage-attribution verdict ----------------------------------------
    print()
    print("=" * 76)
    print("[compare] STAGE-ATTRIBUTION VERDICT")
    print("=" * 76)
    print(f"  scratch status: {scratch_status}")
    print(f"  output  status: {output_status}")

    verdict = "?"
    if scratch_status == "poison":
        verdict = ("Kernel did not write SCRATCH at all. Either auto-mode "
                   "produced an empty body, the GEMM1 TSTORE was elided, the "
                   "stream sync failed, or expert_count[e] == 0 for every e. "
                   "Check main.cpp's expert metadata printout and the "
                   "aclrtSynchronizeStream return code.")
    elif scratch_status == "mismatch":
        verdict = ("GEMM1 / ReLU / TSTORE-to-FP16-scratch is WRONG. This is "
                   "the new-assumption stage (FP32 Acc -> FP16 GM + ReLU in "
                   "ND-layout TSTORE). Stop here — GEMM2's input is wrong, so "
                   "fixing GEMM2 cannot make OUTPUT pass. Suggested next step: "
                   "set `kStopAfterGemm1 = true` in the kernel and re-run to "
                   "isolate. If scratch is still wrong, fall back to the "
                   "documented Strategy: write FP32 scratch (§A17 verbatim) "
                   "and add a separate FP32->FP16 cast pass.")
    elif scratch_status == "pass" and output_status == "poison":
        verdict = ("SCRATCH is correct but OUTPUT was never written. If "
                   "`kStopAfterGemm1` is true in the kernel, this is EXPECTED "
                   "and the milestone is now narrowed to GEMM1-only — that "
                   "stage is PASSING. Flip the toggle back to enable GEMM2 "
                   "and re-run.")
    elif scratch_status == "pass" and output_status == "mismatch":
        verdict = ("SCRATCH matches golden; GEMM1+ReLU+TSTORE is fine. The "
                   "failure is entirely in GEMM2. Compare against "
                   "moe_segmented_gemm_one_layer_kernel.cpp: GlobalDataA2 / "
                   "TileMatA2Data / GlobalDataB2 / TileMatB2Data / AccTile2, "
                   "the TLOAD/TMOV/TMATMUL/TSTORE order, b2 base "
                   "(w2 + e*kF*kH), and c2 base (packed_output + row*kH).")
    elif scratch_status == "pass" and output_status == "pass":
        verdict = "Both stages pass."
    elif scratch_status == "missing" or output_status == "missing":
        verdict = ("One of the actual files is missing. Confirm `bash run.sh "
                   "-r npu -v Ascend910B*` completed without error and that "
                   "main.cpp's WriteFile calls were reached.")
    print()
    print(f"  {verdict}")

    # Exit code: 0 only if BOTH stages pass.
    if scratch_status == "pass" and output_status == "pass":
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
