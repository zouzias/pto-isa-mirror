#!/usr/bin/env python3
# coding=utf-8
# --------------------------------------------------------------------------------
# HCA Compressor - compare_outputs.py
#
# Compares the NPU kernel output against the PyTorch golden reference.
# Both files are raw little-endian float16 arrays with no header.
#
# Usage (run from the build directory, or any dir with output/ as sibling):
#   python3 ../scripts/compare_outputs.py
#
# Exit codes: 0 = PASS, 1 = FAIL.
# --------------------------------------------------------------------------------

import sys

import numpy as np
import torch

GOLDEN_PATH     = "../output/golden_output.bin"
NPU_OUTPUT_PATH = "../output/output_npu.bin"

# Tolerances for FP16/BF16 with a 128-element softmax-weighted sum.
# The accumulation over 128 positions introduces rounding, so we allow
# a moderate absolute error and a small relative error.
ABS_TOL = 0.5
REL_TOL = 0.05


def load_fp16(path: str) -> torch.Tensor:
    arr = np.fromfile(path, dtype=np.float16)
    return torch.from_numpy(arr).float()


def print_error_stats(golden: torch.Tensor, output: torch.Tensor) -> None:
    abs_err = (golden - output).abs()
    tol     = ABS_TOL + REL_TOL * golden.abs()
    mask    = abs_err > tol
    n_total    = golden.numel()
    n_mismatch = int(mask.sum())
    max_err    = float(abs_err.max())
    mean_err   = float(abs_err.mean())

    status = "PASS" if n_mismatch == 0 else "FAIL"
    print(f"[compare] {status}")
    print(f"  elements   : {n_total}")
    print(f"  mismatches : {n_mismatch}")
    print(f"  max_err    : {max_err:.5f}  (atol={ABS_TOL}, rtol={REL_TOL})")
    print(f"  mean_err   : {mean_err:.5f}")

    if n_mismatch > 0:
        flat_g    = golden.ravel()
        flat_o    = output.ravel()
        flat_e    = abs_err.ravel()
        flat_mask = mask.ravel()
        print("  first mismatches:")
        for i in np.where(flat_mask.numpy())[0][:8]:
            print(
                f"    [{i:>7}]  golden={flat_g[i]:+.5f}"
                f"  npu={flat_o[i]:+.5f}  err={flat_e[i]:.5f}"
            )

        bins   = [0, 0.001, 0.01, 0.1, 0.5, 1.0, 10.0, float("inf")]
        labels = ["<0.001", "0.001-0.01", "0.01-0.1", "0.1-0.5", "0.5-1.0", "1-10", ">10"]
        hist, _ = np.histogram(flat_e.numpy(), bins=bins)
        print("  error dist: " + "  ".join(f"{l}:{h}" for l, h in zip(labels, hist) if h > 0))

    return n_mismatch == 0


def main() -> None:
    print("=" * 60)
    print("HCA Compressor output comparison")
    print(f"  golden : {GOLDEN_PATH}")
    print(f"  npu    : {NPU_OUTPUT_PATH}")
    print(f"  tol    : atol={ABS_TOL}, rtol={REL_TOL}")
    print("=" * 60)

    import os
    for p in (GOLDEN_PATH, NPU_OUTPUT_PATH):
        if not os.path.exists(p):
            print(f"[ERROR] File not found: {p}")
            sys.exit(1)

    golden = load_fp16(GOLDEN_PATH)
    output = load_fp16(NPU_OUTPUT_PATH)

    if golden.shape != output.shape:
        print(f"[ERROR] Shape mismatch: golden={golden.shape}, npu={output.shape}")
        sys.exit(1)

    try:
        torch.testing.assert_close(output, golden, atol=ABS_TOL, rtol=REL_TOL)
        ok = True
    except AssertionError:
        ok = False

    ok = print_error_stats(golden, output)

    print("=" * 60)
    if ok:
        print("PASS — NPU output matches PyTorch golden within tolerance")
    else:
        print("FAIL — see mismatches above")
    print("=" * 60)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
