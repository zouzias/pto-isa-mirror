#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# router_topk_small - compare_outputs.py
#
# Compares the device output against the Python golden for two arrays:
#
#   1. topk_values  : float32, GM layout [K, T] (TRANSPOSED — see kernel header)
#   2. topk_indices : uint32 , GM layout [K, T] (TRANSPOSED)
#
# The kernel writes the result transposed because the natural [T, K] layout
# violates the 32-byte UB-burst alignment rule for narrow tiles
# ([pto_tile.hpp:1510-1522]); the ColMajor->RowMajor reshape trick from §A4
# trowsum produces [K, T]. This script reads [K, T] and transposes back to
# [T, K] for the diagnostic prints so the mismatch reports stay readable in
# the (token, k) idiom.
#
# The host driver pre-fills both output device buffers with poison patterns
# BEFORE the launch. This script reports those poison patterns as "kernel
# never wrote" — distinguishing that from "kernel wrote wrong values".
#
# Reads:
#   ./output/golden_topk_values.bin       (K * T float32; TRANSPOSED layout)
#   ./output/golden_topk_indices.bin      (K * T uint32 ; TRANSPOSED layout)
#   ./output/output_topk_values.bin       (K * T float32; device output, transposed)
#   ./output/output_topk_indices.bin      (K * T uint32 ; device output, transposed)
#   ./output/t.txt
#   ./output/k.txt
# --------------------------------------------------------------------------------

import os
import sys
import numpy as np

# Must match main.cpp:
POISON_VAL_BYTE = 0x5A   # FP32 topk_values
POISON_IDX_BYTE = 0x7B   # uint32 topk_indices


def looks_like_poison(raw_bytes: bytes, poison_byte: int) -> bool:
    return all(b == poison_byte for b in raw_bytes[:64])


def read_int(path: str) -> int:
    with open(path) as f:
        return int(f.read().strip())


def main():
    T = read_int("./output/t.txt")
    K = read_int("./output/k.txt")

    val_bytes = T * K * 4
    idx_bytes = T * K * 4

    with open("./output/output_topk_values.bin", "rb") as f:
        out_val_raw = f.read()
    with open("./output/output_topk_indices.bin", "rb") as f:
        out_idx_raw = f.read()

    if looks_like_poison(out_val_raw, POISON_VAL_BYTE):
        print(f"[compare] FAIL: topk_values is all 0x{POISON_VAL_BYTE:02X} — kernel never wrote.")
        sys.exit(1)
    if looks_like_poison(out_idx_raw, POISON_IDX_BYTE):
        print(f"[compare] FAIL: topk_indices is all 0x{POISON_IDX_BYTE:02X} — kernel never wrote.")
        sys.exit(1)

    # Read [K, T] (transposed layout) and transpose back to [T, K] for the
    # token-major diagnostic prints.
    golden_val = np.fromfile("./output/golden_topk_values.bin",  dtype=np.float32).reshape(K, T).T
    golden_idx = np.fromfile("./output/golden_topk_indices.bin", dtype=np.uint32 ).reshape(K, T).T
    out_val    = np.frombuffer(out_val_raw, dtype=np.float32).reshape(K, T).T
    out_idx    = np.frombuffer(out_idx_raw, dtype=np.uint32 ).reshape(K, T).T

    val_ok = np.allclose(out_val, golden_val, atol=1e-5, rtol=1e-5)
    idx_ok = np.array_equal(out_idx, golden_idx)

    if val_ok:
        print("[compare] topk_values  : PASS")
    else:
        diffs = np.abs(out_val - golden_val)
        bad   = np.argwhere(diffs > 1e-5)
        print(f"[compare] topk_values  : FAIL ({len(bad)} mismatches)")
        for (r, c) in bad[:8]:
            print(f"            row={r} k={c}  golden={golden_val[r,c]}  got={out_val[r,c]}")

    if idx_ok:
        print("[compare] topk_indices : PASS")
    else:
        bad = np.argwhere(out_idx != golden_idx)
        print(f"[compare] topk_indices : FAIL ({len(bad)} mismatches)")
        for (r, c) in bad[:8]:
            print(f"            row={r} k={c}  golden={golden_idx[r,c]}  got={out_idx[r,c]}")

    if val_ok and idx_ok:
        print("test success")
        sys.exit(0)
    else:
        print("test failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
