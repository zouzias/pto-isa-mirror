#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# router_topk_small - gen_data.py
#
# Generates inputs and golden for a row-wise small-K TopK kernel:
#
#   input :  scores[T, E]                       float32
#   output:  topk_values[T, K]                  float32   (per-row top-K values, descending)
#   output:  topk_indices[T, K]                 uint32    (per-row top-K column indices, descending)
#
# Tested shape (matches main.cpp / kernel.cpp constants):
#   T = 256
#   E = 16
#   K = 4
#
# Why E = 16 instead of E = 4:
#   - 16 is a comfortable Vec-tile column count and matches the FP16 cube
#     blockAlign on A3 (kept consistent with moe_router_top1 / moe_top1_full).
#   - The kernel template parameter `kK` is configurable up to 10.
#
# Why FP32 (and not FP16):
#   - Router scores are usually computed in FP32 even when the routed weights
#     are FP16. The kernel template can be re-instantiated with `T = half` if
#     needed; the current main.cpp + golden are FP32-only.
#
# GM layout note: the kernel TSTOREs the result as a *transposed* [K, T]
# array (ColMajor [T, K] -> RowMajor [K, T] reshape). This is the §A4
# trowsum trick to satisfy the 32-byte UB-burst alignment for narrow tiles
# ([pto_tile.hpp:1510-1522]). gen_data.py therefore writes [K, T] to disk;
# the kernel writes [K, T]; compare_outputs.py reads [K, T] and transposes
# back to [T, K] for the diagnostic prints.
#
# Output files (all raw little-endian, contiguous, no header):
#   ./input/input_scores.bin              (T * E float32)
#   ./output/golden_topk_values.bin       (K * T float32; TRANSPOSED layout)
#   ./output/golden_topk_indices.bin      (K * T uint32 ; TRANSPOSED layout)
#   ./output/t.txt                        (single int: T)
#   ./output/k.txt                        (single int: K)
# --------------------------------------------------------------------------------

import os
import numpy as np

np.random.seed(23)


def gen_golden_data(kT, kE, kK):
    # Float32 scores; distribution doesn't matter much for correctness but
    # bound-safe values keep printable diagnostics readable.
    scores = np.random.uniform(-10.0, 10.0, size=(kT, kE)).astype(np.float32)

    # Per-row sort descending: argsort then take first K.
    order_desc        = np.argsort(-scores, axis=1, kind="stable")    # [T, E]
    topk_indices_np   = order_desc[:, :kK].astype(np.uint32)          # [T, K]
    topk_values_np    = np.take_along_axis(scores, order_desc[:, :kK], axis=1).astype(np.float32)

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)
    scores.tofile("./input/input_scores.bin")
    # Kernel emits [K, T] (TRANSPOSED); write golden the same way.
    np.ascontiguousarray(topk_values_np.T ).tofile("./output/golden_topk_values.bin")
    np.ascontiguousarray(topk_indices_np.T).tofile("./output/golden_topk_indices.bin")
    with open("./output/t.txt", "w") as f:
        f.write(f"{kT}\n")
    with open("./output/k.txt", "w") as f:
        f.write(f"{kK}\n")

    print(f"[gen_data] kT = {kT}")
    print(f"[gen_data] kE = {kE}")
    print(f"[gen_data] kK = {kK}")
    print("[gen_data] scores[0, :]          =", scores[0, :].tolist())
    print("[gen_data] topk_values[0, :]     =", topk_values_np[0, :].tolist())
    print("[gen_data] topk_indices[0, :]    =", topk_indices_np[0, :].tolist())


if __name__ == "__main__":
    kT = 256
    kE = 16
    kK = 4
    gen_golden_data(kT, kE, kK)
