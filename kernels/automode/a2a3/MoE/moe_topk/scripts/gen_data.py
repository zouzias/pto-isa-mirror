#!/usr/bin/python3
# coding=utf-8
# moe_topk - gen_data.py
#
# Generates input and golden data for the MoE top-K kernel.
# Kernel selects top-K values and indices per row of the logits matrix.
#   logits   : (kRows, kCols) float32  (router logits; one row per token)
#   idx      : (kCols,)       uint32   (identity row [0..kCols-1])
#   outVal   : (kRows, kTopK) float32  (top-K values, descending)
#   outIdx   : (kRows, kTopK) uint32   (matching expert indices)
#
# With kCols=32 (num_experts) and kTopK=2, TSORT32 alone fully sorts each row.
# No merge loop or tail block is needed.
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_logits.bin    kRows * kCols  float32
#   ./input/input_idx.bin       kCols          uint32
#   ./output/golden_val.bin     kRows * kTopK  float32
#   ./output/golden_idx.bin     kRows * kTopK  uint32

import json
import os
from pathlib import Path

import numpy as np

np.random.seed(42)


def _load_case_from_json():
    # Mirror generated_cases.h: first JSON entry drives the build.
    json_path = Path(__file__).resolve().parents[2] / "build" / "generated_cases.json"
    if not json_path.exists():
        return None
    payload = json.loads(json_path.read_text())
    if not payload:
        return None
    c = payload[0]
    return int(c["t"]), int(c["h"]), int(c["f"]), int(c["e"]), int(c["topk"])


_case = _load_case_from_json()
if _case is not None:
    kRows, _kH, _kF, kCols, _kTopK = _case
    # Mirror main.cpp:  kTopK = (kMoeTopK >= 2) ? kMoeTopK : 2;
    # TSORT32 requires at least 2 outputs per row.
    kTopK = _kTopK if _kTopK >= 2 else 2
else:
    kRows = 256  # num_tokens
    kCols = 32   # num_experts
    kTopK = 2    # top-k


def gen_golden_data():
    logits = np.random.uniform(-10.0, 10.0, size=(kRows, kCols)).astype(np.float32)
    idx    = np.arange(kCols, dtype=np.uint32)

    # Per-row descending sort (stable for ties).
    order    = np.argsort(-logits, axis=1, kind='stable')
    topk_idx = order[:, :kTopK].astype(np.uint32)
    topk_val = np.take_along_axis(logits, topk_idx.astype(np.int64), axis=1).astype(np.float32)

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    logits.tofile("./input/input_logits.bin")
    idx.tofile("./input/input_idx.bin")
    topk_val.tofile("./output/golden_val.bin")
    topk_idx.tofile("./output/golden_idx.bin")

    print(f"[gen_data] kRows={kRows}  kCols={kCols}  kTopK={kTopK}")
    print(f"[gen_data] logits.dtype={logits.dtype}")
    print(f"[gen_data] top-2 values row 0: {topk_val[0].tolist()}")
    print(f"[gen_data] top-2 indices row 0: {topk_idx[0].tolist()}")


if __name__ == "__main__":
    gen_golden_data()
