#!/usr/bin/python3
# coding=utf-8
# gate_score_topk — gen_data.py
#
# Generates input and golden data for the DeepSeek-V4 score-branch routing
# step (deepseek/model.py:574-584).
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_scores.bin    T * N_ROUTED    float32   (un-biased)
#   ./input/input_bias.bin      N_ROUTED        float32   (zero if disabled)
#   ./output/golden_indices.bin T * N_ACTIVATED int32
#   ./output/golden_weights.bin T * N_ACTIVATED float32

import os
import sys
from pathlib import Path

import numpy as np

# Shared family case loader.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from case_utils import load_dsmoe_case  # noqa: E402

np.random.seed(42)

_case = load_dsmoe_case()
kT          = _case["t"]
kNRouted    = _case["n_routed"]
kNActivated = _case["n_activated"]

# Match the kernel-side build flags (defaults match ModelArgs).
USE_BIAS         = os.environ.get("DSMOE_BIAS",         "1") != "0"
SCORE_IS_SOFTMAX = os.environ.get("DSMOE_SCORE_SOFTMAX","0") == "1"
ROUTE_SCALE      = float(os.environ.get("DSMOE_ROUTE_SCALE", "1.0"))


def gen_golden_data():
    # original_scores: assume already activated (gate_softmax output), so in [0, 1].
    original = np.random.rand(kT, kNRouted).astype(np.float32)

    if USE_BIAS:
        bias = (np.random.rand(kNRouted) - 0.5).astype(np.float32) * 0.2
    else:
        bias = np.zeros(kNRouted, dtype=np.float32)

    # Biased view drives top-K selection.
    biased = original + bias[None, :]

    # Top-K indices (descending value); numpy returns ascending — flip.
    idx_part = np.argpartition(-biased, kth=kNActivated - 1, axis=-1)[:, :kNActivated]
    # Reorder partition result so values are descending.
    rows = np.arange(kT)[:, None]
    order = np.argsort(-biased[rows, idx_part], axis=-1)
    indices = idx_part[rows, order].astype(np.int32)

    # Gather original (un-biased) scores at the selected positions.
    weights = original[rows, indices].astype(np.float32)

    if not SCORE_IS_SOFTMAX:
        s = weights.sum(axis=-1, keepdims=True)
        s = np.where(s > 0, s, 1.0)
        weights = weights / s

    weights = weights * ROUTE_SCALE

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    original.tofile("./input/input_scores.bin")
    bias    .tofile("./input/input_bias.bin")
    indices .tofile("./output/golden_indices.bin")
    weights .tofile("./output/golden_weights.bin")

    print(f"[gen_data] T={kT} N_ROUTED={kNRouted} N_ACTIVATED={kNActivated}  "
          f"USE_BIAS={USE_BIAS} SCORE_IS_SOFTMAX={SCORE_IS_SOFTMAX} "
          f"ROUTE_SCALE={ROUTE_SCALE}")
    print(f"[gen_data] weight range: [{weights.min():.4f}, {weights.max():.4f}]")


if __name__ == "__main__":
    gen_golden_data()
