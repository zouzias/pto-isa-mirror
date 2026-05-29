#!/usr/bin/python3
# coding=utf-8
# gate_softmax — gen_data.py
#
# Generates input and golden data for the router score activation
# (deepseek/model.py:567-572). The default variant is sqrt(softplus(x));
# switch to softmax / sigmoid by setting SCORE_FUNC=softmax|sigmoid in env.
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_scores.bin   T * N_ROUTED  float32
#   ./output/golden_scores.bin T * N_ROUTED  float32

import os
import sys
from pathlib import Path

import numpy as np

# Shared family case loader.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from case_utils import load_dsmoe_case  # noqa: E402

np.random.seed(42)

_case = load_dsmoe_case()
kT       = _case["t"]
kNRouted = _case["n_routed"]

SCORE_FUNC = os.environ.get("SCORE_FUNC", "sqrtsoftplus").lower()


def _activate(x: np.ndarray) -> np.ndarray:
    if SCORE_FUNC == "softmax":
        # Stable row-softmax over the last axis.
        m = x.max(axis=-1, keepdims=True)
        e = np.exp(x - m)
        return e / e.sum(axis=-1, keepdims=True)
    if SCORE_FUNC == "sigmoid":
        # Stable sigmoid.
        return np.where(x >= 0,
                        1.0 / (1.0 + np.exp(-x)),
                        np.exp(x) / (1.0 + np.exp(x))).astype(np.float32)
    # sqrtsoftplus (default): sqrt(log(1 + exp(x))).
    # Stable softplus: max(0, x) + log(1 + exp(-|x|)).
    sp = np.maximum(0, x) + np.log1p(np.exp(-np.abs(x)))
    return np.sqrt(sp).astype(np.float32)


def gen_golden_data():
    # Logits typical magnitude after a 4096-K GEMM (without weight normalization)
    # can be large; for the prototype keep them in a moderate range so the
    # numpy reference stays well-conditioned.
    scores_in = (np.random.randn(kT, kNRouted) * 1.0).astype(np.float32)
    scores_out = _activate(scores_in)

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    scores_in.tofile ("./input/input_scores.bin")
    scores_out.tofile("./output/golden_scores.bin")

    print(f"[gen_data] T={kT} N_ROUTED={kNRouted}  variant={SCORE_FUNC}")
    print(f"[gen_data] in  range: [{scores_in.min():.3f}, {scores_in.max():.3f}]")
    print(f"[gen_data] out range: [{scores_out.min():.3f}, {scores_out.max():.3f}]")


if __name__ == "__main__":
    gen_golden_data()
