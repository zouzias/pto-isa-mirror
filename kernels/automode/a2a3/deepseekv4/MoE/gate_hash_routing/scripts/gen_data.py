#!/usr/bin/python3
# coding=utf-8
# gate_hash_routing — gen_data.py
#
# Generates input and golden data for the DeepSeek-V4 hash-branch routing
# step (deepseek/model.py:577-578, 581-584).
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_ids.bin       T                   int32
#   ./input/input_tid2eid.bin   VOCAB * N_ACTIVATED int32
#   ./input/input_scores.bin    T * N_ROUTED        float32 (un-biased)
#   ./output/golden_indices.bin T * N_ACTIVATED     int32
#   ./output/golden_weights.bin T * N_ACTIVATED     float32

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
kVocab      = _case["vocab"]

SCORE_IS_SOFTMAX = os.environ.get("DSMOE_SCORE_SOFTMAX", "0") == "1"
ROUTE_SCALE      = float(os.environ.get("DSMOE_ROUTE_SCALE", "1.0"))


def gen_golden_data():
    # Token IDs in [0, VOCAB).
    input_ids = np.random.randint(0, kVocab, size=(kT,)).astype(np.int32)

    # tid2eid: each row picks N_ACTIVATED expert IDs in [0, N_ROUTED).
    # Deduplicate within a row so a token never routes the same expert twice.
    tid2eid = np.empty((kVocab, kNActivated), dtype=np.int32)
    for v in range(kVocab):
        tid2eid[v] = np.random.choice(kNRouted, size=kNActivated, replace=False)

    # original_scores: gate_softmax output range, treat as random in [0, 1].
    original = np.random.rand(kT, kNRouted).astype(np.float32)

    # Golden ops.
    indices = tid2eid[input_ids]                       # (T, N_ACTIVATED)
    rows = np.arange(kT)[:, None]
    weights = original[rows, indices].astype(np.float32)

    if not SCORE_IS_SOFTMAX:
        s = weights.sum(axis=-1, keepdims=True)
        s = np.where(s > 0, s, 1.0)
        weights = weights / s

    weights = weights * ROUTE_SCALE

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    input_ids.tofile("./input/input_ids.bin")
    tid2eid  .tofile("./input/input_tid2eid.bin")
    original .tofile("./input/input_scores.bin")
    indices  .tofile("./output/golden_indices.bin")
    weights  .tofile("./output/golden_weights.bin")

    print(f"[gen_data] T={kT} N_ROUTED={kNRouted} N_ACTIVATED={kNActivated} "
          f"VOCAB={kVocab}  SCORE_IS_SOFTMAX={SCORE_IS_SOFTMAX} "
          f"ROUTE_SCALE={ROUTE_SCALE}")
    print(f"[gen_data] tid2eid bytes: {tid2eid.nbytes}")


if __name__ == "__main__":
    gen_golden_data()
