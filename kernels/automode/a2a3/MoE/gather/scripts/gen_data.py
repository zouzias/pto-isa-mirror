#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# gather - gen_data.py
#
# Generates inputs and the reference golden C for the softmax-weighted gather
# kernel. Generic over kTopK in {1, 2, 4, 8, 16}. Self-contained — no
# dependency on scatter / expert_ffn / moe_topk_padded.
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_B.bin       (kT*kTopK + 16) * kH    float32
#   ./input/input_A_id.bin    (kT*kTopK + 16)         int32   (trailing 16 = -1)
#   ./input/input_rank_id.bin (kT*kTopK + 16)         int32   (trailing 16 = -1, only consulted when kTopK > 1)
#   ./input/input_outVal.bin   kT * kPadded           float32 (cols kTopK..kPadded-1 = -1e30)
#   ./output/golden_C.bin     kT * kH                 float32
#
#   kPadded = max(8, kTopK) — softmax tile column padding for 32-byte UB alignment.
#
# For kTopK == 1 the softmax is degenerate (weight = 1.0) so the golden
# reduces to the v1 unweighted scatter-add (each token row written once,
# no accumulation). For kTopK > 1 the golden does the full softmax-weighted
# scatter-add.
# --------------------------------------------------------------------------------

import json
import os
from pathlib import Path

import numpy as np

np.random.seed(37)


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
    kT, kH, _kF, kE, kTopK = _case
else:
    # v1 shape — must match the C++ side.
    kT    = 256
    kH    = 64
    kE    = 32
    kTopK = 1

kPackedRows   = kT * kTopK
kOverspillPad = 16
kAlloc        = kPackedRows + kOverspillPad
kPadded       = max(8, kTopK)

# Sentinel value for padded outVal columns. Must be very negative so that
# exp(value - max) underflows to 0 in the kernel's softmax pipeline.
kOutValPad    = -1e30


def gen_golden_data():
    # Synthesize a valid A_id / rank_id consistent with what scatter would
    # produce. We don't actually run scatter — we just emulate its pack rule
    # (stable argsort over expert_id, with the kE distribution chosen freely).
    # For the gather test the specific expert values are irrelevant; only the
    # multiplicity pattern (each token t appears exactly kTopK times in A_id)
    # is what makes the kTopK > 1 weighted accumulation meaningful.
    expert_id = np.zeros((kT, kTopK), dtype=np.int32)
    for t in range(kT):
        expert_id[t] = np.random.choice(kE, size=kTopK, replace=False).astype(np.int32)
    expert_id_flat     = expert_id.flatten()
    token_ids_per_pair = np.repeat(np.arange(kT, dtype=np.int32), kTopK)
    ranks_per_pair     = np.tile  (np.arange(kTopK, dtype=np.int32), kT)
    order              = np.argsort(expert_id_flat, kind="stable")
    A_id_valid         = token_ids_per_pair[order].astype(np.int32)
    rank_id_valid      = ranks_per_pair[order].astype(np.int32)

    # B in [-3, 3] float32 (FFN output scale).
    B_valid = np.random.uniform(-3.0, 3.0, size=(kPackedRows, kH)).astype(np.float32)

    # outVal: per-row top-k logits, descending sorted. Range [-5, 5] keeps
    # softmax denominators well-conditioned (exp diffs stay in [exp(-10), 1]).
    outVal_unsorted = np.random.uniform(-5.0, 5.0, size=(kT, kTopK)).astype(np.float32)
    outVal_sorted   = -np.sort(-outVal_unsorted, axis=1)   # descending per row

    # Pad outVal columns kTopK..kPadded-1 with -1e30 so exp underflows to 0
    # in the kernel softmax pipeline.
    outVal_padded = np.full((kT, kPadded), kOutValPad, dtype=np.float32)
    outVal_padded[:, :kTopK] = outVal_sorted

    # Softmax weights per row (numerically stable: subtract per-row max).
    shifted     = outVal_sorted - outVal_sorted.max(axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    weights     = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)  # (kT, kTopK)

    # Golden C: zero-init, then weighted scatter-add.
    C = np.zeros((kT, kH), dtype=np.float32)
    for r in range(kPackedRows):
        t = int(A_id_valid[r])
        k = int(rank_id_valid[r])
        C[t] += weights[t, k] * B_valid[r]

    # Pad B / A_id / rank_id with the kOverspillPad trailing rows that the
    # kernel ignores. B padding = zeros; id arrays = -1 sentinel.
    B       = np.zeros((kAlloc, kH), dtype=np.float32)
    A_id    = np.full((kAlloc,), -1, dtype=np.int32)
    rank_id = np.full((kAlloc,), -1, dtype=np.int32)
    B[:kPackedRows]       = B_valid
    A_id[:kPackedRows]    = A_id_valid
    rank_id[:kPackedRows] = rank_id_valid

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)
    B.tofile            ("./input/input_B.bin")
    A_id.tofile         ("./input/input_A_id.bin")
    rank_id.tofile      ("./input/input_rank_id.bin")
    outVal_padded.tofile("./input/input_outVal.bin")
    C.tofile            ("./output/golden_C.bin")

    print(f"[gen_data] kT={kT} kH={kH} kTopK={kTopK} kPadded={kPadded} kAlloc={kAlloc}")
    print(f"[gen_data] A_id_valid head    = {A_id_valid[:8].tolist()}")
    print(f"[gen_data] rank_id_valid head = {rank_id_valid[:8].tolist()}")
    if kTopK == 1:
        print(f"[gen_data] kTopK == 1: weights are all 1.0 (degenerate softmax)")
    else:
        print(f"[gen_data] weights[0] = {weights[0].tolist()}")
        print(f"[gen_data] weight row sum (should all be ~1.0): "
              f"min={weights.sum(axis=1).min():.6f} max={weights.sum(axis=1).max():.6f}")
    print(f"[gen_data] C[0, :4] = {C[0, :4].tolist()}")


if __name__ == "__main__":
    gen_golden_data()
