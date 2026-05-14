#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# gather - gen_data.py
#
# Generates inputs (B, A_id) and the reference golden C for the scatter-add
# gather kernel. Generic over kTopK in {1, 2, 4, 8, 16}. Self-contained — no
# dependency on scatter / expert_ffn.
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_B.bin     (kT*kTopK + 16) * kH    float32
#   ./input/input_A_id.bin  (kT*kTopK + 16)         int32   (trailing 16 = -1, ignored by kernel)
#   ./output/golden_C.bin   kT * kH                  float32
# --------------------------------------------------------------------------------

import os
import numpy as np

np.random.seed(37)

# v1 shape — must match the C++ side.
kT    = 256
kH    = 64
kTopK = 1

kPackedRows   = kT * kTopK
kOverspillPad = 16
kAlloc        = kPackedRows + kOverspillPad


def gen_golden_data():
    # Synthesize a valid A_id consistent with what scatter would produce:
    # build a random per-token (t, k) -> expert mapping, then stable-sort
    # the (t, k) pairs by expert -> per-pair token id = A_id[r].
    # For the gather test the specific expert values don't matter; only the
    # multiplicity pattern (each token t appears exactly kTopK times in A_id)
    # is what makes the kTopK>1 accumulation meaningful.
    expert_id = np.zeros((kT, kTopK), dtype=np.int32)
    for t in range(kT):
        expert_id[t] = np.random.choice(32, size=kTopK, replace=False).astype(np.int32)
    expert_id_flat     = expert_id.flatten()
    token_ids_per_pair = np.repeat(np.arange(kT, dtype=np.int32), kTopK)
    order              = np.argsort(expert_id_flat, kind="stable")
    A_id_valid         = token_ids_per_pair[order].astype(np.int32)

    # B in [-3, 3] float32 (FFN output scale).
    B_valid = np.random.uniform(-3.0, 3.0, size=(kPackedRows, kH)).astype(np.float32)

    # Pad with kOverspillPad rows: zeros for B (don't care), -1 for A_id
    # (the kernel only reads the first kPackedRows entries of A_id).
    B    = np.zeros((kAlloc, kH), dtype=np.float32)
    A_id = np.full((kAlloc,), -1, dtype=np.int32)
    B[:kPackedRows]    = B_valid
    A_id[:kPackedRows] = A_id_valid

    # Golden C: scatter-add (zero init, then C[A_id[r]] += B[r]).
    C = np.zeros((kT, kH), dtype=np.float32)
    for r in range(kPackedRows):
        C[int(A_id_valid[r])] += B_valid[r]

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)
    B.tofile   ("./input/input_B.bin")
    A_id.tofile("./input/input_A_id.bin")
    C.tofile   ("./output/golden_C.bin")

    print(f"[gen_data] kT={kT} kH={kH} kTopK={kTopK} kAlloc={kAlloc}")
    print(f"[gen_data] A_id_valid head = {A_id_valid[:8].tolist()}")
    print(f"[gen_data] C[0, :4] = {C[0, :4].tolist()}")


if __name__ == "__main__":
    gen_golden_data()
