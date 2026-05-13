#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# moe_router_top1 - compare_outputs.py
#
# Compares device output against the Python golden for:
#
#   1. logits     [T, E] float32   (stage 1 GEMM output; debug)
#   2. expert_id  [T   ] uint32    (stage 2 argmax output; primary)
#
# Stage 1 is debug-only — if logits match but expert_id does not, the bug is
# in TROWARGMAX (likely the §2.7 PR-852 sync issue). If logits do not match,
# the GEMM itself is wrong; debug stage 1 first.
#
# Reads:
#   ./output/golden_logits.bin            (T * E float32)
#   ./output/golden_expert_id.bin         (T     uint32 )
#   ./output/output_logits.bin            (T * E float32; device)
#   ./output/output_expert_id.bin         (T     uint32 ; device)
#   ./output/t.txt
# --------------------------------------------------------------------------------

import os
import sys
import numpy as np

# Must match main.cpp:
POISON_LOGITS_BYTE   = 0x5A
POISON_EXPERTID_BYTE = 0x7B

kE = 16


def looks_like_poison(raw_bytes: bytes, poison_byte: int) -> bool:
    return all(b == poison_byte for b in raw_bytes[:64])


def read_int(path: str) -> int:
    with open(path) as f:
        return int(f.read().strip())


def main():
    T = read_int("./output/t.txt")

    logits_bytes = T * kE * 4
    expert_bytes = T      * 4

    with open("./output/output_logits.bin",    "rb") as f:
        out_logits_raw = f.read()
    with open("./output/output_expert_id.bin", "rb") as f:
        out_expert_raw = f.read()

    if looks_like_poison(out_logits_raw, POISON_LOGITS_BYTE):
        print(f"[compare] FAIL: logits is all 0x{POISON_LOGITS_BYTE:02X} — stage 1 (GEMM) never wrote.")
        sys.exit(1)
    if looks_like_poison(out_expert_raw, POISON_EXPERTID_BYTE):
        print(f"[compare] FAIL: expert_id is all 0x{POISON_EXPERTID_BYTE:02X} — stage 2 (TROWARGMAX) never wrote.")
        sys.exit(1)

    golden_logits  = np.fromfile("./output/golden_logits.bin",    dtype=np.float32).reshape(T, kE)
    golden_expert  = np.fromfile("./output/golden_expert_id.bin", dtype=np.uint32 )
    out_logits     = np.frombuffer(out_logits_raw,  dtype=np.float32).reshape(T, kE)
    out_expert     = np.frombuffer(out_expert_raw,  dtype=np.uint32 )

    logits_ok = np.allclose(out_logits, golden_logits, atol=1.0, rtol=1e-3)  # FP32 GEMM tolerance
    expert_ok = np.array_equal(out_expert, golden_expert)

    if logits_ok:
        print("[compare] logits     : PASS")
    else:
        diffs = np.abs(out_logits - golden_logits)
        bad   = np.argwhere(diffs > 1.0)
        print(f"[compare] logits     : FAIL ({len(bad)} mismatches)")
        for (r, c) in bad[:8]:
            print(f"            row={r} col={c}  golden={golden_logits[r,c]:.2f}  got={out_logits[r,c]:.2f}")

    if expert_ok:
        print("[compare] expert_id  : PASS")
    else:
        bad = np.argwhere(out_expert != golden_expert).flatten()
        print(f"[compare] expert_id  : FAIL ({len(bad)} mismatches)")
        for t in bad[:8]:
            print(f"            t={t}  golden={golden_expert[t]}  got={out_expert[t]}  "
                  f"logits_max_idx_actual={int(out_logits[t].argmax())}")

    if logits_ok and expert_ok:
        print("test success")
        sys.exit(0)
    else:
        print("test failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
