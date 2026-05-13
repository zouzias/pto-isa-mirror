#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# moe_top1_full - compare_outputs.py
#
# Stage-isolation comparator. Runs three comparisons in order; if any
# upstream stage fails, downstream comparisons are diagnostic-only.
#
#   1. logits      [T, E]  float32   (stage 1: router GEMM)
#   2. expert_id   [T   ]  uint32    (stage 2: argmax + permute, argmax part)
#   3. Y           [T, H]  float32   (final MoE output; primary acceptance)
#
# The host driver pre-fills `logits`, `expert_id`, and `Y` device buffers
# with distinct poison patterns BEFORE the launch so we can distinguish
# "kernel never wrote" from "wrote wrong values".
# --------------------------------------------------------------------------------

import os
import sys
import numpy as np

POISON_LOGITS_BYTE   = 0x5A
POISON_EXPERTID_BYTE = 0x7B
POISON_Y_BYTE        = 0x4C

kE = 16
kH = 64


def looks_like_poison(raw_bytes: bytes, poison_byte: int) -> bool:
    return all(b == poison_byte for b in raw_bytes[:64])


def read_int(path: str) -> int:
    with open(path) as f:
        return int(f.read().strip())


def main():
    T = read_int("./output/t.txt")

    with open("./output/output_logits.bin",    "rb") as f: out_logits_raw = f.read()
    with open("./output/output_expert_id.bin", "rb") as f: out_expert_raw = f.read()
    with open("./output/output_Y.bin",         "rb") as f: out_Y_raw      = f.read()

    if looks_like_poison(out_logits_raw, POISON_LOGITS_BYTE):
        print(f"[compare] FAIL: logits is all 0x{POISON_LOGITS_BYTE:02X} — stage 1 (router GEMM) never wrote.")
        sys.exit(1)
    if looks_like_poison(out_expert_raw, POISON_EXPERTID_BYTE):
        print(f"[compare] FAIL: expert_id is all 0x{POISON_EXPERTID_BYTE:02X} — stage 2 (argmax+permute) never wrote.")
        sys.exit(1)
    if looks_like_poison(out_Y_raw, POISON_Y_BYTE):
        print(f"[compare] FAIL: Y is all 0x{POISON_Y_BYTE:02X} — pipeline never reached unpermute.")
        sys.exit(1)

    golden_logits  = np.fromfile("./output/golden_logits.bin",    dtype=np.float32).reshape(T, kE)
    golden_expert  = np.fromfile("./output/golden_expert_id.bin", dtype=np.uint32 )
    golden_Y       = np.fromfile("./output/golden_Y.bin",         dtype=np.float32).reshape(T, kH)
    out_logits     = np.frombuffer(out_logits_raw, dtype=np.float32).reshape(T, kE)
    out_expert     = np.frombuffer(out_expert_raw, dtype=np.uint32 )
    out_Y          = np.frombuffer(out_Y_raw,      dtype=np.float32).reshape(T, kH)

    logits_ok = np.allclose(out_logits, golden_logits, atol=1.0, rtol=1e-3)
    expert_ok = np.array_equal(out_expert, golden_expert)
    y_ok      = np.allclose(out_Y,        golden_Y,      atol=1e-3, rtol=1e-3)

    if logits_ok:
        print("[compare] logits     : PASS")
    else:
        print("[compare] logits     : FAIL — debug stage 1 (router GEMM) first.")

    if expert_ok:
        print("[compare] expert_id  : PASS")
    else:
        bad = np.argwhere(out_expert != golden_expert).flatten()
        print(f"[compare] expert_id  : FAIL ({len(bad)} mismatches; if logits PASS, suspect TROWARGMAX §2.7).")
        for t in bad[:8]:
            print(f"            t={t}  golden={golden_expert[t]}  got={out_expert[t]}")

    if y_ok:
        print("[compare] Y          : PASS")
    else:
        diffs = np.abs(out_Y - golden_Y)
        bad   = np.argwhere(diffs > 1e-3)
        print(f"[compare] Y          : FAIL ({len(bad)} mismatches)")
        for (r, c) in bad[:8]:
            print(f"            t={r} col={c}  golden={golden_Y[r,c]:.4f}  got={out_Y[r,c]:.4f}")

    if logits_ok and expert_ok and y_ok:
        print("test success")
        sys.exit(0)
    else:
        print("test failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
