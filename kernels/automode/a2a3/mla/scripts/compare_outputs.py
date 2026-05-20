#!/usr/bin/env python3
# compare_outputs.py — post-mortem comparison of all MLA pipeline stages.
#
# Usage (run from the build directory, or any dir with output/ and input/ siblings):
#   python ../scripts/compare_outputs.py
#
# Each stage is compared against its Python golden. A "POISONED" line means the
# kernel never wrote that buffer (all bytes equal the host poison constant).

import os
import sys
import numpy as np

HALF = np.float16

# Poison constants matching main.cpp (DeepSeek-V2 layout).
POISON = {
    "Q_nope":      0xA1,
    "C_kv":        0xA2,
    "C_cache":     0xA3,
    "K_nope":      0xA4,
    "V":           0xA5,
    "scores_nope": 0xA6,
    "probs":       0xA7,
    "out":         0xA8,
    "Q_rope":      0xB1,
    "K_rope":      0xB2,
    "Q_rope_rot":  0xB3,
    "K_rope_rot":  0xB4,
    "scores_rope": 0xB5,
    "C_q":         0xC1,
}

# (stage_name, golden_path, device_path, dtype, abs_tol, rel_tol)
STAGES = [
    # DeepSeek-V2 compressed Q path
    ("C_q",         "../output/golden_c_q.bin",         "../output/output_c_q.bin",         HALF, 0.05, 0.02),
    ("Q_nope",      "../output/golden_q.bin",           "../output/output_q.bin",           HALF, 0.10, 0.02),
    # KV path
    ("C_kv",        "../output/golden_c_kv.bin",        "../output/output_c_kv.bin",        HALF, 0.10, 0.02),
    # C_cache is a copy of C_kv; compare against golden_c_kv.
    ("C_cache",     "../output/golden_c_kv.bin",        "../output/output_c_cache.bin",     HALF, 0.10, 0.02),
    ("K_nope",      "../output/golden_k.bin",           "../output/output_k.bin",           HALF, 0.10, 0.02),
    ("V",           "../output/golden_v.bin",           "../output/output_v.bin",           HALF, 0.10, 0.02),
    # The cube path writes nope-only scores; the rope+nope sum lives only
    # inside the softmax kernel.
    ("scores_nope", "../output/golden_scores_nope.bin", "../output/output_scores.bin",      HALF, 0.20, 0.05),
    # Decoupled RoPE stages.
    ("Q_rope",      "../output/golden_q_rope.bin",      "../output/output_q_rope.bin",      HALF, 0.05, 0.02),
    ("K_rope",      "../output/golden_k_rope.bin",      "../output/output_k_rope.bin",      HALF, 0.05, 0.02),
    ("Q_rope_rot",  "../output/golden_q_rope_rot.bin",  "../output/output_q_rope_rot.bin",  HALF, 0.05, 0.02),
    ("K_rope_rot",  "../output/golden_k_rope_rot.bin",  "../output/output_k_rope_rot.bin",  HALF, 0.05, 0.02),
    ("scores_rope", "../output/golden_scores_rope.bin", "../output/output_scores_rope.bin", HALF, 0.20, 0.05),
    ("probs",       "../output/golden_probs.bin",       "../output/output_probs.bin",       HALF, 0.10, 0.02),
    ("out",         "../output/golden_out.bin",         "../output/output_out.bin",         HALF, 0.50, 0.05),
]

def load(path, dtype):
    if not os.path.exists(path):
        return None
    return np.fromfile(path, dtype=dtype)

def is_poisoned(arr_u8, poison_byte, check=256):
    sample = arr_u8[:check]
    return bool(np.all(sample == poison_byte))

def compare_stage(name, golden_path, device_path, dtype, abs_tol, rel_tol):
    golden = load(golden_path, dtype)
    device_raw = load(device_path, np.uint8)

    if device_raw is None:
        print(f"[{name:<14}] ERROR   — output file missing: {device_path}")
        return False

    poison_byte = POISON.get(name, 0xFF)
    if is_poisoned(device_raw, poison_byte):
        print(f"[{name:<14}] POISONED — kernel never wrote (poison=0x{poison_byte:02X})")
        return False

    device = device_raw.view(dtype)

    if golden is None:
        print(f"[{name:<14}] WRITTEN  — no golden to compare ({device_path} exists, not poisoned)")
        return True

    if golden.shape != device.shape:
        print(f"[{name:<14}] SHAPE MISMATCH  golden={golden.shape} device={device.shape}")
        return False

    g32 = golden.astype(np.float32)
    d32 = device.astype(np.float32)
    abs_err = np.abs(g32 - d32)
    tol = abs_tol + rel_tol * np.abs(g32)
    mask = abs_err > tol
    n_mismatch = int(mask.sum())
    max_err = float(abs_err.max())
    mean_err = float(abs_err.mean())
    ok = (n_mismatch == 0)

    status = "PASS" if ok else "FAIL"
    print(f"[{name:<14}] {status:<5}  n={golden.size:<8}  mismatches={n_mismatch:<8}"
          f"  max_err={max_err:.5f}  mean_err={mean_err:.5f}"
          f"  (tol={abs_tol}+{rel_tol}*|g|)")

    if not ok:
        # Show first few mismatches with context
        flat_g = g32.ravel()
        flat_d = d32.ravel()
        flat_e = abs_err.ravel()
        flat_mask = mask.ravel()
        shown = 0
        for i in np.where(flat_mask)[0][:8]:
            print(f"              [{i:>7}]  golden={flat_g[i]:+.5f}  device={flat_d[i]:+.5f}"
                  f"  err={flat_e[i]:.5f}")
            shown += 1

        # Error distribution
        bins = [0, 0.001, 0.01, 0.1, 0.5, 1.0, 10.0, float("inf")]
        labels = ["<0.001", "0.001-0.01", "0.01-0.1", "0.1-0.5", "0.5-1.0", "1-10", ">10"]
        hist, _ = np.histogram(flat_e, bins=bins)
        print(f"              error dist: " +
              "  ".join(f"{l}:{h}" for l, h in zip(labels, hist) if h > 0))

    return ok


def main():
    all_ok = True
    print("=" * 80)
    print("MLA pipeline per-stage comparison")
    print("=" * 80)
    for stage_args in STAGES:
        ok = compare_stage(*stage_args)
        all_ok = all_ok and ok
    print("=" * 80)
    if all_ok:
        print("ALL STAGES PASS")
    else:
        print("SOME STAGES FAILED — see FAIL/POISONED lines above")
    print("=" * 80)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
