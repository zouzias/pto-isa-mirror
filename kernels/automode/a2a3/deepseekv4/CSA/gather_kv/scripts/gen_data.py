#!/usr/bin/python3
# coding=utf-8
# gather_kv — gen_data.py
#
# Generates input + golden data for the gather-KV stage of sparse_attn.
# Mirrors kernel.py:322-325 for ONE pipelined block t = 0.
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_kv.bin             kv_rows * D   bfloat16
#   ./input/input_topk_idxs.bin      TOPK          int32
#   ./output/golden_kv_gathered.bin  BLOCK   * D   bfloat16

import os
import sys
from pathlib import Path

import numpy as np

# Shared family case loader.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from case_utils import load_csa_case  # noqa: E402

np.random.seed(42)

_case = load_csa_case()
kH         = _case["h"]
kD         = _case["d"]
kS         = _case["s"]            # TOPK length
kBlock     = _case["block"]
kNumBlocks = _case["num_blocks"]
# Sized so every valid topk_idx falls in [0, kKvRows).
kKvRows    = kS
# Compile-time block index baked into the binary.
kTblock    = 0


def _f32_to_bf16_bytes(x_f32: np.ndarray) -> bytes:
    """Cast FP32 -> BF16 bit pattern (uint16, round-to-nearest-even approx)."""
    raw = x_f32.astype(np.float32).view(np.uint32)
    raw_bf16 = ((raw + 0x8000) >> 16).astype(np.uint16)
    return raw_bf16.tobytes()


def _bf16_round(x_f32: np.ndarray) -> np.ndarray:
    """Return FP32 array rounded to BF16 precision (still stored as FP32)."""
    raw = x_f32.astype(np.float32).view(np.uint32)
    raw = (raw + 0x8000) & 0xFFFF0000
    return raw.view(np.float32).copy()


def gen_golden_data():
    # KV cache: small integers stay bit-exact in BF16.
    KV_f32 = np.random.randint(-4, 5, size=(kKvRows, kD)).astype(np.float32)
    KV_f32 = _bf16_round(KV_f32)

    # Topk indices: random in [0, kKvRows), with a few -1 sentinels sprinkled
    # in the tail of this block to exercise the zero-row path.
    idx = np.random.randint(0, kKvRows, size=(kS,)).astype(np.int32)
    # Sentinel injection: last 2 indices of the current block become -1.
    start = kTblock * kBlock
    end   = min(start + kBlock, kS)
    if end - start >= 2:
        idx[end - 2] = -1
        idx[end - 1] = -1

    # Golden gather for this block.
    out = np.zeros((kBlock, kD), dtype=np.float32)
    for i in range(kBlock):
        global_i = start + i
        if global_i >= kS:
            continue  # tail rows stay zero
        ii = int(idx[global_i])
        if ii == -1:
            continue
        out[i, :] = KV_f32[ii, :]

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    with open("./input/input_kv.bin",        "wb") as f:
        f.write(_f32_to_bf16_bytes(KV_f32))
    with open("./input/input_topk_idxs.bin", "wb") as f:
        f.write(idx.tobytes())
    with open("./output/golden_kv_gathered.bin", "wb") as f:
        f.write(_f32_to_bf16_bytes(out))

    print(f"[gen_data] H={kH} D={kD} S={kS} BLOCK={kBlock}  "
          f"t_block={kTblock}  kv_rows={kKvRows}")
    print(f"[gen_data] sentinel positions in idx: -1 count = "
          f"{int((idx == -1).sum())}")


if __name__ == "__main__":
    gen_golden_data()
