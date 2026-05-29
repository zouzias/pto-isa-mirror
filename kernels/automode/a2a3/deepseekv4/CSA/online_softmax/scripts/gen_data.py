#!/usr/bin/python3
# coding=utf-8
# online_softmax — gen_data.py
#
# Generates input + golden data for the FA online softmax update stage.
# Mirrors kernel.py:331-340 for ONE pipelined block (t = 0).
#
# At t=0, scores_max is initialized to -infinity (kernel.py:318), sum_exp to
# zero (kernel.py:317). To exercise the running-update path correctly we
# expose scores_max / sum_exp as inputs so this leaf can be tested at any
# iteration index.

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from case_utils import load_csa_case  # noqa: E402

np.random.seed(42)

_case = load_csa_case()
kH         = _case["h"]
kD         = _case["d"]
kS         = _case["s"]
kBlock     = _case["block"]
kTblock    = 0

# Mask penalty used by the vector mask fold. Must match the kernel constant.
MASK_NEG = -1.0e30


def _f32_to_bf16_bytes(x_f32: np.ndarray) -> bytes:
    raw = x_f32.astype(np.float32).view(np.uint32)
    raw_bf16 = ((raw + 0x8000) >> 16).astype(np.uint16)
    return raw_bf16.tobytes()


def _bf16_round(x_f32: np.ndarray) -> np.ndarray:
    raw = x_f32.astype(np.float32).view(np.uint32)
    raw = (raw + 0x8000) & 0xFFFF0000
    return raw.view(np.float32).copy()


def gen_golden_data():
    # Random acc_s from qk_matmul — small range so exp stays finite.
    acc_s = np.random.uniform(-2.0, 2.0, size=(kH, kBlock)).astype(np.float32)

    # Running state. Use the t=0 init values from kernel.py:317-318.
    scores_max = np.full((kH,), -np.float32(np.inf), dtype=np.float32)
    sum_exp    = np.zeros((kH,), dtype=np.float32)

    # topk_idxs slice for THIS block. Inject -1 in the last 2 positions to
    # exercise the mask path.
    idx = np.random.randint(0, kS, size=(kBlock,)).astype(np.int32)
    if kBlock >= 2:
        idx[-2] = -1
        idx[-1] = -1

    # ---- golden -----------------------------------------------------------
    # 1) Mask fold: where idx[j] == -1 or t*block + j >= topk, push acc_s to
    #    MASK_NEG. (Mirror of the deferred mask from qk_matmul.)
    s = acc_s.copy()
    base = kTblock * kBlock
    for j in range(kBlock):
        global_j = base + j
        if idx[j] == -1 or global_j >= kS:
            s[:, j] = MASK_NEG

    # 2) Online update.
    scores_max_prev = scores_max.copy()
    sum_exp_prev    = sum_exp.copy()
    block_max = s.max(axis=1)  # [H]
    # running max
    new_max = np.maximum(scores_max_prev, block_max)
    # scores_scale: where prev_max was -inf, exp(-inf - new) -> 0, which is
    # the correct rescale for the very first block. Use safe_diff.
    safe_diff = np.where(np.isfinite(scores_max_prev),
                         scores_max_prev - new_max,
                         np.float32(-np.inf))
    scores_scale = np.where(np.isfinite(safe_diff),
                            np.exp(safe_diff), np.float32(0.0)).astype(np.float32)

    s_post = np.exp(s - new_max[:, None]).astype(np.float32)
    s_cast = _bf16_round(s_post)

    sum_local = s_post.sum(axis=1).astype(np.float32)
    new_sum   = (sum_exp_prev * scores_scale + sum_local).astype(np.float32)

    # ---- write inputs -----------------------------------------------------
    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    acc_s.tofile("./input/input_acc_s.bin")
    scores_max.tofile("./input/input_scores_max.bin")
    sum_exp.tofile("./input/input_sum_exp.bin")
    idx.tofile("./input/input_topk_idxs.bin")

    # ---- write golden -----------------------------------------------------
    s_post.tofile("./output/golden_acc_s.bin")
    with open("./output/golden_acc_s_cast.bin", "wb") as f:
        f.write(_f32_to_bf16_bytes(s_cast))
    new_max.tofile("./output/golden_scores_max.bin")
    new_sum.tofile("./output/golden_sum_exp.bin")
    scores_scale.tofile("./output/golden_scores_scale.bin")

    print(f"[gen_data] H={kH} BLOCK={kBlock}  t_block={kTblock}")
    print(f"[gen_data] s range pre-mask:  [{acc_s.min():.4f}, {acc_s.max():.4f}]")
    print(f"[gen_data] s range post-exp:  [{s_post.min():.4e}, {s_post.max():.4e}]")
    print(f"[gen_data] new_max:           {new_max}")
    print(f"[gen_data] new_sum:           {new_sum}")


if __name__ == "__main__":
    gen_golden_data()
