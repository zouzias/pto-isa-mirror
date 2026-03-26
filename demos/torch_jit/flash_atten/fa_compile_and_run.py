#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the
# terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance
# with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY,
# OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import random
import math
import torch
import torch_npu
from jit_util_flash import jit_compile_flash
from util.device import get_test_device
from util.bench import do_bench

_DEVICE = get_test_device()
torch.npu.set_device(_DEVICE)

NUM_ITERATIONS = 50
WARMUP = 10
SEED = 1

random.seed(SEED)
torch.manual_seed(SEED)
torch.npu.manual_seed(SEED)


# ---------------------------
# 2) Reference attention (pure PyTorch, fp32)
# ---------------------------
def fa_reference(q, k, v, is_causal=False):
    scale = 1.0 / math.sqrt(q.shape[1])
    scores = q.float() @ k.float().T * scale
    if is_causal:
        mask = torch.triu(
            torch.ones(scores.shape, device=q.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return attn @ v.float()


def fused_attention(q, k, v, is_causal=False):
    scale = 1.0 / math.sqrt(q.shape[1])
    # npu_fused_infer_attention_score expects BSH: (1, S, H)
    out, _ = torch_npu.npu_fused_infer_attention_score(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        num_heads=1,
        input_layout="BSH",
        scale=scale,
        next_tokens=0 if is_causal else 65535,
    )
    return out.squeeze(0)


def test_flash():
    s0, s1, head = 128, 2048, 128

    dtype = torch.float16

    # ==========================
    # Inputs
    # ==========================
    q2d = torch.randn((s0, head), dtype=dtype).npu()
    k2d = torch.randn((s1, head), dtype=dtype).npu()
    v2d = torch.randn((s1, head), dtype=dtype).npu()

    # ==========================
    # Compile flash ONCE
    # ==========================
    flash = jit_compile_flash(verbose=False)

    # ==========================
    # Benchmark reference ops
    # ==========================
    ref_ms = do_bench(
        lambda: fa_reference(q2d, k2d, v2d),
        warmup_iters=WARMUP,
        benchmark_iters=NUM_ITERATIONS,
        unit="ms",
    )
    npu_ms = do_bench(
        lambda: fused_attention(q2d, k2d, v2d),
        warmup_iters=WARMUP,
        benchmark_iters=NUM_ITERATIONS,
        unit="ms",
    )
    flash_ms = do_bench(
        lambda: flash(q2d, k2d, v2d),
        warmup_iters=WARMUP,
        benchmark_iters=NUM_ITERATIONS,
        unit="ms",
    )

    # ==========================
    # Correctness check
    # ==========================
    o_out = flash(q2d, k2d, v2d)
    o_ref = fa_reference(q2d, k2d, v2d).to(torch.float32)
    o_npu = fused_attention(q2d, k2d, v2d).to(torch.float32)

    print(f"JIT flash kernel           : {flash_ms:.3f} ms/iter")
    print(f"npu_fused_infer_attention  : {npu_ms:.3f} ms/iter")
    print(f"torch reference            : {ref_ms:.3f} ms/iter")
    torch.testing.assert_close(o_out, o_ref, rtol=1e-3, atol=1e-3)
    print("vs torch reference: PASSED")
    torch.testing.assert_close(o_out, o_npu, rtol=1e-3, atol=1e-3)
    print("vs npu_fused_attention: PASSED")


if __name__ == "__main__":
    test_flash()
