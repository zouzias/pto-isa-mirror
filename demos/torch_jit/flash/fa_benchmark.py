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

import math
import torch
import torch_npu
from jit_util_flash import jit_compile_flash


def attn_matmul_flops(batch_size, s_q, s_k, h):
    return 4 * batch_size * s_q * s_k * h


def tflops(flops, ms):
    return flops / (ms * 1e-3) / 1e12


def time_npu(fn, iters=200, warmup=50):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.npu.synchronize()

    return start.elapsed_time(end) / iters


# ---------------------------
# 1) Torch GEMM reference
# ---------------------------
def gemm_reference_no_copy(q, k, v, out, scale=True):
    scores = q @ k.transpose(0, 1)
    if scale:
        scores = scores * (1.0 / math.sqrt(q.shape[-1]))
    p = torch.softmax(scores, dim=-1)
    torch.matmul(p, v, out=out)


# ---------------------------
# 2) Fused attention
# ---------------------------
def fused_only_no_copy(q_bsh, k_bsh, v_bsh):
    o, _ = torch_npu.npu_fused_infer_attention_score(
        q_bsh, k_bsh, v_bsh, input_layout="BSH"
    )
    return o


def bench_all(s_q=128, s_k=1024, h=128, scale=True, iters=200, warmup=50):
    device = "npu"
    torch.npu.set_device(device)

    dtype = torch.float16

    # Inputs
    q = torch.rand((s_q, h), device=device, dtype=dtype).contiguous()
    k = torch.rand((s_k, h), device=device, dtype=dtype).contiguous()
    v = torch.rand((s_k, h), device=device, dtype=dtype).contiguous()

    # FLOPs baseline (matmuls only)
    batch_size = 1
    flops = attn_matmul_flops(batch_size, s_q, s_k, h)

    # 1) GEMM output buffer (fp16)
    out_gemm = torch.empty((s_q, h), device=device, dtype=dtype)

    # 2) Fused inputs in BSH
    q_bsh = q.unsqueeze(0).contiguous()  # (1,Sq,H)
    k_bsh = k.unsqueeze(0).contiguous()  # (1,Sk,H)
    v_bsh = v.unsqueeze(0).contiguous()  # (1,Sk,H)

    # 3) JIT flash kernel compile ONCE
    flash = jit_compile_flash(verbose=False)

    # JIT flash buffers
    # NOTE: o_out is fp32 output; keep ref in fp32 for checks.
    num_tiles = s_q // 128
    o_out = torch.empty((s_q, h), device=device, dtype=torch.float32)

    out_device = torch.empty((s_q, s_k), device=device, dtype=torch.float32)
    xexp_device = torch.empty((s_q, s_k), device=device, dtype=torch.float16)
    p_out_fp32_device = torch.empty((s_q, s_k), device=device, dtype=torch.float32)

    out_2d_device = torch.empty((num_tiles, s_q, h), device=device, dtype=torch.float32)
    g_sum_device = torch.empty((num_tiles, s_q), device=device, dtype=torch.float32)
    exp_max_device = torch.empty((num_tiles, s_q), device=device, dtype=torch.float32)
    o_parts_device = torch.empty((num_tiles, s_q, h), device=device, dtype=torch.float32)

    # --- Bench 1: GEMM reference
    ms_gemm = time_npu(
        lambda: gemm_reference_no_copy(q, k, v, out_gemm, scale=scale),
        iters=iters,
        warmup=warmup,
    )

    # --- Bench 2: fused attention
    ms_fused = time_npu(
        lambda: fused_only_no_copy(q_bsh, k_bsh, v_bsh),
        iters=iters,
        warmup=warmup,
    )

    # --- Bench 3: jit flash
    ms_jit = time_npu(
        lambda: flash(
            q,
            k,
            v,
            o_out,
            out_device,
            xexp_device,
            p_out_fp32_device,
            out_2d_device,
            g_sum_device,
            exp_max_device,
            o_parts_device,
        ),
        iters=iters,
        warmup=warmup,
    )

    print("\n=== Matmul-only FLOPs baseline ===")
    print(f"B={batch_size}, Sq={s_q}, Sk={s_k}, H={h}")
    print(f"FLOPs = 4*B*Sq*Sk*H = {flops:,}")

    def show(name, ms):
        print(f"\n{name}")
        print(f"  time:    {ms:.6f} ms/iter")
        print(f"  TFLOP/s: {tflops(flops, ms):.6f}")

    show("1) GEMM attention (q@k^T + softmax + p@v)", ms_gemm)
    show("2) Fused npu_fused_infer_attention_score", ms_fused)
    show("3) JIT compiled flash kernel", ms_jit)

    # ref in fp32
    scores = q @ k.transpose(0, 1)
    if scale:
        scores = scores * (1.0 / math.sqrt(h))
    ref_fp32 = (torch.softmax(scores, dim=-1) @ v).to(torch.float32)

    # gemm output is fp16; compare in fp32
    torch.testing.assert_close(
        out_gemm.to(torch.float32), ref_fp32, rtol=1.0, atol=8e-2
    )

    # fused returns (1,Sq,H) fp16 typically
    fused_out = fused_only_no_copy(q_bsh, k_bsh, v_bsh).squeeze(0).to(torch.float32)
    torch.testing.assert_close(fused_out, ref_fp32, rtol=1.0, atol=8e-2)

    # jit out already fp32
    torch.testing.assert_close(o_out, ref_fp32, rtol=1.0, atol=8e-2)

    print("\nAll outputs match reference (within tolerance)")


if __name__ == "__main__":
    bench_all(s_q=1024, s_k=2048, h=128)
