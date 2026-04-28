#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os
import numpy as np
np.random.seed(19)
import ml_dtypes
bfloat16 = ml_dtypes.bfloat16
import torch


def engram_gate_ref_forward(x, k, v, wh, we, clamp_value, eps, save_for_backward=True):
    hidden_size = x.shape[-1]
    scalar = hidden_size ** -0.5

    x_f = x.float()
    k_f = k.float()
    wh_f = wh.float().unsqueeze(0)
    we_f = we.float().unsqueeze(0)

    rstd_x = torch.rsqrt(x_f.pow(2).mean(-1) + eps)
    rstd_k = torch.rsqrt(k_f.pow(2).mean(-1) + eps)

    raw_dot = torch.einsum('...d,...d->...', x_f * wh_f, k_f * we_f)
    dot = raw_dot * rstd_x * rstd_k * scalar
    signed_sqrt = dot.abs().clamp_min(clamp_value).sqrt() * dot.sign()
    gate_score = signed_sqrt.sigmoid()

    output = x_f + gate_score.unsqueeze(-1) * v.unsqueeze(-2).float()
    output = output.bfloat16()

    if save_for_backward:
        return output, raw_dot, gate_score, rstd_x, rstd_k
    return output


def gen_golden_data_engram_gate_bwd(case_name, param):
    hidden_size = param.hidden_size
    dtype = param.dtype
    clamp_value = 1e-6
    eps = 1e-20

    x_data = torch.randn(1, hidden_size, dtype=torch.bfloat16)
    k_data = torch.randn(1, hidden_size, dtype=torch.bfloat16)
    v_data = torch.randn(1, hidden_size, dtype=torch.bfloat16)
    wh_data = torch.randn(1, hidden_size, dtype=torch.bfloat16)
    we_data = torch.randn(1, hidden_size, dtype=torch.bfloat16)
    w_fused = wh_data.float() * we_data.float()
    grad_out_data = torch.randn(1, hidden_size, dtype=torch.bfloat16)

    x_ref = x_data.clone().requires_grad_(True)
    k_ref = k_data.clone().requires_grad_(True)
    v_ref = v_data.clone().requires_grad_(True)
    wh_ref = wh_data.float().requires_grad_(True)
    we_ref = we_data.float().requires_grad_(True)

    o_ref, raw_dot_ref, gate_score_ref, rstd_x_ref, rstd_k_ref = engram_gate_ref_forward(
        x_ref, k_ref, v_ref, wh_ref, we_ref, clamp_value, eps, save_for_backward=True
    )
    o_ref.backward(grad_out_data.float())

    golden_grad_x = x_ref.grad.bfloat16().numpy().astype(dtype)
    golden_grad_k = k_ref.grad.bfloat16().numpy().astype(dtype)
    golden_grad_v = v_ref.grad.bfloat16().numpy().astype(dtype)
    grad_w_fused = wh_ref.grad * we_data.float() + we_ref.grad * wh_data.float()
    golden_grad_w = grad_w_fused.numpy().astype(np.float32)

    x_data.numpy().astype(dtype).tofile("x.bin")
    k_data.numpy().astype(dtype).tofile("k.bin")
    v_data.numpy().astype(dtype).tofile("v.bin")
    w_fused.numpy().astype(np.float32).tofile("w.bin")
    grad_out_data.numpy().astype(dtype).tofile("grad_out.bin")

    gate_score_ref.numpy().astype(np.float32).tofile("gate.bin")
    rstd_x_ref.numpy().astype(np.float32).tofile("rstd_x.bin")
    rstd_k_ref.numpy().astype(np.float32).tofile("rstd_k.bin")
    raw_dot_ref.numpy().astype(np.float32).tofile("raw_dot.bin")

    golden_grad_x.tofile("golden_grad_x.bin")
    golden_grad_k.tofile("golden_grad_k.bin")
    golden_grad_v.tofile("golden_grad_v.bin")
    golden_grad_w.tofile("golden_grad_w.bin")


class EngramGateBwdParams:
    def __init__(self, dtype, hidden_size):
        self.dtype = dtype
        self.hidden_size = hidden_size


def generate_case_name(param):
    dtype_str = {
        np.float32: 'float',
        np.float16: 'half',
    }.get(param.dtype, 'bfloat16')
    return f"EngramGateBwdTest.case_{dtype_str}_{param.hidden_size}"


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        EngramGateBwdParams(np.float32, 4096),
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        case_path = os.path.join(testcases_dir, case_name)
        if not os.path.exists(case_path):
            os.makedirs(case_path)
        original_dir = os.getcwd()
        os.chdir(case_path)
        gen_golden_data_engram_gate_bwd(case_name, param)
        os.chdir(original_dir)