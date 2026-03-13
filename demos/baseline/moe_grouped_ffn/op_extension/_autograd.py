#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import torch
import torch.nn.functional as F


_AUTOGRAD_REGISTERED = False


def _group_ranges(group_offsets):
    offsets = group_offsets.detach().cpu().to(torch.int64).tolist()
    return zip(offsets[:-1], offsets[1:])


def _setup_context(ctx, inputs, output):
    x, gate_weight_dn, up_weight_dn, group_offsets = inputs
    ctx.save_for_backward(x, gate_weight_dn, up_weight_dn, group_offsets)


def _setup_context_with_intermediates(ctx, inputs, output):
    x, gate_weight_dn, up_weight_dn, group_offsets = inputs
    _, gate_proj, up_proj = output
    ctx.save_for_backward(x, gate_weight_dn, up_weight_dn, group_offsets, gate_proj, up_proj)


def _allocate_input_grads(ctx, x, gate_weight_dn, up_weight_dn):
    needs_x, needs_gate, needs_up, _ = ctx.needs_input_grad
    grad_x = torch.zeros(x.shape, dtype=torch.float32, device=x.device) if needs_x else None
    grad_gate = (
        torch.zeros(gate_weight_dn.shape, dtype=torch.float32, device=gate_weight_dn.device) if needs_gate else None
    )
    grad_up = torch.zeros(up_weight_dn.shape, dtype=torch.float32, device=up_weight_dn.device) if needs_up else None
    return needs_x, needs_gate, needs_up, grad_x, grad_gate, grad_up


def _accumulate_input_grads(
    x,
    gate_weight_dn,
    up_weight_dn,
    group_offsets,
    total_grad_gate_proj,
    total_grad_up_proj,
    needs_x,
    needs_gate,
    needs_up,
    grad_x,
    grad_gate,
    grad_up,
):
    for expert, (start, end) in enumerate(_group_ranges(group_offsets)):
        if start == end:
            continue

        x_e = x[start:end].float()
        gate_e = gate_weight_dn[expert].float()
        up_e = up_weight_dn[expert].float()
        grad_gate_proj = total_grad_gate_proj[start:end].float()
        grad_up_proj = total_grad_up_proj[start:end].float()

        if needs_x:
            grad_x[start:end] += torch.matmul(grad_gate_proj, gate_e) + torch.matmul(grad_up_proj, up_e)
        if needs_gate:
            grad_gate[expert].copy_(torch.matmul(grad_gate_proj.transpose(0, 1), x_e))
        if needs_up:
            grad_up[expert].copy_(torch.matmul(grad_up_proj.transpose(0, 1), x_e))

    return (
        grad_x.to(x.dtype) if needs_x else None,
        grad_gate.to(gate_weight_dn.dtype) if needs_gate else None,
        grad_up.to(up_weight_dn.dtype) if needs_up else None,
        None,
    )


def _backward(ctx, grad_out):
    x, gate_weight_dn, up_weight_dn, group_offsets = ctx.saved_tensors
    needs_x, needs_gate, needs_up, grad_x, grad_gate, grad_up = _allocate_input_grads(
        ctx, x, gate_weight_dn, up_weight_dn
    )
    total_grad_gate_proj = torch.zeros(
        (x.shape[0], gate_weight_dn.shape[1]), dtype=torch.float32, device=x.device
    )
    total_grad_up_proj = torch.zeros_like(total_grad_gate_proj)

    for expert, (start, end) in enumerate(_group_ranges(group_offsets)):
        if start == end:
            continue

        x_e = x[start:end].float()
        gate_e = gate_weight_dn[expert].float()
        up_e = up_weight_dn[expert].float()
        grad_out_e = grad_out[start:end].float()

        gate_proj = torch.matmul(x_e, gate_e.transpose(0, 1))
        up_proj = torch.matmul(x_e, up_e.transpose(0, 1))
        sigmoid_gate = torch.sigmoid(gate_proj)
        silu_gate = F.silu(gate_proj)

        grad_gate_proj = grad_out_e * up_proj * sigmoid_gate * (1.0 + gate_proj * (1.0 - sigmoid_gate))
        grad_up_proj = grad_out_e * silu_gate

        total_grad_gate_proj[start:end].copy_(grad_gate_proj)
        total_grad_up_proj[start:end].copy_(grad_up_proj)

    return _accumulate_input_grads(
        x,
        gate_weight_dn,
        up_weight_dn,
        group_offsets,
        total_grad_gate_proj,
        total_grad_up_proj,
        needs_x,
        needs_gate,
        needs_up,
        grad_x,
        grad_gate,
        grad_up,
    )


def _backward_with_intermediates(ctx, grad_out, grad_gate_proj_out, grad_up_proj_out):
    x, gate_weight_dn, up_weight_dn, group_offsets, gate_proj, up_proj = ctx.saved_tensors
    needs_x, needs_gate, needs_up, grad_x, grad_gate, grad_up = _allocate_input_grads(
        ctx, x, gate_weight_dn, up_weight_dn
    )
    total_grad_gate_proj = torch.zeros_like(gate_proj, dtype=torch.float32)
    total_grad_up_proj = torch.zeros_like(up_proj, dtype=torch.float32)

    if grad_gate_proj_out is not None:
        total_grad_gate_proj.add_(grad_gate_proj_out.float())
    if grad_up_proj_out is not None:
        total_grad_up_proj.add_(grad_up_proj_out.float())

    if grad_out is not None:
        gate_proj_fp32 = gate_proj.float()
        up_proj_fp32 = up_proj.float()
        sigmoid_gate = torch.sigmoid(gate_proj_fp32)
        total_grad_gate_proj.add_(
            grad_out.float()
            * up_proj_fp32
            * sigmoid_gate
            * (1.0 + gate_proj_fp32 * (1.0 - sigmoid_gate))
        )
        total_grad_up_proj.add_(grad_out.float() * F.silu(gate_proj_fp32))

    return _accumulate_input_grads(
        x,
        gate_weight_dn,
        up_weight_dn,
        group_offsets,
        total_grad_gate_proj,
        total_grad_up_proj,
        needs_x,
        needs_gate,
        needs_up,
        grad_x,
        grad_gate,
        grad_up,
    )


def register_autograd():
    global _AUTOGRAD_REGISTERED

    if _AUTOGRAD_REGISTERED or not hasattr(torch.library, "register_autograd"):
        return

    torch.library.register_autograd(
        "npu::pto_moe_grouped_ffn",
        _backward,
        setup_context=_setup_context,
    )
    torch.library.register_autograd(
        "npu::pto_moe_grouped_ffn_with_intermediates",
        _backward_with_intermediates,
        setup_context=_setup_context_with_intermediates,
    )
    _AUTOGRAD_REGISTERED = True
