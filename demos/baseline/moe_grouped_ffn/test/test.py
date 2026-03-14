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

import contextlib
import os
import pathlib
import sys
import unittest

import torch
import torch.nn.functional as F
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

TEST_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

import op_extension


HIDDEN_SIZE = 4096
INTER_SIZE = 1920

_SPLIT_ENV = "PTO_MOE_GROUPED_FFN_USE_CUSTOM_SPLIT"
_FUSED_ENV = "PTO_MOE_GROUPED_FFN_USE_FUSED"


def ref_grouped_ffn(x_cpu, gate_dn_cpu, up_dn_cpu, group_offsets):
    chunks = []
    for expert in range(group_offsets.numel() - 1):
        start = int(group_offsets[expert].item())
        end = int(group_offsets[expert + 1].item())
        if start == end:
            continue
        x_e = x_cpu[start:end].float()
        gate = torch.matmul(x_e, gate_dn_cpu[expert].float().t())
        up = torch.matmul(x_e, up_dn_cpu[expert].float().t())
        chunks.append(F.silu(gate) * up)
    if not chunks:
        return torch.empty((0, INTER_SIZE), dtype=torch.float32)
    return torch.cat(chunks, dim=0)


def ref_grouped_ffn_with_intermediates(x_cpu, gate_dn_cpu, up_dn_cpu, group_offsets):
    outputs = []
    gate_projs = []
    up_projs = []
    for expert in range(group_offsets.numel() - 1):
        start = int(group_offsets[expert].item())
        end = int(group_offsets[expert + 1].item())
        if start == end:
            continue
        x_e = x_cpu[start:end].float()
        gate_proj = torch.matmul(x_e, gate_dn_cpu[expert].float().t())
        up_proj = torch.matmul(x_e, up_dn_cpu[expert].float().t())
        gate_projs.append(gate_proj)
        up_projs.append(up_proj)
        outputs.append(F.silu(gate_proj) * up_proj)

    if not outputs:
        empty = torch.empty((0, INTER_SIZE), dtype=torch.float32)
        return empty, empty.clone(), empty.clone()

    return torch.cat(outputs, dim=0), torch.cat(gate_projs, dim=0), torch.cat(up_projs, dim=0)


def ref_grouped_ffn_stage1_input(x_cpu, gate_dn_cpu, up_dn_cpu, group_offsets):
    _, gate_proj, up_proj = ref_grouped_ffn_with_intermediates(x_cpu, gate_dn_cpu, up_dn_cpu, group_offsets)
    if gate_proj.numel() == 0:
        return torch.empty((0, INTER_SIZE * 2), dtype=torch.bfloat16)
    return torch.cat([gate_proj.to(torch.bfloat16), up_proj.to(torch.bfloat16)], dim=-1)


@contextlib.contextmanager
def impl_env(split=False, fused=False):
    prev_split = os.environ.get(_SPLIT_ENV)
    prev_fused = os.environ.get(_FUSED_ENV)
    try:
        if split:
            os.environ[_SPLIT_ENV] = "1"
        else:
            os.environ.pop(_SPLIT_ENV, None)
        if fused:
            os.environ[_FUSED_ENV] = "1"
        else:
            os.environ.pop(_FUSED_ENV, None)
        yield
    finally:
        if prev_split is None:
            os.environ.pop(_SPLIT_ENV, None)
        else:
            os.environ[_SPLIT_ENV] = prev_split
        if prev_fused is None:
            os.environ.pop(_FUSED_ENV, None)
        else:
            os.environ[_FUSED_ENV] = prev_fused


class TestPtoMoeGroupedFFN(TestCase):

    def _make_demo_inputs(self, seed):
        torch.manual_seed(seed)

        group_sizes = torch.tensor([48, 64, 0, 32], dtype=torch.int32)
        group_offsets = torch.cat(
            [torch.tensor([0], dtype=torch.int32), torch.cumsum(group_sizes, dim=0)],
            dim=0,
        )
        total_rows = int(group_offsets[-1].item())
        num_experts = group_sizes.numel()

        x = torch.randn((total_rows, HIDDEN_SIZE), dtype=torch.bfloat16) * 0.05
        gate_dn = torch.randn((num_experts, INTER_SIZE, HIDDEN_SIZE), dtype=torch.bfloat16) * 0.02
        up_dn = torch.randn((num_experts, INTER_SIZE, HIDDEN_SIZE), dtype=torch.bfloat16) * 0.02
        return x, gate_dn, up_dn, group_offsets

    def _make_large_inputs(self, seed):
        torch.manual_seed(seed)

        group_sizes = torch.full((8,), 256, dtype=torch.int32)
        group_offsets = torch.cat(
            [torch.tensor([0], dtype=torch.int32), torch.cumsum(group_sizes, dim=0)],
            dim=0,
        )
        total_rows = int(group_offsets[-1].item())
        num_experts = group_sizes.numel()

        x = torch.randn((total_rows, HIDDEN_SIZE), dtype=torch.bfloat16) * 0.05
        gate_dn = torch.randn((num_experts, INTER_SIZE, HIDDEN_SIZE), dtype=torch.bfloat16) * 0.02
        up_dn = torch.randn((num_experts, INTER_SIZE, HIDDEN_SIZE), dtype=torch.bfloat16) * 0.02
        return x, gate_dn, up_dn, group_offsets

    def _assert_repeated_large_forward(self, seed, repeats=10, split=False, fused=False):
        x, gate_dn, up_dn, group_offsets = self._make_large_inputs(seed=seed)

        x_npu = x.npu()
        gate_dn_npu = gate_dn.npu()
        up_dn_npu = up_dn.npu()
        group_offsets_npu = group_offsets.npu()
        ref = ref_grouped_ffn(x, gate_dn, up_dn, group_offsets)

        ref_out = None
        with impl_env(split=split, fused=fused):
            for idx in range(repeats):
                out = torch.ops.npu.pto_moe_grouped_ffn(x_npu, gate_dn_npu, up_dn_npu, group_offsets_npu)
                torch.npu.synchronize()
                self.assertTrue(bool(torch.isfinite(out).all().cpu().item()))
                if idx == 0:
                    self.assertRtolEqual(out.cpu(), ref, prec=2.0e-2)
                if ref_out is None:
                    ref_out = out.detach()
                    continue

                max_diff = torch.max(torch.abs(out - ref_out)).cpu().item()
                self.assertLessEqual(max_diff, 1.0e-5)

    def test_pto_moe_grouped_ffn(self):
        x, gate_dn, up_dn, group_offsets = self._make_demo_inputs(seed=0)

        ref = ref_grouped_ffn(x, gate_dn, up_dn, group_offsets)

        x_npu = x.npu()
        gate_dn_npu = gate_dn.npu()
        up_dn_npu = up_dn.npu()
        group_offsets_npu = group_offsets.npu()

        out = torch.ops.npu.pto_moe_grouped_ffn(x_npu, gate_dn_npu, up_dn_npu, group_offsets_npu)
        self.assertRtolEqual(out.cpu(), ref, prec=2.0e-2)

    def test_pto_moe_grouped_ffn_with_intermediates(self):
        x, gate_dn, up_dn, group_offsets = self._make_demo_inputs(seed=9)
        ref_out, ref_gate, ref_up = ref_grouped_ffn_with_intermediates(x, gate_dn, up_dn, group_offsets)

        out, gate_proj, up_proj = torch.ops.npu.pto_moe_grouped_ffn_with_intermediates(
            x.npu(), gate_dn.npu(), up_dn.npu(), group_offsets.npu()
        )
        self.assertRtolEqual(out.cpu(), ref_out, prec=2.0e-2)
        self.assertRtolEqual(gate_proj.cpu(), ref_gate, prec=2.0e-2)
        self.assertRtolEqual(up_proj.cpu(), ref_up, prec=2.0e-2)

    def test_pto_moe_grouped_ffn_stage1_input_custom_split(self):
        x, gate_dn, up_dn, group_offsets = self._make_demo_inputs(seed=10)
        ref = ref_grouped_ffn(x, gate_dn, up_dn, group_offsets)

        with impl_env(split=True):
            stage1_input = torch.ops.npu.pto_moe_grouped_ffn_stage1_input(
                x.npu(), gate_dn.npu(), up_dn.npu(), group_offsets.npu()
            )
            out = torch_npu.npu_swiglu(stage1_input, dim=-1)
        self.assertRtolEqual(out.cpu().float(), ref.float(), prec=2.0e-2)

    def test_pto_moe_grouped_ffn_stage1_input_nd_custom_split(self):
        x, gate_dn, up_dn, group_offsets = self._make_demo_inputs(seed=11)
        ref = ref_grouped_ffn(x, gate_dn, up_dn, group_offsets)
        weight_nd = torch.cat([gate_dn.transpose(1, 2), up_dn.transpose(1, 2)], dim=-1).contiguous()

        with impl_env(split=True):
            stage1_input = torch.ops.npu.pto_moe_grouped_ffn_stage1_input_nd(
                x.npu(), weight_nd.npu(), group_offsets.npu()
            )
            out = torch_npu.npu_swiglu(stage1_input, dim=-1)
        self.assertRtolEqual(out.cpu().float(), ref.float(), prec=2.0e-2)

    def test_pto_moe_grouped_ffn_custom_split_forward(self):
        x, gate_dn, up_dn, group_offsets = self._make_demo_inputs(seed=3)
        ref = ref_grouped_ffn(x, gate_dn, up_dn, group_offsets)

        with impl_env(split=True):
            out = torch.ops.npu.pto_moe_grouped_ffn(x.npu(), gate_dn.npu(), up_dn.npu(), group_offsets.npu())
            self.assertRtolEqual(out.cpu(), ref, prec=2.0e-2)

    def test_pto_moe_grouped_ffn_fused_forward(self):
        x, gate_dn, up_dn, group_offsets = self._make_demo_inputs(seed=4)
        ref = ref_grouped_ffn(x, gate_dn, up_dn, group_offsets)

        with impl_env(fused=True):
            out = torch.ops.npu.pto_moe_grouped_ffn(x.npu(), gate_dn.npu(), up_dn.npu(), group_offsets.npu())
            self.assertRtolEqual(out.cpu(), ref, prec=2.0e-2)

    def test_pto_moe_grouped_ffn_impl_consistency(self):
        x, gate_dn, up_dn, group_offsets = self._make_demo_inputs(seed=8)
        ref = ref_grouped_ffn(x, gate_dn, up_dn, group_offsets)

        outputs = {}
        configs = {
            "aclnn": {},
            "custom_split": {"split": True},
        }
        for name, env_kwargs in configs.items():
            with impl_env(**env_kwargs):
                out = torch.ops.npu.pto_moe_grouped_ffn(x.npu(), gate_dn.npu(), up_dn.npu(), group_offsets.npu())
            out_cpu = out.cpu()
            self.assertRtolEqual(out_cpu, ref, prec=2.0e-2)
            outputs[name] = out_cpu

        baseline = outputs["aclnn"]
        for name, out_cpu in outputs.items():
            self.assertRtolEqual(out_cpu, baseline, prec=2.0e-2)

    def test_pto_moe_grouped_ffn_backward(self):
        torch.manual_seed(1)

        group_sizes = torch.tensor([16, 48, 32, 64], dtype=torch.int32)
        group_offsets = torch.cat(
            [torch.tensor([0], dtype=torch.int32), torch.cumsum(group_sizes, dim=0)],
            dim=0,
        )
        total_rows = int(group_offsets[-1].item())
        num_experts = group_sizes.numel()

        x = (torch.randn((total_rows, HIDDEN_SIZE), dtype=torch.bfloat16) * 0.05).requires_grad_()
        gate_dn = (torch.randn((num_experts, INTER_SIZE, HIDDEN_SIZE), dtype=torch.bfloat16) * 0.02).requires_grad_()
        up_dn = (torch.randn((num_experts, INTER_SIZE, HIDDEN_SIZE), dtype=torch.bfloat16) * 0.02).requires_grad_()
        grad_out = torch.randn((total_rows, INTER_SIZE), dtype=torch.float32) * 0.03

        ref = ref_grouped_ffn(x, gate_dn, up_dn, group_offsets)
        ref.backward(grad_out)

        x_npu = x.detach().npu().requires_grad_()
        gate_dn_npu = gate_dn.detach().npu().requires_grad_()
        up_dn_npu = up_dn.detach().npu().requires_grad_()
        group_offsets_npu = group_offsets.npu()

        out = torch.ops.npu.pto_moe_grouped_ffn(x_npu, gate_dn_npu, up_dn_npu, group_offsets_npu)
        out.backward(grad_out.npu())

        self.assertRtolEqual(out.detach().cpu(), ref.detach(), prec=2.0e-2)
        self.assertRtolEqual(x_npu.grad.cpu().float(), x.grad.cpu().float(), prec=5.0e-2)
        self.assertRtolEqual(gate_dn_npu.grad.cpu().float(), gate_dn.grad.cpu().float(), prec=5.0e-2)
        self.assertRtolEqual(up_dn_npu.grad.cpu().float(), up_dn.grad.cpu().float(), prec=5.0e-2)

    def test_pto_moe_grouped_ffn_repeated_large_forward(self):
        self._assert_repeated_large_forward(seed=2, repeats=10)

    def test_pto_moe_grouped_ffn_custom_split_repeated_large_forward(self):
        self._assert_repeated_large_forward(seed=5, repeats=5, split=True)

    def test_pto_moe_grouped_ffn_fused_repeated_large_forward(self):
        self._assert_repeated_large_forward(seed=6, repeats=5, fused=True)


if __name__ == "__main__":
    run_tests()
