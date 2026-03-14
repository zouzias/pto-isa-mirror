#!/usr/bin/env python3
# coding=utf-8

import argparse
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace

import torch
import torch_npu


PTO_REPO = "/home/llx/pto-isa"
MS_REPO = "/home/llx/MindSpeed"
LLM_REPO = "/home/llx/lzm_Mindspeed-LLm"
PTO_MOE_DEMO_REPO = f"{PTO_REPO}/demos/baseline/moe_grouped_ffn"

for repo in (MS_REPO, LLM_REPO, PTO_REPO, PTO_MOE_DEMO_REPO):
    if repo not in sys.path:
        sys.path.insert(0, repo)

from mindspeed.args_utils import get_mindspeed_args
from mindspeed.core.transformer.moe.grouped_matmul_util import get_gmm_op_cls
from megatron.core.transformer.moe.pto_grouped_ffn import (
    maybe_run_pto_grouped_ffn_stage1_overlap,
    pto_grouped_ffn_stage1_backward_prepare,
)


def _init_args():
    args = get_mindspeed_args()
    defaults = {
        "fp8": False,
        "use_gmm_fp8": False,
        "gemm_gradient_accumulation_fusion": False,
        "overlap_grad_reduce": False,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)


def _median(values):
    return statistics.median(values)


def _make_group_list(num_experts, rows_per_expert, device):
    return torch.tensor(
        [(idx + 1) * rows_per_expert for idx in range(num_experts)],
        dtype=torch.int64,
        device=device,
    )


def _make_group_offsets_cpu(num_experts, rows_per_expert):
    return torch.tensor(
        [idx * rows_per_expert for idx in range(num_experts + 1)],
        dtype=torch.int32,
        device="cpu",
    )


def _make_config():
    return SimpleNamespace(
        moe_zero_memory="disable",
        gated_linear_unit=True,
    )


def _make_inputs(num_experts, rows_per_expert, hidden_size, inter_size, device):
    total_rows = num_experts * rows_per_expert
    inputs = (torch.randn((total_rows, hidden_size), dtype=torch.bfloat16, device=device) * 0.05).contiguous()
    weights1 = (
        torch.randn((num_experts, hidden_size, inter_size * 2), dtype=torch.bfloat16, device=device) * 0.02
    ).contiguous()
    probs = (torch.rand((total_rows,), dtype=torch.bfloat16, device=device) * 0.9 + 0.1).contiguous()
    grad_stage1_output = (
        torch.randn((total_rows, inter_size), dtype=torch.bfloat16, device=device) * 0.03
    ).contiguous()
    group_list = _make_group_list(num_experts, rows_per_expert, device)
    group_offsets_cpu = _make_group_offsets_cpu(num_experts, rows_per_expert)
    return inputs, weights1, probs, grad_stage1_output, group_list, group_offsets_cpu


def _baseline_forward(gmm_cls, inputs, weights1, probs, group_list):
    mm1_out = gmm_cls.op_forward(inputs, weights1, group_list)[0]
    act_without_probs = torch_npu.npu_swiglu(mm1_out, dim=-1)
    act_out = act_without_probs * probs.unsqueeze(-1)
    return mm1_out, act_without_probs, act_out


def _baseline_backward(gmm_cls, inputs, mm1_out, grad_stage1_output, weights1, group_list):
    mm1_out_local = mm1_out.detach().requires_grad_(True)
    act_out = torch_npu.npu_swiglu(mm1_out_local, dim=-1)
    act_out.backward(grad_stage1_output)
    grad_proj = mm1_out_local.grad
    mm1_inputs_grad = gmm_cls.op_dx(grad_proj, weights1, group_list)[0]
    mm1_weights_grad = gmm_cls.op_dw(inputs, grad_proj, group_list)[0]
    return mm1_inputs_grad, mm1_weights_grad, grad_proj


def _pto_forward(inputs, weights1, probs, group_list, group_offsets_cpu):
    config = _make_config()
    outputs = maybe_run_pto_grouped_ffn_stage1_overlap(
        weights1,
        inputs,
        group_list,
        weights1,
        config,
        activation_recompute=False,
        group_offsets_cpu=group_offsets_cpu,
    )
    stage1_input = outputs["stage1_input"]
    act_without_probs = torch_npu.npu_swiglu(stage1_input, dim=-1)
    act_out = act_without_probs * probs.unsqueeze(-1)
    return stage1_input, act_without_probs, act_out


def _pto_backward(gmm_cls, inputs, weights1, grad_stage1_output, stage1_input, group_list):
    mm1_inputs_grad, grad_proj = pto_grouped_ffn_stage1_backward_prepare(
        inputs,
        grad_stage1_output,
        weights1,
        None,
        stage1_input=stage1_input,
        group_list=group_list,
        gmm_cls=gmm_cls,
    )
    mm1_weights_grad = gmm_cls.op_dw(inputs, grad_proj, group_list)[0]
    return mm1_inputs_grad, mm1_weights_grad, grad_proj


def _time_cuda(fn, warmup, iters):
    for _ in range(warmup):
        tensors = fn()
        torch.npu.synchronize()
        del tensors

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        tensors = fn()
        torch.npu.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
        del tensors
    return times


def main():
    parser = argparse.ArgumentParser(description="Benchmark overlap stage1 baseline vs PTO split.")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--rows-per-expert", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--inter-size", type=int, default=1920)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    _init_args()
    torch.npu.set_device(args.device)
    torch.manual_seed(0)
    os.environ.setdefault("ENABLE_PTO_MOE_GROUPED_FFN", "1")
    os.environ.setdefault("PTO_MOE_GROUPED_FFN_USE_CUSTOM_SPLIT", "1")
    os.environ.setdefault("PTO_MOE_GROUPED_FFN_CACHE_DN_WEIGHT", "1")
    os.environ.setdefault("PTO_MOE_GROUPED_FFN_SO_PATH", f"{PTO_MOE_DEMO_REPO}/build/libop_extension.so")

    gmm_cls = get_gmm_op_cls()
    inputs, weights1, probs, grad_stage1_output, group_list, group_offsets_cpu = _make_inputs(
        args.num_experts, args.rows_per_expert, args.hidden_size, args.inter_size, args.device
    )

    baseline_forward = _time_cuda(
        lambda: _baseline_forward(gmm_cls, inputs, weights1, probs, group_list),
        args.warmup,
        args.iters,
    )
    pto_forward = _time_cuda(
        lambda: _pto_forward(inputs, weights1, probs, group_list, group_offsets_cpu),
        args.warmup,
        args.iters,
    )

    mm1_out, _, _ = _baseline_forward(gmm_cls, inputs, weights1, probs, group_list)
    stage1_input, _, _ = _pto_forward(inputs, weights1, probs, group_list, group_offsets_cpu)

    baseline_backward = _time_cuda(
        lambda: _baseline_backward(gmm_cls, inputs, mm1_out, grad_stage1_output, weights1, group_list),
        args.warmup,
        args.iters,
    )
    pto_backward = _time_cuda(
        lambda: _pto_backward(gmm_cls, inputs, weights1, grad_stage1_output, stage1_input, group_list),
        args.warmup,
        args.iters,
    )

    results = {
        "shape": {
            "num_experts": args.num_experts,
            "rows_per_expert": args.rows_per_expert,
            "hidden_size": args.hidden_size,
            "inter_size": args.inter_size,
        },
        "forward_ms": {
            "baseline_median": _median(baseline_forward),
            "pto_median": _median(pto_forward),
        },
        "backward_ms": {
            "baseline_median": _median(baseline_backward),
            "pto_median": _median(pto_backward),
        },
    }
    results["total_ms"] = {
        "baseline_median": results["forward_ms"]["baseline_median"] + results["backward_ms"]["baseline_median"],
        "pto_median": results["forward_ms"]["pto_median"] + results["backward_ms"]["pto_median"],
    }

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
