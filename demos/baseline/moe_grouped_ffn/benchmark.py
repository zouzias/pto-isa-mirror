#!/usr/bin/python3
# coding=utf-8

import argparse
import contextlib
import json
import os
import time

import torch
import torch.nn.functional as F
import torch_npu

import op_extension


HIDDEN_SIZE = 4096
INTER_SIZE = 1920

_SPLIT_ENV = "PTO_MOE_GROUPED_FFN_USE_CUSTOM_SPLIT"

_IMPL_ENVS = {
    "aclnn": {},
    "custom_split": {_SPLIT_ENV: "1"},
}

_IMPL_ALIASES = {
    "custom": "aclnn",
    "grouped_matmul": "aclnn",
    "split": "custom_split",
}

_IMPL_SETS = {
    "both": ["aclnn", "eager"],
    "pto": ["custom_split"],
    "all": ["aclnn", "custom_split", "eager"],
}


CASES = {
    "demo_small": [48, 64, 0, 32],
    "aligned_2k_8e": [256] * 8,
    "aligned_4k_16e": [256] * 16,
    "ragged_1664_8e": [320, 288, 256, 224, 192, 160, 128, 96],
}


def make_group_offsets(group_sizes, device):
    group_sizes = torch.tensor(group_sizes, dtype=torch.int32, device=device)
    return torch.cat(
        [torch.tensor([0], dtype=torch.int32, device=device), torch.cumsum(group_sizes, dim=0)],
        dim=0,
    )


def make_inputs(group_sizes, device):
    total_rows = sum(group_sizes)
    num_experts = len(group_sizes)
    x = torch.randn((total_rows, HIDDEN_SIZE), dtype=torch.bfloat16, device=device) * 0.05
    gate_dn = torch.randn((num_experts, INTER_SIZE, HIDDEN_SIZE), dtype=torch.bfloat16, device=device) * 0.02
    up_dn = torch.randn((num_experts, INTER_SIZE, HIDDEN_SIZE), dtype=torch.bfloat16, device=device) * 0.02
    group_offsets = make_group_offsets(group_sizes, device)
    return x.contiguous(), gate_dn.contiguous(), up_dn.contiguous(), group_offsets.contiguous()


def eager_grouped_ffn(x, gate_dn, up_dn, offsets_cpu):
    chunks = []
    for expert, (start, end) in enumerate(zip(offsets_cpu[:-1], offsets_cpu[1:])):
        if start == end:
            continue
        x_e = x[start:end]
        gate = torch.matmul(x_e, gate_dn[expert].transpose(0, 1)).float()
        up = torch.matmul(x_e, up_dn[expert].transpose(0, 1)).float()
        chunks.append(F.silu(gate) * up)
    if not chunks:
        return torch.empty((0, INTER_SIZE), dtype=torch.float32, device=x.device)
    return torch.cat(chunks, dim=0)


@contextlib.contextmanager
def impl_env(impl):
    prev_split = os.environ.get(_SPLIT_ENV)

    try:
        os.environ.pop(_SPLIT_ENV, None)
        for key, value in _IMPL_ENVS.get(impl, {}).items():
            os.environ[key] = value
        yield
    finally:
        if prev_split is None:
            os.environ.pop(_SPLIT_ENV, None)
        else:
            os.environ[_SPLIT_ENV] = prev_split


def parse_impls(spec):
    tokens = [token.strip() for token in spec.split(",") if token.strip()]
    if not tokens:
        raise ValueError("Implementation list must not be empty")

    impls = []
    for token in tokens:
        normalized = _IMPL_ALIASES.get(token, token)
        if normalized in _IMPL_SETS:
            impls.extend(_IMPL_SETS[normalized])
            continue
        if normalized not in _IMPL_ENVS and normalized != "eager":
            valid = sorted(list(_IMPL_ENVS.keys()) + list(_IMPL_ALIASES.keys()) + list(_IMPL_SETS.keys()) + ["eager"])
            raise ValueError(f"Unsupported impl '{token}'. Valid choices: {', '.join(valid)}")
        impls.append(normalized)

    deduped = []
    seen = set()
    for impl in impls:
        if impl in seen:
            continue
        seen.add(impl)
        deduped.append(impl)
    return deduped


def run_forward(impl, x, gate_dn, up_dn, group_offsets, offsets_cpu):
    if impl in _IMPL_ENVS:
        return torch.ops.npu.pto_moe_grouped_ffn(x, gate_dn, up_dn, group_offsets)
    if impl == "eager":
        return eager_grouped_ffn(x, gate_dn, up_dn, offsets_cpu)
    raise ValueError(f"Unsupported impl: {impl}")


def grouped_ffn_forward_flops(total_rows):
    matmul_flops = 4.0 * float(total_rows) * float(HIDDEN_SIZE) * float(INTER_SIZE)
    silu_flops = 5.0 * float(total_rows) * float(INTER_SIZE)
    mul_flops = 1.0 * float(total_rows) * float(INTER_SIZE)
    return matmul_flops + silu_flops + mul_flops


def grouped_ffn_train_flops(total_rows, impl):
    forward_flops = grouped_ffn_forward_flops(total_rows)
    if impl != "eager":
        # The op-level autograd recomputes gate/up projections in backward for all torch.ops.npu backends.
        recompute_flops = 4.0 * float(total_rows) * float(HIDDEN_SIZE) * float(INTER_SIZE)
        grad_matmul_flops = 8.0 * float(total_rows) * float(HIDDEN_SIZE) * float(INTER_SIZE)
        pointwise_flops = 8.0 * float(total_rows) * float(INTER_SIZE)
        return forward_flops + recompute_flops + grad_matmul_flops + pointwise_flops
    # Eager autograd keeps the forward intermediates, so no extra projection recompute term here.
    grad_matmul_flops = 8.0 * float(total_rows) * float(HIDDEN_SIZE) * float(INTER_SIZE)
    pointwise_flops = 8.0 * float(total_rows) * float(INTER_SIZE)
    return forward_flops + grad_matmul_flops + pointwise_flops


def tflops(flops, time_ms):
    return flops / (time_ms * 1.0e-3) / 1.0e12


def default_peak_tflops():
    env = os.getenv("PTO_MOE_GROUPED_FFN_PEAK_TFLOPS")
    if env:
        return float(env)
    return None


def bench_forward(impl, x, gate_dn, up_dn, group_offsets, offsets_cpu, warmup, iters):
    with impl_env(impl):
        for _ in range(warmup):
            out = run_forward(impl, x, gate_dn, up_dn, group_offsets, offsets_cpu)
            torch.npu.synchronize()
            del out

        times = []
        for _ in range(iters):
            start = time.perf_counter()
            out = run_forward(impl, x, gate_dn, up_dn, group_offsets, offsets_cpu)
            torch.npu.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)
            del out
    return times


def bench_train(impl, x, gate_dn, up_dn, group_offsets, offsets_cpu, warmup, iters):
    grad_out = torch.randn((x.size(0), INTER_SIZE), dtype=torch.float32, device=x.device) * 0.03

    def run_once():
        x_local = x.detach().requires_grad_(True)
        gate_local = gate_dn.detach().requires_grad_(True)
        up_local = up_dn.detach().requires_grad_(True)
        out = run_forward(impl, x_local, gate_local, up_local, group_offsets, offsets_cpu)
        out.backward(grad_out)
        return out, x_local, gate_local, up_local

    with impl_env(impl):
        for _ in range(warmup):
            tensors = run_once()
            torch.npu.synchronize()
            del tensors

        times = []
        for _ in range(iters):
            start = time.perf_counter()
            tensors = run_once()
            torch.npu.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)
            del tensors
    return times


def summarize(times_ms):
    sorted_times = sorted(times_ms)
    mid = len(sorted_times) // 2
    median = sorted_times[mid] if len(sorted_times) % 2 == 1 else 0.5 * (sorted_times[mid - 1] + sorted_times[mid])
    return {
        "mean_ms": sum(sorted_times) / len(sorted_times),
        "median_ms": median,
        "min_ms": min(sorted_times),
        "max_ms": max(sorted_times),
    }


def attach_perf_metrics(summary, flops, peak_tflops):
    summary["tflops_mean"] = tflops(flops, summary["mean_ms"])
    summary["tflops_median"] = tflops(flops, summary["median_ms"])
    summary["flops_total"] = int(flops)
    if peak_tflops is not None and peak_tflops > 0.0:
        summary["mfu_mean"] = summary["tflops_mean"] / peak_tflops
        summary["mfu_median"] = summary["tflops_median"] / peak_tflops


def attach_speedups(mode_results, baseline):
    baseline_result = mode_results.get(baseline)
    if baseline_result is None:
        return

    baseline_median = baseline_result.get("median_ms")
    if baseline_median is None:
        return
    speedups = {}
    for impl, summary in mode_results.items():
        if impl == baseline or not isinstance(summary, dict):
            continue
        # Skip non-summary dict entries such as previously attached speedup maps.
        median = summary.get("median_ms")
        if median is None:
            continue
        speedups[impl] = baseline_median / median

    if speedups:
        mode_results[f"speedup_vs_{baseline}"] = speedups


def main():
    parser = argparse.ArgumentParser(description="Benchmark grouped MoE FFN custom op on torch_npu.")
    parser.add_argument("--case", choices=sorted(CASES.keys()), default="aligned_2k_8e")
    parser.add_argument(
        "--impl",
        type=str,
        default="both",
        help="Comma-separated impls or aliases: aclnn, custom_split, eager, both, pto, all",
    )
    parser.add_argument("--mode", choices=["forward", "train", "both"], default="both")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--device", type=str, default="npu:0")
    parser.add_argument("--peak-tflops", type=float, default=default_peak_tflops())
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not torch.npu.is_available():
        raise RuntimeError("NPU is not available")

    torch.npu.set_device(args.device)
    torch.manual_seed(0)

    group_sizes = CASES[args.case]
    x, gate_dn, up_dn, group_offsets = make_inputs(group_sizes, args.device)
    offsets_cpu = group_offsets.cpu().to(torch.int64).tolist()

    impls = parse_impls(args.impl)
    modes = ["forward", "train"] if args.mode == "both" else [args.mode]

    results = {
        "case": args.case,
        "group_sizes": group_sizes,
        "impls": impls,
        "total_tokens": sum(group_sizes),
        "num_experts": len(group_sizes),
        "warmup": args.warmup,
        "iters": args.iters,
        "results": {},
    }

    for mode in modes:
        results["results"][mode] = {}
        for impl in impls:
            if mode == "forward":
                times = bench_forward(impl, x, gate_dn, up_dn, group_offsets, offsets_cpu, args.warmup, args.iters)
                flops = grouped_ffn_forward_flops(x.size(0))
            else:
                times = bench_train(impl, x, gate_dn, up_dn, group_offsets, offsets_cpu, args.warmup, args.iters)
                flops = grouped_ffn_train_flops(x.size(0), impl)
            results["results"][mode][impl] = summarize(times)
            attach_perf_metrics(results["results"][mode][impl], flops, args.peak_tflops)

        attach_speedups(results["results"][mode], "eager")
        attach_speedups(results["results"][mode], "aclnn")

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
        return

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
