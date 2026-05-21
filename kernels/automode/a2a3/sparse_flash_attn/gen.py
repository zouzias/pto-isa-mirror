#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def float_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    rounding_bias = ((u >> 16) & 1) + 0x7FFF
    return ((u + rounding_bias) >> 16).astype(np.uint16)


def gather_sparse_kv(kv_states: torch.Tensor, topk_idxs: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, topk = topk_idxs.shape
    batch_idx = torch.arange(batch_size).view(batch_size, 1, 1).expand(-1, seq_len, topk)
    safe_topk_idxs = torch.where(topk_idxs == -1, 0, topk_idxs).long()
    gathered = kv_states[batch_idx, safe_topk_idxs, :]
    gathered_mask = (topk_idxs != -1).unsqueeze(-1).to(gathered.dtype)
    return gathered * gathered_mask


def sparse_softmax_with_sink(scores: torch.Tensor, attn_sink: torch.Tensor, head_dim: int, softmax_dim: int = -1) -> torch.Tensor:
    max_vals = torch.max(scores, dim=softmax_dim, keepdim=True).values
    exp_scores = torch.exp(scores - max_vals)
    sum_exp = torch.sum(exp_scores, dim=softmax_dim, keepdim=True)

    sink_view_shape = [1] * scores.dim()
    sink_view_shape[head_dim if head_dim >= 0 else scores.dim() + head_dim] = scores.shape[head_dim]
    sink_term = torch.exp(attn_sink.view(sink_view_shape) - max_vals)
    return exp_scores / (sum_exp + sink_term)


def sparse_attn(query_states: torch.Tensor,
                kv_states: torch.Tensor,
                attn_sink: torch.Tensor,
                topk_idxs: torch.Tensor,
                softmax_scale: float) -> torch.Tensor:
    sparse_kv = gather_sparse_kv(kv_states, topk_idxs)
    score_mask = torch.where((topk_idxs == -1).unsqueeze(-2), -torch.inf, 0.0).to(dtype=torch.float32)
    scores = torch.matmul(query_states, sparse_kv.transpose(-2, -1)).to(torch.float32)
    probs = sparse_softmax_with_sink(scores * softmax_scale + score_mask, attn_sink, head_dim=-2)
    return torch.matmul(probs, sparse_kv.to(torch.float32))


def make_topk_indices(b: int, m: int, n: int, topk: int, rng: np.random.Generator) -> np.ndarray:
    out = np.full((b, m, topk), -1, dtype=np.int32)
    for bi in range(b):
        for mi in range(m):
            if topk <= n:
                vals = rng.choice(n, size=topk, replace=False)
            else:
                vals = np.concatenate([rng.permutation(n), np.full(topk - n, -1, dtype=np.int32)])
            out[bi, mi] = vals.astype(np.int32, copy=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--b", type=int, default=1)
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--h", type=int, default=64)
    parser.add_argument("--d", type=int, default=512)
    parser.add_argument("--topk", type=int, default=128)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    q = rng.standard_normal((args.b, args.m, args.h, args.d), dtype=np.float32)
    kv = rng.standard_normal((args.b, args.n, args.d), dtype=np.float32)
    attn_sink = rng.standard_normal((args.h,), dtype=np.float32)
    topk_idxs = make_topk_indices(args.b, args.m, args.n, args.topk, rng)

    q_t = torch.from_numpy(q.astype(np.float32))
    kv_t = torch.from_numpy(kv.astype(np.float32))
    attn_sink_t = torch.from_numpy(attn_sink.astype(np.float32))
    topk_idxs_t = torch.from_numpy(topk_idxs.astype(np.int32))

    softmax_scale = args.d ** -0.5
    output_golden = sparse_attn(q_t, kv_t, attn_sink_t, topk_idxs_t, softmax_scale).to(torch.float32).numpy()

    float_to_bf16_bits(q).tofile(out_dir / "q.bin")
    float_to_bf16_bits(kv).tofile(out_dir / "kv.bin")
    attn_sink.astype(np.float32).tofile(out_dir / "attn_sink.bin")
    topk_idxs.astype(np.int32).tofile(out_dir / "topk_idxs.bin")
    output_golden.astype(np.float32).tofile(out_dir / "golden.bin")

    shape = {
        "b": args.b,
        "m": args.m,
        "n": args.n,
        "h": args.h,
        "d": args.d,
        "topk": args.topk,
    }
    with open(out_dir / "shape.json", "w", encoding="utf-8") as f:
        json.dump(shape, f, indent=2)

    print(f"Generated q        : {out_dir / 'q.bin'}")
    print(f"Generated kv       : {out_dir / 'kv.bin'}")
    print(f"Generated attn_sink: {out_dir / 'attn_sink.bin'}")
    print(f"Generated topk_idxs: {out_dir / 'topk_idxs.bin'}")
    print(f"Generated golden   : {out_dir / 'golden.bin'}")
    print(f"Generated shape    : {out_dir / 'shape.json'}")
    print(f"Shape              : b={args.b}, m={args.m}, n={args.n}, h={args.h}, d={args.d}, topk={args.topk}")


if __name__ == "__main__":
    main()
