# HC — Hyper-Connections (DeepSeek-V4 residual mixing)

Auto-mode A3 split of the `Block.hc_pre` / `Block.hc_post` and the underlying
`hc_split_sinkhorn_kernel` ([kernel.py:371](../../../../../deepseek/kernel.py),
[model.py:648](../../../../../deepseek/model.py)).

Hyper-Connections (HC) maintain `hc_mult = 4` copies of the hidden state.
Instead of a simple residual, `hc_pre` collapses 4 → 1 via learned weights,
and `hc_post` expands 1 → 4 via a separate set of weights plus a combination
matrix.

## Reference

```python
# Block.hc_pre: model.py:674-682
mixes = F.linear(x_flat, hc_fn) * rsqrt(mean(x_flat^2) + eps)  # -- hc_mix_gemm (cube)
pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base,
                                     hc_mult=4, iters=20, eps=1e-6)  # -- hc_sinkhorn (vector)
y = sum(pre.unsqueeze(-1) * x.view(b,s,hc,d), dim=2)              # -- hc_pre_reduce (vector)

# Block.hc_post: model.py:684-687
y = post.unsqueeze(-1) * x.unsqueeze(-2)
    + sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)     # -- hc_post_combine (vector)
```

## Leaves

| Leaf | HW | Op | Shapes (defaults) |
|------|----|----|-------------------|
| [hc_mix_gemm/](hc_mix_gemm/) | cube | `mixes = x_flat @ hc_fn^T * rsqrt(...)` | `x_flat: [b*s, hc_mult*dim=16384]`, `hc_fn: [mix_hc=24, hc_mult*dim]`, out `[b*s, 24]` |
| [hc_sinkhorn/](hc_sinkhorn/) | vector | sigmoid + softmax + N Sinkhorn iterations on `comb[hc,hc]` | `mixes: [n, mix_hc=24]`, out `pre[n,4]`, `post[n,4]`, `comb[n,4,4]` |
| [hc_pre_reduce/](hc_pre_reduce/) | vector | `y = sum(pre.unsqueeze(-1) * x.view(b,s,hc,d), dim=2)` | `pre: [b,s,hc=4]`, `x: [b,s,hc=4,dim=4096]`, out `[b,s,dim]` |
| [hc_post_combine/](hc_post_combine/) | vector | `y = post*x + sum(comb*residual, dim=2)` | `x: [b,s,dim]`, `residual: [b,s,hc,dim]`, `post: [b,s,hc]`, `comb: [b,s,hc,hc]`, out `[b,s,hc,dim]` |

## Family shape contract

Generated header: `build/generated_cases.h` (see [scripts/generate_cases.py](scripts/generate_cases.py)).
Case tuple: `B, S, DIM, HC_MULT, SINKHORN_ITERS`.

Defaults from [ModelArgs](../../../../../deepseek/model.py):

- `dim=4096`, `hc_mult=4`, `hc_sinkhorn_iters=20`, `hc_eps=1e-6`,
  `norm_eps=1e-6`.

## Auto-mode constraints (HC family)

- `hc_mult = 4` is fixed in the model. The Sinkhorn `comb` matrix is `[4,4]`
  — small enough to live entirely in UB.
- The Sinkhorn loop (`sinkhorn_iters - 1` iterations of row+column normalize)
  needs care: alternating reductions on a 4×4 tile are well within UB but
  may stress the auto-mode pipeline if loop-unrolled. `Assumption`: keep
  loop in source as `T.serial` equivalent.
- `hc_mix_gemm` weight shape is `mix_hc × (hc_mult * dim) = 24 × 16384`
  per layer — moderate. FP32 weights per model.py:666-672 (`set_dtype(float32)`).
