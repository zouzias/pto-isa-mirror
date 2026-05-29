# HCA/Indexer — Top-k compressed-cache selector

Auto-mode A3 split of the `Indexer.forward` path
([model.py:381](../../../../../../deepseek/model.py)).

## Reference

```python
# Indexer.__init__: model.py:385-401
# Indexer.forward:  model.py:403-434
q = self.wq_b(qr)                  # qr: [b,s,q_lora_rank=1024]
                                   # q:  [b,s, n_heads*head_dim] then unflatten
                                   #     to [b,s, n_heads=64, head_dim=128]
                                   # -- wq_b_gemm (cube)

apply_rotary_emb(q[..., -rd:], freqs_cis)        # rope on last rope_head_dim=64
q = rotate_activation(q)                          # Hadamard rotation
fp4_act_quant(q, fp4_block_size, True)            # inplace FP4 quant+dequant

self.compressor(x, start_pos)                     # produces self.kv_cache

weights = self.weights_proj(x) * (self.softmax_scale * n_heads**-0.5)
                                   # weights: [b,s, n_heads]   -- weights_proj_gemm (cube)

index_score = einsum("bshd,btd->bsht",
                     q, self.kv_cache[:bsz, :end_pos // ratio])  # cube (heavy)
index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
                                   # -- index_score (vector reduce)

# Then masking + topk + offset adjustment (host-side or vector-only ops)
topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
```

## Leaves

| Leaf | HW | Op | Shapes (defaults) |
|------|----|----|-------------------|
| [wq_b_gemm/](wq_b_gemm/) | cube | `q = qr @ Wq_b^T` low-rank Q expansion | `qr: [b*s, 1024]`, `Wq_b: [n_heads*head_dim=64*128=8192, 1024]` |
| [weights_proj_gemm/](weights_proj_gemm/) | cube | `weights = x @ W_wproj^T * (scale * n_heads^-0.5)` | `x: [b*s, dim=4096]`, `W_wproj: [n_heads=64, 4096]`, BF16 |
| [index_score/](index_score/) | vector + cube hybrid (split into 2 leaves later if needed) | `relu(einsum(q, kv)) * weights → reduce_sum over n_heads` | `q: [b,s,h,d]`, `kv: [b,t,d]`, `weights: [b,s,h]`, out `[b,s,t]` |

## Out of scope (handled elsewhere)

- `apply_rotary_emb` on Q — same kernel as `Compressor/rope/`; reuse via the
  `_kernel.cpp` declaration in that leaf, no duplicate scaffold.
- `rotate_activation` (Hadamard) — small constant-matrix multiply; may be
  added as a future leaf if it shows up in profiling.
- `fp4_act_quant` — see `HCA/quant/act_quant_fp4/`.
- The inner `Compressor` of `Indexer` (rotate=True branch) — same kernels as
  the outer `HCA/Compressor/` family, no duplicate scaffold.
- `topk_idxs.topk(...)` — top-k is already covered by the
  [topkv2/](../../../topkv2/) kernel; we wire it into the test harness rather
  than re-implementing.

## Family shape contract

Generated header: `build/generated_cases.h` (see [scripts/generate_cases.py](scripts/generate_cases.py)).
Case tuple: `B, S, T, DIM, Q_LORA_RANK, N_HEADS, HEAD_DIM, INDEX_TOPK`.

Defaults from [ModelArgs](../../../../../../deepseek/model.py):

- `dim=4096`, `q_lora_rank=1024`, `index_n_heads=64`, `index_head_dim=128`,
  `index_topk=512`.
