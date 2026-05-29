# index_score — Indexer relu(einsum) * weights, reduce over heads

Auto-mode A3 **vector** kernel that scaffolds the index-score compute of
the [Indexer](../README.md):

```
index_score = einsum("bshd,btd->bsht", q, kv_cache)               # model.py:421
index_score = (relu(index_score) * weights.unsqueeze(-1)).sum(2)  # model.py:422
```

The einsum is cube-heavy, but we are scaffolding this as a **vector**
leaf because the dominant follow-up op is the reduction over `H`. A
follow-up split into `einsum_cube + reduce_vector` is anticipated and
documented below.

## Reference (CUDA / TileLang)

There is no dedicated TileLang kernel for `index_score`. Model code:
- [model.py:421](../../../../../../../deepseek/model.py) — `einsum("bshd,btd->bsht", q, kv_cache)`
- [model.py:422](../../../../../../../deepseek/model.py) — `(F.relu(.) * weights[..., None]).sum(dim=2)`

`Assumption`: the einsum is **small-D enough** for the prototype shape
(`HEAD_DIM=64` in the tiny default) that a per-`(s, t)` inner product
streamed through UB is acceptable for a first pass. For the realistic
shape (`HEAD_DIM=128`), this Assumption is unlikely to hold and the
follow-up split is required.

## I/O

| Name | Shape | dtype | Source line |
|------|-------|-------|-------------|
| `q`         | `(B, S, H, D)`    | BF16  | model.py:418 (reshape of `wq_b` output) |
| `kv_cache`  | `(B, T, D)`       | BF16  | model.py:417 (`k_cache_compressed`) |
| `weights`   | `(B, S, H)`       | FP32  | model.py:419 (output of weights_proj) |
| `score`     | `(B, S, T)`       | FP32  | model.py:422 (reduced over H) |

`H = N_HEADS`, `D = HEAD_DIM`, `T = kv-cache length`.

## Memory-budget-first plan

Per [CLAUDE.md §Memory-budget-first kernel planning](../../../../../../../CLAUDE.md):

```text
Memory budgets:
- L1 custom budget:    not used (single-core vector, all live data in UB)
- L0A custom budget:   not used in vector path
- L0B custom budget:   not used in vector path
- L0C custom budget:   not used in vector path
- UB custom budget:    q_tile (BF16, H*D), kv_tile (BF16, T_inner*D),
                       weights_tile (FP32, H), partial (FP32, H*T_inner),
                       out_tile (FP32, T_inner)
                       Target ≤ 192 KB to leave slack for the auto-mode
                       pass's working set.

Live tiles by memory level:
- L1:  none
- L0A: none
- L0B: none
- L0C: none
- UB:  q_sh (BF16, [H, D]),
       kv_slice (BF16, [T_inner, D]),
       weights_sh (FP32, [H]),
       partial (FP32, [H, T_inner]),
       score_st (FP32, [T_inner])

Smallest hardware operation:
- cube operation shape: N/A (vector leaf for the prototype)
- vector operation shape:
    - element-wise relu and elementwise scalar multiply over (H, T_inner)
      UB tile
    - reduce-sum across the H axis (axis=0 of partial)

Loop tiling plan:
- tile-and-loop dimensions:
    outer:   (b, s) — flat 1D loop over B*S query rows
    middle:  tTile  — over T in blocks of kTileT (= 16 or 32)
    inner:   per-head inner product over D
- inferred tile sizes: kTileT = 16  (Assumption — fits in UB with H*D = 8*64)
- compile-time unroll/peel strategy: unroll H if H ≤ 8; otherwise leave rolled
- tail handling only where needed: T tail when T % kTileT != 0

Test-shape plan:
- tiny debug shape:        B=1, S=64,  T=32,  H=8,  D=64
- medium tiling shape:     B=1, S=64,  T=128, H=16, D=64
- model-inspired realistic: B=1, S=128, T=4096, H=64, D=128
                            (likely needs the einsum_cube + reduce_vector split)
- tail shape:              B=1, S=64,  T=37,  H=8,  D=64
```

## Auto-mode constraints

- A3 only; vector path: `--cce-aicore-arch=dav-c220-vec`.
- Single AICORE. No `block_idx` work split.
- No `TASSIGN`, `TPipe`, `TPUSH`/`TPOP`, no raw `set_flag`/`wait_flag`,
  no `Tile::data()` pointer casts.
- Pattern anchor: `kernels/automode/a2a3/add_tile_array/` (Vec auto baseline).

## Follow-up plan

If profiler shows the per-(s, t) inner product dominates, split this
leaf into:
- `index_einsum_cube/`  — cube GEMM batched over (b, s) producing
  `[B, S, H, T]` FP32.
- `index_reduce_vector/` — vector relu + scale + reduce over H to
  `[B, S, T]` FP32.

## Build & run

```bash
bash run.sh -r npu -v Ascend910B1
```

## Comparison policy

- Tolerance-based: relative ≤ `1e-2`, absolute ≤ `1e-3` on FP32 output.

## Status

`Unknown` whether this builds. **No claim of compile / run success.**
