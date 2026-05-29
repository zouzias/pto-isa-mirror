# weights_proj_gemm — Indexer per-head weights projection

Auto-mode A3 **cube** kernel that produces the per-head index weight
inside `Indexer.forward`:
`weights[B*S, N_HEADS] = (X[B*S, DIM] @ W_wproj[N_HEADS, DIM]^T) * scale`
where `scale = softmax_scale * n_heads^-0.5`
([model.py:419](../../../../../../../deepseek/model.py),
[model.py:395](../../../../../../../deepseek/model.py) for scale).

The output is the multiplier later used in `index_score`
([model.py:422](../../../../../../../deepseek/model.py)).

## Reference (CUDA / TileLang)

No dedicated TileLang kernel for `wproj`; dispatched via the generic
`linear()` plus a post-scale. We scaffold the post-scale either fused
into the cube accumulator drain (preferred for a future single-leaf
implementation) or as a follow-on vector op
(`Assumption`: fusion saves a UB round-trip; not yet confirmed).

For the prototype, defaults to **BF16 inputs, FP32 accumulator**. The
scalar `scale` is stored in `generated_cases.h` as an FP32 constant
computed at host time.

## I/O

| Name | Shape | dtype | Source line |
|------|-------|-------|-------------|
| `X`       | `(M = B*S, K = DIM)`        | BF16 | model.py:404 (`x` passed as Indexer arg) |
| `W_wproj` | `(N = N_HEADS, K = DIM)`    | BF16 | model.py:409 (`weights_proj`) |
| `weights` | `(M, N)`                    | FP32 | model.py:419 (then `* softmax_scale * n_heads^-0.5`) |

`softmax_scale = head_dim^-0.5` per model.py:395 (`self.softmax_scale`).
We pass it via the family case header (`kIdxSoftmaxScale` /
`kIdxWeightsScale` — `Assumption`: emitted later; the gen_data computes
the same constant and the launcher reads it from the header). For now
the post-multiply is folded into the FP32 numpy golden; the kernel
pseudocode notes it as a TODO follow-on.

## Memory-budget-first plan

Per [CLAUDE.md §Memory-budget-first kernel planning](../../../../../../../CLAUDE.md):

```text
Memory budgets:
- L1 custom budget:    BF16 A panel + BF16 B panel + slack (target ≤ 256 KB)
                       N is small (= n_heads ≤ 64), so B fits trivially
- L0A custom budget:   one BF16 A fractal-tile [kTileM, kTileK]
- L0B custom budget:   one BF16 B fractal-tile [kTileN, kTileK]
- L0C custom budget:   one FP32 C fractal-tile [kTileM, kTileN]
- UB custom budget:    one FP32 C tile staged for write-back +
                       (if post-scale done in vector) a scaled copy

Live tiles by memory level:
- L1:  Atile (BF16, M×K_inner), Btile (BF16, N×K_inner)  (B persistent)
- L0A: ATile (BF16, kTileM × kTileK fractal)
- L0B: BTile (BF16, kTileN × kTileK fractal)
- L0C: CTile (FP32, kTileM × kTileN fractal accumulator)
- UB:  CTile_fp32 + scale_broadcast (single FP32 scalar)

Smallest hardware operation:
- cube operation shape: 16×16×16 fractal MMA (BF16 × BF16 → FP32)
- vector operation shape: 1 FP32 scalar broadcast multiply on (M×N) UB tile
  if scale is applied as a follow-on vector op

Loop tiling plan:
- tile-and-loop dimensions: M tiled by kTileM=64, N tiled by kTileN=64
  (n_heads default 64 fits in one tile), K reduced by kTileK=64
- inferred tile sizes: kTileM=64, kTileN=64, kTileK=64 (Assumption)
- compile-time unroll/peel strategy: none
- tail handling only where needed: M tail when (B*S) % kTileM != 0

Test-shape plan:
- tiny debug shape:        B=1, S=64,  DIM=128, N_HEADS=8   (single N tile)
- medium tiling shape:     B=1, S=128, DIM=512, N_HEADS=16
- model-inspired realistic: B=1, S=128, DIM=4096, N_HEADS=64
- tail shape:              B=1, S=137, DIM=128, N_HEADS=8
```

## Auto-mode constraints

- A3 only; cube path: `--cce-aicore-arch=dav-c220-cube`.
- Single AICORE. No `block_idx` work split.
- No `TASSIGN`, `TPipe`, `TPUSH`/`TPOP`, no raw `set_flag`/`wait_flag`,
  no `Tile::data()` pointer casts.
- W_wproj can stay resident across M-tile loop (small N).
- Reason in 16×16 fractals.
- Scale handling: **either** fuse into the cube drain by multiplying the
  L0C → UB copy, **or** apply as a post-pass FP32 vector multiply over
  the (M×N) UB tile. Decision deferred to implementation.

## Build & run

```bash
bash run.sh -r npu -v Ascend910B1
```

## Comparison policy

- Tolerance-based: relative ≤ `1e-2`, absolute ≤ `1e-3` on FP32 output
  (BF16 GEMM with FP32 accumulator; one extra FP32 multiply).

## Status

`Unknown` whether this builds. **No claim of compile / run success.**
