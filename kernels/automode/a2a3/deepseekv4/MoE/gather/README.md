# gather — DeepSeek-V4 expert-output → token-position recombine

Auto-mode A3 **vector** kernel that scatters per-expert FFN outputs back
into per-token rows, applying the per-token routing weight on the way.
Mirrors the I/O contract of the top-level
[MoE/gather/gather_kernel.cpp](../../../MoE/gather/gather_kernel.cpp);
the only DeepSeek-V4 difference is the absence of the FA-style softmax
pass — DeepSeek-V4 produces the routing `weights` upstream in
`gate_score_topk/` or `gate_hash_routing/`, so this leaf is pure
weighted scatter-add.

```python
# Conceptual:
# Y = zeros((T, DIM), dtype=fp32)
# for r in packed_rows:
#     t = A_id[r]
#     k = rank_id[r]
#     Y[t] += weights[t, k] * B[r]            # weighted accumulate
```

## I/O

| Name | Shape | dtype | Notes |
|------|-------|-------|-------|
| `B`        | `(T*N_ACTIVATED + 16, DIM)` | BF16 | per-expert FFN outputs (downcast from FP32 in expert_ffn) |
| `A_id`     | `(T*N_ACTIVATED + 16)`      | INT32 | original token positions (from scatter) |
| `rank_id`  | `(T*N_ACTIVATED + 16)`      | INT32 | k slot per packed row (from scatter) |
| `weights`  | `(T, N_ACTIVATED)`          | FP32  | per-(token, k) routing weights (from gate) |
| `Y`        | `(T, DIM)`                  | BF16 | out; weighted sum over the N_ACTIVATED outputs per token |

Shape constants (from `../build/generated_cases.h`):
`kDsmoeT, kDsmoeDim, kDsmoeNActivated`.

> Note: the top-level MoE/gather uses FP32 for `B` and `Y` and embeds
> the FA softmax pass internally (because its upstream `moe_topk`
> doesn't produce normalized weights). DeepSeek-V4 normalizes upstream,
> so this leaf is the **post-softmax fast path** from MoE/gather —
> just the weighted scatter-add.

## Memory-budget-first plan

```text
Memory budgets:
- UB custom budget:    Two BF16 row tiles [1, DIM] + one FP32 scaled
                       row tile + small scalar registers for w_t_k.
                       At DIM=4096 (realistic): 8 KB BF16 + 8 KB BF16 +
                       16 KB FP32 = 32 KB. Well under UB cap.
                       At DIM=128 (tiny): ~1 KB total.
- L1, L0A/B/C:         N/A — vector-only leaf.

Live tiles by memory level:
- UB: bTile        [1, DIM] BF16   (loaded B[r] row)
      yTile        [1, DIM] BF16   (loaded Y[t] row for accumulate)
      scaledTile   [1, DIM] FP32   (w * b, before accumulate)
      sumTile      [1, DIM] BF16   (yTile + scaledTile, written back)

Smallest hardware operation:
- vector operation shape: full-row TLOAD / TMULS / TADD / TSTORE over
  DIM BF16 elements.
- cube operation shape: N/A.

Loop tiling plan:
- tile-and-loop dimensions: per packed-row loop over r in [0, T*N_ACTIVATED).
- inferred tile sizes: one DIM-wide row at a time in UB.
- compile-time unroll/peel strategy: none.
- tail handling only where needed: none (output Y is host-zeroed; only
  the first T*N_ACTIVATED packed rows are consulted).

Test-shape plan:
- tiny debug shape:         T=64,  DIM=128,  N_ACTIVATED=2
- model-inspired realistic: T=256, DIM=4096, N_ACTIVATED=2
```

## Auto-mode constraints

- A3 only; vector path.
- Single AICORE.
- Per-iter `pipe_barrier(PIPE_ALL)` at the start of each packed-row
  iteration (cross-iter auto-sync guard; same hardware-confirmed
  pattern from `MoE/scatter` and `MoE/gather`).
- `Y` must be **host-zero-initialized** before kernel launch
  (the kernel does `Y[t] += weighted_b`).
- No `TASSIGN`, no `Tile::data()` from kernel, no `*_IMPL` calls.

## Risks (`Assumption` / `Unknown`)

- `Assumption`: BF16 `+` FP32 mix is handled — the kernel either
  up-casts `B[r]` to FP32 before the scale and accumulate, or operates
  entirely in BF16 and accepts the precision drop. The top-level
  MoE/gather uses FP32 throughout. For this prototype, we use a FP32
  scratch tile for the scaled value and accumulate into a BF16 Y row;
  if the precision is unacceptable, switch `Y` to FP32.
- `Unknown`: whether `TMULS(BF16_out, BF16_in, FP32_scalar)` is
  directly supported on A3 vector or needs an intermediate FP32
  cast tile.

## Build & run

```bash
bash run.sh -r npu -v Ascend910B1
```

## Comparison policy

- Tolerance-based on `Y` (BF16): relative ≤ `1e-2`, absolute ≤ `1e-3`
  (BF16 GEMM upstream + BF16 weighted accumulate downstream).

## Status

`Unknown` whether this builds. **No claim of compile / run success**
until user runs `run.sh` on the compiler server and reports the output.
