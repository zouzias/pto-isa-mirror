# hc_post_combine — `y = post*x + sum(comb*residual, dim=hc_axis)`

Auto-mode A3 **vector** kernel that expands the single-stream residual back
into `hc_mult` copies, using the `post` and `comb` weights produced by
`hc_sinkhorn`.

## Reference (PyTorch)

Source: [model.py:685-686](../../../../../deepseek/model.py)
(`Block.hc_post`):

```python
y = post.unsqueeze(-1) * x.unsqueeze(-2) \
  + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
# x:        [B, S, DIM]
# residual: [B, S, HC_MULT, DIM]
# post:     [B, S, HC_MULT]
# comb:     [B, S, HC_MULT, HC_MULT]
# y:        [B, S, HC_MULT, DIM]
```

Explicit index form (per `[b, s, h, d]`):

```text
y[b, s, h, d] = post[b, s, h]  * x[b, s, d]
              + sum_{h2=0..HC_MULT-1}  comb[b, s, h2, h] * residual[b, s, h2, d]
```

(`comb.unsqueeze(-1)` broadcasts over `d`; `residual.unsqueeze(-2)`
broadcasts over the output `h` dim — so the `sum` over `dim=2` reduces over
the `h2` axis of `comb`'s first hc dim. The output `h` matches the second
`comb` axis. See model.py:686.)

## I/O

| Name      | Shape                       | dtype | Source line |
|-----------|-----------------------------|-------|-------------|
| `x`       | `(B, S, DIM)`               | FP32  | model.py:685 |
| `residual`| `(B, S, HC_MULT, DIM)`      | FP32  | model.py:685 |
| `post`    | `(B, S, HC_MULT)`           | FP32  | model.py:685 |
| `comb`    | `(B, S, HC_MULT, HC_MULT)`  | FP32  | model.py:685 |
| `y`       | `(B, S, HC_MULT, DIM)`      | FP32  | model.py:685 |

## Memory-budget-first plan

```text
Memory budgets:
- L1 custom budget:  N/A (vector path)
- L0A/L0B/L0C:       N/A
- UB custom budget:  per-token working set
                       - ubX        : DIM_TILE * 4 B
                       - ubRes      : HC_MULT * DIM_TILE * 4 B
                       - ubPost     : HC_MULT * 4 B
                       - ubComb     : HC_MULT * HC_MULT * 4 B
                       - ubY        : HC_MULT * DIM_TILE * 4 B (per-h accumulator)
                      For DIM_TILE=128, HC_MULT=4: ~3 KB per tile -> easy fit.

Live tiles by memory level:
- UB:  ubX (DIM_TILE), ubRes (HC_MULT, DIM_TILE), ubPost (HC_MULT),
       ubComb (HC_MULT, HC_MULT), ubY (HC_MULT, DIM_TILE).

Smallest hardware operation:
- vector operation shape: VL FP32 (scalar-broadcast FMA over DIM_TILE; the
  inner reduction is over HC_MULT (= 4) products per output element).
- cube operation shape: none

Loop tiling plan:
- tile-and-loop dimensions: outer serial loop over tokens t = 0..(B*S)-1;
  inner loop over DIM tiled by kDimTile.
- inferred tile sizes: kDimTile = 128 (smaller than hc_pre_reduce because we
  hold both ubRes [HC_MULT, DIM_TILE] and ubY [HC_MULT, DIM_TILE]
  simultaneously).
- compile-time unroll/peel: HC_MULT=4 axes are static; the inner double loop
  over `h, h2` is a candidate for full unroll.
- tail handling: DIM tail when DIM % kDimTile != 0.

Test-shape plan:
- tiny debug shape:        B=1, S=16,  HC_MULT=4, DIM=64
- medium shape:            B=1, S=64,  HC_MULT=4, DIM=1024
- model-inspired realistic: B=1, S=128, HC_MULT=4, DIM=4096
- tail shape: DIM=130 forces a 128-aligned tail of 2 lanes.
```

## Auto-mode constraints

- A3 only; vector path: `--cce-aicore-arch=dav-c220-vec` (see [CMakeLists.txt](CMakeLists.txt)).
- Single AICORE. No `block_idx` work split (v1).
- No `TASSIGN`, `TPipe`, `TPUSH`/`TPOP`, no raw `set_flag`/`wait_flag`,
  no `Tile::data()` pointer casts.
- The combination matrix `comb` is `[HC_MULT, HC_MULT] = [4, 4]` per token,
  loaded once per token. The cross-axis indexing (sum over `comb[h2, h] *
  residual[h2, d]`) means the inner h2 loop must read both `comb` (with `h`
  as the outer broadcast index) and `residual` slice (h2, :).

## Build & run

```bash
bash run.sh -r npu -v Ascend910B1
```

## Comparison policy

Tolerance-based: relative <= `1e-4`. Sum is over `HC_MULT=4` products
per output element plus one `post * x` term.

## Known limitations

- Single AICORE; multi-core token/hc split is a follow-up.
- Output is `[B, S, HC_MULT, DIM]` — `HC_MULT * DIM = 4 * 4096 = 16384` FP32
  values per token in the realistic case.

## Status

`Unknown` whether this builds — kernel body is a pseudocode skeleton, not a
real implementation. **No claim of compile / run success** until user runs
`run.sh` on the compiler server and reports the output.
