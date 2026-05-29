# hc_pre_reduce — `y = sum(pre.unsqueeze(-1) * x.view(b,s,hc,d), dim=2)`

Auto-mode A3 **vector** kernel that collapses the `hc_mult` HC copies of the
hidden state into a single residual, weighted by the per-head `pre`
coefficients produced by `hc_sinkhorn`.

## Reference (PyTorch)

Source: [model.py:681](../../../../../deepseek/model.py)
(`Block.hc_pre` final line):

```python
y = torch.sum(pre.unsqueeze(-1) * x.view(b, s, hc, d), dim=2)
# pre: [B, S, HC_MULT],  x: [B, S, HC_MULT, DIM]  ->  y: [B, S, DIM]
```

The `pre` argument is the FP32 output of `hc_split_sinkhorn` (`pre[b, s, j] =
sigmoid(...) + eps`).

## I/O

| Name  | Shape                       | dtype | Source line |
|-------|-----------------------------|-------|-------------|
| `pre` | `(B, S, HC_MULT)`           | FP32  | model.py:681 (lhs unsqueeze) |
| `x`   | `(B, S, HC_MULT, DIM)`      | FP32  | model.py:681 (rhs `x.view(shape)` — shape is the original `x.size()` before flatten) |
| `y`   | `(B, S, DIM)`               | FP32  | model.py:681 |

`y[b, s, d] = sum_{h=0..HC_MULT-1} pre[b, s, h] * x[b, s, h, d]`.

## Memory-budget-first plan

```text
Memory budgets:
- L1 custom budget:  N/A (vector path)
- L0A/L0B/L0C:       N/A
- UB custom budget:  per-token working set
                       - pre_token   : HC_MULT * 4 B  = 16 B
                       - x_token     : HC_MULT * DIM_TILE * 4 B
                                       (e.g. 4 * 256 * 4 = 4096 B for DIM_TILE=256)
                       - y_token     : DIM_TILE * 4 B (256 -> 1024 B)
                      Total per-tile UB << 1 KB - 8 KB depending on DIM_TILE.

Live tiles by memory level:
- UB:  ubPre   (HC_MULT scalars per token; broadcast across DIM_TILE),
       ubX_h   (one slice (DIM_TILE,) per HC_MULT lane),
       ubY     (DIM_TILE accumulator per token).

Smallest hardware operation:
- vector operation shape: VL FP32 (multiply-add over DIM_TILE; reduction is
  a sum over HC_MULT (= 4) scalars per output element).
- cube operation shape: none

Loop tiling plan:
- tile-and-loop dimensions: outer serial loop over tokens t = 0..(B*S)-1;
  inner loop over DIM tiled by kDimTile.
- inferred tile sizes: kDimTile = 256 (Assumption — keep UB <= few KB)
- compile-time unroll/peel: HC_MULT=4 fan-in is a small known constant;
  candidate for compile-time unroll of the HC_MULT-axis FMA chain.
- tail handling: DIM tail when DIM % kDimTile != 0 (auto-mode valid-region).

Test-shape plan:
- tiny debug shape:        B=1, S=16,  HC_MULT=4, DIM=64
- medium shape:            B=1, S=64,  HC_MULT=4, DIM=1024
- model-inspired realistic: B=1, S=128, HC_MULT=4, DIM=4096
- tail shape: DIM=130 forces a 256-aligned tail of 2 lanes.
```

## Auto-mode constraints

- A3 only; vector path: `--cce-aicore-arch=dav-c220-vec` (see [CMakeLists.txt](CMakeLists.txt)).
- Single AICORE. No `block_idx` work split (v1).
- No `TASSIGN`, `TPipe`, `TPUSH`/`TPOP`, no raw `set_flag`/`wait_flag`,
  no `Tile::data()` pointer casts.
- The reduction (`sum over hc`) is over a tiny static axis (HC_MULT=4),
  so it's a chain of 4 FMAs per (token, dim_tile) — not a true vector
  reduce intrinsic.

## Build & run

```bash
bash run.sh -r npu -v Ascend910B1
```

## Comparison policy

Tolerance-based: relative <= `1e-4`. Sum is over only `HC_MULT=4`
products per output element; numerical drift is small.

## Known limitations

- Single AICORE; multi-core token split is a follow-up.
- `pre` is loaded from a separate buffer (it is the output of `hc_sinkhorn`
  in the full pipeline; this leaf treats it as an independent input).

## Status

`Unknown` whether this builds — kernel body is a pseudocode skeleton, not a
real implementation. **No claim of compile / run success** until user runs
`run.sh` on the compiler server and reports the output.
