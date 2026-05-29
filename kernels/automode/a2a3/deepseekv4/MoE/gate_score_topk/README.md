# gate_score_topk — DeepSeek-V4 score-branch routing

Auto-mode A3 **vector** kernel that selects the top `N_ACTIVATED` experts
per token from a (biased) activated-score matrix, then gathers the
**original** un-biased scores at those positions and normalizes
([deepseek/model.py:574-584](../../../../../deepseek/model.py)):

```python
if self.bias is not None:
    scores = scores + self.bias        # bias-shifted scores for selection only
indices = scores.topk(self.topk, dim=-1)[1]
weights = original_scores.gather(1, indices)  # gather from *un-biased* scores
if self.score_func != "softmax":
    weights /= weights.sum(dim=-1, keepdim=True)
weights *= self.route_scale
```

This leaf consumes `gate_softmax` output and produces the `indices` /
`weights` pair that drives `scatter` and `expert_ffn`.

## I/O

| Name | Shape | dtype | Notes |
|------|-------|-------|-------|
| `original_scores` | `(T, N_ROUTED)` | FP32 | gate_softmax output (un-biased) |
| `bias` (optional) | `(N_ROUTED)`    | FP32 | model.py:561-563; per-expert |
| `indices`         | `(T, N_ACTIVATED)` | INT32 | top-K expert IDs per token |
| `weights`         | `(T, N_ACTIVATED)` | FP32  | gathered + normalized + scaled |

Shape constants (from `../build/generated_cases.h`):
`kDsmoeT, kDsmoeNRouted, kDsmoeNActivated`.

Build-time configurable scalars:
- `kRouteScale` — `ModelArgs.route_scale` (default `1.0`)
- `kScoreFunc`  — `"softmax"` skips the normalize step

## Memory-budget-first plan

```text
Memory budgets:
- UB custom budget:    FP32 (T, N_ROUTED) scores + bias + packed
                       (T, kPackedTopK) topk output + (T, N_ACTIVATED)
                       indices / weights tiles.
                       At T=64 N_ROUTED=8 N_ACTIVATED=2: ~6 KB total.
                       At T=256 N_ROUTED=8 N_ACTIVATED=2: ~24 KB.
- L1, L0A/B/C:         N/A — vector-only leaf.

Live tiles by memory level:
- UB: scoresTile     [T, N_ROUTED]  FP32 (biased)
      biasTile       [N_ROUTED]     FP32 (broadcast row)
      origScoresTile [T, N_ROUTED]  FP32 (un-biased; for gather)
      sortDstTile    [T, kPackedTopK] FP32 (TSORT32 packed (val,idx))
      indicesTile    [T, N_ACTIVATED] INT32
      weightsTile    [T, N_ACTIVATED] FP32
      rowSumTile     [T, 1] FP32     (for normalize)

Smallest hardware operation:
- vector operation shape: TSORT32 (one block per token row) + TGATHER /
  TROWSUM / TROWEXPANDDIV for normalize.
- cube operation shape: N/A.

Loop tiling plan:
- tile-and-loop dimensions: per-row scalar loop over T (mirrors
  MoE/moe_topk_kernel.cpp); no inner tiling required at these shapes.
- inferred tile sizes: full (T, N_ROUTED) UB-resident.
- compile-time unroll/peel strategy: none.
- tail handling only where needed: none (T need not be a multiple of any
  vector lane size since the row loop is per-token).

Test-shape plan:
- tiny debug shape:         T=64,  N_ROUTED=8, N_ACTIVATED=2
- model-inspired realistic: T=256, N_ROUTED=8, N_ACTIVATED=2
```

## Building blocks

This leaf can lean on the existing `MoE/moe_topk_kernel.cpp` pattern
for the `TSORT32 → TSUBVIEW → TGATHER` block (kCols=32, kTopK=K). For
DeepSeek-V4 with `N_ROUTED=8` < 32, the input is zero-padded to 32 cols
before TSORT32 (host-pad with `-FLT_MAX` so padded entries never
top-K).

> Note: `N_ROUTED=8` produces a packed-tile width of 16 elements after
> TSORT32; the existing top-level `moe_topk` was sized for `kCols=32`.
> Confirm the alignment constants in `pto_tile.hpp` accept the smaller
> width or stick with the 32-col-padding approach.

## Auto-mode constraints

- A3 only; vector path.
- Single AICORE.
- For the bias-add and normalize composition, mirror `MoE/gather/`'s
  `TROWMAX → TROWEXPANDSUB → TEXP → TROWSUM → TROWEXPANDDIV` recipe
  (replace the exp() phase with the identity for plain row-normalize).
- No `TASSIGN`, `Tile::data()` in kernel, `*_IMPL` calls, raw CCE
  intrinsics, manual sync, or `TPipe`/`TPUSH`/`TPOP`.

## Risks (`Assumption` / `Unknown`)

- `Assumption`: TSORT32 accepts FP32 input of width `kPaddedNRouted=32`.
  The existing `MoE/moe_topk` confirms this for `kCols=32`.
- `Assumption`: the bias add can fuse with the topk path; if not, run it
  as a separate pre-pass and pass the biased view to TSORT32.
- `Unknown`: whether the gather step (`weights = original_scores.gather(1, indices)`)
  is best implemented as a per-row scalar gather (mirrors
  `scatter_kernel.cpp` pass 3) or as a TGATHER over packed `(val, idx)`.
  Per-row scalar is the safer prototype.

## Build & run

```bash
bash run.sh -r npu -v Ascend910B1
```

## Comparison policy

- `indices`: exact byte equality (memcmp on INT32).
- `weights`: relative ≤ `1e-3`, absolute ≤ `1e-4` (FP32 add / divide /
  scale — small ULP drift).

## Status

`Unknown` whether this builds. **No claim of compile / run success**
until user runs `run.sh` on the compiler server and reports the output.
