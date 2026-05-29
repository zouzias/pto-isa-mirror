# gate_softmax — DeepSeek-V4 router score activation

Auto-mode A3 **vector** kernel that converts router logits into routing
scores ([deepseek/model.py:567-572](../../../../../deepseek/model.py)):

```python
if   self.score_func == "softmax":      scores = scores.softmax(dim=-1)
elif self.score_func == "sigmoid":      scores = scores.sigmoid()
else:                                   scores = F.softplus(scores).sqrt()  # default
```

The third branch — `sqrt(softplus(x))` — is the **ModelArgs default**
(`score_func="sqrtsoftplus"`, model.py earlier `ModelArgs` section). This
scaffold picks that variant by default; the other two are gated behind
compile-time flags described below.

## I/O

| Name | Shape | dtype | Notes |
|------|-------|-------|-------|
| `scores_in`   | `(T, N_ROUTED)` | FP32 | gate_logits output |
| `scores_out`  | `(T, N_ROUTED)` | FP32 | activated scores; `original_scores` downstream |

Shape constants (from `../build/generated_cases.h`):
`kDsmoeT, kDsmoeNRouted`.

## Variants behind compile flags

| Flag | Op |
|------|----|
| (none — default) | `sqrt(softplus(x)) = sqrt(log(1 + exp(x)))` |
| `-DSCORE_FUNC_SOFTMAX` | row-softmax over `N_ROUTED` |
| `-DSCORE_FUNC_SIGMOID` | elementwise `sigmoid(x)` |

Only one flag at a time; documented at the top of the kernel.

## Memory-budget-first plan

```text
Memory budgets:
- UB custom budget:    FP32 (T, N_ROUTED) input tile + scratch + output tile.
                       At T=256, N_ROUTED=8 this is ~24 KB total (well under
                       UB cap). At T=64, N_ROUTED=8 it is ~6 KB.
- L1, L0A/B/C:         N/A — vector-only leaf.

Live tiles by memory level:
- UB: inTile  [T, N_ROUTED]  FP32
      tmpTile [T, N_ROUTED]  FP32
      outTile [T, N_ROUTED]  FP32
  For softmax: + rowMaxTile [T, 1], rowSumTile [T, 1] FP32 (broadcast tiles).

Smallest hardware operation:
- vector operation shape: per-row across N_ROUTED for softmax; pure
  elementwise for sigmoid and sqrtsoftplus.
- cube operation shape: N/A.

Loop tiling plan:
- tile-and-loop dimensions: single (T × N_ROUTED) tile (T is small —
  64 or 256 in defaults). No tiling required for the prototype shapes.
- inferred tile sizes: full (T, N_ROUTED) UB-resident.
- compile-time unroll/peel strategy: none.
- tail handling only where needed: not needed at default shapes.

Test-shape plan:
- tiny debug shape:         T=64,  N_ROUTED=8
- model-inspired realistic: T=256, N_ROUTED=8
- stress shape (UB tile):   T=2048, N_ROUTED=8  (still ~64 KB UB — OK)
```

## Auto-mode constraints

- A3 only; vector path: `--cce-aicore-arch=dav-c220-vec`.
- Single AICORE. No `block_idx` work split.
- For the softmax variant: composition mirrors the FA softmax recipe used
  in `MoE/gather/gather_kernel.cpp`:
  `TROWMAX → TROWEXPANDSUB → TEXP → TROWSUM → TROWEXPANDDIV`.
- For `sqrtsoftplus`: `TEXP` + `TADDS`(1) + `TLN` (or equivalent) + `TSQRT`
  composition. Auto-mode is expected to wire RAW dependencies; if it
  fails, that is the first thing to suspect.
- No `TASSIGN`, no `Tile::data()` from kernel, no `*_IMPL` calls, no raw
  CCE intrinsics.

## Risks (`Assumption` / `Unknown`)

- `Assumption`: `TSOFTPLUS` / `TSQRT` exist as auto-mode primitives on A3
  (or are composable from `TLN + TADDS + TEXP + TSQRT`). Confirm against
  `docs_for_ai/tile_type_reference.md` and the manual-mode FA / softmax
  helpers before committing the pseudocode body.
- `Unknown`: whether `softplus(x) = log(1 + exp(x))` overflow guard is
  needed at `x > ~88` (FP32). Realistic logits from `X @ W` with FP32
  weights at `DIM=4096` can exceed `±100` easily; we likely need
  `softplus(x) ≈ max(0, x) + log(1 + exp(-|x|))` for numerical safety.

## Build & run

```bash
bash run.sh -r npu -v Ascend910B1                  # default sqrtsoftplus
bash run.sh -r npu -v Ascend910B1 -- -DSCORE_FUNC_SOFTMAX
```

## Comparison policy

- Tolerance-based: relative ≤ `1e-3`, absolute ≤ `1e-4` (FP32 elementwise
  transcendental — modest ULP drift vs numpy reference).

## Status

`Unknown` whether this builds. **No claim of compile / run success** until
user runs `run.sh` on the compiler server and reports the output.
