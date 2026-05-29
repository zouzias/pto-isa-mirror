# gate_hash_routing — DeepSeek-V4 hash-table routing

Auto-mode A3 **vector** kernel that looks up routing indices in a fixed
hash table `tid2eid` ([deepseek/model.py:577-578](../../../../../deepseek/model.py)):

```python
indices = self.tid2eid[input_ids]          # pure gather; no topk, no reduce
weights = original_scores.gather(1, indices)
if self.score_func != "softmax":
    weights /= weights.sum(dim=-1, keepdim=True)
weights *= self.route_scale
```

This leaf is taken when `layer_id < args.n_hash_layers` (model.py:557, 559).
**It is functionally distinct from [gate_score_topk/](../gate_score_topk/)**:

| | gate_score_topk | gate_hash_routing |
|---|---|---|
| selection input | `(T, N_ROUTED)` biased scores | `(T,)` token IDs |
| algorithm       | TSORT32 + TGATHER over E axis | direct lookup in `(VOCAB, N_ACTIVATED)` table |
| reduce          | yes (per-row max + sum) | none |
| sort            | yes (per-row top-K) | none |
| GM access       | regular (T × N_ROUTED) | scattered (one row of `tid2eid` per token) |

Treating them as one kernel would force a `if (self.hash)` branch that
the auto-mode compiler cannot fold cleanly. Keeping them separate also
lets each leaf be tuned for its access pattern: the score branch wants
N_ROUTED-wide rows packed densely; the hash branch is a per-token
indirect load.

Both downstream consumers (`scatter`, `expert_ffn`, `gather`) take the
same `(indices, weights)` contract regardless of which branch produced
them.

## I/O

| Name | Shape | dtype | Notes |
|------|-------|-------|-------|
| `input_ids`       | `(T)`              | INT32 | token IDs |
| `tid2eid`         | `(VOCAB, N_ACTIVATED)` | INT32 | static hash table |
| `original_scores` | `(T, N_ROUTED)`   | FP32  | gate_softmax output (still needed for weights) |
| `indices`         | `(T, N_ACTIVATED)`| INT32 | out |
| `weights`         | `(T, N_ACTIVATED)`| FP32  | out (gathered + normalized + scaled) |

Shape constants (from `../build/generated_cases.h`):
`kDsmoeT, kDsmoeNRouted, kDsmoeNActivated, kDsmoeVocab`.

## Memory-budget-first plan

```text
Memory budgets:
- UB custom budget:    Per-token row of tid2eid (kNActivated INT32) +
                       original_scores row (kNRouted FP32) + small
                       indices/weights output tile. Per-row footprint
                       ~64 bytes for N_ROUTED=8, N_ACTIVATED=2. The
                       whole tid2eid table (VOCAB × N_ACTIVATED = 129280 ×
                       2 × 4 = 1 MB at realistic shape) lives in GM —
                       NOT preloaded to UB.
- L1, L0A/B/C:         N/A — vector-only leaf.

Live tiles by memory level:
- UB: tidLookupTile   [1, N_ACTIVATED] INT32 (one token's expert IDs)
      origRowTile     [1, N_ROUTED]    FP32  (one token's un-biased scores)
      idxRowTile      [1, N_ACTIVATED] INT32 (output indices for this token)
      wRowTile        [1, N_ACTIVATED] FP32  (output weights for this token)

Smallest hardware operation:
- vector operation shape: per-row scalar gather (mirrors
  MoE/scatter_kernel.cpp pass 3 — GM scalar reads inside a per-token loop).
- cube operation shape: N/A.

Loop tiling plan:
- tile-and-loop dimensions: per-token loop over t in [0, T).
- inferred tile sizes: 1 row at a time; UB footprint independent of T.
- compile-time unroll/peel strategy: none (T iterations).
- tail handling only where needed: none.

Test-shape plan:
- tiny debug shape:         T=64,  VOCAB=1024,   N_ROUTED=8, N_ACTIVATED=2
- model-inspired realistic: T=256, VOCAB=129280, N_ROUTED=8, N_ACTIVATED=2
```

## Auto-mode constraints

- A3 only; vector path.
- Single AICORE.
- Mirror the per-iter `pipe_barrier(PIPE_ALL)` guard from
  [MoE/scatter/scatter_kernel.cpp](../../../MoE/scatter/scatter_kernel.cpp)
  (pass 3 inner loop).
- GM scalar reads for `tid2eid[input_ids[t] * N_ACTIVATED + k]`.
- No `TASSIGN`, no `Tile::data()` from kernel, no `*_IMPL` calls.

## Risks (`Assumption` / `Unknown`)

- `Assumption`: `tid2eid` fits in GM. At realistic shape `VOCAB=129280 ×
  N_ACTIVATED=2 × int32 = 1.03 MB` — well under any practical GM budget.
- `Assumption`: `original_scores` is the **un-biased** output of
  gate_softmax (no bias is added on the hash branch since
  `self.bias is None`, model.py:561).
- `Unknown`: whether per-row scalar GM reads through `int32 tid2eid[...]`
  hit the auto-mode RAW-detection limits at large `T`. If they do, a
  block-fetch of the tid2eid rows via `TLOAD` of a Shape{1, kNActivated}
  vector tile per iter would replace the scalar reads.

## Build & run

```bash
bash run.sh -r npu -v Ascend910B1
```

## Comparison policy

- `indices`: exact byte equality (memcmp on INT32).
- `weights`: relative ≤ `1e-3`, absolute ≤ `1e-4` (FP32 divide / scale —
  small ULP drift).

## Status

`Unknown` whether this builds. **No claim of compile / run success**
until user runs `run.sh` on the compiler server and reports the output.
