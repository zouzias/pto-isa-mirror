# hc_sinkhorn — split + sigmoid + softmax + Sinkhorn iterations

Auto-mode A3 **vector** kernel that consumes `mixes [B*S, MIX_HC]` and emits
three tensors: `pre [B*S, HC_MULT]`, `post [B*S, HC_MULT]`, and a doubly
stochastic `comb [B*S, HC_MULT, HC_MULT]` (Sinkhorn-normalized).

## Reference (TileLang)

Source: [deepseek/kernel.py:371-427](../../../../../deepseek/kernel.py) —
`hc_split_sinkhorn_kernel`. Per-row (`i = 0..N-1` where `N = B*S`):

```python
mixes_shared = mixes[i, :]          # [MIX_HC], MIX_HC = (2+hc)*hc = 24
for j in range(hc):
    pre[i, j]  = sigmoid(mixes_shared[j]            * hc_scale[0]
                         + hc_base[j])              + eps
for j in range(hc):
    post[i, j] = 2 * sigmoid(mixes_shared[j+hc]     * hc_scale[1]
                             + hc_base[j+hc])
for j, k in product(range(hc), range(hc)):
    comb_frag[j, k] = (mixes_shared[j*hc + k + 2*hc] * hc_scale[2]
                       + hc_base[j*hc + k + 2*hc])

# First normalization: row-softmax + eps, then column-normalize
row_max = reduce_max(comb_frag, dim=1)
comb_frag = exp(comb_frag - row_max[:, None])
row_sum = reduce_sum(comb_frag, dim=1)
comb_frag = comb_frag / row_sum[:, None] + eps

col_sum = reduce_sum(comb_frag, dim=0)
comb_frag = comb_frag / (col_sum[None, :] + eps)

# Remaining (sinkhorn_iters - 1) iterations: row-normalize, col-normalize
for _ in range(sinkhorn_iters - 1):
    row_sum = reduce_sum(comb_frag, dim=1)
    comb_frag = comb_frag / (row_sum[:, None] + eps)
    col_sum = reduce_sum(comb_frag, dim=0)
    comb_frag = comb_frag / (col_sum[None, :] + eps)

comb[i, :, :] = comb_frag
```

## I/O

| Name      | Shape                  | dtype | Source line |
|-----------|------------------------|-------|-------------|
| `mixes`   | `(N = B*S, MIX_HC)`    | FP32  | kernel.py:379 |
| `hc_scale`| `(3,)`                 | FP32  | kernel.py:380 |
| `hc_base` | `(MIX_HC,)`            | FP32  | kernel.py:381 |
| `pre`     | `(N, HC_MULT)`         | FP32  | kernel.py:382 |
| `post`    | `(N, HC_MULT)`         | FP32  | kernel.py:383 |
| `comb`    | `(N, HC_MULT, HC_MULT)`| FP32  | kernel.py:384 |

With model defaults (`HC_MULT=4`): `MIX_HC=24`, comb is `[4,4]` per row.

## Memory-budget-first plan

```text
Memory budgets:
- L1 custom budget:   N/A (vector path; everything goes through UB)
- L0A custom budget:  N/A
- L0B custom budget:  N/A
- L0C custom budget:  N/A
- UB custom budget:   ~few KB per row
                       - mixes_shared : MIX_HC * 4 B = 96 B
                       - hc_scale     : 3 * 4 B = 12 B (broadcast as scalar)
                       - hc_base      : MIX_HC * 4 B = 96 B
                       - comb_frag    : HC_MULT^2 * 4 B = 64 B
                       - row_max, row_sum, col_sum : HC_MULT * 4 B each (16 B)
                       - pre, post    : HC_MULT * 4 B each (16 B)
                      Total per-row working set << 1 KB. Easy fit.

Live tiles by memory level:
- UB:  mixes_shared (MIX_HC), hc_scale (3), hc_base (MIX_HC), comb_frag (HC_MULT^2),
       row_max / row_sum / col_sum (HC_MULT), pre_row (HC_MULT), post_row (HC_MULT)

Smallest hardware operation:
- vector operation shape: VL FP32. Many of the reductions are over only
  HC_MULT=4 elements, well under the vector width — auto mode handles the
  packing.
- cube operation shape: none

Loop tiling plan:
- tile-and-loop dimensions: outer serial loop over rows i = 0..N-1
  (single AICORE, no block_idx split in v1).
- inferred tile sizes: per-row in-UB processing; `comb_frag [4,4]` resident
  in UB across the Sinkhorn iterations within a row, freed before next row.
- compile-time unroll/peel: HC_MULT=4 is small enough that the inner
  (j, k in range(hc)) double loops are good candidates for compile-time
  unroll. Leave for the implementer.
- tail handling: none (N = B*S is a clean integer; only the row loop has a
  trip count, no inner partial tiles).

Test-shape plan:
- tiny debug shape:        B=1, S=16,  HC_MULT=4, SINKHORN_ITERS=4
                            -> N=16, MIX_HC=24
- medium shape:            B=1, S=128, HC_MULT=4, SINKHORN_ITERS=20
                            -> N=128, MIX_HC=24
- model-inspired realistic: B=1, S=128, HC_MULT=4, SINKHORN_ITERS=20
                            (matches ModelArgs default; HC_MULT is fixed to 4)
- tail shape: N/A (no inner partial tile)
```

## Auto-mode constraints

- A3 only; vector path: `--cce-aicore-arch=dav-c220-vec` (see [CMakeLists.txt](CMakeLists.txt)).
- Single AICORE. No `block_idx` work split (v1).
- No `TASSIGN`, `TPipe`, `TPUSH`/`TPOP`, no raw `set_flag`/`wait_flag`,
  no `Tile::data()` pointer casts.
- **`comb [HC_MULT, HC_MULT]` is `4x4` for the model default** — entire
  matrix fits in UB. The Sinkhorn loop carries no inter-iteration
  dependency that needs flag/sync (auto mode should be happy).
- `sigmoid` and `exp` are needed. Cross-check `docs_for_ai/qualifier_reference.md`
  for the auto-mode-friendly variants (TLOAD/TSIGMOID/TEXP shapes) before
  committing the body.

## Build & run

```bash
bash run.sh -r npu -v Ascend910B1
```

## Comparison policy

Tolerance-based: relative <= `1e-4` for `pre`, `post`; relative <= `1e-3`
for `comb` (more iterations of exp/div compound rounding).

## Known limitations

- Single AICORE; multi-core row-split is a follow-up.
- `hc_mult` is assumed fixed to 4 by the model
  ([model.py:651](../../../../../deepseek/model.py)).
- The model uses `sinkhorn_iters=20` ([model.py:hc_sinkhorn_iters](../../../../../deepseek/model.py)).
  Tiny case uses 4 for faster CI.

## Status

`Unknown` whether this builds — kernel body is a pseudocode skeleton, not a
real implementation. **No claim of compile / run success** until user runs
`run.sh` on the compiler server and reports the output.
