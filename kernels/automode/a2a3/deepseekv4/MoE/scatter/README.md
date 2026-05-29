# scatter — DeepSeek-V4 token-to-expert reorder

Auto-mode A3 **vector** kernel that packs tokens by their assigned
expert. Mirrors the I/O contract of the top-level
[MoE/scatter/scatter_kernel.cpp](../../../MoE/scatter/scatter_kernel.cpp),
re-stated against the DeepSeek-V4 shape constants.

```python
# MoE.forward inner part (model.py:635-641):
counts = bincount(indices.flatten(), minlength=n_routed_experts).tolist()
for i in routed_expert_range:
    idx, top = torch.where(indices == i)
    # tokens belonging to expert i: x[idx], routing weight weights[idx, top]
```

For the kernel, this means: given `expert_id[t, k]` of shape
`(T, N_ACTIVATED)` produced by either `gate_score_topk/` or
`gate_hash_routing/`, rearrange `X[T, DIM]` into a per-expert packed
layout `A[T*N_ACTIVATED + 16, DIM]` plus auxiliary back-maps used by
`gather/`.

## I/O

| Name | Shape | dtype | Notes |
|------|-------|-------|-------|
| `X`            | `(T, DIM)`                 | BF16  | token features |
| `expert_id`    | `(T, N_ACTIVATED)`         | INT32 | from gate |
| `A`            | `(T*N_ACTIVATED + 16, DIM)` | BF16 | out; per-expert packed rows; +16 overspill pad |
| `A_id`         | `(T*N_ACTIVATED + 16)`     | INT32 | out; original token positions |
| `rank_id`      | `(T*N_ACTIVATED + 16)`     | INT32 | out; k slot (0..N_ACTIVATED-1) per packed row |
| `expert_start` | `(N_ROUTED)`               | INT32 | out; prefix-sum offsets |
| `expert_count` | `(N_ROUTED)`               | INT32 | out; per-expert counts |

The trailing 16-row overspill pad is undefined and is overwritten by the
downstream `expert_ffn/` outer loop (or left as scratch beyond the
gather's valid range). This matches the top-level scatter contract
(see [MoE/scatter/scatter_kernel.cpp](../../../MoE/scatter/scatter_kernel.cpp)
top-of-file comment).

Shape constants (from `../build/generated_cases.h`):
`kDsmoeT, kDsmoeDim, kDsmoeNRouted, kDsmoeNActivated`.

> Difference from top-level MoE scatter: DeepSeek-V4 uses BF16 token
> features (vs FP16 in the top-level kernel). The algorithm is
> identical; only the row dtype changes. Mirror the same three-pass
> structure (histogram → prefix sum → pack) and the per-iter
> `pipe_barrier(PIPE_ALL)` discipline.

## Memory-budget-first plan

```text
Memory budgets:
- UB custom budget:    One BF16 row tile [1, DIM] for row-copy. At
                       DIM=128 (tiny) this is 256 B; at DIM=4096
                       (realistic) it is 8 KB — well under UB cap.
                       Small INT32 scratch arrays `count[N_ROUTED]`,
                       `start[N_ROUTED]`, `counter[N_ROUTED]` live
                       in scalar memory (each 32 B at N_ROUTED=8).
- L1, L0A/B/C:         N/A — vector-only.

Live tiles by memory level:
- UB: rowTile [1, DIM] BF16   (used for the GM→GM row copy in pass 3)

Smallest hardware operation:
- vector operation shape: row-wide TLOAD/TSTORE (DIM BF16 elements).
- cube operation shape: N/A.

Loop tiling plan:
- tile-and-loop dimensions: pass 1: (t × k) over [0, T)×[0, N_ACTIVATED);
                            pass 2: e over [0, N_ROUTED);
                            pass 3: same (t × k) order as pass 1.
- inferred tile sizes: one row at a time in pass 3.
- compile-time unroll/peel strategy: none.
- tail handling only where needed: none (kT*kTopK fixed valid count;
  16-row overspill pad simply not written).

Test-shape plan:
- tiny debug shape:         T=64,  DIM=128,  N_ROUTED=8, N_ACTIVATED=2
- model-inspired realistic: T=256, DIM=4096, N_ROUTED=8, N_ACTIVATED=2
```

## Auto-mode constraints

- A3 only; vector path.
- Single AICORE; no `block_idx` split.
- Per-row `pipe_barrier(PIPE_ALL)` at the start of the pass-3 inner
  iteration (mirrors `MoE/scatter/scatter_kernel.cpp` and
  `MoE/gather/gather_kernel.cpp`).
- Per-expert histogram / prefix-sum / counter in small local INT32
  arrays (sized `kNRouted`).
- No `TASSIGN`, no `Tile::data()` from kernel, no `*_IMPL`, no raw
  CCE intrinsics, no `TPipe`/`TPUSH`/`TPOP`, no double buffering.

## Risks (`Assumption` / `Unknown`)

- `Assumption`: BF16 row-copy through `TLOAD`/`TSTORE` is bit-exact
  (it should be — gather is pure data movement, no arithmetic).
- `Unknown`: whether `host TU is -xc++` quirk for the `bfloat16_t`
  type-name on the device kernel TU forces the same `uint8_t*`
  reinterpret-cast pattern used in `MoE/scatter` for `half`. Likely
  yes — mirror the pattern.

## Build & run

```bash
bash run.sh -r npu -v Ascend910B1
```

## Comparison policy

- `A`: byte-exact memcmp on the first `T*N_ACTIVATED` rows (BF16 — no
  arithmetic).
- `A_id`, `rank_id`: byte-exact on the first `T*N_ACTIVATED` entries.
- `expert_count`, `expert_start`: exact equality on `N_ROUTED` entries.

## Status

`Unknown` whether this builds. **No claim of compile / run success**
until user runs `run.sh` on the compiler server and reports the output.
