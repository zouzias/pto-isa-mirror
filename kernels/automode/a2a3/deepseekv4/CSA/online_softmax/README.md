# online_softmax — sparse_attn FA running max/exp/rescale/sum

Auto-mode A3 **vector** kernel scaffold. Stage 3 of the FlashAttention-style
sparse-attention pipeline (`sparse_attn_kernel` at
[deepseek/kernel.py:277-352](../../../../../../deepseek/kernel.py)).

## Pipeline position

Runs once per pipelined inner-block iteration (kernel.py:321). Consumes
`acc_s` from `qk_matmul`, updates the per-head running statistics
(`scores_max`, `sum_exp`), produces `acc_s_cast` for `pv_matmul`, and emits
`scores_scale` for the prior-`acc_o` rescale (which the FA driver applies
before invoking `pv_matmul`). This binary runs **one iteration**; the
driver iterates `t` over `[0, num_blocks)`.

## Op (kernel.py:331-340)

```python
scores_max_prev = scores_max
T.reduce_max(acc_s -> scores_max, dim=1, clear=False)
scores_scale = T.exp(scores_max_prev - scores_max)
acc_s = T.exp(acc_s - scores_max[:, None])
acc_s_cast = (bfloat16) acc_s
T.reduce_sum(acc_s -> scores_sum, dim=1)
sum_exp = sum_exp * scores_scale + scores_sum
```

### Mask fold

The upstream `qk_matmul` leaf does **not** apply the `idx==-1 → -inf` mask
(kernel.py:326-327); we fold it in here. Where `topk_idxs[j] == -1` or
`t*block + j >= topk`, we push `acc_s[:, j]` to a large negative bias
(`MASK_NEG = -1e30f`) before the running max so the resulting `exp(.)`
underflows to ~0. Using pure -inf would risk poisoning the running max when
every column in the block is masked; large-negative is robust.

## I/O

| Name | Shape | dtype | Direction | Source |
|------|-------|-------|-----------|--------|
| `acc_s` | `(H, BLOCK)` | FP32 | in/out | kernel.py:308 |
| `acc_s_cast` | `(H, BLOCK)` | BF16 | out | kernel.py:305 |
| `scores_max` | `(H,)` | FP32 | in/out | kernel.py:310 |
| `sum_exp` | `(H,)` | FP32 | in/out | kernel.py:314 |
| `scores_scale` | `(H,)` | FP32 | out | kernel.py:312 |
| `topk_idxs` (this block slice) | `(BLOCK,)` | INT32 | in | kernel.py:299 |

## Memory-budget-first plan

Per [CLAUDE.md §Memory-budget-first kernel planning](../../../../../../CLAUDE.md):

```text
Memory budgets:
- L1 custom budget:    none
- L0A custom budget:   none
- L0B custom budget:   none
- L0C custom budget:   none
- UB custom budget:    1x [H, BLOCK] FP32 acc_s tile
                       + 1x [H, BLOCK] BF16 acc_s_cast tile
                       + 5x [H] FP32 vectors (mPrev, m, scale, sumPrev, sumLocal)
                       + 1x [BLOCK] INT32 idx tile
                       <= ~24 KB for tiny shape, <= ~256 KB for realistic (well within UB)

Live tiles by memory level:
- L1:  none
- L0A: none
- L0B: none
- L0C: none
- UB:  sTile, sCastTile, mPrev, m, scale, sumPrev, sumLocal, idxTile

Smallest hardware operation:
- cube operation shape: N/A
- vector operation shape: row-wise reduce_max + reduce_sum on [H, BLOCK]

Loop tiling plan:
- tile-and-loop dimensions: single tile (H, BLOCK both small)
- inferred tile sizes: full [H, BLOCK] tile UB-resident
- compile-time unroll/peel strategy: none for prototype
- tail handling only where needed: mask fold handles global_j >= topk_len

Test-shape plan:
- tiny debug shape:        H=4,  BLOCK=32        (DEFAULT)
- medium tiling shape:     H=16, BLOCK=64
- model-inspired realistic: H=16, BLOCK=64       (matches kernel.py defaults)
- tail shape:              global_j >= topk path exercised at t = num_blocks - 1
```

## Auto-mode constraints

- A3 only; vector path: `--cce-aicore-arch=dav-c220-vec` (see
  [CMakeLists.txt](CMakeLists.txt)).
- Single AICORE. No `block_idx` work split.
- No `TASSIGN`, `TPipe`, `TPUSH`/`TPOP`, no raw `set_flag`/`wait_flag`,
  no `Tile::data()` pointer casts.
- Reduce primitives (TREDUCE_MAX / TREDUCE_SUM) come from pto/pto-inst.hpp.

## Build & run

```bash
bash run.sh -r npu -v Ascend910B1
```

## Comparison policy

- FP32 outputs: relative <= `1e-2`, absolute <= `1e-3`. Exp + running update
  accumulates a few ULPs per iteration.
- BF16 `acc_s_cast`: compared as bit-exact uint16 against the python BF16
  rounding (`(uint32 + 0x8000) >> 16`).

## Known limitations

- One block per binary; outer pipelined loop deferred.
- t=0 init values (scores_max=-inf, sum_exp=0) hard-coded by `gen_data.py`;
  for non-first blocks the user must override the input files manually.

## Status

`Unknown` whether this builds — kernel body is a pseudocode skeleton, not a
real implementation. **No claim of compile / run success** until user runs
`run.sh` on the compiler server and reports the output.
