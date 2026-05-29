# output_rescale — sparse_attn final /sum_exp + attn_sink tail

Auto-mode A3 **vector** kernel scaffold. Stage 5 of the FlashAttention-style
sparse-attention pipeline (`sparse_attn_kernel` at
[deepseek/kernel.py:277-352](../../../../../../deepseek/kernel.py)).

## Pipeline position

Runs **once per (b, m) tuple**, AFTER the pipelined inner-block loop
completes (kernel.py:345-350). The driver invokes this leaf only once
(no `t` index), passing the post-loop `acc_o`, `scores_max`, `sum_exp`, plus
the per-head learnable `attn_sink`. Output is the final BF16 `o` tile for
`o[by, bx, :, :]`.

## Op (kernel.py:345-350)

```python
for i in T.Parallel(h):
    sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])
for i, j in T.Parallel(h, d):
    acc_o[i, j] /= sum_exp[i]
T.copy(acc_o, o_shared)               # FP32 -> BF16 cast
T.copy(o_shared, o[by, bx, :, :])     # GM write
```

## I/O

| Name | Shape | dtype | Direction | Source |
|------|-------|-------|-----------|--------|
| `acc_o` | `(H, D)` | FP32 | in | kernel.py:309 (post-loop accumulator) |
| `scores_max` | `(H,)` | FP32 | in | kernel.py:310 |
| `sum_exp` | `(H,)` | FP32 | in/out | kernel.py:314 |
| `attn_sink` | `(H,)` | FP32 | in | kernel.py:298 |
| `o` | `(H, D)` | BF16 | out | kernel.py:297 (`o[by, bx, :, :]`) |

## Memory-budget-first plan

Per [CLAUDE.md §Memory-budget-first kernel planning](../../../../../../CLAUDE.md):

```text
Memory budgets:
- L1 custom budget:    none
- L0A custom budget:   none
- L0B custom budget:   none
- L0C custom budget:   none
- UB custom budget:    [H, D] FP32 acc + [H, D] BF16 out + 3x [H] FP32 vectors
                       <= ~6 KB for tiny shape, <= ~96 KB for realistic

Live tiles by memory level:
- L1:  none
- L0A: none
- L0B: none
- L0C: none
- UB:  accTile (FP32, H x D), oTile (BF16, H x D), maxV/sumV/sinkV (FP32, H)

Smallest hardware operation:
- cube operation shape: N/A
- vector operation shape: per-head broadcast division along D, scalar exp,
                          BF16 cast

Loop tiling plan:
- tile-and-loop dimensions: single tile (H, D both small)
- inferred tile sizes: full [H, D] UB-resident
- compile-time unroll/peel strategy: none for prototype
- tail handling only where needed: none — this leaf has no inner loop

Test-shape plan:
- tiny debug shape:        H=4,  D=32        (DEFAULT)
- medium tiling shape:     H=16, D=64
- model-inspired realistic: H=16, D=128      (matches kernel.py defaults)
- tail shape:              n/a
```

## Auto-mode constraints

- A3 only; vector path: `--cce-aicore-arch=dav-c220-vec` (see
  [CMakeLists.txt](CMakeLists.txt)).
- Single AICORE. No `block_idx` work split.
- No `TASSIGN`, `TPipe`, `TPUSH`/`TPOP`, no raw `set_flag`/`wait_flag`,
  no `Tile::data()` pointer casts.
- Per-head broadcast division along D: either explicit per-row vector
  multiply by `1/sum[i]` (preferred) or a broadcast primitive — match
  whatever `add_tile_array` uses for elementwise vector ops.

## Build & run

```bash
bash run.sh -r npu -v Ascend910B1
```

## Comparison policy

- BF16 `o`: bit-exact uint16 against the python BF16 rounding
  (`(uint32 + 0x8000) >> 16`).

## Known limitations

- Single (b, m); the FA driver iterates over `(b, m)` outside this leaf.
- BF16-vs-FP32 cast uses round-half-to-even approximation in the python
  reference (matching the project convention); if the kernel rounds
  differently, the bit-exact compare may fail and the comparison should be
  relaxed to a tolerance.

## Status

`Unknown` whether this builds — kernel body is a pseudocode skeleton, not a
real implementation. **No claim of compile / run success** until user runs
`run.sh` on the compiler server and reports the output.
