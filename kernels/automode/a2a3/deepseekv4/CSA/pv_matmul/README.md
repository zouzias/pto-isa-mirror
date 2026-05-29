# pv_matmul — sparse_attn acc_o += acc_s_cast @ V

Auto-mode A3 **cube** kernel scaffold. Stage 4 of the FlashAttention-style
sparse-attention pipeline (`sparse_attn_kernel` at
[deepseek/kernel.py:277-352](../../../../../../deepseek/kernel.py)).

## Pipeline position

Runs once per pipelined inner-block iteration (kernel.py:321). Consumes
`acc_s_cast` from `online_softmax` and the same `kv_block` that `qk_matmul`
used (the FA driver keeps `kv_shared` resident across qk/pv). Updates
`acc_o` in place. This binary runs **one iteration** of the pipeline.

## Driver-side prelude (NOT in this kernel)

Before invoking `pv_matmul`, the FA driver applies a vector rescale:

```python
for i, j in T.Parallel(h, d):
    acc_o[i, j] *= scores_scale[i]
```

This is the kernel.py:341-342 step. Because this leaf is a **cube-only**
AICORE binary, vector ops are not legal here; the rescale is performed by
a separate vector pass (or fused into `output_rescale` if that turns out
cheaper). The input `./input/input_acc_o.bin` file already encodes the
post-rescale FP32 values.

## Op (kernel.py:343)

```python
T.gemm(acc_s_cast, kv_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
```

## I/O

| Name | Shape | dtype | Direction | Source |
|------|-------|-------|-----------|--------|
| `acc_o` | `(H, D)` | FP32 | in/out | kernel.py:309 (RMW accumulator) |
| `acc_s_cast` | `(H, BLOCK)` | BF16 | in | kernel.py:305 (from online_softmax) |
| `kv_block` | `(BLOCK, D)` | BF16 | in | kernel.py:303 (same gathered block as qk_matmul) |

## Memory-budget-first plan

Per [CLAUDE.md §Memory-budget-first kernel planning](../../../../../../CLAUDE.md):

```text
Memory budgets:
- L1 custom budget:    BF16 [H, BLOCK] panel + BF16 [BLOCK, D] panel
- L0A custom budget:   BF16 [H, BLOCK] fractal panel
- L0B custom budget:   BF16 [BLOCK, D] fractal panel
- L0C custom budget:   FP32 [H, D] fractal accumulator (RMW: preload from gO)
- UB custom budget:    none

Live tiles by memory level:
- L1:  scL1 (BF16, H x BLOCK), vL1 (BF16, BLOCK x D)
- L0A: scL0A (BF16, H x BLOCK fractal)
- L0B: vL0B  (BF16, BLOCK x D fractal)
- L0C: oL0C  (FP32, H x D fractal, RMW)
- UB:  N/A

Smallest hardware operation:
- cube operation shape: 16x16x16 fractal MMA (BF16 x BF16 -> FP32)
- vector operation shape: N/A in this leaf

Loop tiling plan:
- tile-and-loop dimensions: single tile for tiny shape
- inferred tile sizes: kTileM = H, kTileN = D, kTileK = BLOCK (one-shot)
- compile-time unroll/peel strategy: none for prototype
- tail handling only where needed: none — sizes are multiples of 16 after
  head-padding at kernel.py:359

Test-shape plan:
- tiny debug shape:        H=4,  D=32,  BLOCK=32        (DEFAULT — single fractal)
- medium tiling shape:     H=16, D=64,  BLOCK=64
- model-inspired realistic: H=16, D=128, BLOCK=64       (matches kernel.py defaults)
- tail shape:              n/a (multi-tile tiling deferred)
```

## Auto-mode constraints

- A3 only; cube path: `--cce-aicore-arch=dav-c220-cube` (see
  [CMakeLists.txt](CMakeLists.txt)).
- Single AICORE. No `block_idx` work split.
- No `TASSIGN`, `TPipe`, `TPUSH`/`TPOP`, no raw `set_flag`/`wait_flag`,
  no `Tile::data()` pointer casts.
- Reason in 16x16 fractals in L0A/L0B/L0C; **do not** treat tiles as flat
  row-major matrices.
- RMW on L0C: preload from gO via `TLOAD(oL0C, gO)` before `TMATMUL`, rather
  than `TCLEAR`. Anchor in `docs_for_ai/tile_type_reference.md`.

## Build & run

```bash
bash run.sh -r npu -v Ascend910B1
```

## Comparison policy

- Tolerance-based: relative <= `1e-2`. BF16 GEMM with FP32 accumulator
  plus one prior-FP32 addend keeps drift bounded.

## Known limitations

- One block per binary; outer pipelined loop deferred.
- Vector-side rescale `acc_o *= scores_scale[h]` is NOT in this kernel; it
  is encoded in the input file by the test driver.

## Status

`Unknown` whether this builds — kernel body is a pseudocode skeleton, not a
real implementation. **No claim of compile / run success** until user runs
`run.sh` on the compiler server and reports the output.
