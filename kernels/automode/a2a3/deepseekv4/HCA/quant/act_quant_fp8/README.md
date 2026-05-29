# act_quant_fp8 — block-wise FP8 activation quant (inplace)

Auto-mode A3 **vector** kernel that scaffolds the block-wise FP8
activation quantization used by the DeepSeek-V4 compressor / linear
dispatch.

Op (per [kernel.py:41-102](../../../../../../../deepseek/kernel.py),
TileLang `act_quant_kernel`):

```
along the last axis in blocks of BLOCK_SIZE:
  amax = max(|x|, dim=block)
  amax = max(amax, 1e-4)                            # tiny-block guard
  s    = fast_round_scale(amax, 1/448)              # nearest power-of-2
  y    = clamp(x / s, ±448)
  if inplace:  x  = y * s                           # write back as BF16
  else:        out_fp8 = y cast-to-FP8(E4M3)
return s : [M, ceildiv(N, BLOCK_SIZE)]  float32
```

For this prototype, we scaffold the `inplace=True` variant — the call
shape the model actually uses (model.py:373 — Compressor's non-rope
dims call `act_quant(..., 64)`; the FP8 path uses BLOCK_SIZE=128 by
default).

## Reference (TileLang)

[kernel.py:41-102](../../../../../../../deepseek/kernel.py) defines
`act_quant_kernel` with:
- `T.symbolic` block shape `[1, BLK]` over the last axis,
- `fp8_max = 448.0`, `fp8_min_scale = 1e-4`,
- `fast_round_scale` implements an IEEE-754 bit-trick to round to the
  nearest power-of-2 exponent (see `kernel.py:13-39` for the helper).

## I/O

| Name | Shape | dtype | Source line |
|------|-------|-------|-------------|
| `x` (in/out) | `(M, N)`                       | BF16    | kernel.py:60 (input) |
| `s`          | `(M, ceildiv(N, BLOCK_SIZE))`  | FP32    | kernel.py:97 (scale output) |

For non-inplace the additional output would be FP8 `(M, N)`. We do not
scaffold that path here.

`Assumption`: `N % BLOCK_SIZE == 0` in the default case so no last-block
tail handling is needed in the prototype. The case generator warns
otherwise.

## Memory-budget-first plan

Per [CLAUDE.md §Memory-budget-first kernel planning](../../../../../../../CLAUDE.md):

```text
Memory budgets:
- L1 custom budget:    not used
- L0A custom budget:   not used
- L0B custom budget:   not used
- L0C custom budget:   not used
- UB custom budget:    one row of x (BF16, N) + per-block amax/scale
                       (FP32, ceildiv(N, BLK)) + one row of scaled-back
                       BF16 for write-back. Target ≤ 32 KB for the
                       tiny default (M=32, N=128).

Live tiles by memory level:
- L1:  none
- L0A: none
- L0B: none
- L0C: none
- UB:  x_row  (BF16, [N])
       amax  (FP32, [N / BLK])
       scale (FP32, [N / BLK])
       y_row (BF16, [N]) — quantized then dequant for inplace write-back

Smallest hardware operation:
- cube operation shape: N/A
- vector operation shape:
    - abs over [BLK] segment
    - reduce-max over [BLK] segment -> scalar
    - elementwise divide by scale -> elementwise clamp [-448, 448]
    - elementwise multiply by scale (inplace dequant)

Loop tiling plan:
- tile-and-loop dimensions: outer loop over M (rows), inner loop over
  block index b in [0, N/BLK)
- inferred tile sizes: 1 row × 1 block per iteration. Auto-mode pass can
  fuse the per-row sequence.
- compile-time unroll/peel strategy: unroll over blocks if N/BLK small
- tail handling only where needed: last block when N % BLK != 0 — not in
  the prototype default

Test-shape plan:
- tiny debug shape:        M=32,  N=128, BLK=128
- medium tiling shape:     M=128, N=512, BLK=128
- model-inspired realistic: M=B*S=128, N=4096, BLK=128 (and BLK=64 for the
                            compressor's non-rope path, model.py:373)
- tail shape:              M=32,  N=144, BLK=128  (1 full + 1 partial block)
```

## Auto-mode constraints

- A3 only; vector path: `--cce-aicore-arch=dav-c220-vec`.
- Single AICORE. No `block_idx` work split.
- No `TASSIGN`, `TPipe`, `TPUSH`/`TPOP`, no raw `set_flag`/`wait_flag`,
  no `Tile::data()` pointer casts.
- FP8 is emulated in the numpy golden (clamp + round to E4M3-friendly
  values); see `scripts/gen_data.py`. The golden BF16 inplace output
  matches the dequantized BF16 the kernel produces.

## Build & run

```bash
bash run.sh -r npu -v Ascend910B1
```

## Comparison policy

- Tolerance-based: relative ≤ `5e-2`, absolute ≤ `1e-2` on the BF16
  inplace output (FP8 round-trip drops ~3 mantissa bits) and on the FP32
  scale.

## Status

`Unknown` whether this builds. **No claim of compile / run success.**
