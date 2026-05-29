# act_quant_fp4 — block-wise FP4 activation quant (inplace)

Auto-mode A3 **vector** kernel that scaffolds the block-wise FP4
activation quantization used by the DeepSeek-V4 FP4 GEMM path.

Op (per [kernel.py:128-200](../../../../../../../deepseek/kernel.py),
TileLang `fp4_quant_kernel`):

```
along the last axis in blocks of BLOCK_SIZE=32:
  amax = max(|x|, dim=block)
  amax = max(amax, 6 * 2^-126)                # tiny-subnormal guard
  s    = pow2(ceil(log2(amax * (1/6))))       # nearest pow2 >= amax/6
  y    = clamp(x / s, ±6)
  if inplace:  x  = y * s                     # write back as BF16
  else:        out_fp4 = y cast-to-FP4
return s : [M, ceildiv(N, BLOCK_SIZE)]  uint8 (E8M0 exponent)
```

For this prototype, we scaffold the `inplace=True` variant. The scale
is exposed as **uint8** in the I/O contract (E8M0 — biased exponent
only) per the kernel.py contract; gen_data emits it as uint8 for
clarity.

## Reference (TileLang)

[kernel.py:128-200](../../../../../../../deepseek/kernel.py) defines
`fp4_quant_kernel` with:
- block shape `[1, 32]` over the last axis,
- `fp4_max = 6.0`,
- `fp4_min_scale = 6 * 2^-126`,
- power-of-2 scale via the same `fast_round_scale` IEEE-754 bit trick.

The output scale is the **biased exponent** (E8M0) of the pow2 scale —
8 bits, no sign, no mantissa.

## I/O

| Name | Shape | dtype | Source line |
|------|-------|-------|-------------|
| `x` (in/out) | `(M, N)`                       | BF16    | kernel.py:147 (input) |
| `s`          | `(M, ceildiv(N, BLOCK_SIZE))`  | uint8 (E8M0) | kernel.py:195 (scale output) |

`Assumption`: `N % BLOCK_SIZE == 0` (BLOCK_SIZE=32 default) — no tail
handling in the prototype.

## Memory-budget-first plan

Per [CLAUDE.md §Memory-budget-first kernel planning](../../../../../../../CLAUDE.md):

```text
Memory budgets:
- L1 custom budget:    not used
- L0A custom budget:   not used
- L0B custom budget:   not used
- L0C custom budget:   not used
- UB custom budget:    one row of x (BF16, N) + per-block amax (FP32) +
                       per-block scale (uint8 E8M0 + materialized FP32
                       pow2 used for the dequant). Target ≤ 32 KB for
                       the tiny default.

Live tiles by memory level:
- L1:  none
- L0A: none
- L0B: none
- L0C: none
- UB:  x_row       (BF16, [N])
       amax       (FP32, [N / BLK])
       scale_f32  (FP32, [N / BLK])  — pow2 expanded
       scale_u8   (uint8, [N / BLK]) — E8M0 exponent for GM write
       y_row      (BF16, [N])         — quantized + dequant for inplace

Smallest hardware operation:
- cube operation shape: N/A
- vector operation shape:
    - abs / reduce-max over [BLK=32] segment
    - elementwise divide by scale_f32 / clamp ±6 / elementwise multiply

Loop tiling plan:
- tile-and-loop dimensions: outer over M, inner over block index
- inferred tile sizes: 1 row × 1 block per iteration; BLK=32 is small
- compile-time unroll/peel strategy: unroll blocks if N/BLK ≤ 16
- tail handling only where needed: last block if N % BLK != 0 — not in
  prototype default

Test-shape plan:
- tiny debug shape:        M=32,  N=128,  BLK=32 (4 blocks per row)
- medium tiling shape:     M=128, N=512,  BLK=32
- model-inspired realistic: M=128, N=4096, BLK=32
- tail shape:              M=32,  N=160,  BLK=32 (with a few extra cols)
```

## Auto-mode constraints

- A3 only; vector path: `--cce-aicore-arch=dav-c220-vec`.
- Single AICORE. No `block_idx` work split.
- No `TASSIGN`, `TPipe`, `TPUSH`/`TPOP`, no raw `set_flag`/`wait_flag`,
  no `Tile::data()` pointer casts.
- FP4 is emulated in the numpy golden (clamp + round to a 4-value grid
  per sign — see `scripts/gen_data.py`).

## Build & run

```bash
bash run.sh -r npu -v Ascend910B1
```

## Comparison policy

- Tolerance-based: relative ≤ `2e-1`, absolute ≤ `5e-2` on the BF16
  inplace round-trip (FP4 only has ~2-3 bits of mantissa). Scale
  uint8 (E8M0) compared bit-exact.

## Status

`Unknown` whether this builds. **No claim of compile / run success.**
