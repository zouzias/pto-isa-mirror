# HCA/quant — Block-wise activation quantization

Auto-mode A3 vector-only kernels mirroring `act_quant_kernel`
([kernel.py:41](../../../../../../deepseek/kernel.py)) and `fp4_quant_kernel`
([kernel.py:128](../../../../../../deepseek/kernel.py)).

Both kernels support **inplace=True** (fused quant + dequant back to BF16),
which is the variant the deepseek model actually invokes (see model.py:373,
model.py:507, model.py:417).

## Leaves

| Leaf | HW | Op | Block size | Scale dtype |
|------|----|----|------------|-------------|
| [act_quant_fp8/](act_quant_fp8/) | vector | `s = amax / fp8_max`, `y = clamp(x/s, ±448)` (optional `round_scale` to pow2) | 128 (default) or 64 (model.py:373) | FP32 or E8M0 (UE8M0) |
| [act_quant_fp4/](act_quant_fp4/) | vector | `s = pow2(ceil(log2(amax * fp4_max_inv)))`, `y = clamp(x/s, ±6)` | 32 | E8M0 only |

## Algorithmic notes

- **Power-of-2 scale (`round_scale=True`)** uses IEEE-754 bit manipulation
  (kernel.py:22-37) — no `log` / `ceil` intrinsic needed. This is auto-mode
  friendly: pure integer + reinterpret + arithmetic on FP32 lanes.
- **Inplace path** writes `y_local = (out_dtype)((compute_dtype)((out_dtype)(clamp(...))) * s)`
  — a 3-stage casting roundtrip that is sensitive to FP overflow; document
  the exact cast chain in the leaf README.
- **Amax floor** prevents `s = 0`:
  - FP8: `amax = max(amax, 1e-4)`
  - FP4: `amax = max(amax, 6 * 2^-126)`

## Family shape contract

Generated header: `build/generated_cases.h` (see [scripts/generate_cases.py](scripts/generate_cases.py)).
Case tuple: `M, N, BLOCK_SIZE, INPLACE`.

The kernel-launch shape is `(ceildiv(M, blk_m), ceildiv(N, block_size))`,
where `blk_m = 32` is fixed in both kernels.
