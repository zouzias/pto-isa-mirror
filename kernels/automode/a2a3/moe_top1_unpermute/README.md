# moe_top1_unpermute

## Purpose

Reverse token movement after the passing forward
[moe_top1_permute](../moe_top1_permute/). Restores packed expert output
back to original token order.

This is **top-1 unpermute only**. It does **not** do:

- router / argmax / top-K logits selection,
- expert FFN (no GEMM, no activation),
- per-expert capacity, drop policy, or fallback expert,
- weighted combine (`topK > 1` blending),
- backward pass,
- multi-core dispatch.

It assumes `token_to_packed` is the mapping the passing forward kernel
emits. This kernel completes the forward token-movement pair:

```text
tokens + expert_id
    ↓  moe_top1_permute      (device-side dispatch)
packed_tokens + token_to_packed
    ↓  (fake packed_output = packed_tokens + 1.0 in v1)
packed_output
    ↓  moe_top1_unpermute    (this kernel)
output
```

## Target platform

A3 only (`PTO_NPU_ARCH_A2A3`, compiled with `--cce-aicore-arch=dav-c220-vec`).
A3 auto mode, single AICORE.

## What the kernel does

```cpp
for (t = 0; t < T; ++t) {
    packed_pos      = token_to_packed[t];                // GM scalar read
    output[t, :]    = packed_output[packed_pos, :];      // TLOAD then TSTORE
}
```

`packed_pos` is `int32_t`; the per-iteration GM offset is
`size_t(packed_pos) * H` (matches the discipline used in
[moe_top1_permute](../moe_top1_permute/)).

## Inputs / outputs

**Inputs (GM)**
- `packed_output [T, H]` — `float32`
- `token_to_packed [T]` — `int32`

**Output (GM)**
- `output [T, H]` — `float32`

## Auto-mode constraints honored

- Single AICORE (`<<<1, nullptr, stream>>>`); no `block_idx` work split.
- Static row tile `Tile<TileType::Vec, float, 1, H, BLayout::RowMajor, 1, H>`
  declared once and reused across all `T` iterations; UB address pinned by
  the auto allocator (same as [moe_top1_permute](../moe_top1_permute/)).
- `GlobalTensor` reconstructed per iteration with `base + runtime_offset`.
- No `TASSIGN` aliasing tricks; no `Tile::data()` in kernel; no `*_IMPL`
  calls; no raw CCE intrinsics; no `Event<>`; no manual sync; no `TPipe` /
  `TPUSH` / `TPOP`; no double buffering; no A5-only instructions.

## Shape

- `T = 256` tokens
- `H = 64` hidden dim
- `E = 4` experts (used only by `scripts/gen_data.py` to construct the
  reference `token_to_packed`; the kernel itself does not depend on `E`)
- dtype: `float32` tokens, `int32` metadata

## How to generate data

```bash
python ./scripts/gen_data.py
```

This writes:

```text
./input/input_packed_output.bin     (T * H float32)
./input/input_token_to_packed.bin   (T      int32)
./output/golden_output.bin          (T * H float32)
```

The Python reference uses the **same recipe** the device permute kernel
implements (stable argsort by `expert_id` → `packed_tokens`, then
`packed_output = packed_tokens + 1.0`, then unpermute via
`output[t, :] = packed_output[token_to_packed[t], :]`). It prints debug
info: `expert_count`, `expert_start`, first 32 `expert_id`, first 32
`token_to_packed`, and a few output rows.

## How to build / run

```bash
bash run.sh -r npu -v Ascend910B1
```

(or `-r sim` for the simulator). Mirrors the
[add_tile_array](../add_tile_array/) /
[moe_top1_permute](../moe_top1_permute/) build harnesses; only the
executable name differs.

On success the driver prints:

```text
test data success
test success
```

## How to compare against the Python reference

The C++ driver runs `ResultCmp(golden, dev, 0.001f)` internally. For a
richer post-mortem diff (max abs error, mismatching row count, first few
differing rows side-by-side):

```bash
python ./scripts/compare_outputs.py
```

This script reads `./output/golden_output.bin` and
`./output/output_output.bin` (the bin the C++ driver wrote).

## Output meaning

`output[t, :]` is the row of `packed_output` that originally belonged to
token `t`, i.e., the inverse of the permute step. After this kernel
runs, `output` should be element-wise equal to the original
`packed_tokens` plus 1.0, but in the **original token order** (not
expert-grouped order).

For the integer-valued input recipe in `gen_data.py`, this equality is
bit-exact in IEEE-754 float32.

## Known limitations (v1)

- Fixed shape (`T = 256`, `H = 64`); no dynamic shape, no tail handling.
- Single AICORE; no multi-core split.
- `float32` tokens only; `int32` `token_to_packed` only.
- Assumes `token_to_packed` is a valid permutation of `[0, T)` — the
  contract `moe_top1_permute` provides. Garbage input (out-of-range
  indices) is undefined.
- The "expert FFN" is faked as `+1.0` — replaced by real per-expert
  compute in M3 / M4 / M5.

## Patterns reused from earlier confirmed kernels

This kernel reuses patterns already resolved by experiment (see
[docs_for_ai/assumptions_to_verify.md §11.4](../../../../docs_for_ai/assumptions_to_verify.md)
and [docs_for_ai/known_good_kernel_examples.md §A13](../../../../docs_for_ai/known_good_kernel_examples.md)):

- scalar GM read of `int32_t` metadata from kernel code,
- runtime scalar used to compute GM row offset for `GlobalTensor`,
- single static row tile reused across `T` runtime-offset iterations,
- no manual sync.

No new auto-mode capability is required for this milestone; it is a
strictly smaller subset of what `moe_top1_permute` already proved.

## Failure protocol

If it fails, report:

1. The first meaningful compiler / runtime error verbatim.
2. Whether the failure is in:
   - the scalar GM read of `token_to_packed[t]`,
   - the runtime GM-offset construction `packed_output + packed_pos * H`,
   - the `TLOAD` / `TSTORE` pair,
   - the host harness (file I/O, allocation, `aclrtMemcpy`),
   - the comparison / golden mismatch (`ResultCmp` or `compare_outputs.py`).
3. The smallest reduced reproducer or fallback.

Do not proceed to segmented identity / GEMM / FFN until this unpermute
kernel passes.

## After it passes

The next milestone is **moe_segmented_identity** (not yet implemented):

```cpp
for (e = 0; e < num_experts; ++e) {
    start = expert_start[e];
    count = expert_count[e];
    for (m0 = 0; m0 < count; m0 += TILE_M) {
        validM = min(TILE_M, count - m0);
        packed_output[start + m0 : start + m0 + validM, :]
            = packed_tokens[start + m0 : start + m0 + validM, :] + 1.0f;
    }
}
```

with `TILE_M = 128`. Do not implement that until this unpermute passes.
