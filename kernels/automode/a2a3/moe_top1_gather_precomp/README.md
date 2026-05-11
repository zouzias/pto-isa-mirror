# moe_top1_gather_precomp (Milestone 1a)

## Purpose

Host-precomputed packed gather. **This is not real MoE permute.** It is an
A3 auto-mode compiler / API verification step that retires two assumptions
the real M1b kernel will depend on:

- runtime GM scalar read of an index array from kernel code
  (`uint32_t r = packed_to_token[p];`), and
- runtime-offset GM-row `TLOAD` / `TSTORE` driven by that scalar
  (`GlobalTensor srcGlobal(tokens + r * H);`).

## Target platform

A3 only (`PTO_NPU_ARCH_A2A3`, compiled with `--cce-aicore-arch=dav-c220-vec`).

## What the kernel does

```cpp
for (p = 0; p < T; ++p) {
    src_row = packed_to_token[p];        // GM scalar read
    packed_tokens[p, :] = tokens[src_row, :];
}
```

The host pre-computes `packed_to_token[T]` by `argsort(expert_id, stable)` —
i.e., the same permutation the future M1b kernel will compute on the device.

## Auto-mode constraints honored

- Single AICORE (`<<<1, nullptr, stream>>>`); no `block_idx` work split.
- Static row tile (`Tile<Vec, T, 1, H, RowMajor, 1, H>`) declared once and
  reused across iterations; UB address pinned by the auto allocator.
- `GlobalTensor` reconstructed per iteration with `base + runtime_offset`
  (same shape as the confirmed-built [add_tile_array](../add_tile_array)
  pattern).
- No `TASSIGN` aliasing tricks; no `Tile::data()` in kernel code;
  no `*_IMPL` calls; no raw CCE intrinsics; no `Event<>`; no manual sync.

## Shape

- `T   = 256` tokens
- `H   = 64`  hidden dim
- `E   = 4`   experts (only used by `gen_data.py` to build a representative
  permutation; the kernel itself does not know about experts)
- dtype: `float32`

## Build

```bash
bash run.sh -r npu -v Ascend910B1
```

(or `-r sim` for the simulator). Mirrors the
[add_tile_array](../add_tile_array) build harness; only the executable
name differs.

## Run / compare against Python reference

`run.sh` first invokes `scripts/gen_data.py`, which writes:

- `./input/input_tokens.bin`
- `./input/input_packed_to_token.bin`
- `./output/golden_packed_tokens.bin`

The driver writes `./output/output_packed_tokens.bin` and prints
`test data success` / `test success` on match (tolerance `0.001f`, exact
under integer-valued inputs).

## Known limitations (v1)

- Single AICORE; no multi-core split.
- Fixed shape (`T=256`, `H=64`); no dynamic shape, no tail handling.
- `float32` only.
- Not real MoE — the permutation is computed on the host.
- No `expert_count` / `expert_start` / `token_to_packed` emitted; the
  kernel is purely a gather. Those outputs land in M1b.

## Assumptions this milestone verifies

- **A1**: scalar GM read (`__gm__ uint32_t *idx; uint32_t r = idx[p];`) is
  auto-mode-safe under `--cce-enable-pto-passes`.
- **A2**: a runtime scalar (`r`) is usable as a multiplier in
  `GlobalTensor srcGlobal(tokens + r * H);` constructed inside the loop.

If both pass, M1b can be implemented with confidence. If either fails,
report the first compiler/runtime error before proceeding.
