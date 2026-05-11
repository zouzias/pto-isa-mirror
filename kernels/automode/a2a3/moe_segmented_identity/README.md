# moe_segmented_identity

## Purpose

First A3 auto-mode kernel that walks **dynamic expert segments** described by
`(expert_start[e], expert_count[e])` and runs a fixed `TILE_M = 128`
microtile inside each segment. The microtile op is a simple elementwise
`dst = src + 1.0f` (`TADDS`) — this prototype proves the per-expert outer
loop + per-microtile inner loop shape that the future expert FFN
(GEMM1 → activation → GEMM2) will reuse, with no GEMM yet.

This is **not** the real expert FFN. It is the segmented-loop shape only.

## Where this sits in the MoE pipeline

After the passing forward token-movement pair
([§A13 moe_top1_permute](../moe_top1_permute/) and
[§A14 moe_top1_unpermute](../moe_top1_unpermute/)), this kernel exercises
the per-expert segmented compute that will replace the fake
`packed_output = packed_tokens + 1.0` test fixture used by the unpermute
project:

```text
moe_top1_permute       (proven)
    ↓ packed_tokens, expert_count, expert_start
moe_segmented_identity (THIS — proves the segmented loop shape)
    ↓ packed_output = packed_tokens + 1.0
moe_top1_unpermute     (proven)
    ↓ output
```

## Target platform

A3 only (`PTO_NPU_ARCH_A2A3`, compiled with `--cce-aicore-arch=dav-c220-vec`).
A3 auto mode, single AICORE.

## What the kernel does

```cpp
for (e = 0; e < kNumExperts; ++e) {
    int32_t start = expert_start[e];     // GM scalar read (PADDED)
    int32_t count = expert_count[e];     // GM scalar read (PADDED)

    for (int32_t m0 = 0; m0 < count; m0 += kTileM) {
        size_t off = (size_t(start) + size_t(m0)) * kH;
        SegGlobal srcGlobal(packed_tokens + off);
        SegGlobal dstGlobal(packed_output + off);

        TLOAD(segTile, srcGlobal);
        TADDS(segTile, segTile, 1.0f);   // in-place
        TSTORE(dstGlobal, segTile);
    }
}
```

`TADDS` is the user-facing public wrapper at
[include/pto/common/pto_instr.hpp:1517-1524](../../../../include/pto/common/pto_instr.hpp);
[tests/npu/a2a3/src/st/testcase/tadds/tadds_kernel.cpp](../../../../tests/npu/a2a3/src/st/testcase/tadds/tadds_kernel.cpp)
uses the same three-argument form and is in `ALL_TESTCASES` (auto-mode
build list).

## Tail policy (v1)

Host pads every expert segment length up to a multiple of `TILE_M`. Padded
rows are initialised to `0.0`, and the golden adds `1.0` to them too — so
the kernel processes padded rows identically to real rows. This sidesteps
`SetValidRow` / partial-tile stores entirely (still an open Assumption,
see [docs_for_ai/tile_type_reference.md §6 / §11 item 12](../../../../docs_for_ai/tile_type_reference.md)
and [docs_for_ai/assumptions_to_verify.md §1.5](../../../../docs_for_ai/assumptions_to_verify.md)).

A follow-up `moe_segmented_identity_tail_validM` milestone will retire that
assumption after this one passes.

## Inputs / outputs

**Inputs (GM)**
- `packed_tokens [T_PADDED, H]` — `float32` (real rows first per expert,
  remainder zero-padded to a multiple of `TILE_M`)
- `expert_count [E]` — `int32` (**padded** counts; multiples of `TILE_M`)
- `expert_start [E]` — `int32` (**padded** starts; prefix sum of padded counts)

**Output (GM)**
- `packed_output [T_PADDED, H]` — `float32` (equal to `packed_tokens + 1.0`
  for every row, padded rows included)

`T_PADDED` is determined at runtime by `scripts/gen_data.py` and persisted
to `./output/t_padded.txt`; `main.cpp` reads that file to size the device
buffers.

## Shape

- `kT = 256` real tokens (host-only)
- `kH = 64` hidden dim
- `kE = 4` experts
- `kTileM = 128` microtile rows
- dtype: `float32` data, `int32` metadata

The deliberately **skewed** expert distribution in `scripts/gen_data.py`
(`p = [0.55, 0.20, 0.15, 0.10]`) is chosen so at least one expert ends up
with padded count > `TILE_M` — i.e., the inner `m0` loop runs more than
once for that expert. This exercises nested-loop control flow, not just
the outer.

## Auto-mode constraints honored

- Single AICORE (`<<<1, nullptr, stream>>>`); no `block_idx` work split.
- Static segment tile
  `Tile<TileType::Vec, float, kTileM, kH, BLayout::RowMajor, kTileM, kH>`
  declared once and reused across all inner iterations; UB address pinned
  by the auto allocator. Size: 128 × 64 × 4 B = 32 KB (2× the
  [add_tile_array](../add_tile_array/) tile; comfortably below UB).
- `GlobalTensor` reconstructed per inner iter with `base + runtime offset`.
- `TADDS` used in-place (`dst == src`): one tile, one liveness range across
  `TLOAD` → `TADDS` → `TSTORE`. No redundant `TLOAD` on dst.
- No `TASSIGN` aliasing tricks; no `Tile::data()` in kernel; no `*_IMPL`
  calls; no raw CCE intrinsics; no `Event<>`; no manual sync; no `TPipe` /
  `TPUSH` / `TPOP`; no double buffering; no `SetValidRow` / `SetValidShape`;
  no partial-tile stores; no A5-only instructions.

## Patterns reused from earlier confirmed kernels

This kernel reuses patterns already resolved by experiment (see
[docs_for_ai/assumptions_to_verify.md §11.1 / §11.4 / §11.5](../../../../docs_for_ai/assumptions_to_verify.md)
and [docs_for_ai/known_good_kernel_examples.md §A11 / §A13 / §A14](../../../../docs_for_ai/known_good_kernel_examples.md)):

- scalar GM read of `int32_t` metadata from kernel code,
- runtime scalar used to compute GM row offset for `GlobalTensor`,
- single reused static Vec tile across many runtime-offset iterations,
- `TLOAD` → elementwise op → `TSTORE` with no manual sync.

## NEW: what this kernel is the first to validate

If it passes, these become Resolved-by-experiment for the tested shape:

- **Nested loop**: outer over experts (`kNumExperts = 4`), inner over
  microtiles within an expert's padded segment, with both bounds coming
  from scalar GM reads of `int32_t` metadata.
- **`128 × 64` float static Vec tile** (32 KB UB allocation, vs
  `add_tile_array`'s 16 KB and `moe_top1_permute`'s 256 B-per-tile).
- **`TADDS` in-place form** (`dst == src`) from auto-mode kernel code.
- **Per-expert padded segment processing** as a step toward the real
  expert FFN outer schedule.

## How to generate data

```bash
python ./scripts/gen_data.py
```

Writes:

```text
./input/input_packed_tokens.bin       (T_PADDED * H float32)
./input/input_expert_count.bin        (E         int32)  PADDED counts
./input/input_expert_start.bin        (E         int32)  PADDED starts
./output/golden_packed_output.bin     (T_PADDED * H float32)
./output/t_padded.txt                 (int)  consumed by main.cpp
```

And prints debug info: real vs padded expert counts, padded starts,
`T_PADDED`, per-expert inner-loop iteration count, the boundary row
between real and padded data for expert 0.

## How to build / run

```bash
bash run.sh -r npu -v Ascend910B1
```

(or `-r sim` for the simulator). Mirrors the
[moe_top1_permute](../moe_top1_permute/) build harness; only the
executable name differs.

On success the driver prints:

```text
test data success
test success
```

## How to compare against the Python reference

The C++ driver runs `ResultCmp(golden, dev, 0.001f)` internally. For a
richer post-mortem diff (max abs error, mismatching row count, segment
context for the first few differing rows — including whether each
mismatch falls in a padded or real row):

```bash
python ./scripts/compare_outputs.py
```

## Output meaning

`packed_output[row, :]` is `packed_tokens[row, :] + 1.0` for every row,
real or padded. Under the integer-valued token recipe this equality is
bit-exact in IEEE-754 float32.

## Known limitations (v1) — what this kernel does NOT prove

Even if it passes, do not claim any of these as resolved:

- Dynamic tail handling with `validM` / `SetValidRow` / `SetValidShape`.
- Partial-tile stores.
- GEMM / cube path / `TMATMUL`.
- Real expert FFN (GEMM1 → activation → GEMM2).
- Router / argmax / top-K selection.
- `topK > 1` (weighted combine).
- Per-expert capacity / drop policy / fallback expert.
- Multi-core (`block_idx`) work split.
- `fp16` / `bfloat16` data; non-multiple-of-block `H`.
- Larger `kNumExperts` (stack-array sizing implications).
- Backward pass.

## Failure protocol

If it fails, report:

1. The first meaningful compiler / runtime error verbatim.
2. Whether the failure is in:
   - scalar GM read of `expert_start` / `expert_count`,
   - the nested `e → m0` loop control flow,
   - the `128 × 64` Vec tile allocation in UB,
   - the `TADDS` elementwise add instruction (in-place form),
   - `TLOAD` / `TSTORE` with the runtime segment offset,
   - the host harness (file I/O, `t_padded.txt` parsing, allocation),
   - the comparison / golden mismatch (`ResultCmp` or `compare_outputs.py`).
3. The smallest reduced reproducer or fallback (e.g., a 10-line kernel
   that only does the outer-only loop, or a hard-coded single-expert
   single-tile variant).

Do not silently fall back to a non-segmented shape and call this milestone
done.

## After it passes

The next milestone is either:

- **`moe_segmented_identity_tail_validM`** — same shape but no host
  padding; use a dynamic-valid-region tile and `SetValidRow` on the tail
  iteration. This retires the open `SetValidRow` Assumption.
- **`moe_segmented_gemm_one_layer`** — replace the in-place `TADDS` with a
  single GEMM tile per inner step (`A: [TILE_M, H] × B: [H, O] → C:
  [TILE_M, O]`). Switches the build target to `dav-c220-cube` and uses the
  `TileLeft` / `TileRight` / `TileAcc` skeleton from
  [tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp](../../../../tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp).

Do not implement either of those until this kernel passes.
