# moe_segmented_gemm_one_layer

## Purpose

First **cube/GEMM** milestone inside the working MoE expert-segment loop.
Reuses the host-padded segment layout from the passing
[moe_segmented_identity](../moe_segmented_identity/) but replaces the
elementwise `+1.0` with one expert-specific matrix multiply per inner
microtile iteration:

```text
packed_output[row : row + TILE_M, 0:O]
  = packed_tokens[row : row + TILE_M, 0:H]  @  expert_weight[e, 0:H, 0:O]
```

This is **not** the full MoE FFN. It is GEMM1 only — no bias, no
activation, no GEMM2, no unpermute inside this kernel.

## Where this sits in the MoE pipeline

```text
moe_top1_permute             (proven, §A13)
    ↓
moe_top1_unpermute           (proven, §A14)
moe_segmented_identity       (proven, §A15) — fake +1.0 expert compute
moe_segmented_gemm_one_layer (THIS) — one expert-specific GEMM tile
    ↓ (eventually)
moe_segmented_ffn            (future) — GEMM1 → activation → GEMM2
```

## Target platform

A3 only (`PTO_NPU_ARCH_A2A3`, compiled with
`--cce-aicore-arch=dav-c220-cube` — note: **cube** arch, not vec).
A3 auto mode, single AICORE.

## Reference copied

The cube tile aliases, GlobalTensor shape/stride, and
TLOAD → TLOAD → TMOV → TMOV → TMATMUL → TSTORE sequence are lifted
verbatim from `RunTMATMUL<float, half, half, float, validM, validK,
validN, false>` in
[tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp](../../../../tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp)
(the `isBias = false` branch, matching `LaunchTMATMUL<1>`'s
instantiation). That testcase is in `ALL_TESTCASES` — it builds and
runs in auto mode.

The only structural delta vs the reference is the outer `(expert,
microtile)` loop wrapping the body, lifted from the passing §A15
`moe_segmented_identity` kernel.

## What the kernel does

```cpp
// Tile aliases (exact copy from tmatmul reference)
using TileMatAData = Tile<TileType::Mat, half,  M, K, BLayout::ColMajor,
                          kTileM, kH, SLayout::RowMajor, 512>;
using TileMatBData = Tile<TileType::Mat, half,  K, N, BLayout::ColMajor,
                          kH,     kO, SLayout::RowMajor, 512>;
using LeftTile  = TileLeft <half,  M, K, kTileM, kH>;     // L0A
using RightTile = TileRight<half,  K, N, kH,     kO>;     // L0B
using AccTile   = TileAcc  <float, M, N, kTileM, kO>;     // L0C

TileMatAData aMatTile;
TileMatBData bMatTile;
LeftTile     aTile;
RightTile    bTile;
AccTile      cTile;

for (e = 0; e < kNumExperts; ++e) {
    int32_t start = expert_start[e];
    int32_t count = expert_count[e];

    GlobalDataB bGlobal(expert_weight + e * (kH * kO));

    for (m0 = 0; m0 < count; m0 += kTileM) {
        size_t row  = size_t(start) + size_t(m0);
        GlobalDataA aGlobal(packed_tokens + row * kH);
        GlobalDataC cGlobal(packed_output + row * kO);

        TLOAD(aMatTile, aGlobal);   // GM -> L1
        TLOAD(bMatTile, bGlobal);   // GM -> L1
        TMOV (aTile, aMatTile);     // L1 -> L0A
        TMOV (bTile, bMatTile);     // L1 -> L0B
        TMATMUL(cTile, aTile, bTile);
        TSTORE(cGlobal, cTile);     // L0C -> GM
    }
}
```

## Chosen dtype and why

`float (out) × half × half  →  float` accumulator, no bias.

This matches `RunTMATMUL<float, half, half, float, ..., false>` in the
reference (the `LaunchTMATMUL<1>` instantiation) — the canonical
A3 auto-mode-eligible cube combo. The reference does NOT show a pure
`<float, float, float, float, false>` non-TF32 path; FP32×FP32 GEMM in
that file is only present via `RunTMATMUL_TF32` (`LaunchTMATMUL<7>`,
`<8>`) which uses TF32 mode and explicit `RoundMode`. To keep this
milestone narrow, we stick with the FP16-input cube combo.

## Chosen tile shapes and why

```text
kTileM = 128     (M)
kH     = 64      (K)
kO     = 64      (N)
```

- M = 128 is the working `TILE_M` from §A15.
- K = N = 64 matches the existing `H = 64` token width used by every
  earlier MoE milestone in this stack.
- For `half` inputs, the reference's alignment rule is
  `blockAlign = C0_SIZE_BYTE / sizeof(half) = 16`. Required:
  `validM` aligned to 16, `validK / validN` aligned to `blockAlign`.
  `(128, 64, 64)` are already aligned; no padding inside the tile.
- L0/L1 footprint per inner iter:
  - `aMatTile` (L1, half): 128 × 64 × 2 B = 16 KB
  - `bMatTile` (L1, half): 64 × 64 × 2 B = 8 KB
  - `aTile`    (L0A, half): 16 KB
  - `bTile`    (L0B, half): 8 KB
  - `cTile`    (L0C, float): 128 × 64 × 4 B = 32 KB
  Comfortably within A3's L0A / L0B / L0C budgets per general
  knowledge; if the auto allocator pushes back, the fallback is to
  reduce `kTileM` to 64.

## Inputs / outputs

**Inputs (GM)**
- `packed_tokens [T_PADDED, H]` — `float16` (real rows first per expert,
  zero-padded tail per expert)
- `expert_count [E]` — `int32` (PADDED counts; multiples of `TILE_M`)
- `expert_start [E]` — `int32` (PADDED starts; prefix sum of padded counts)
- `expert_weight [E, H, O]` — `float16` (per-expert weights, flat layout)

**Output (GM)**
- `packed_output [T_PADDED, O]` — `float32` (cube FP32 accumulator)

`T_PADDED` is determined at runtime by `scripts/gen_data.py` and
persisted to `./output/t_padded.txt`; `main.cpp` reads that file to
size the device buffers.

## Tail policy

Same as §A15: host pads every expert segment to a multiple of
`TILE_M = 128`. Padded rows in `packed_tokens` are zero; their GEMM
output is therefore exactly zero (no bias). The kernel processes
padded rows identically to real rows; no `SetValidRow` / partial-tile
stores.

## How to generate data

```bash
python ./scripts/gen_data.py
```

Writes:

```text
./input/input_packed_tokens.bin       (T_PADDED * H float16)
./input/input_expert_count.bin        (E         int32)
./input/input_expert_start.bin        (E         int32)
./input/input_expert_weight.bin       (E * H * O float16)
./output/golden_packed_output.bin     (T_PADDED * O float32)
./output/t_padded.txt                 (int)
./output/expert_count_real.bin        (E int32, debug only)
```

The Python script uses small integer-valued inputs (in `[-4, 4]`) cast
to `float16`, so products fit exactly in FP16 and the FP32 cube
accumulation is bit-exact. Skewed expert distribution
(`p = [0.55, 0.20, 0.15, 0.10]`) so at least one expert exceeds
`TILE_M` and the inner `m0` loop runs `> 1` iter for that expert.

## How to build / run

```bash
bash run.sh -r npu -v Ascend910B1
```

(or `-r sim` for the simulator). Mirrors the
[moe_segmented_identity](../moe_segmented_identity/) build harness;
the **only** delta is that the kernel target uses
`--cce-aicore-arch=dav-c220-cube` (cube arch) — the function
`pto_example_cube_auto` in `CMakeLists.txt` replaces the vec form.

On success the driver prints:

```text
test data success
test success
```

## How to compare against the Python reference

The C++ driver runs `ResultCmp(golden, dev, 0.001f)` internally. For a
richer post-mortem diff (max abs error, mismatching row count, first
mismatch's `{flat_index, row, col, expert, segment, tile_m0, real-vs-pad}`
label, per-expert per-tile mismatch counts):

```bash
python ./scripts/compare_outputs.py
```

## Auto-mode constraints honored

- Single AICORE; no `block_idx` work split.
- Static tile shapes throughout the inner loop; no `SetValidRow` /
  `SetValidShape`; no partial-tile stores.
- Each tile declared once outside both loops; auto allocator pins L1
  and L0 addresses.
- No `TASSIGN` literal addresses (the reference uses them but they are
  no-ops in auto mode; we omit them for clarity).
- No `#ifndef __PTO_AUTO__ set_flag / wait_flag / TFILLPAD` blocks
  (auto-sync inserts MTE2 → MTE1 → M → FIX fences).
- No `Tile::data()` in kernel; no `*_IMPL` calls; no raw CCE
  intrinsics; no `Event<>`; no `TPipe` / `TPUSH` / `TPOP`; no double
  buffering; no manual L0 buffer macros; no A5-only instructions.
- Host boundary uses `uint8_t*` for typed buffers (mirrors §A12 topk's
  pattern that resolved compile-error E11; metadata still uses bare
  `int32_t*`).

## Patterns reused from earlier confirmed kernels

- scalar GM read of `int32_t` metadata                       — §11.4 / §11.6
- runtime scalar GM row offset for `GlobalTensor`            — §11.4 / §11.5 / §11.6
- nested (expert, microtile) loop                            — §11.6
- host-padded expert segment layout                          — §11.6
- `uint8_t*` host boundary + in-kernel `__gm__ T*` cast      — §A12 topk

## NEW: what this kernel is the first to validate

If it passes, these become Resolved-by-experiment for the tested shape:

- **Cube path (`TMATMUL`) inside the auto-mode segmented loop** —
  TileMatA/B (L1) + TileLeft/Right (L0A/L0B) + TileAcc (L0C) all
  declared once outside the loops and reused across inner iters; no
  manual sync; auto allocator pins all five tile addresses.
- **FP16 × FP16 → FP32 cube combo in auto mode** at the chosen
  `(M, K, N) = (128, 64, 64)` static tile shape.
- **Runtime data-dependent GM offset for cube inputs** —
  `aGlobal = packed_tokens + (start + m0) * H` per inner iter,
  `bGlobal = expert_weight + e * (H * O)` per outer iter. Per-expert
  weight selection by `e` works through GM-pointer arithmetic, no UB
  copy.
- **Per-expert weight reload per outer iter** — `bMatTile` and `bTile`
  are reused across inner iters within an expert but reloaded from a
  different GM region for each new expert.

## What this prototype does NOT prove

Even if it passes:

- Full MoE FFN; GEMM1 → activation → GEMM2 fused into one kernel.
- Dynamic tail handling with `validM` / `SetValidRow` / `SetValidShape`.
- TF32 path; FP32 × FP32 GEMM (auto-mode A3 cube only supports it via
  TF32 mode in the reference, not validated here).
- `int8` quantised path; bias path; `bfloat16` inputs.
- Larger or smaller `kTileM`, `H`, `O`, `kE`.
- Multi-core (`block_idx`) split.
- `topK > 1`; weighted combine; router; capacity / drop policy;
  backward pass.
- Performance.

## Failure protocol

If it fails, report:

1. The first meaningful compiler / runtime error verbatim.
2. Whether the failure is likely:
   - `TMATMUL` tile shape / layout mismatch (e.g.,
     `ColMajor` / `RowMajor` mismatch, `SFractalSize = 512` wrong for
     this dtype),
   - `TLOAD` / `TSTORE` layout mismatch (the
     `GlobalTensor<..., Shape, Stride>` slice arithmetic),
   - `expert_weight` GM offset arithmetic
     (`e * kH * kO`),
   - the nested loop + cube op interaction (auto-sync not inserting
     correct fences when both A and B reload across iters),
   - dtype mismatch at the host boundary (uint8_t cast,
     `aclFloat16` vs `half` ABI),
   - host harness / golden mismatch,
   - L1 / L0 budget exhaustion (the auto allocator may report
     insufficient resources for the chosen tile shape).
3. The smallest reduced reproducer or fallback.

**Prepared fallbacks (do NOT apply until first failure is known)**:

- Reduce `kTileM` from 128 to 64 → cuts `aMatTile` / `aTile` from 16 KB
  to 8 KB, `cTile` from 32 KB to 16 KB. Update consistently in
  `gen_data.py`, the kernel, this README, and `compare_outputs.py`.
- Reduce `kO` from 64 to 32 → cuts `bMatTile` and `cTile` width.
- If `half` ABI fails at the host boundary, swap `uint8_t*` for
  `aclFloat16*` (cf. `tadds_kernel.cpp`'s shape).

## After it passes

The next milestone is **moe_segmented_ffn** (full GEMM1 → activation →
GEMM2 per expert segment tile). Do not implement until this kernel
passes.
