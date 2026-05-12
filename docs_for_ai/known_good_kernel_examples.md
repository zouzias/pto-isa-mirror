# Known-Good Kernel Examples (auto-mode A3/A5)

Source-grounded shortlist of repo files useful when generating, reviewing, or
debugging auto-mode kernels for A3 and A5. Each entry is anchored to a path
inside this repo; nothing here has been compiled. Companion to
[repo_kernel_map.md](repo_kernel_map.md).

How to read each entry:
1. **File**
2. **Why it is useful**
3. **Pattern demonstrated**
4. **Auto-mode compatibility** — Yes / Mostly / Manual-only / Problematic. Each label is grounded in source evidence.
5. **Copy these parts**
6. **Do NOT copy these parts**
7. **Confidence** — High / Medium / Low

Conventions used in source that drive these labels (Known unless noted):
- Auto mode is selected by `__PTO_AUTO__` and `--cce-enable-pto-passes -O2`. (Known: [docs/auto_mode/Auto_Mode_Overview.md](../docs/auto_mode/Auto_Mode_Overview.md), [demos/auto_mode/baseline/add/CMakeLists.txt:65-67](../demos/auto_mode/baseline/add/CMakeLists.txt#L65-L67).)
- In auto mode, `TASSIGN`, `TSYNC`, and `Event<>` are no-ops. (Known: [docs/auto_mode/Auto_Mode_Overview.md](../docs/auto_mode/Auto_Mode_Overview.md), [include/pto/npu/a2a3/TSync.hpp](../include/pto/npu/a2a3/TSync.hpp).)
- Auto-mode rules: prefer `TRESHAPE`/`TSUBVIEW` over `TASSIGN` for aliasing; do not call CCE intrinsics from kernel code; do not `TLOAD` a destination-only tile; do not use double-buffering today; `set_flag`/`wait_flag` need `#ifndef __PTO_AUTO__` guards in kernel code (or use `PtoSetWaitFlag`/`TSYNC` which are auto-safe). (Known: [docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md).)

---

## Group A — Tier 1 references (copy-quality)

These compile under both `AUTO_MODE=ON` and `AUTO_MODE=OFF` per the test
harness (Known: enumerated in `ALL_TESTCASES` of
[tests/npu/a2a3/src/st/testcase/CMakeLists.txt](../tests/npu/a2a3/src/st/testcase/CMakeLists.txt) and
[tests/npu/a5/src/st/testcase/CMakeLists.txt](../tests/npu/a5/src/st/testcase/CMakeLists.txt)).

### A1. Auto-mode add demo

1. **File** — [demos/auto_mode/baseline/add/csrc/kernel/add_custom.cpp](../demos/auto_mode/baseline/add/csrc/kernel/add_custom.cpp)
2. **Why** — The single end-to-end auto-mode kernel committed to the repo. Confirmed compiled with `--cce-enable-pto-passes -O2` per [demos/auto_mode/baseline/add/CMakeLists.txt:65-67](../demos/auto_mode/baseline/add/CMakeLists.txt#L65-L67).
3. **Pattern** — Multi-core elementwise `TLOAD → TADD → TSTORE`. Block tiling via `block_idx`; no `TASSIGN`, no `set_flag`/`wait_flag`, no `Event<>`.
4. **Auto-mode compatibility** — Yes (Known).
5. **Copy** — `__global__ AICORE` entry signature; templated `AICORE` helper; `block_idx`-based GM offset; `Tile<TileType::Vec, T, R, C, BLayout::RowMajor, -1, -1>` declaration with **dynamic valid region**; bare `TLOAD`/`TADD`/`TSTORE` sequence with no manual sync.
6. **Do not copy** — `set_mask_norm()` / `set_vector_mask(-1, -1)` at function top: those are CCE intrinsics. They appear here because this is the entry kernel, but moving them into a tile-shaped helper or into a refactored auto-mode kernel may conflict with [docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md §3.2](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md). The `#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)` guard is **A3-only** (Inferred from [include/pto/common/arch_macro.hpp:14-15](../include/pto/common/arch_macro.hpp#L14-L15) — A2/A3 → `__NPU_ARCH__ == 2201` and the cube/vec macros). Replace it for A5.
7. **Confidence** — High.

### A2. TADD ST (A3) — minimal dual-mode elementwise

1. **File** — [tests/npu/a2a3/src/st/testcase/tadd/tadd_kernel.cpp](../tests/npu/a2a3/src/st/testcase/tadd/tadd_kernel.cpp)
2. **Why** — Smallest dual-mode kernel that passes auto-mode build (`tadd` is in `ALL_TESTCASES`). Mirrors the canonical [docs/auto_mode/Examples.md](../docs/auto_mode/Examples.md) `TADD` example but adds GM stride and template specializations.
3. **Pattern** — Static-shape elementwise: define `Shape`/`Stride`/`GlobalTensor`, declare three vec tiles, `TASSIGN` (no-op in auto), `TLOAD`/`TLOAD`/`TADD`/`TSTORE` chained through `Event<>` (no-op in auto).
4. **Auto-mode compatibility** — Yes (Known: in `ALL_TESTCASES`; relies on `TASSIGN`/`Event<>` being no-ops per overview doc).
5. **Copy** — Templated `Tile<TileType::Vec, T, R, C, BLayout::RowMajor, -1, -1>` declaration with runtime valid region from constructor args; the `Event<Op::TLOAD, Op::TADD>` typedef pattern (it is auto-mode-safe because `Event<>` is dropped); the launcher `LaunchTAdd` template with explicit instantiations; the `aclFloat16 → half` casting wrapper.
6. **Do not copy** — The `TASSIGN(src0Tile, 0x0)` style with hard-coded addresses if you intend to add aliasing in a derived kernel: in auto mode, two tiles given the same `TASSIGN` address do **not** alias (that is a manual-mode trick). Use `TRESHAPE`/`TSUBVIEW` (Known: [docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md §2.1, §2.2](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md)).
7. **Confidence** — High.

### A3. TPOW ST (A3) — `TASSIGN<Addr>` overload + Event-chained pipeline

1. **File** — [tests/npu/a2a3/src/st/testcase/tpow/tpow_kernel.cpp](../tests/npu/a2a3/src/st/testcase/tpow/tpow_kernel.cpp)
2. **Why** — Demonstrates the **compile-time `TASSIGN<Addr>(tile)`** overload (defined at [include/pto/common/pto_instr.hpp](../include/pto/common/pto_instr.hpp), the `std::enable_if_t<is_tile_data_v<T> || is_conv_tile_v<T>>` template). Auto-compatible elementwise op with a tmp tile.
3. **Pattern** — `TASSIGN<0x0>(baseTile)` style, four-tile static layout (`base`, `exp`, `dst`, `tmp`), `Event<Op::TLOAD, Op::TPOW>` chaining.
4. **Auto-mode compatibility** — Yes (Known: `tpow` is in `ALL_TESTCASES`).
5. **Copy** — `TASSIGN<offset>(tile)` static-address overload; explicit `tmpTile` for ops that require scratch; per-type `LaunchTPow` instantiations covering `float`, `int8/16/32`, `uint8/16/32`. `TPOW` is the canonical "auto-compatible op with a tmp tile argument" example.
6. **Do not copy** — Hard-coded offsets relative to `TileData::Numel * sizeof(T)`: in auto mode the compiler picks addresses, so this expression is irrelevant. Keep the `TileData` declarations and let `TASSIGN` become a no-op.
7. **Confidence** — High.

### A4. TROWSUM ST (A3) — reduction with explicit `TRESHAPE`

1. **File** — [tests/npu/a2a3/src/st/testcase/trowsum/trowsum_kernel.cpp](../tests/npu/a2a3/src/st/testcase/trowsum/trowsum_kernel.cpp)
2. **Why** — Uses correct `#ifndef __PTO_AUTO__` guards around `set_flag`/`wait_flag`. Includes a `TRESHAPE` example to express that two tiles share an address ([line 87](../tests/npu/a2a3/src/st/testcase/trowsum/trowsum_kernel.cpp#L87): `TRESHAPE(dstTileND, dstTile)`), which is the auto-mode-safe alias pattern (Known: [docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md §2.4](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md)).
3. **Pattern** — Row reduction `TROWSUM(dstTile, srcTile, tmpTile)`; **DN-layout destination** variant via `TRESHAPE` to a row-major view before `TSTORE`. Multiple template specializations + `extern "C" __global__ AICORE` launcher per case.
4. **Auto-mode compatibility** — Yes (Known: `trowsum` in `ALL_TESTCASES`; manual-only sync is properly guarded).
5. **Copy** — The `#ifndef __PTO_AUTO__ set_flag(...) wait_flag(...) #endif` skeleton; the `TRESHAPE(dstTileND, dstTile)` aliasing for layout swap right before `TSTORE`; the `tmpTile` argument to `TROWSUM` (reductions take scratch); `TileShape2D` / `BaseShape2D` / `GlobalTensor<..., Layout::ND>` style in `runTRowSumDNDst`.
6. **Do not copy** — `TASSIGN(tmpTile, row * srcCol * sizeof(T))` and the other byte-offset arithmetic — in auto mode it does nothing; keep the `tmpTile` declaration.
7. **Confidence** — High.

### A5. TAXPY ST (A3) — read-modify-write destination

1. **File** — [tests/npu/a2a3/src/st/testcase/taxpy/taxpy_kernel.cpp](../tests/npu/a2a3/src/st/testcase/taxpy/taxpy_kernel.cpp)
2. **Why** — Shows the **legitimate** dst-side `TLOAD` pattern: `TAXPY` does `dst += alpha*src`, so `TLOAD(dstTile, dstGlobal)` is **not** redundant. Useful contrast with the rule in [docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md §3.1](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md), which targets dst tiles "with no copied-in data from GM".
3. **Pattern** — Two-tile RMW elementwise; mixed-precision overload (`__gm__ U` source, `__gm__ T` destination); accumulator-style elementwise.
4. **Auto-mode compatibility** — Yes (Known: `taxpy` in `ALL_TESTCASES`).
5. **Copy** — The mixed-type template signature `<typename T, typename U, ...>`; the launcher pattern `<half, ...>` cast for `aclFloat16`.
6. **Do not copy** — Do not generalize the dst-side `TLOAD` to ops where the dst is write-only (e.g., `TADD`, `TPOW`); that would trigger the §3.1 rule. The hard-coded `0x0` / `0x10000` addresses are inert in auto mode.
7. **Confidence** — High.

### A6. TMATMUL ST (A3) — Cube tiles + Split-K loop

1. **File** — [tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp](../tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp)
2. **Why** — Closest dual-mode (manual + auto) GEMM-shaped reference for A3. Both a single-tile `RunTMATMUL` and a `RunTMATMULSplitK` loop variant ([line 217 onward](../tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp#L217)). Demonstrates `TileLeft`/`TileRight`/`TileAcc` aliases, `TileMatA/B` with `SLayout::RowMajor` fractal, `TMOV` between L1 and L0, optional bias path via `if constexpr (isBias)`.
3. **Pattern** — `TLOAD → TFILLPAD (manual only) → TMOV → TMATMUL/TMATMUL_BIAS/TMATMUL_ACC → TSTORE`, each stage protected by `#ifndef __PTO_AUTO__` set_flag/wait_flag fences.
4. **Auto-mode compatibility** — Yes (Known: `tmatmul` in `ALL_TESTCASES`). The Split-K loop’s `if (i == 0) TMATMUL else TMATMUL_ACC` is exactly the **statically peelable first-iteration guard** recommended in [docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md §1.1](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md).
5. **Copy** — All `using TileLeft = TileLeft<U, M, K, validM, validK>` style aliases; the `CeilAlign<int>(validX, …)` constexpr; the if-constexpr bias branch; the Split-K loop with `i == 0 ? TMATMUL : TMATMUL_ACC`. The whole skeleton transfers cleanly to auto mode.
6. **Do not copy** — `TFILLPAD(aMatTile, aMatTile)` is wrapped in `#ifndef __PTO_AUTO__` ([lines 80-83](../tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp#L80-L83), [274-275](../tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp#L274-L275)) — i.e., the manual kernel relies on it but the auto kernel **omits** it. Inferred reason: auto mode does L1 padding differently. Do not unconditionally `TFILLPAD` in a new auto kernel.
7. **Confidence** — High.

### A7. TMATMUL ST (A5) — A5 mirror of A6

1. **File** — [tests/npu/a5/src/st/testcase/tmatmul/tmatmul_kernel.cpp](../tests/npu/a5/src/st/testcase/tmatmul/tmatmul_kernel.cpp)
2. **Why** — A5 dual-mode matmul. Same skeleton as A6, but with A5 alignment (`blockAlign = (sizeof(AType) == 1) ? 32 : 16`) and `BlockCubeMN = 16`.
3. **Pattern** — Same as A6.
4. **Auto-mode compatibility** — Yes (Known: `tmatmul` in A5 `ALL_TESTCASES`).
5. **Copy** — A5-specific `blockAlign` rule for INT8/FP4/FP8 vs FP16/BF16; the GlobalTensor stride/shape templates.
6. **Do not copy** — Hard-coded UB offsets `0x0/0x10000/0x20000`.
7. **Confidence** — High.

### A8. TMATMUL_MX ST (A5) — MX FP4/FP8 GEMM

1. **File** — [tests/npu/a5/src/st/testcase/tmatmul_mx/tmatmul_mx_kernel.cpp](../tests/npu/a5/src/st/testcase/tmatmul_mx/tmatmul_mx_kernel.cpp)
2. **Why** — A5-only MX (microscaled) GEMM dual-mode reference. Builds in auto mode (Known: `tmatmul_mx` in [tests/npu/a5/src/st/testcase/CMakeLists.txt:110](../tests/npu/a5/src/st/testcase/CMakeLists.txt#L110)).
3. **Pattern** — Same `TileLeft/TileRight/TileAcc` pipeline plus MX scale tiles. Required when targeting `float4_e1m2x2_t` / `float4_e2m1x2_t` per [include/pto/npu/a5/datatype.hpp](../include/pto/npu/a5/datatype.hpp).
4. **Auto-mode compatibility** — Yes (Known).
5. **Copy** — MX scale tile types; A5 datatype enums.
6. **Do not copy** — Anything specific to manual-mode L0 tiling that the file may inherit; verify by reading the file before reusing.
7. **Confidence** — Medium (file not yet read end-to-end).

### A9. Auto-mode `TMATMUL` example (docs)

1. **File** — [docs/auto_mode/Examples.md](../docs/auto_mode/Examples.md)
2. **Why** — Side-by-side auto vs manual `TADD` and `TMATMUL` snippets authored as the canonical "what an auto-mode kernel looks like" reference.
3. **Pattern** — Both elementwise and GEMM minimums; explicit demonstration that auto-mode drops `TASSIGN` and `Event<>`.
4. **Auto-mode compatibility** — Yes (the doc is the source).
5. **Copy** — Function signature shape (`__global__ AICORE void run...(__gm__ T __out__ *out, __gm__ T __in__ *src, ...)`); the `using DynShapeDim5 = Shape<1, 1, 1, R, C>` 5-D shape style.
6. **Do not copy** — Manual-mode side of the diff (the `TASSIGN(...)` calls and `Event<...> e0; e0 = TLOAD(...)` chains).
7. **Confidence** — High.

### A10. Dual-mode aliasing recipe — `TASSIGN(...) + TRESHAPE(other, base)` (PR-852, NOT yet merged)

1. **File** — [tests/npu/a2a3/src/st/testcase/tcvt/tcvt_kernel.cpp](../tests/npu/a2a3/src/st/testcase/tcvt/tcvt_kernel.cpp) **after PR-852 lands**. The current source still has the bug pattern; see [external_context/pr_852_notes.md §T3](external_context/pr_852_notes.md) for the full diff.
2. **Why** — Canonical answer to [auto_mode_bad_patterns.md §1.1](auto_mode_bad_patterns.md): how to write a kernel that aliases two tiles correctly in **both** modes. Manual mode honors the `TASSIGN(a, X); TASSIGN(b, X);` pair; auto mode no-ops both `TASSIGN`s but honors the explicit `TRESHAPE(b, a)` hint.
3. **Pattern** —
   ```cpp
   TASSIGN(srcTileFull, 0x0  + 0x400 * block_idx);
   TASSIGN(dstTileFull, 0x20000 + 0x400 * block_idx);
   TASSIGN(srcTile,     0x0  + 0x400 * block_idx);  // aliases srcTileFull in manual
   TASSIGN(dstTile,     0x20000 + 0x400 * block_idx); // aliases dstTileFull in manual
   TRESHAPE(srcTile, srcTileFull);                  // aliases in auto mode
   TRESHAPE(dstTile, dstTileFull);                  // aliases in auto mode
   ```
   Place the `TRESHAPE` calls **immediately after** the `TASSIGN` block (per kernel rules §2.4: a tile cannot be the destination of multiple `TRESHAPE`/`TSUBVIEW`, so do it once at declaration).
4. **Auto-mode compatibility** — Yes (post-merge). The same shape lands in five `runTCVT*` kernels in PR-852 across `runTCVT`, `runTCVT_fp16_to_s4`, `runTCVT_s4_to_fp16`, `runTCVTNonSatTorch`.
5. **Copy** — The `TASSIGN(...) + TRESHAPE(...)` pair pattern; the order (TASSIGN block first, then TRESHAPE block); use for any layout-swap aliasing (e.g., `srcS4Tile` aliased onto `srcBytesTile` for in-place type re-views).
6. **Do not copy** — Do not omit either side. Auto-only kernels can drop the `TASSIGN`s; cross-mode kernels need both. Do not move `TRESHAPE` into a loop — see kernel rules §2.4.
7. **Confidence** — High (the recipe is consistent with the existing `TQuant.hpp` library pattern at lines 108-114). Low for "this is in the current branch" (it is not; PR-852 not merged).

### A11. add_tile_array — first confirmed-built auto-mode kernel with an in-kernel serial loop (A3)

1. **File** — [kernels/automode/a2a3/add_tile_array/add_tile_array_kernel.cpp](../kernels/automode/a2a3/add_tile_array/add_tile_array_kernel.cpp)
2. **Why** — first in-tree auto-mode kernel project that **(a) uses a serial in-kernel `for` loop over chunk indices**, and **(b) has been confirmed to build and produce numerically-correct output** by the user under `bash run.sh -r npu -v Ascend910B1` on A3. The closest references (A1, A2) only do a single tile per AICORE; this entry validates that the same auto-mode patterns extend to a serial chunk loop. Status: **Known** (user-confirmed). Inputs: integer-valued floats in `[1, 10]` cast to FP32, expected `max_abs_error == 0`; the user reported `test data success` / `test success`.
3. **Pattern** — single AICORE (`<<<1, nullptr, stream>>>`); `for (i = 0; i < NUM_TILES; ++i) { GlobalData<...>(base + offset); TLOAD; TLOAD; TADD; TSTORE; }` with `TileData a, b, c` declared **once outside the loop** and reused; static valid region `Tile<TileType::Vec, T, R, C, BLayout::RowMajor, R, C>` (no `DYNAMIC = -1`); GM access via `GlobalTensor<T, Shape<1,1,1,R,C>, Stride<1,1,1,C,1>>` reconstructed per iteration with the `offset` baked into the GM pointer. **Build harness mirrors topk** ([kernels/manual/a2a3/topk/CMakeLists.txt](../kernels/manual/a2a3/topk/CMakeLists.txt)): bisheng-direct compiler, the topk `pto_example_vec` function plus `--cce-enable-pto-passes` on the kernel target. `-O2` carries from the global `add_compile_options(...)`.
4. **Auto-mode compatibility** — Yes (Known, user-confirmed). Demonstrates several patterns that were Inferred-only before this build:
   - Static valid region in auto mode for a Vec tile of `(64, 64)` `float` ([tile_type_reference.md §6](tile_type_reference.md)).
   - In-kernel serial loop reusing the same Tile across iterations ([auto_mode_bad_patterns.md §1.2 corollary](auto_mode_bad_patterns.md): tile addresses pinned by the auto allocator do not require fresh declarations per iteration).
   - `GlobalTensor` reconstruction per iteration with `base + offset`.
   - `TLOAD → TADD → TSTORE` with **no manual sync** (no `set_flag` / `wait_flag` / `pipe_barrier` / `Event<>`) producing exact output.
   - Auto mode enabled by adding **only** `--cce-enable-pto-passes` to the kernel target's compile options (no separate `-D__PTO_AUTO__`; `-O2` from the project-global options is sufficient).
   - **No** kernel-arch guard `#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)` is needed when the CMake target sets `--cce-aicore-arch=dav-c220-vec` directly (matches topk's pattern).
5. **Copy** — the entire project layout for any new auto-mode prototype:
   - `<name>_kernel.cpp` with includes `<pto/common/constants.hpp>` and `<pto/pto-inst.hpp>` only (NOT `kernel_operator.h` — see [compile_error_logbook.md §E8](compile_error_logbook.md));
   - the topk-style `pto_example_vec_auto(NAME)` function in CMakeLists.txt (one-line delta from topk: add `--cce-enable-pto-passes` to the kernel target's `target_compile_options`);
   - run.sh that takes `-r npu|sim -v Ascend910B*`;
   - `scripts/gen_data.py` writing to `./input/` and `./output/`;
   - main.cpp using `tests/common/test_common.h` `ReadFile` / `WriteFile` / `ResultCmp`. **Important**: declare the file-size local as plain `size_t`, not `const size_t` (see [compile_error_logbook.md §E9](compile_error_logbook.md)).
6. **Do not copy** — single-AICORE launch is a v1 simplification; production-quality kernels should partition work across `BLOCK_DIM` cores via `block_idx` ([demos/auto_mode/baseline/add/csrc/kernel/add_custom.cpp:39-42](../demos/auto_mode/baseline/add/csrc/kernel/add_custom.cpp#L39-L42)). Tail handling is absent in v1 — total length must be a multiple of `TILE_ROWS * TILE_COLS`. No double / multi-buffering; no `block_idx`-based work split. These are deliberate v1 simplifications; future iterations will introduce optimization techniques.
7. **Confidence** — High (user-confirmed build and `test data success` on Ascend910B1).

### A12. topk (auto-mode A3, full TopK with values + indices) — confirmed-built single-row prototype

1. **File** — [kernels/automode/a2a3/topk/topk_kernel.cpp](../kernels/automode/a2a3/topk/topk_kernel.cpp)
2. **Why** — second in-tree confirmed-built auto-mode A3 kernel. First end-to-end auto-mode kernel that uses the full **TSORT32 + TMRGSORT + TGATHER** pipeline (vector-style; no cube/matmul) and produces both top-K values and matching original indices from a fully-random unsorted input. User-confirmed PASS on Ascend910B1 (`test value success` / `test index success` / `test success`). Status: **Known** (user-confirmed, fixed shape).
3. **Pattern** — Single AICORE (`<<<1, nullptr, stream>>>`); single row; `kCols=1280`, `kTopK=512`, `T=float`. Pipeline:
   - `TLOAD` `srcTile` (1×kCols float) and `idxTile` (1×kCols uint32 identity)
   - `TSORT32(packed, src, idx, scratch)` per 32-block, output (val, idx) packed
   - main `TMRGSORT` 4-way self-merge loop with **independent** ping-pong tile (`mrgScratchTile`) + `TMOV`-back to `sort32DstTile` prefix; loop bound and widths in **packed-element units**
   - `SortTailBlock` for non-power-of-4 residuals — `FillMrgArray<kPackedCols>(...)` schedules 2-list merges, each writing into `mrgScratchTile` (independent dst) and `TMOV`-back to `sort32DstTile`
   - `TGATHER<…, MaskPattern::P0101>` extracts values; `TGATHER<…, MaskPattern::P1010>` extracts indices via a **type-pun** view (`TRESHAPE` between `Tile<Vec, float, ...>` and `Tile<Vec, uint32, ...>`, mirroring [TQuant.hpp:108-114](../include/pto/npu/a2a3/TQuant.hpp#L108-L114)'s auto branch)
   - `TSTORE` two outputs (values + indices)

   **Packed-buffer layout and TMRGSORT semantics** (after `TSORT32`):
   - Each logical value becomes a packed `(value, index)` structure: `[val0, idx0, val1, idx1, ...]`.
   - For `float` (`TYPE_COEF=1`), the first sorted run is 32 structures = **64 packed elements** wide. That is why `blockLen = 64 * TYPE_COEF` is the starting block length.
   - `TMRGSORT(dst, src, blockLen)` is the **4-way merge** form. `blockLen` is in packed-element units; the implementation requires the merge width to be divisible by `blockLen * 4`, and each Phase-2 pass grows the sorted run length by **4×**. Hence the loop does `blockLen *= 4`.
   - `TMRGSORT(curDst, executedNumList, tmp, src0, src1)` is the **explicit 2-source** form used by `SortTailBlock` to handle residual / tail runs that are not a power of 4 of `blockLen`.

   **View-operator convention** ([§14 in tile_type_reference.md](tile_type_reference.md)):
   - Same-type prefix slice → `TSUBVIEW(view, tile, 0, 0)`
   - Same-type non-zero slice → `TSUBVIEW(view, tile, 0, offset)`
   - Reshape / reinterpret / type-pun → `TRESHAPE(view, tile)`

   Phases 2-4 of this kernel use `TSUBVIEW(..., 0, 0)` for same-type prefix views (`srcSortedView`, `tmpSortedView`, `src0View`, `curDstView`, `copyBackView`, `sortedTopKView`); Phase 5 retains `TRESHAPE` because it reinterprets float-typed storage as `uint32` for `TGATHER P1010`.
4. **Auto-mode compatibility** — Yes (Known, user-confirmed). Confirms several Inferred items:
   - `TSORT32`, `TMRGSORT` (4-way self-merge AND 2-list explicit forms with `MrgSortExecutedNumList`), and `TGATHER` (template form `<DstTile, SrcTile, MaskPattern>(dst, src)`) are all auto-callable from kernel code on A3.
   - `TGATHER` masks `P0101` (values from float-typed packed buffer) and `P1010` (indices from uint32 type-pun view) work as documented in the manual TopK.
   - `TRESHAPE` between tiles of **different element types** (float ↔ uint32) is auto-mode-safe at the kernel-level wrapper.
   - `TSORT32`'s `tmp` parameter is pure scratch — content is irrelevant; an uninitialized independent tile suffices (no `TLOAD` of tmp from GM).
   - Multi-iter `SortTailBlock` (3 iters in our `kCols=1280` shape) with `TMOV`-back is correct: each iteration's prefix-view `src0View` reads the previous iteration's merged result from `sort32DstTile`.
   - Auto-allocator places ~45 KB of independent UB tiles (srcTile, idxTile, sort32TmpTile, sort32DstTile, mrgScratchTile, tmp1Tile, outValTile, outIdxTile) without overlap.
5. **Copy** — the entire project layout (`pto_example_vec_auto`, `bash run.sh -r npu -v Ascend910B*` CLI, `scripts/gen_data.py` writing `input/` + `output/`); the packed-width discipline (`kPackedCols`/`kPackedTopK` derived from `TYPE_COEF`); the `TSUBVIEW`-for-same-type-slices / `TRESHAPE`-only-for-reinterpret rule (see view convention above); the **independent destination + TMOV-back** pattern for `SortTailBlock` (sidesteps the in-place TMRGSORT-with-dst-aliased-to-src risk that surfaced in the abandoned shortcut). Use as the auto-mode A3 baseline for any sort/argsort/select-k variant.
6. **Do not copy** —
   - The fixed shape (`kCols=1280`, `kTopK=512`, `float`) is what was tested; other shapes have not been exercised. Half (`TYPE_COEF=2`) was deferred (mask patterns differ).
   - Single AICORE only; no `block_idx` work split.
   - The kernel build observed **two real compile errors during development** that are now in the logbook ([§E11 host-side `__gm__` cast](compile_error_logbook.md), [§E12 `Topk` naming collision](compile_error_logbook.md)). Subsequent ports should mirror the resolved patterns.
   - The `__cce_tinit` / `__cce_alias` / `matrix-types-extension` errors that appeared transiently during development were caused by a misconfigured local bisheng-CCE toolchain, **not** a project bug; see [§E10 (WITHDRAWN)](compile_error_logbook.md). Do not add `-fenable-matrix` or fake `__cce_*` macros to any auto-mode project on the basis of those errors.
   - An earlier values-only shortcut (which assumed Python pre-sorted 64-element blocks and skipped `TSORT32` / `idxTile` / `TGATHER`) was abandoned. **Do not treat that variant as known-good TopK.** It produced wrong answers (interleaved output, dropped tail blocks) for reasons the full pipeline above does not share.
7. **Confidence** — High for the fixed shape; behavior at other `kCols`/`kTopK` or other dtypes is **Unknown**.

### A13. moe_top1_permute (auto-mode A3, device-side top-1 MoE forward dispatch) — confirmed-built fixed-shape prototype

> **Do not confuse with** the sibling directory [kernels/automode/a2a3/moe_top1_gather_precomp/](../kernels/automode/a2a3/moe_top1_gather_precomp/). That folder contains a **reduced sanity/debug kernel only** — it consumes a host-precomputed `packed_to_token[T]` index array and executes a pure indexed gather. It validates runtime scalar GM index reads and runtime row-offset `TLOAD`/`TSTORE`, but it is **NOT** the real MoE dispatch kernel because the packing order is computed on the host, not on the device. Treat `moe_top1_gather_precomp/` as a fallback/debug reference; the real MoE permute milestone is `moe_top1_permute/` (this entry).

1. **File** — [kernels/automode/a2a3/moe_top1_permute/moe_top1_permute_kernel.cpp](../kernels/automode/a2a3/moe_top1_permute/moe_top1_permute_kernel.cpp)
2. **Why** — third in-tree confirmed-built auto-mode A3 kernel; first one that does **device-side data-dependent control flow on GM scalar metadata** rather than purely tile-shaped compute. Implements the forward MoE token movement (histogram → prefix-sum → pack) for `topK = 1`, unlimited capacity, single AICORE. User-confirmed PASS on Ascend910B1 (all four GM outputs — `packed_tokens`, `expert_count`, `expert_start`, `token_to_packed` — match the Python golden). Status: **Known** (user-confirmed, fixed shape).
3. **Pattern** — Single AICORE (`<<<1, nullptr, stream>>>`); single static row tile `Tile<TileType::Vec, float, 1, kH, BLayout::RowMajor, 1, kH>` declared once and reused across all 256 token iterations. Three sequential passes:
   - **Pass 1 — histogram**: `for (t = 0..T) count[expert_id[t]]++;` with `int32_t count[kNumExperts]` as a small device-local stack array; result written to GM via scalar stores.
   - **Pass 2 — prefix sum**: `start[0] = 0; start[e] = start[e-1] + count[e-1];` on the same local stack array; result written to GM.
   - **Pass 3 — pack**: re-reads `expert_id[t]`, advances `counter[e]++`, computes `packed_pos = start[e] + slot`, writes scalar `token_to_packed[t] = packed_pos` to GM, then `TLOAD/TSTORE` the row into `packed_tokens[packed_pos * kH ..]`.
4. **Auto-mode compatibility** — Yes (Known, user-confirmed). Confirms the following Inferred / Assumption items together (all are auto-mode-safe at this shape):
   - **Scalar GM read of `int32_t` from kernel code** (`int32_t e = expert_id[t];`).
   - **Scalar GM write of `int32_t` from kernel code** (`expert_count[e] = count[e]; token_to_packed[t] = packed_pos;`) — auto-sync correctly orders these against the surrounding `TLOAD`/`TSTORE` on a different GM buffer in the same iteration.
   - **Small device-local `int32_t arr[E]` indexed by a runtime scalar** with mutating updates (`arr[e]++`). `E = 4` was tested.
   - **Runtime-scalar GM offset for `GlobalTensor`** — `GlobalTensor srcGlobal(tokens + size_t(t) * kH); GlobalTensor dstGlobal(packed_tokens + size_t(packed_pos) * kH);` reconstructed per iteration. The offset comes from a kernel-local `int32_t`, not a compile-time induction variable — different from the [add_tile_array §A11](#a11-add_tile_array--first-confirmed-built-auto-mode-kernel-with-an-in-kernel-serial-loop-a3) case where the offset was `i * stride`. Both work; A13 closes the data-dependent variant.
   - **Single row tile reused across 256 sequential write destinations**, each at a runtime-different `packed_pos`, with no manual sync. Auto-allocator pins the UB address; auto-sync inserts the MTE2/MTE3 fences.
   - **Multiple GM output buffers written in the same kernel** (`packed_tokens`, `expert_count`, `expert_start`, `token_to_packed`) without manual sync between them; auto-sync handles the buffer-level liveness.
5. **Copy** — the entire project layout for any new auto-mode A3 prototype that needs **device-computed metadata + indexed row copy**:
   - the [moe_top1_permute](../kernels/automode/a2a3/moe_top1_permute/) directory shape (`<name>_kernel.cpp`, `main.cpp`, `CMakeLists.txt`, `run.sh`, `scripts/gen_data.py`, `README.md`);
   - the three-pass histogram → prefix-sum → pack shape — directly applicable to any "group tokens by key, emit per-group start/count, then permute rows" problem (e.g., bucket-sort dispatch, top-1 MoE, gathered embedding lookups, segment-by-id reductions);
   - the `int32_t arr[E]` stack-array pattern for small per-group state — avoids any UB tile for what is fundamentally a small register working set;
   - the `size_t off = static_cast<size_t>(scalar) * kH;` discipline for constructing per-iter GM offsets (avoids `int * unsigned` width pitfalls);
   - the Python golden that mirrors the kernel exactly: `np.bincount(expert_id)` for `expert_count`, `np.cumsum(...)` for `expert_start`, `np.argsort(expert_id, kind="stable")` for the packing order — these match the kernel's pass-1/2/3 semantics byte-for-byte.
6. **Do not copy** —
   - The fixed shape (`T = 256`, `H = 64`, `E = 4`, `float32` tokens, `int32` metadata) is what was tested; **larger T**, **larger E**, **larger or non-multiple-of-block H**, **half tokens**, **uint8 expert IDs** have NOT been exercised. Especially: a runtime `E` (vs the compile-time `kNumExperts = 4` used here) is not validated — the `int32_t count[kNumExperts]` stack array depends on `kNumExperts` being a compile-time constant.
   - **Single AICORE only**; no `block_idx` work split. Multi-core dispatch requires a per-core histogram workspace (`workspace[num_cores][num_experts]`) and a cross-core prefix-sum / barrier — neither is implemented. Do NOT generalize this entry to multi-core MoE without a separate confirmed build.
   - **`topK = 1` only**; no router GEMM, no top-K selection, no weighted combine, no per-expert capacity / drop policy, no fallback expert. `topK > 1` changes the packing rule (each token contributes `topK` rows) and is out of scope for this entry.
   - **Forward only**; no backward. The mapping `token_to_packed` is emitted to support a future unpermute, not a backward derivative.
   - The kernel reads `expert_id[t]` **twice** (pass 1 histogram, pass 3 pack). Acceptable for `T = 256`; for large `T` an optimization could cache it in a UB tile. The current shape does NOT validate that optimization.
7. **Confidence** — High for the fixed shape; behavior at other `T` / `H` / `E` / dtype / multi-core / `topK > 1` is **Unknown**.

### A14. moe_top1_unpermute (auto-mode A3, reverse token movement after top-1 permute) — confirmed-built fixed-shape prototype

> Companion to [§A13 moe_top1_permute](#a13-moe_top1_permute-auto-mode-a3-device-side-top-1-moe-forward-dispatch--confirmed-built-fixed-shape-prototype). Together the two kernels close the forward token-movement pair (`tokens` → packed → unpermute → `output`).

1. **File** — [kernels/automode/a2a3/moe_top1_unpermute/moe_top1_unpermute_kernel.cpp](../kernels/automode/a2a3/moe_top1_unpermute/moe_top1_unpermute_kernel.cpp)
2. **Why** — fourth in-tree confirmed-built auto-mode A3 kernel; completes the **inverse** of §A13's expert-grouped packing by restoring `packed_output` to original token order. Where §A13 reads `expert_id[t]` and computes the destination slot on-device, this kernel reads the precomputed `token_to_packed[t]` mapping and copies the matching packed row back to the token's original position. User-confirmed PASS on Ascend910B1 (`output` matches the Python golden bit-exact at the tested shape). Status: **Known** (user-confirmed, fixed shape).
3. **Pattern** — Single AICORE (`<<<1, nullptr, stream>>>`); single static row tile `Tile<TileType::Vec, float, 1, kH, BLayout::RowMajor, 1, kH>` declared once and reused across all 256 token iterations. Single pass:
   ```cpp
   for (t = 0; t < T; ++t) {
       int32_t packed_pos = token_to_packed[t];     // scalar GM read
       size_t  src_off    = size_t(packed_pos) * H;
       size_t  dst_off    = size_t(t)          * H;
       TLOAD (rowTile, GlobalTensor<...>(packed_output + src_off));
       TSTORE(GlobalTensor<...>(output    + dst_off), rowTile);
   }
   ```
   No on-device histogram / prefix-sum logic (the mapping was already produced by §A13); no scalar GM writes; no device-local `int32_t arr[E]`.
4. **Auto-mode compatibility** — Yes (Known, user-confirmed). Reconfirms patterns already resolved by [§A13 / §11.4](#a13-moe_top1_permute-auto-mode-a3-device-side-top-1-moe-forward-dispatch--confirmed-built-fixed-shape-prototype):
   - Scalar GM read of `int32_t` (`token_to_packed[t]`) — same shape as `expert_id[t]` in §A13.
   - Runtime scalar usable as the source-side row offset multiplier in `GlobalTensor srcGlobal(packed_output + size_t(packed_pos) * kH);` — symmetric to §A13's destination-side use of `packed_pos` for `packed_tokens`.
   - Single static row tile reused across `T` runtime-offset iterations.
   - `TLOAD` from runtime-offset GM, `TSTORE` to runtime-offset GM in the same iter, no manual sync.
5. **Copy** — the entire project layout for any new auto-mode A3 prototype that needs **gather-from-runtime-index then write-sequential** (or, by argument swap, **read-sequential then scatter-to-runtime-index**):
   - the [moe_top1_unpermute](../kernels/automode/a2a3/moe_top1_unpermute/) directory shape;
   - the kernel skeleton (loop, scalar GM read, two `size_t` offsets, one row tile, `TLOAD` + `TSTORE`);
   - the Python reference that **replays the forward permute internally** so the unpermute can be exercised standalone — the recipe (`np.argsort(expert_id, kind="stable")` + `token_to_packed[order[p]] = p`) matches the §A13 kernel's pass-3 packing rule byte-for-byte.
   - the optional [scripts/compare_outputs.py](../kernels/automode/a2a3/moe_top1_unpermute/scripts/compare_outputs.py) standalone diff utility for post-mortem when ResultCmp reports failure.
6. **Do not copy** —
   - The fixed shape (`T = 256`, `H = 64`, `float32` data, `int32` metadata) is what was tested; other `T` / `H` / dtypes are Unknown.
   - **Assumes the input `token_to_packed` is a valid permutation of `[0, T)`** — the contract §A13 provides. Garbage indices (out-of-range, duplicates) are undefined.
   - Single AICORE only; no `block_idx` work split.
   - The fake `packed_output = packed_tokens + 1.0` is a **test fixture**, not part of the production unpermute. Replace it with the actual per-expert FFN output in the real MoE pipeline.
   - Does NOT validate `topK > 1` (which requires weighted combine: `output[t, :] = sum_k prob[t, k] * packed_output[token_to_packed_topk[t, k], :]` — a different kernel shape with a multiplication and an accumulation per slot).
7. **Confidence** — High for the fixed shape; behavior at other `T` / `H` / dtype / multi-core / `topK > 1` (weighted combine) is **Unknown**.

### A15. moe_segmented_identity (auto-mode A3, per-expert segmented microtile loop with elementwise op) — confirmed-built fixed-shape prototype

> First in-tree A3 auto-mode kernel that walks **dynamic expert segments** (`expert_start[e]`, `expert_count[e]`) with a fixed `TILE_M = 128` microtile inside each segment. Mirrors the future expert-FFN outer schedule but uses a simple elementwise op (`TADDS`, `dst = src + 1.0f`) in place of GEMM — no cube path yet. Companion to [§A13 moe_top1_permute](#a13-moe_top1_permute-auto-mode-a3-device-side-top-1-moe-forward-dispatch--confirmed-built-fixed-shape-prototype) and [§A14 moe_top1_unpermute](#a14-moe_top1_unpermute-auto-mode-a3-reverse-token-movement-after-top-1-permute--confirmed-built-fixed-shape-prototype).

1. **File** — [kernels/automode/a2a3/moe_segmented_identity/moe_segmented_identity_kernel.cpp](../kernels/automode/a2a3/moe_segmented_identity/moe_segmented_identity_kernel.cpp)
2. **Why** — fifth in-tree confirmed-built auto-mode A3 kernel. First one that nests **two** runtime loops where both bounds come from scalar GM metadata (outer over experts, inner over microtiles within an expert's padded segment); first that holds **two concurrently-live** 32 KB Vec tiles in UB; first that calls `TADDS` from inside a nested loop with runtime-driven GM offsets. User-confirmed PASS on Ascend910B1 (after a one-revision fix described in the "gotcha" note below). Status: **Known** (user-confirmed, fixed shape).
3. **Pattern** — Single AICORE (`<<<1, nullptr, stream>>>`); two static segment tiles declared once and reused across all inner iterations:
   ```cpp
   using SegTile = Tile<TileType::Vec, float, 128, 64,
                         BLayout::RowMajor, 128, 64>;
   SegTile srcTile;
   SegTile dstTile;
   for (e = 0; e < kNumExperts; ++e) {
       int32_t start = expert_start[e];    // GM scalar read (PADDED)
       int32_t count = expert_count[e];    // GM scalar read (PADDED, multiple of TILE_M)
       for (m0 = 0; m0 < count; m0 += kTileM) {
           size_t off = (size_t(start) + size_t(m0)) * kH;
           SegGlobal srcGlobal(packed_tokens + off);
           SegGlobal dstGlobal(packed_output + off);
           TLOAD (srcTile, srcGlobal);
           TADDS (dstTile, srcTile, 1.0f);
           TSTORE(dstGlobal, dstTile);
       }
   }
   ```
   v1 tail policy: host pads every expert segment length up to a multiple of `TILE_M`; padded rows initialised to `0.0`; golden adds `1.0` to all rows (real or padded). Kernel never needs `SetValidRow` / partial-tile stores.
4. **Auto-mode compatibility** — Yes (Known, user-confirmed). Confirms the following Inferred / Assumption items at the tested shape:
   - **Nested loop with both bounds from GM scalar metadata** — outer 4 experts × inner up to `padded_count / kTileM` microtiles; `expert_start[e]` and `expert_count[e]` are re-read each outer iter.
   - **Two `128 × 64` float Vec tiles concurrently live** (64 KB total UB allocation, 4× the [§A11 add_tile_array](#a11-add_tile_array--first-confirmed-built-auto-mode-kernel-with-an-in-kernel-serial-loop-a3) tile budget). Auto allocator pins both addresses; no manual `TASSIGN`.
   - **`TADDS(dstTile, srcTile, 1.0f)` with distinct src/dst tiles** from inside a nested loop with runtime-driven `GlobalTensor` offsets. Matches the shape in [tests/npu/a2a3/src/st/testcase/tadds/tadds_kernel.cpp](../tests/npu/a2a3/src/st/testcase/tadds/tadds_kernel.cpp) (which is in `ALL_TESTCASES`).
   - **Host-padded expert segment layout** as a workable v1 tail policy — kernel sees only padded `expert_count` / `expert_start`; padded rows are processed identically to real rows; `SetValidRow` / `SetValidShape` / partial stores remain unused (and still Unknown for auto mode — see [tile_type_reference.md §6 / §11 item 12](tile_type_reference.md)).
5. **Copy** — the entire project layout for any auto-mode A3 prototype that needs **per-expert segmented compute on host-padded segments**:
   - the [moe_segmented_identity](../kernels/automode/a2a3/moe_segmented_identity/) directory shape;
   - the nested loop skeleton above (it transfers cleanly to a GEMM body once the elementwise call is swapped out);
   - the Python reference shape (`np.bincount` for real counts, `ceil(count / TILE_M) * TILE_M` for padded counts, stable argsort for the packing order, zero-pad the tail of each expert's slot in `packed_tokens`);
   - the v1 host-padded golden rule (`golden = packed_tokens + 1.0` for ALL rows including padded) — extends naturally to GEMM where padded zero-input rows produce zero output;
   - the `compare_outputs.py` first-mismatch report that maps flat index → `{row, col, expert, padded_start, padded_count, real_count, offset_in_segment, tile_m0, tile_idx, is_padded_row}` — useful debugging surface for any segmented-loop bug.
6. **Do not copy** —
   - The fixed shape (`T = 256`, `H = 64`, `kE = 4`, `kTileM = 128`, `float32`) is what was tested.
   - **In-place `TADDS(t, t, 1.0f)` failed and is NOT known-good.** The first revision of this kernel used a single `segTile` for both src and dst (`TADDS(segTile, segTile, 1.0f)`). It produced a zero-filled mismatch region around flat indices ~`0x1088..0x1132`. The passing revision uses distinct `srcTile` / `dstTile`. **Do not copy the in-place form as a known-good auto-mode pattern** — its safety is Unknown / suspicious until separately tested. This applies specifically to the observed `TADDS` case; do NOT generalize to every PTO instruction without evidence. The kernel sources keep a history comment recording the regression.
   - Single AICORE only; no `block_idx` work split.
   - `SetValidRow` / `SetValidShape` are NOT validated by this milestone — host padding sidesteps them.
   - No cube path (no `TMATMUL` / `TileLeft` / `TileRight` / `TileAcc`); no second tile or fused op; no activation; no real FFN. Replaces the test-fixture `+1.0` with the real per-expert compute in `moe_segmented_gemm_one_layer` / `moe_segmented_ffn`.
   - Larger `kNumExperts` may pressure register / stack allocation for the local `count[]`, `start[]`, `counter[]` arrays in the upstream permute kernel and the larger inner-loop bound here — Unknown.
7. **Confidence** — High for the fixed shape; behavior at other `T_PADDED` / `H` / `kE` / `kTileM` / dtype / multi-core / dynamic-tail / GEMM is **Unknown**.

### A16. moe_segmented_gemm_one_layer (auto-mode A3, one cube GEMM per expert microtile) — confirmed-built fixed-shape prototype

> First in-tree A3 auto-mode kernel that fires the **cube path** (`TMATMUL`) inside the working per-expert segmented loop. Replaces §A15's `TADDS` body with one expert-specific GEMM tile; everything outside the inner body (host-padded segments, nested loop, runtime GM offsets) is identical. FP16 × FP16 → FP32 (the canonical A3 auto-mode-eligible cube combo from §A6 `tmatmul`).

1. **File** — [kernels/automode/a2a3/moe_segmented_gemm_one_layer/moe_segmented_gemm_one_layer_kernel.cpp](../kernels/automode/a2a3/moe_segmented_gemm_one_layer/moe_segmented_gemm_one_layer_kernel.cpp)
2. **Why** — sixth in-tree confirmed-built auto-mode A3 kernel. First **cube** auto-mode kernel in this MoE stack; first to combine the §A15 nested expert/microtile loop with the §A6 `TMATMUL` skeleton. User-confirmed PASS on Ascend910B1 after a host-side compile-error fix described in the gotcha below. Status: **Known** (user-confirmed, fixed shape).
3. **Pattern** — Single AICORE (`<<<1, nullptr, stream>>>`); build target uses `--cce-aicore-arch=dav-c220-cube` (not vec) plus `--cce-enable-pto-passes` for auto mode. Tile aliases lifted from `tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp` `RunTMATMUL<float, half, half, float, validM, validK, validN, false>`:
   ```cpp
   using TileMatAData = Tile<TileType::Mat, half,  M, K, BLayout::ColMajor,
                             kTileM, kH, SLayout::RowMajor, 512>;
   using TileMatBData = Tile<TileType::Mat, half,  K, N, BLayout::ColMajor,
                             kH,     kO, SLayout::RowMajor, 512>;
   using LeftTile  = TileLeft <half,  M, K, kTileM, kH>;   // L0A
   using RightTile = TileRight<half,  K, N, kH,     kO>;   // L0B
   using AccTile   = TileAcc  <float, M, N, kTileM, kO>;   // L0C
   ```
   All five tiles declared once outside both loops; auto allocator pins all five addresses. Body per inner iter:
   ```cpp
   GlobalDataA aGlobal(packed_tokens + (start + m0) * kH);
   GlobalDataC cGlobal(packed_output + (start + m0) * kO);
   GlobalDataB bGlobal(expert_weight + e * (kH * kO));   // hoisted to outer iter
   TLOAD (aMatTile, aGlobal);
   TLOAD (bMatTile, bGlobal);
   TMOV  (aTile, aMatTile);
   TMOV  (bTile, bMatTile);
   TMATMUL(cTile, aTile, bTile);
   TSTORE(cGlobal, cTile);
   ```
   Host-padded expert segments (same recipe as §A15) make every inner iter a full `TILE_M`-row tile — no `SetValidRow`, no partial stores.
4. **Auto-mode compatibility** — Yes (Known, user-confirmed). Confirms the following at the tested shape:
   - **`TMATMUL` inside the segmented expert/microtile loop** with runtime-driven A and C offsets per inner iter and per-expert weight offset per outer iter.
   - **FP16 × FP16 → FP32 cube combo** (`<float, half, half, float, M=128, K=64, N=64>`) — matches `LaunchTMATMUL<1>` reference instantiation. Cube FP32 accumulator over small-int FP16 inputs is bit-exact in the tested distribution.
   - **Expert-specific weight selection** by GM-pointer arithmetic `expert_weight + e * (kH * kO)`; no UB copy of weights; `bMatTile` and `bTile` reused across inner iters within an expert but driven by a fresh `GlobalDataB` per outer iter.
   - **Mat/Left/Right/Acc tiles reused across nested loop iterations** without manual sync — auto-sync inserts the MTE2 → MTE1 → M → FIX fences.
   - **Host-padded expert segment layout** continues to work for the cube path; padded rows are zero, so their GEMM output is exactly zero with no bias.
   - **Cube-arch + auto-mode compile recipe**: kernel target gets `--cce-aicore-arch=dav-c220-cube --cce-enable-pto-passes -O2`; the rest of the harness is unchanged from the vec-arch sibling projects.
   - **Non-template host launcher wrapper hides `half` from `main.cpp`** — see the gotcha below.
5. **Copy** — the entire project layout for any new auto-mode A3 prototype that needs **per-expert segmented cube GEMM**:
   - the [moe_segmented_gemm_one_layer](../kernels/automode/a2a3/moe_segmented_gemm_one_layer/) directory shape;
   - the `pto_example_cube_auto` CMake function (one-line delta from `pto_example_vec_auto`: swap `dav-c220-vec` for `dav-c220-cube`);
   - the nested-loop + TMATMUL body above; it transfers cleanly to a GEMM1 + activation step (see §A17 / `moe_segmented_gemm_relu`) by changing the `TSTORE` template args;
   - the `uint8_t*` host boundary for FP16 / FP32 typed buffers + bare `int32_t*` for metadata + non-template `…Fp16` wrapper at the kernel TU boundary;
   - the Python golden recipe (per-expert `np.float32` GEMM over `np.float16` inputs; padded rows are zero so their golden output is zero).
6. **Do not copy** —
   - The fixed shape (`T = 256`, `H = K = 64`, `O = N = 64`, `kE = 4`, `kTileM = M = 128`, FP16 × FP16 → FP32) is what was tested. Other `M/K/N` combos, dtypes (int8, bf16, fp32 × fp32), bias path, TF32 path, SplitK loop, TGEMV — all Unknown for this MoE-segmented context until separately confirmed.
   - **Do not expose `half` in `main.cpp` template arguments.** The host TU is compiled with plain `-xc++` and cannot see `half` (a bisheng-CCE compiler-provided type only visible in `-xcce` translation units that include `<pto/pto-inst.hpp>`). The first compile attempt — `launchMoeSegmentedGemmOneLayer<float, half, half>(...)` at the host call site — failed with `use of undeclared identifier 'half'`. The passing fix is to keep all `half` template parameters inside the kernel TU and expose a non-template launcher wrapper (`launchMoeSegmentedGemmOneLayerFp16(uint8_t*, uint8_t*, int32_t*, int32_t*, uint8_t*, void*)`) that the host calls. **Rule (narrowly scoped to this pattern)**: *For host drivers using raw `uint8_t*` FP16 buffers, do not expose `half` in `main.cpp` template arguments; hide device scalar types behind a non-template launcher wrapper.* Do not overgeneralize to other host-boundary patterns.
   - Single AICORE only; no `block_idx` work split.
   - No bias, no activation, no second GEMM, no fused L0C-side post-processing other than the plain `TSTORE`. Use §A17 once it lands for the GEMM + ReLU shape.
   - The `TASSIGN` literal-address calls and `#ifndef __PTO_AUTO__` manual-sync / `TFILLPAD` blocks from the `tmatmul` reference are omitted here (auto-mode no-ops); do not reintroduce.
   - `SetValidRow` / `SetValidShape` / partial-tile stores still NOT validated — host padding sidesteps them.
7. **Confidence** — High for the fixed shape; behavior at other shapes, dtypes, bias / activation paths, multi-core, dynamic-tail is **Unknown**.

### A17. moe_segmented_gemm_relu (auto-mode A3, one cube GEMM + FIX-pipe ReLU per expert microtile) — confirmed-built fixed-shape prototype

> Adds ReLU activation after the §A16 per-expert GEMM by **fusing it into the L0C → GM TSTORE** via the `ReluPreMode::NormalRelu` overload of the public `TSTORE` wrapper. No cube → vec handoff, no separate vector kernel, no mix-arch build, no UB intermediate.

1. **File** — [kernels/automode/a2a3/moe_segmented_gemm_relu/moe_segmented_gemm_relu_kernel.cpp](../kernels/automode/a2a3/moe_segmented_gemm_relu/moe_segmented_gemm_relu_kernel.cpp)
2. **Why** — seventh in-tree confirmed-built auto-mode A3 kernel. First MoE-stack kernel that wires an **activation** onto the cube GEMM output without any UB roundtrip. Validates the FIX-pipe activation fusion in the per-expert segmented loop. User-confirmed PASS on Ascend910B1. Status: **Known** (user-confirmed, fixed shape).
3. **Pattern** — Single AICORE; cube arch (`--cce-aicore-arch=dav-c220-cube`) + `--cce-enable-pto-passes`. Kernel body is byte-for-byte the §A16 GEMM kernel except the final TSTORE adds two template args:
   ```cpp
   TSTORE<AccTile, GlobalDataC,
          AtomicType::AtomicNone,
          ReluPreMode::NormalRelu>(cGlobal, cTile);
   ```
   The pre-ReLU accumulator stays on L0C; the FIX pipe applies `max(x, 0)` while writing FP32 to GM. All five cube tiles (`Mat ×2`, `Left`, `Right`, `Acc`) declared once outside both loops and reused across iters; auto allocator pins all addresses; auto-sync inserts the MTE2 → MTE1 → M → FIX fences.
4. **Auto-mode compatibility** — Yes (Known, user-confirmed). Confirms at the tested shape:
   - **Fused ReLU in the L0C → GM TSTORE** via `TSTORE<TileData, GlobalData, AtomicType::AtomicNone, ReluPreMode::NormalRelu>(dst, src)` inside the per-expert segmented loop. Matches the shape of [tests/npu/a2a3/src/st/testcase/tstore_acc2gm/tstore_acc2gm_kernel.cpp:88-93](../tests/npu/a2a3/src/st/testcase/tstore_acc2gm/tstore_acc2gm_kernel.cpp#L88-L93) (`LaunchTStoreAcc2gmNz2nd<21>` / `LaunchTStoreAcc2gmNz2nz<21>`).
   - **FIX-pipe activation fusion** is auto-mode-safe in the per-expert segmented context (not just for a single-tile TSTORE as in the standalone `tstore_acc2gm` test).
   - **`ReluPreMode::NormalRelu`** template arg propagates correctly through the auto-mode `TSTORE` template dispatch in a kernel built with `--cce-aicore-arch=dav-c220-cube`.
5. **Copy** — for any new auto-mode A3 prototype that needs **GEMM + ReLU as a single op** without leaving cube arch:
   - the entire [moe_segmented_gemm_relu](../kernels/automode/a2a3/moe_segmented_gemm_relu/) project shape;
   - the explicit-template-arg form (`<AccTile, GlobalDataC, AtomicType::AtomicNone, ReluPreMode::NormalRelu>`) — the `reluPreMode` parameter has no default in [include/pto/common/pto_instr.hpp:251-258](../include/pto/common/pto_instr.hpp#L251-L258), so it must be spelled explicitly along with the `atomicType` arg before it;
   - the Python data recipe (inputs in `[-4, 4]` so the pre-ReLU GEMM output contains both negatives (clipped) and positives (passed through) — without negatives ReLU degenerates to identity and the test cannot distinguish "ReLU fired" from "ReLU was a no-op");
   - the `compare_outputs.py` "failure-mode breakdown" (CLIPPED vs PASS-THROUGH counts) — distinguishes "device skipped ReLU" from "GEMM itself regressed".
6. **Do not copy** —
   - The fixed shape (`T = 256`, `H = K = 64`, `O = N = 64`, `kE = 4`, `kTileM = 128`, FP16 × FP16 → FP32) is what was tested.
   - **Activation other than `ReluPreMode::NormalRelu`** is NOT validated. Only the two-value enum `{NoRelu, NormalRelu}` exists in [include/pto/common/type.hpp:255-259](../include/pto/common/type.hpp#L255-L259); GELU, SiLU, LeakyReLU, etc. are not options here. The standalone `TRELU` / `TMAXS` wrappers on Vec tiles are also unexercised by this milestone — the ReLU happens entirely in the FIX pipe.
   - The **combination** of `ReluPreMode::NormalRelu` AND down-cast `AccTile<float>` → GM `half` in the same TSTORE is NOT validated. `tstore_acc2gm` shows the two features individually (`tilingKey=4` does FP32→FP16 without ReLU; `tilingKey=21` does ReLU with no dtype change) but not combined. Do not assume the combined form works without a separate experiment (this is the open assumption that gates §A18 / the FFN milestone — see `moe_segmented_ffn_top1`).
   - Single AICORE only; no `block_idx`; no SetValidRow / partial stores; no second GEMM; no bias; no SplitK; no TF32; no INT8 / BF16.
7. **Confidence** — High for the fixed shape; behavior at other shapes / dtypes / activation modes / combined Acc-down-cast-with-ReLU / multi-core / dynamic-tail is **Unknown**.

### A18. moe_segmented_ffn_top1 (auto-mode A3, full top-1 segmented MoE FFN via two stream-serialised kernels) — confirmed-built fixed-shape prototype

> First in-tree auto-mode A3 kernel that runs the **full FFN** (`GEMM1 → ReLU → scratch → GEMM2`) per expert microtile. Composes §A17 and §A16 verbatim through a host-level stream handoff; the scratch dtype is FP16, written by the §A17 fused-TSTORE pattern with the GM destination dtype changed from float to half. The previous-iteration assumption (FP32 Acc → FP16 GM + ReLU in **ND-layout** TSTORE) is now Resolved at this shape.

1. **File** — [kernels/automode/a2a3/moe_segmented_ffn_top1/moe_segmented_ffn_top1_kernel.cpp](../kernels/automode/a2a3/moe_segmented_ffn_top1/moe_segmented_ffn_top1_kernel.cpp)
2. **Why** — eighth in-tree confirmed-built auto-mode A3 kernel. First to chain two cube GEMMs with an FP16 scratch hand-off in a single device computation. User-confirmed PASS on Ascend910B1 (`test data success` / `test success`). Status: **Known** (user-confirmed, fixed shape).
3. **Pattern** — Two `__global__ AICORE` template functions in the SAME TU:
   - `runFfnStage1Gemm1Relu` — byte-for-byte the §A17 kernel, with the GM destination dtype changed from `float` to `half` (`AccTile` stays FP32; ReLU + FP32→FP16 downcast both happen inside the FIX-pipe TSTORE).
   - `runFfnStage2Gemm2` — byte-for-byte the §A16 kernel, with the A buffer pointing at the scratch produced by stage 1.

   Each kernel declares 5 cube tiles (Mat ×2, Left, Right, Acc) — identical budget to §A16 / §A17. The host wrapper fires both kernels on the **same stream**:
   ```cpp
   launchFfnStage1Gemm1Relu<half, half, half>(scratch, packed_tokens, count, start, w1, stream);
   launchFfnStage2Gemm2<float, half, half>(packed_output, scratch, count, start, w2, stream);
   ```
   ACL stream-order semantics guarantee stage 2 only starts after stage 1's TSTORE-to-scratch has fully drained to GM. The cross-GEMM hand-off is an ACL stream dependency, NOT a PTO auto-sync dependency.
4. **Auto-mode compatibility** — Yes (Known, user-confirmed). Confirms at the tested shape:
   - **Fused FP32 Acc → FP16 GM + `ReluPreMode::NormalRelu` in a SINGLE ND-layout TSTORE** — this was §A17's "Do not copy" gating assumption; it is now Resolved for ND layout at this shape. The combo was already in `ALL_TESTCASES` for NZ layout via [tstore_acc2gm_kernel.cpp:627 `LaunchTStoreAcc2gmNz2nz<21>` `<0, float, float, half, ..., 1>`](../tests/npu/a2a3/src/st/testcase/tstore_acc2gm/tstore_acc2gm_kernel.cpp#L627). ND-layout variant of the same combo now also works.
   - **Two cube kernels chained through an FP16 GM scratch buffer on the same stream**. Each kernel sees its own well-formed auto-sync graph; the cross-kernel ordering is provided by ACL, not PTO.
   - **Host-managed temporary device buffer** (`scratchDev`) that no host code ever copies INTO — kernel writes it, kernel reads it back. Host only `aclrtMalloc`s the size, optionally poisons it, and `aclrtFree`s it.
5. **Copy** — for any new auto-mode A3 prototype that needs **multi-stage cube compute with a GM hand-off**:
   - the entire [moe_segmented_ffn_top1](../kernels/automode/a2a3/moe_segmented_ffn_top1/) project layout;
   - the **split-kernel + same-stream** composition shape — one `__global__ AICORE` per stage in the same TU, with a single `extern "C"` host wrapper that calls them in order on the user's stream;
   - the **single-isolated-new-assumption discipline** — only change ONE thing per milestone (here, the GM dest dtype of the §A17 TSTORE);
   - the **stage-isolation host driver shape** — poison the scratch and output device buffers with distinct byte patterns before the launch; copy both back after sync; let `compare_outputs.py` distinguish "kernel never wrote" / "kernel wrote zeros" / "kernel wrote wrong values" per stage;
   - the **Python golden recipe** — mirror the kernel's three computational steps (GEMM1 in FP32 acc → cast to FP16 with `np.maximum(., 0)` → GEMM2 in FP32 acc) so the golden scratch is bit-exact comparable.
6. **Do not copy** —
   - The fixed shape (`T = 256`, `H = K = 64`, `F = N = 64`, `kE = 4`, `kTileM = 128`, FP16 × FP16 → FP32 for both GEMMs, FP16 scratch). `kH == kF` is enforced by `static_assert` in the kernel; widening either independently is **Unknown** because the cube tile shapes for stage 1 and stage 2 would diverge.
   - **The fused single-kernel form was attempted first and hung the device once**. It used either (a) two independent 5-tile cube sets concurrently (10 cube tiles total — produced wrong output, didn't hang) or (b) a single 5-tile cube set reused across both GEMMs in the same inner iteration (cTile drained to scratch GM then aMatTile reloaded from the same scratch GM — hung the device once during testing). Splitting into two kernels sidesteps both. **Root cause of the (b) hang is unverified** — it could be auto-sync producing an unmatched `wait_flag` for the cross-GEMM tile-reuse pattern, OR a transient hardware/driver issue unrelated to the kernel. The split-kernel form is the Known-good path either way; reproducing the hang to attribute root cause is a follow-up (see `assumptions_to_verify.md §11.9 "What is NOT proven"`).
   - **Single AICORE per kernel only**; no `block_idx` work split.
   - Activations other than `NormalRelu`; bias; SplitK; TF32; INT8 / BF16; non-`(128, 64, 64)` cube shapes; multi-F-tile / multi-K-tile; `topK > 1`; weighted combine; backward; performance.
   - `SetValidRow` / `SetValidShape` / partial-tile stores — still NOT validated; host padding sidesteps them.
   - The `kSkipStage2` host-wrapper toggle is a debug aid, not a production switch.
7. **Confidence** — High for the fixed shape; behavior at other shapes / dtypes / `kH != kF` / multi-core / dynamic-tail / fused-single-kernel form / non-NormalRelu activations is **Unknown**.

---

## Group B — Pattern references (use semantics, not source as-is)

### B1. Row-softmax tutorial (single-tile)

1. **File** — [docs/coding/tutorials/row-softmax.md](../docs/coding/tutorials/row-softmax.md)
2. **Why** — Smallest correct **softmax** decomposition: `TROWMAX → TROWEXPAND/TSUB → TEXP → TROWSUM → TROWEXPAND/TDIV`. Single tile, no buffering.
3. **Pattern** — Row reduction → broadcast subtract → exp → row reduction → broadcast divide. Matches the canonical FA softmax pieces.
4. **Auto-mode compatibility** — Mostly (Inferred). The tutorial uses no manual sync/`TASSIGN`/`Event<>`. The exact `TROWEXPAND` overload here passes a `tmp` and a `Col1` tile — verify the signature exists in [include/pto/npu/a2a3/TRowExpand.hpp](../include/pto/npu/a2a3/TRowExpand.hpp) before reusing.
5. **Copy** — The 5-step decomposition; the `Col1 = Tile<..., M, 1, BLayout::RowMajor, DYNAMIC, DYNAMIC>` declaration; ordering of operations.
6. **Do not copy** — Treat as pseudocode for shape/order — overload arities should be cross-checked in the per-arch `T*.hpp` first.
7. **Confidence** — Medium.

### B2. GEMM tutorial (single-tile skeleton)

1. **File** — [docs/coding/tutorials/gemm.md](../docs/coding/tutorials/gemm.md)
2. **Why** — Documents tile roles (`Mat`, `TileLeft`, `TileRight`, `TileAcc`) and the `TLOAD → TMOV → TMATMUL → TSTORE` skeleton in plain prose. Useful as a doc anchor when writing a new auto-mode GEMM.
3. **Pattern** — Same as A6/A7 minus pipelining.
4. **Auto-mode compatibility** — Yes (Inferred — written as a generic intrinsic skeleton; no manual sync).
5. **Copy** — Tile-role naming and shapes.
6. **Do not copy** — N/A (not source code).
7. **Confidence** — Medium.

### B3. FA softmax helper macros (FA-internal, library-level)

1. **File** — [tests/npu/a2a3/src/st/testcase/tfa/pto_macro_fa_softmax.hpp](../tests/npu/a2a3/src/st/testcase/tfa/pto_macro_fa_softmax.hpp)
2. **Why** — Concrete FA-2.0 online-softmax math expressed in PTO instructions, including the init/non-init distinction (rescale prior global sum by `exp_max`). Matches the math in [tests/npu/a2a3/src/st/testcase/tfa/TFA_kernel.md](../tests/npu/a2a3/src/st/testcase/tfa/TFA_kernel.md).
3. **Pattern** — `TROWMAX → TROWEXPANDSUB_IMPL → TMULS(scale) → TEXP → TCVT → TROWSUM → TADD/TMUL` with a numerically-stable max/sum recurrence. Uses `TRESHAPE` to alias 1-D and 2-D views of the same tile (auto-friendly).
4. **Auto-mode compatibility** — Manual-only when used as-is (Known). It calls `pipe_barrier(PIPE_V)` directly and `*_IMPL` variants — both are library-developer-only per [docs/auto_mode/Library_Developer_Rules_And_Limitations.md §3, §5](../docs/auto_mode/Library_Developer_Rules_And_Limitations.md). Also relies on the FA driver kernel that is **not** in `ALL_TESTCASES` (Known: missing from [tests/npu/a2a3/src/st/testcase/CMakeLists.txt](../tests/npu/a2a3/src/st/testcase/CMakeLists.txt)).
5. **Copy** — The math/operation sequence; the `TRESHAPE` view-aliasing for column-vs-1D access; the FA-2.0 init vs non-init split.
6. **Do not copy** — `pipe_barrier(PIPE_V)` calls and `TROWEXPANDSUB_IMPL` from this file. Use the user-facing `TROWEXPAND + TSUB` instead per [docs/coding/tutorials/row-softmax.md](../docs/coding/tutorials/row-softmax.md). Note: `__in__`/`__out__` qualifiers ARE meaningful on A3/A5 (compiler-provided keywords; only `#define`d as empty for kirin, CPU-sim, and cost-model — see [qualifier_reference.md](qualifier_reference.md)); keep them on tile-function parameter declarations.
7. **Confidence** — Medium for math; Low for direct code reuse in auto mode.

### B4. CPU FA demo (host-runnable functional reference)

1. **File** — [demos/cpu/flash_attention_demo/flash_attention_demo.cpp](../demos/cpu/flash_attention_demo/flash_attention_demo.cpp)
2. **Why** — Clean end-to-end FA pattern designed for the CPU simulator backend. No NPU pipelining concerns; useful for verifying intended semantics before targeting auto mode.
3. **Pattern** — FA-style attention against the `__CPU_SIM` backend.
4. **Auto-mode compatibility** — Manual-only as-is (Inferred — built in `tests/cpu/...` style, not under `--cce-enable-pto-passes`).
5. **Copy** — Math layout, Q/K/V shape conventions, reference outputs.
6. **Do not copy** — Backend-specific includes/macros that only fire under `__CPU_SIM`.
7. **Confidence** — Medium.

---

## Group C — Manual-mode references (semantic source only; do not copy code)

### C1. Manual A3 GEMM performance kernel

1. **File** — [kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp](../kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp)
2. **Why** — Hand-tuned high-performance GEMM with explicit L1/L0 ping-pong buffers and per-pipe events. Authoritative for what a fully tuned A3 GEMM looks like.
3. **Pattern** — `BUFFER_NUM = 2` ping-pong on L1/L0; explicit `set_flag`/`wait_flag` (NOT guarded by `__PTO_AUTO__`); manual `TASSIGN(aTile[0], 0x0); TASSIGN(aTile[1], 0x0 + L0_PINGPONG_BYTES);` to pin pong buffers; `block_idx`-based partitioning.
4. **Auto-mode compatibility** — Manual-only (Known: extensive ungauarded `set_flag`/`wait_flag`; double buffering, which [docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md §1.4](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md) explicitly tells kernel devs to avoid in auto mode today).
5. **Copy** — High-level **partitioning math** (`mIter`, `nIter`, gmOffsetA/B/C); the `MatmulAcc(cTile, aTile, bTile, k)` helper that branches `if (k == 0) TMATMUL else TMATMUL_ACC`.
6. **Do not copy** — Ping-pong buffer pairs `aMatTile[BUFFER_NUM]` plus their hard-coded byte offsets; manual `set_flag`/`wait_flag`; `getPingPong()` static state. None of this works under `--cce-enable-pto-passes`.
7. **Confidence** — High.

### C2. Manual cross-platform Flash Attention

1. **File** — [kernels/manual/common/flash_atten/fa_performance_kernel.cpp](../kernels/manual/common/flash_atten/fa_performance_kernel.cpp) (1014 lines)
2. **Why** — Most complete FA reference (A2/A3/A5 per [kernels/manual/README.md](../kernels/manual/README.md)). Includes pipeline diagrams (`fa_pipeline*.svg`) and macro helpers `pto_macro_fa_gu.hpp`, `pto_macro_fa_softmax.hpp`, `pto_macro_matmul.hpp`.
3. **Pattern** — Multi-buffer FFTS pipeline using `TPipe`, `TPUSH`, `TPOP` (Known: see [tests/npu/a2a3/src/st/testcase/tfa/pto_macro_matmul.hpp](../tests/npu/a2a3/src/st/testcase/tfa/pto_macro_matmul.hpp)). Manual `set_flag`/`wait_flag` between MTE2/MTE1/M/FIX/V pipes; raw `__ca__ half *` casts to literal addresses.
4. **Auto-mode compatibility** — Manual-only (Known). `TPUSH`/`TPOP` are explicitly excluded from auto-mode builds in [tests/npu/a2a3/src/st/testcase/CMakeLists.txt:213-220](../tests/npu/a2a3/src/st/testcase/CMakeLists.txt#L213-L220) and [tests/npu/a5/src/st/testcase/CMakeLists.txt:243-250](../tests/npu/a5/src/st/testcase/CMakeLists.txt#L243-L250).
5. **Copy** — The FA-2.0 stage decomposition (`compute_qk → compute_p → compute_pv → compute_gu`) as documented in [tests/npu/a2a3/src/st/testcase/tfa/TFA_kernel.md](../tests/npu/a2a3/src/st/testcase/tfa/TFA_kernel.md). Use it for design notes only.
6. **Do not copy** — `TPipe`/`TPUSH`/`TPOP`; raw L0 buffer casts; `set_flag`/`wait_flag`; double buffering; `wait_flag_dev(...)` FFTS handshakes.
7. **Confidence** — High.

### C3. A5 Flash Attention manual

1. **File** — [kernels/manual/a5/flash_atten/fa_performance_kernel.cpp](../kernels/manual/a5/flash_atten/fa_performance_kernel.cpp) (1239 lines) plus `fa_performance_dn_kernel.cpp` and `pto_macro_fa_dn_*.hpp`.
2. **Why** — Same as C2 but A5-specific, including a "dn" (different-network) variant.
3. **Pattern** — As C2, plus A5 mxN intrinsics and dn-specific macros.
4. **Auto-mode compatibility** — Manual-only.
5. **Copy** — Stage decomposition; A5 fractal layout choices.
6. **Do not copy** — Same as C2.
7. **Confidence** — High.

### C4. tfa ST kernel (orphaned testcase)

1. **File** — [tests/npu/a2a3/src/st/testcase/tfa/tfa_kernel.cpp](../tests/npu/a2a3/src/st/testcase/tfa/tfa_kernel.cpp) (788 lines)
2. **Why** — Most complete in-tree FA implementation that ALSO documents stages (`compute_qk`, `compute_p`, `compute_pv`, `compute_gu`) with a math companion in `TFA_kernel.md`. Note: NOT in `ALL_TESTCASES` (Known), so it neither builds nor runs in CI today.
3. **Pattern** — Cube/Vec subcore split (`__DAV_CUBE__` / `__DAV_VEC__`); FFTS device-side flags via `wait_flag_dev`; `TPipe` ping-pong slots; `assign_tile_buffers` helper that loops over `TASSIGN(tiles[idx], offset)`.
4. **Auto-mode compatibility** — Manual-only (Known: heavy `set_flag`/`wait_flag` and `TASSIGN` arithmetic, no `__PTO_AUTO__` guards anywhere).
5. **Copy** — The math and shape conventions; the buffer-allocation accounting via `tile_storage_bytes<TileType>()` constexpr (auto-mode kernels can keep this constexpr math even though `TASSIGN` becomes a no-op).
6. **Do not copy** — `assign_tile_buffers`'s `TASSIGN` loop, `wait_flag_dev`, `TPipe`/`TPUSH`/`TPOP`, sub-core FFTS handshake protocol.
7. **Confidence** — High.

---

## Group D — Library helpers (read for semantics, not as kernel templates)

### D1. TPow library implementation

1. **File** — [include/pto/npu/a2a3/TPow.hpp](../include/pto/npu/a2a3/TPow.hpp), [include/pto/npu/a5/TPow.hpp](../include/pto/npu/a5/TPow.hpp)
2. **Why** — Reference for an op whose **kernel** ([tests/npu/a2a3/src/st/testcase/tpow/tpow_kernel.cpp](../tests/npu/a2a3/src/st/testcase/tpow/tpow_kernel.cpp)) is auto-compatible. Library code uses `pipe_barrier(PIPE_V)` and `PTO_INTERNAL` correctly per [docs/auto_mode/Library_Developer_Rules_And_Limitations.md §3, §5](../docs/auto_mode/Library_Developer_Rules_And_Limitations.md).
3. **Pattern** — Branchy library helper that handles integer/float/inf/NaN exponents, then composes existing PTO primitives.
4. **Auto-mode compatibility** — Yes for kernel use; the IMPL itself is library-side (`PTO_INTERNAL`, with `pipe_barrier` allowed because it is inside a tile-function-equivalent context).
5. **Copy** — `IsInteger`/`IsInf` utility patterns; the `TPOW_IMPL(dst, base, exp, tmp)` four-argument shape (matches the user-facing wrapper requiring a `tmpTile`).
6. **Do not copy** — `pipe_barrier(PIPE_V)` into kernel code.
7. **Confidence** — Medium (Tier-1 for understanding the op contract; not a kernel template).

### D2. TQuant library implementation — auto-mode-aware aliasing

1. **File** — [include/pto/npu/a2a3/TQuant.hpp](../include/pto/npu/a2a3/TQuant.hpp), [include/pto/npu/a5/TQuant.hpp](../include/pto/npu/a5/TQuant.hpp)
2. **Why** — Concrete example of an `_IMPL` that **was modified for auto mode**. Lines [108-114](../include/pto/npu/a2a3/TQuant.hpp#L108-L114) explicitly branch:

   ```cpp
   #ifndef __PTO_AUTO__
       TASSIGN_IMPL(src_f16, reinterpret_cast<uintptr_t>(src.data()));
       TASSIGN_IMPL(src_s32, reinterpret_cast<uintptr_t>(src.data()));
   #else
       TRESHAPE_IMPL(src_f16, src);
       TRESHAPE_IMPL(src_s32, src);
   #endif
   ```

   This is the **canonical pattern** for upgrading a `TASSIGN`-based aliasing IMPL to be auto-mode-safe (Known: matches [docs/auto_mode/Library_Developer_Rules_And_Limitations.md §4](../docs/auto_mode/Library_Developer_Rules_And_Limitations.md)).
3. **Pattern** — Multi-stage `TROWEXPANDMUL_IMPL → optional TROWEXPANDADD_IMPL → TCVT_IMPL (fp32→s32) → TQuantCvtS32ToFp16 → TCVT_IMPL (fp16→int8)` with `pipe_barrier(PIPE_V)` between stages; aliasing of an in-place buffer to multiple element types via `TRESHAPE` in auto mode.
4. **Auto-mode compatibility** — Yes for the IMPL (auto-mode branch is present).
5. **Copy** — The exact `#ifndef __PTO_AUTO__ / TASSIGN_IMPL ... / #else / TRESHAPE_IMPL ... / #endif` pattern when porting any other `_IMPL` that needs in-place type re-views. Also: `static_assert` on dst/src/scale dtypes.
6. **Do not copy** — `pipe_barrier(PIPE_V)` into kernel-level code; do not call `TQUANT_IMPL` directly from a kernel — call `TQUANT<...>(dst, src, scale[, offset])` per the user wrapper.
7. **Confidence** — High (the file shows the pattern, in-tree).

### D3. TQuant ST kernel — example of an auto-mode-risky kernel pattern

1. **File** — [tests/npu/a2a3/src/st/testcase/tquant/tquant_kernel.cpp](../tests/npu/a2a3/src/st/testcase/tquant/tquant_kernel.cpp)
2. **Why** — `tquant` IS in `ALL_TESTCASES` (Known), so it builds in auto mode, **but** the same kernel does:

   ```cpp
   TASSIGN(srcTile, 0x0);
   TASSIGN(dstS8Tile, 0x0);     // same address as srcTile
   ```

   In auto mode, `TASSIGN` is a no-op and these two tiles are independent allocations. In manual mode, the manual aliasing trick (overlap by giving the same address) is being used. This is exactly the kind of subtle "works in both modes via different mechanisms" pattern called out in [docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md §2.3](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md) ("a tile's memory shouldn't change at runtime in auto mode").
3. **Pattern** — Quant kernel skeleton with scale/offset tiles; `TLOAD(srcTile)` + `TLOAD(scaleTile)` + `TQUANT<QuantType::INT8_SYM>(dst, src, scale)`; `#ifndef __PTO_AUTO__` guards around `set_flag`/`wait_flag`.
4. **Auto-mode compatibility** — Mostly (Inferred). It compiles per the test list, but the `TASSIGN(..., 0x0)` aliasing trick has manual-mode semantics that are silently dropped in auto mode. Verify the kernel still computes correctly under auto mode before treating it as a clean reference.
5. **Copy** — Quant call shape `TQUANT<pto::QuantType::INT8_SYM, DstTile, SrcTile, ParaTile>(dstS8Tile, srcTile, scaleTile)`; per-row scale tile shape `Tile<TileType::Vec, float, validRows, 1, BLayout::ColMajor, -1, -1>` with `Layout::DN` stride on `GlobalTensor`; the `PTO_CEIL`-style col padding to `BLOCK_BYTE_SIZE / sizeof(T)`.
6. **Do not copy** — Two `TASSIGN(...)` calls with the same address (`0x0` for both `srcTile` and `dstS8Tile`). If you want aliasing, use `TRESHAPE`/`TSUBVIEW` per [docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md §2.1, §2.2](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md).
7. **Confidence** — Medium.

### D4. TDequant ST kernel — example of likely-redundant `TLOAD` on dst

1. **File** — [tests/npu/a2a3/src/st/testcase/tdequant/tdequant_kernel.cpp](../tests/npu/a2a3/src/st/testcase/tdequant/tdequant_kernel.cpp)
2. **Why** — `tdequant` IS in `ALL_TESTCASES`, **but** [line 50](../tests/npu/a2a3/src/st/testcase/tdequant/tdequant_kernel.cpp#L50) does `TLOAD(dstTile, dstGlobal)` on a tile that is **only an output** of `TDEQUANT`. This is exactly the "redundant `TLOAD` on dst" anti-pattern from [docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md §3.1](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md), which warns about data races in auto mode.
3. **Pattern** — Standard dequant skeleton; `TASSIGN` with computed offsets via `size_t srcOffset = ...`.
4. **Auto-mode compatibility** — Problematic (Inferred). Builds per the test list, but the `TLOAD(dstTile, ...)` is an explicitly-warned-against pattern. Either the test happens to pass because the compiler does not coalesce `srcTile` and `dstTile` here, or the result is silently incorrect — verify with the user before reusing.
5. **Copy** — The scale/offset GM stride conventions; the `Tile<TileType::Vec, ...>` declarations.
6. **Do not copy** — `TLOAD(dstTile, dstGlobal)` for write-only outputs. If you need to initialize a dst tile, use a deliberate value (e.g., `TFILLPAD`) or skip it.
7. **Confidence** — Medium.

---

## Quick lookup by kernel family

| Family | Tier 1 (copy) | Pattern only | Manual-only |
|---|---|---|---|
| Elementwise | A1 add demo, A2 tadd, A5 taxpy | A9 docs example | — |
| Power / scalar-broadcast | A3 tpow | D1 TPow library | — |
| Reductions | A4 trowsum | B1 row-softmax tutorial | — |
| GEMM | A6 tmatmul A3, A7 tmatmul A5, A8 tmatmul_mx A5 | B2 GEMM tutorial | C1 gemm_performance |
| Softmax | (none in-tree fully auto) | B1 row-softmax, B3 fa_softmax math | — |
| Attention | (none in-tree fully auto) | B3 fa_softmax, B4 cpu FA | C2 common FA, C3 A5 FA, C4 tfa ST |
| Quant / Dequant | D2 TQuant library auto branch | — | D3 tquant ST aliasing trick, D4 tdequant TLOAD-on-dst |
| Aliasing recipes | A10 dual-mode `TASSIGN`+`TRESHAPE` (PR-852, not yet merged) | — | — |

---

## Cross-cutting risks observed (link back when reviewing)

- **`__tf__` IS meaningful on A3/A5** (Known). The repo `#define`s `__tf__` as empty only for kirin ([include/pto/common/arch_macro.hpp:29-34](../include/pto/common/arch_macro.hpp#L29-L34)), CPU-sim ([include/pto/common/cpu_stub.hpp:34](../include/pto/common/cpu_stub.hpp#L34)), and cost-model ([include/pto/costmodel/common/qualifiers.hpp:27](../include/pto/costmodel/common/qualifiers.hpp#L27)). On A3/A5 it is a bisheng-CCE keyword, used pervasively in [include/pto/npu/a2a3/](../include/pto/npu/a2a3/) and [include/pto/npu/a5/](../include/pto/npu/a5/). Add `__tf__` on helpers that contain raw CCE intrinsics; do not add it to user-facing kernel entries. See [qualifier_reference.md](qualifier_reference.md).
- **`__in__`/`__out__` ARE meaningful on A3/A5** (Inferred). Same `#define`-as-empty pattern in arch_macro.hpp / cpu_stub.hpp / qualifiers.hpp. The library spec ([docs/auto_mode/Library_Developer_Rules_And_Limitations.md §6](../docs/auto_mode/Library_Developer_Rules_And_Limitations.md)) requires them on `TileDType` parameters of tile functions. Preserve when copying helper signatures.
- **`TPUSH` / `TPOP` are not safe in auto mode** ([tests/npu/a2a3/src/st/testcase/CMakeLists.txt:213-220](../tests/npu/a2a3/src/st/testcase/CMakeLists.txt#L213-L220), [docs/auto_mode/Library_Developer_Rules_And_Limitations.md §4](../docs/auto_mode/Library_Developer_Rules_And_Limitations.md)). Avoid in any auto-mode kernel.
- **Double buffering is not supported for kernel devs today** ([docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md §1.4](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md)). Do not transplant ping-pong buffer logic from C1/C2/C3.
- **`set_flag` / `wait_flag` / `Event<>` from manual kernels** must be either dropped or wrapped in `#ifndef __PTO_AUTO__` (canonical guard pattern: A4, A6, A7, D3).
- **Aliasing**: in auto mode, `TASSIGN(a, addr)` followed by `TASSIGN(b, addr)` does NOT alias `a` and `b`. Use `TRESHAPE(b, a)` (same base) or `TSUBVIEW(b, a, row, col)` (offset). Examples: A4 (`TRESHAPE`), D2 (auto branch), A10 (dual-mode recipe from PR-852, not yet merged).
- **Inside `__tf__` bodies, sync rules invert** — use raw `set_flag`/`wait_flag`/`pipe_barrier` (or guard `PtoSetWaitFlag` with `#ifndef __PTO_AUTO__` and emit raw flags in the `#else`). Auto-sync does NOT walk into tile functions, so `PtoSetWaitFlag` becoming a no-op silently drops sync. See [auto_mode_bad_patterns.md §2.7](auto_mode_bad_patterns.md) and [external_context/pr_852_notes.md](external_context/pr_852_notes.md).
- **Several existing-source bug patterns are scheduled for fix in PR-852** (not yet merged). See [external_context/pr_852_notes.md](external_context/pr_852_notes.md) for the per-file mapping. Until then, the current source still has the bugs and the entries in [auto_mode_bad_patterns.md](auto_mode_bad_patterns.md) still apply.

---

## Open items

- Verify that `tquant_kernel.cpp` and `tdequant_kernel.cpp` compute correctly under `AUTO_MODE=ON`. Both rely on patterns the auto-mode rules warn against; "passes the build" is not "produces correct output". (Unknown.)
- Verify that A5 `tmatmul_mx` does not depend on manual-mode-only L0 layout tricks. (Unknown — file not yet read end-to-end.)
- ~~Confirm whether `__in__`/`__out__` are silently no-ops on A3/A5~~ — Resolved. They are bisheng-CCE keywords on A3/A5 and `#define`d empty only for kirin / CPU-sim / cost-model. See [qualifier_reference.md](qualifier_reference.md).
- Confirm whether the orphaned `tests/npu/a2a3/src/st/testcase/tfa/` will be re-listed in `ALL_TESTCASES` for either mode. (Unknown.)
