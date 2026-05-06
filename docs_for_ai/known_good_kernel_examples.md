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
