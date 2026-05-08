# Assumptions to Verify (auto-mode A3/A5)

Open assumptions, unknowns, and verification questions extracted from the
existing knowledge base. Used to drive the next round of source audits and
user-confirmation requests before generating kernels.

Sources synthesized:
- [auto_mode_bad_patterns.md](auto_mode_bad_patterns.md)
- [tile_type_reference.md](tile_type_reference.md)
- [a3_a5_differences.md](a3_a5_differences.md)
- [external_context/pr_852_notes.md](external_context/pr_852_notes.md)

Each entry uses:
1. Question or assumption
2. Why it matters
3. Current evidence
4. Affected platform: A3 / A5 / both / unknown
5. Priority: high / medium / low
6. Suggested next action
7. Status: `Known` / `Inferred` / `Assumption` / `Unknown`

---

## 1. Compiler / auto-mode support

### 1.1 `tile_size(N)` is a bisheng-CCE compiler keyword consuming an element count

1. Assumption: `tile_size(...)` is a CCE builtin/keyword (not `#define`d in repo) that takes an **element count**, not bytes, and produces a sized vector type when applied to a memory-qualified element type.
2. Why it matters — drives the `TileDType` shape on A3/A5 in auto mode ([tile_type_reference.md §1.1, §1.2](tile_type_reference.md)). If we're wrong about element-vs-byte semantics, every `Tile`/`ConvTile` allocation in auto mode is mis-sized.
3. Evidence — used unconditionally at [include/pto/common/pto_tile.hpp:1530-1540, 1166-1170](../include/pto/common/pto_tile.hpp#L1530-L1540); not `#define`d in any repo header (verified by `grep`). PR-852 description for `ConvTile` ([§T4b](external_context/pr_852_notes.md)) implicitly treats it as an element count: passing `bytes` "inflates the UB allocation by `sizeof(T)`".
4. Platform — both.
5. Priority — high.
6. Next action — confirm with the user / bisheng-CCE compiler docs that `tile_size(N)` consumes an element count and is a backend keyword.
7. Status — Assumption.

### 1.2 `__cce_tinit(...)` is a CCE builtin used to dummy-initialize `TileDType` against SROA undef

1. Assumption: `__cce_tinit` is a CCE backend builtin (not repo-defined) used at constructor sites in `pto_tile.hpp` to suppress SROA-pass undefs.
2. Why — anything we generate that constructs a `Tile` ends up calling this; we should not add other constructor shapes that bypass it.
3. Evidence — appears at [pto_tile.hpp:1135, 1454, 1464, 1475, 1485](../include/pto/common/pto_tile.hpp#L1135); not `#define`d anywhere in repo ([tile_type_reference.md §12](tile_type_reference.md)).
4. Platform — both.
5. Priority — medium.
6. Next action — confirm builtin status; capture in `qualifier_reference.md` once verified.
7. Status — Assumption.

### 1.3 Whether kernel-scope `pipe_barrier(PIPE_ALL)` actually compiles under `--cce-enable-pto-passes`

1. Question — does the auto-mode toolchain accept `pipe_barrier(PIPE_ALL)` at kernel scope (outside any `__tf__`, outside `#ifndef __PTO_AUTO__`), or is it silently a no-op, or does it fail to compile?
2. Why — `tcolexpandadd/div/max/mul/sub_kernel.cpp` all leave `pipe_barrier(PIPE_ALL)` outside the `#ifndef __PTO_AUTO__` block while still being listed in `ALL_TESTCASES` ([auto_mode_bad_patterns.md §2.2](auto_mode_bad_patterns.md)). Either CCE tolerates it, or these tests are passing for the wrong reason.
3. Evidence — five testcases in `ALL_TESTCASES` exhibit the pattern; rules doc explicitly forbids it ([Kernel rules §3.2, §3.3](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md)).
4. Platform — A3 (the listed kernels live under `tests/npu/a2a3/`); A5 status not checked.
5. Priority — high (resolves a class of "is this rule strict?" questions).
6. Next action — request a build log of one of these testcases under `--cce-enable-pto-passes -O2 __PTO_AUTO__` to confirm; if it compiles, decide whether to relax the rule or treat as accidentally-tolerated.
7. Status — Unknown.

### 1.4 Whether a vector-aware overload of `__cce_get_tile_ptr(tile.data())` exists in auto mode

1. Question — `tload_gm2mat` (A3) and `tload_shape2d` (A5) both pass `tile.data()` into `__cce_get_tile_ptr` from kernel code, yet are in `ALL_TESTCASES`. Is there a vector-overload that handles this, or do they pass for a coincidental reason (or get excluded silently)?
2. Why — if such an overload exists, the `__cce_get_tile_ptr(tile.data())` pattern is not categorically broken in auto mode and several entries in `auto_mode_bad_patterns.md` need softening.
3. Evidence — [tests/npu/a2a3/src/st/testcase/tload_gm2mat/tload_gm2mat_kernel.cpp:23,31](../tests/npu/a2a3/src/st/testcase/tload_gm2mat/tload_gm2mat_kernel.cpp#L23), [tests/npu/a5/src/st/testcase/tload_shape2d/tload_shape2d_kernel.cpp:20](../tests/npu/a5/src/st/testcase/tload_shape2d/tload_shape2d_kernel.cpp#L20). `texpands_mat` is excluded for the same construct ([CMakeLists.txt:213-220](../tests/npu/a2a3/src/st/testcase/CMakeLists.txt#L213-L220)).
4. Platform — both.
5. Priority — high.
6. Next action — request a build log; grep `include/pto/` for any `__cce_get_tile_ptr` overload set; ask the user.
7. Status — Unknown.

### 1.5 Whether `SetValidRow` / `SetValidCol` / `SetValidShape` interact safely with auto-sync inside a `__tf__`

1. Question — the header comment says PIPE_S sync is required when calling these. Does that requirement co-exist with auto-mode sync analysis when the call sits inside a `__tf__` body?
2. Why — dynamic valid regions are common in tile-loop kernels; if `Set*` is unsafe inside `__tf__`, all dynamic-shape tile setup must happen in kernel code, not helpers.
3. Evidence — [pto_tile.hpp:1608, 1616, 1624](../include/pto/common/pto_tile.hpp#L1608) ("Call this function need PIPE_S wait"); existing test kernels prefer constructor-time supply ([trowsum_kernel.cpp:31](../tests/npu/a2a3/src/st/testcase/trowsum/trowsum_kernel.cpp#L31)).
4. Platform — both.
5. Priority — medium.
6. Next action — ask the user; alternatively scan for a `__tf__` helper that calls `SetValid*` and check the build list.
7. Status — Unknown.

---

## 2. A3 vs A5 differences

### 2.1 A5 cube compile flag

1. Assumption — A5 cube target uses `--cce-aicore-arch=dav-c310-cube` (parallel to confirmed `dav-c310-vec`).
2. Why — needed to write any A5 cube CMake target or to interpret A5 GEMM build configs.
3. Evidence — A5 vec is `dav-c310-vec` (Inferred from [docs/auto_mode/Auto_Mode_Overview.md](../docs/auto_mode/Auto_Mode_Overview.md)); cube flag is not visible in [tests/npu/a5/src/st/testcase/CMakeLists.txt](../tests/npu/a5/src/st/testcase/CMakeLists.txt) ([a3_a5_differences.md §1](a3_a5_differences.md)).
4. Platform — A5.
5. Priority — medium.
6. Next action — read A5 CMakeLists end-to-end; confirm with the user.
7. Status — Assumption.

### 2.2 A5 cube macro name (analogue of `__DAV_C220_CUBE__`)

1. Question — A3 gates `Bias` storage on `__DAV_C220_CUBE__` ([memory.hpp:74](../include/pto/common/memory.hpp#L74)); what is the A5 equivalent that prevents `MemoryQualifier<Bias, T>::type` falling back to `uint64_t`?
2. Why — without the right macro, an A5 cube kernel silently gets `uint64_t` bias storage (Group 5.3 hazard).
3. Evidence — [memory.hpp:73-83](../include/pto/common/memory.hpp#L73-L83) only shows the A3 macro ([a3_a5_differences.md §6](a3_a5_differences.md), [auto_mode_bad_patterns.md §5.3](auto_mode_bad_patterns.md)).
4. Platform — A5.
5. Priority — high (silent fallback to wrong type is dangerous).
6. Next action — grep `include/pto/common/` and `include/pto/npu/a5/` for `__DAV_C310_CUBE__` or similar; ask the user.
7. Status — Unknown.

### 2.3 Why A5 `Tile<TileType::Bias>::TileDType` skips `tile_size(...)`

1. Question — `pto_tile.hpp:1534-1536` has an A5-only `std::conditional_t` that drops the `tile_size(Rows*Cols)` modifier for `Bias`. What is the A5 backend reason, and what storage shape does it use instead?
2. Why — informs whether bias buffers can be sized dynamically on A5 and whether any helper must avoid `tile_size`-aware code paths for `Bias`.
3. Evidence — [pto_tile.hpp:1530-1540](../include/pto/common/pto_tile.hpp#L1530-L1540) ([a3_a5_differences.md §4.4](a3_a5_differences.md)).
4. Platform — A5.
5. Priority — medium.
6. Next action — bisheng-CCE A5 docs; the user.
7. Status — Inferred (A5 cube exposes a fixed-size construct; not verified).

### 2.4 `TileLeft` BLayout split — does `TMOV` cope when `TileMatA` is declared `BLayout::ColMajor` on both archs?

1. Question — `TileLeft<...>` is `BLayout::RowMajor` on A3 and `BLayout::ColMajor` on A5 ([a3_a5_differences.md §4.1](a3_a5_differences.md), [pto_tile.hpp:1681-1699](../include/pto/common/pto_tile.hpp#L1681-L1699)). Yet both A3 and A5 in-tree `tmatmul_kernel.cpp` declare `TileMatA` as `BLayout::ColMajor`. Does the per-arch `TMOV` dispatch handle the layout swap correctly in both directions?
2. Why — porting a GEMM design between archs depends on this. If `TMOV` does the swap implicitly, copying `TileMatA = BLayout::ColMajor` is portable. If not, A3 needs `BLayout::RowMajor` for `TileMatA`.
3. Evidence — [a3 tmatmul:51](../tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp#L51), [a5 tmatmul:53](../tests/npu/a5/src/st/testcase/tmatmul/tmatmul_kernel.cpp#L53); per-arch dispatch via [pto_instr_impl.hpp](../include/pto/common/pto_instr_impl.hpp).
4. Platform — both.
5. Priority — high (load-bearing for any cross-arch GEMM design).
6. Next action — read `pto_instr_impl.hpp` `TMOV` dispatch; ideally request a build log for one A3 and one A5 GEMM with both layout choices.
7. Status — Inferred.

### 2.5 Whether A5 ST testcases have an analogous "auto-mode failures" surface

1. Question — PR-852 was triggered by failing A3 ST tests under auto mode. Has the A5 ST set been run under auto mode, and what is the failure surface?
2. Why — drives whether we should plan an A5-side audit equivalent to PR-852.
3. Evidence — PR-852 scope statement says A3-only ([external_context/pr_852_notes.md "Scope"](external_context/pr_852_notes.md), [a3_a5_differences.md §11](a3_a5_differences.md)).
4. Platform — A5.
5. Priority — medium.
6. Next action — ask the user; if no A5 auto-mode test sweep exists, recommend one.
7. Status — Unknown.

---

## 3. Tile / memory / aliasing

### 3.1 `tquant_kernel.cpp` numerical correctness in auto mode

1. Question — does the kernel produce correct numbers in auto mode, despite using overlapping `TASSIGN(srcTile, 0x0); TASSIGN(dstS8Tile, 0x0)` (manual-mode aliasing trick)?
2. Why — `tquant` is in `ALL_TESTCASES` and would be used as a positive auto-mode example. If results are wrong, it must be downgraded to "builds, but mixed".
3. Evidence — [tests/npu/a2a3/src/st/testcase/tquant/tquant_kernel.cpp:45-47](../tests/npu/a2a3/src/st/testcase/tquant/tquant_kernel.cpp#L45-L47); flagged in [auto_mode_bad_patterns.md §1.1, §6.1](auto_mode_bad_patterns.md).
4. Platform — A3.
5. Priority — high.
6. Next action — request a run + numerical-diff log under auto mode; cross-check whether `TQuant.hpp`'s `__PTO_AUTO__` `TRESHAPE_IMPL` branch covers the aliasing.
7. Status — Inferred risk; Unknown empirically.

### 3.2 `tdequant_kernel.cpp` numerical correctness — redundant `TLOAD` on a write-only dst

1. Question — does the kernel compute correct numbers despite `TLOAD(dstTile, dstGlobal)` on a pure-output tile?
2. Why — same as 3.1; ALL_TESTCASES inclusion is build-only evidence.
3. Evidence — [tests/npu/a2a3/src/st/testcase/tdequant/tdequant_kernel.cpp:50](../tests/npu/a2a3/src/st/testcase/tdequant/tdequant_kernel.cpp#L50); [auto_mode_bad_patterns.md §1.5, §6.2](auto_mode_bad_patterns.md).
4. Platform — A3.
5. Priority — high.
6. Next action — same as 3.1.
7. Status — Inferred risk; Unknown empirically.

### 3.3 `ConvTile` dynamic-shape support in auto mode

1. Question — auto mode's `GetShape(dim)` returns `staticShape[dim]` only ([pto_tile.hpp:1124-1129](../include/pto/common/pto_tile.hpp#L1124-L1129)). Some manual-mode tests (e.g., dynamic `srcG`/`srcN` in `TTrans*`) rely on the runtime branch. Are any of those kernels auto-mode-eligible without rewriting to static shapes?
2. Why — bounds the kernel-design space for conv/fa kernels using `ConvTile`.
3. Evidence — [tile_type_reference.md §1.2, §12](tile_type_reference.md), [include/pto/npu/a2a3/TTrans.hpp](../include/pto/npu/a2a3/TTrans.hpp).
4. Platform — both (`ConvTile` definition shared; A5 has extra `dstStride_`/`dstMposition_` fields per [a3_a5_differences.md §4.6](a3_a5_differences.md)).
5. Priority — medium.
6. Next action — read `TTrans*` end-to-end for the dynamic vs static branches; identify which kernels break in auto mode.
7. Status — Unknown.

### 3.4 `Tile<TileType::Bias, ...>` silently falling back to `uint64_t`

1. Assumption — on a build without `__DAV_C220_CUBE__` (or the A5 equivalent — see 2.2), declaring a `Tile<TileType::Bias, ...>` returns a `uint64_t` storage type silently.
2. Why — silent type degradation is a class of bug invisible at the call site.
3. Evidence — [memory.hpp:73-83](../include/pto/common/memory.hpp#L73-L83); [auto_mode_bad_patterns.md §5.3](auto_mode_bad_patterns.md), [tile_type_reference.md §11 item 13](tile_type_reference.md).
4. Platform — both, with A5 macro Unknown (see 2.2).
5. Priority — medium.
6. Next action — confirm via a small build with cube macro undefined; document the right `#if` guard for kernels.
7. Status — Inferred.

### 3.5 Post-PR-852 `TRESHAPE` silently allows mismatched `TileType`

1. Assumption — after PR-852 merges, `TRESHAPE_IMPL` drops the `Loc == NewLoc` assert in auto mode ([external_context/pr_852_notes.md §L5](external_context/pr_852_notes.md)). Mismatched dst/src `TileType` then compiles silently.
2. Why — a soft gotcha for any kernel that uses `TRESHAPE` to alias between storage classes.
3. Evidence — PR-852 description and [auto_mode_bad_patterns.md §1.1 note](auto_mode_bad_patterns.md), [tile_type_reference.md §10.1](tile_type_reference.md).
4. Platform — A3 (post-merge); A5 mirror Unknown.
5. Priority — low (only matters once PR-852 lands).
6. Next action — once merged, add a check in code review for `TRESHAPE(a, b)` where `a.Loc != b.Loc`.
7. Status — Inferred (post-merge).

---

## 4. Qualifier / `__tf__` / `TileDType` rules

### 4.1 `tile_size(N)` element-count semantics

See 1.1. Reiterated here because it is the load-bearing assumption behind the `ConvTile<..., bufferSize, ...>` element-vs-byte rule ([auto_mode_bad_patterns.md §5.6](auto_mode_bad_patterns.md), [tile_type_reference.md §10.2](tile_type_reference.md)).

### 4.2 `__cce_get_tile_ptr` accepts only the bare `TileDType` (not vector arithmetic on it)

1. Assumption — extraction-then-arithmetic is the rule: `__cce_get_tile_ptr(tmp) + N`. Doing `__cce_get_tile_ptr(tmp + N)` crashes the libexpand pass during RAUW.
2. Why — drives every `__tf__` body that indexes into a parameter.
3. Evidence — PR-852 description ([§L1](external_context/pr_852_notes.md)) plus the eight rewrite sites in [include/pto/npu/a2a3/TCI.hpp:59,108,147,149,151,203](../include/pto/npu/a2a3/TCI.hpp#L59); [auto_mode_bad_patterns.md §3.2.1](auto_mode_bad_patterns.md).
4. Platform — A3 (verified PR scope); A5 spot-check negative ([a3_a5_differences.md §11](a3_a5_differences.md)) but full audit pending.
5. Priority — high.
6. Next action — promote this rule to its own one-liner in `qualifier_reference.md` once PR-852 merges; full A5 audit.
7. Status — Inferred (very strong); pending PR merge to become Known.

### 4.3 Canonical `__tf__` migration shape: pass `TileDType` by value with `__in__`/`__out__`

1. Assumption — when migrating a helper that contains raw CCE intrinsics from `Tile&` to `__tf__`, the right shape is:
   - parameters typed `typename TileData::TileDType __in__`/`__out__`,
   - `__cce_get_tile_ptr(arg)` directly inside the body (no `.data()`),
   - caller passes `tile.data()` (vector value) at the kernel/library boundary.
2. Why — both PR-852 migrations (`TQuantCvtS32ToFp16` and `TSTORE_MAT2GM_CONVTILE`) follow this exact shape, but the rule's authoritative source is [Library Developer Rules §6](../docs/auto_mode/Library_Developer_Rules_And_Limitations.md), not the PR.
3. Evidence — [external_context/pr_852_notes.md §L4b, §T4a](external_context/pr_852_notes.md); [auto_mode_bad_patterns.md §3.3 fix](auto_mode_bad_patterns.md).
4. Platform — both.
5. Priority — high (this is the recipe for any future `_IMPL`-to-`__tf__` migration).
6. Next action — once PR-852 merges, cite as canonical example in `qualifier_reference.md`.
7. Status — Inferred (will be Known post-merge).

### 4.4 Inside a `__tf__` body, `set_flag`/`wait_flag`/`pipe_barrier` are required even in auto mode

1. Assumption — the auto-mode compiler does not look inside tile functions, so `PtoSetWaitFlag` (a no-op in auto mode) silently drops sync there. Library helpers must use raw CCE sync intrinsics.
2. Why — refines the kernel-rules guidance which says "prefer `PtoSetWaitFlag`" — that guidance applies at kernel scope only.
3. Evidence — [docs/auto_mode/Library_Developer_Rules_And_Limitations.md §3](../docs/auto_mode/Library_Developer_Rules_And_Limitations.md); PR-852 [§L2, §L3, §L6, §L7b](external_context/pr_852_notes.md); [auto_mode_bad_patterns.md §2.7](auto_mode_bad_patterns.md).
4. Platform — A3 (PR scope); A5 spot-check found no `PtoSetWaitFlag` inside tile-function bodies, so possibly already correct or using a different sync style — Unknown.
5. Priority — high.
6. Next action — full A5 audit of `TConcat.hpp`, `TFillPad.hpp`, `TRowReduceIdx.hpp`, `TTrans.hpp` for sync style.
7. Status — Known anti-pattern (A3); Unknown for A5.

---

## 5. Instruction support

### 5.1 A3 `TRSQRT` — exists, composed, or absent?

1. Question — A5 has a dedicated `TRsqrt.hpp`. Does A3 expose `TRSQRT` (composed via `TRECIP` + `TSQRT`), or is the user-API absent on A3?
2. Why — softmax / FA / norm kernels often need rsqrt; if A3 has only the composition, performance and precision differ.
3. Evidence — [include/pto/npu/a5/TRsqrt.hpp](../include/pto/npu/a5/TRsqrt.hpp) exists; A3 has no equivalent file ([a3_a5_differences.md §3.5, §9.1, §12.8](a3_a5_differences.md)).
4. Platform — A3.
5. Priority — medium.
6. Next action — grep [include/pto/common/pto_instr.hpp](../include/pto/common/pto_instr.hpp) and [pto_instr_impl.hpp](../include/pto/common/pto_instr_impl.hpp) for `TRSQRT` declarations and per-arch dispatch.
7. Status — Unknown.

### 5.2 A5 `TSCATTER` semantics divergence from A3

1. Question — A5 `TSCATTER` IMPL body is much larger than A3's (line 136 vs line 56). Does the user-facing `TSCATTER(...)` contract differ at the bit level?
2. Why — sparse-attention and embedding-style kernels rely on scatter semantics being uniform across archs.
3. Evidence — [a3_a5_differences.md §3.2, §8, §12.7](a3_a5_differences.md).
4. Platform — both.
5. Priority — medium.
6. Next action — read [include/pto/npu/a5/TScatter.hpp](../include/pto/npu/a5/TScatter.hpp) end-to-end; compare semantic comment blocks against [a2a3/TScatter.hpp](../include/pto/npu/a2a3/TScatter.hpp).
7. Status — Inferred divergence; Unknown content.

### 5.3 A5 `*_Custom` and `*Hp` helpers — kernel-callable or library-internal?

1. Question — files like [a5/custom/Div754.hpp](../include/pto/npu/a5/custom/Div754.hpp), `TExp_Custom.hpp`, `TFmodRemHp.hpp`, `TLog_Custom.hpp`, `TSqrtHp.hpp`. Are these user-API entry points, or library-internal `_IMPL`-style helpers?
2. Why — determines whether a kernel that needs IEEE-754 division or high-precision math can call them directly in auto mode.
3. Evidence — [a3_a5_differences.md §2, §9.3, §12.9](a3_a5_differences.md).
4. Platform — A5.
5. Priority — medium (gates A5 attention high-precision designs).
6. Next action — read each file's public surface; classify as user wrapper / `_IMPL` / `__tf__` library helper.
7. Status — Unknown.

### 5.4 A5 `MXFP8` quant semantics

1. Question — A5 `QuantType::MXFP8` is unique to A5. Does A3-style `TQUANT<INT8_SYM, ...>` produce identical output bit-for-bit on A3 and A5, given the larger A5 quant header and helpers like `AbsReduceMax_Naive`?
2. Why — affects whether A5 INT8 quant kernels are drop-in for A3 INT8 kernels.
3. Evidence — [a5/TQuant.hpp:23-28](../include/pto/npu/a5/TQuant.hpp#L23-L28); [a3_a5_differences.md §3.4, §12.6](a3_a5_differences.md).
4. Platform — both.
5. Priority — low to medium.
6. Next action — compare A3 and A5 `TQUANT_IMPL` bodies for `INT8_SYM` and `INT8_ASYM` paths.
7. Status — Inferred-same; Unknown bit-level.

### 5.5 A5 `PTO_URMA` path effect on auto-mode kernel generation

1. Question — `PTO_URMA_SUPPORTED` is gated to `__NPU_ARCH__ == 3510` only ([arch_macro.hpp:18-20](../include/pto/common/arch_macro.hpp#L18-L20)). Does this affect auto-mode kernel generation, or strictly comm-side ops?
2. Why — if comm-only, we can ignore for kernel work.
3. Evidence — [include/pto/comm/a5/async/](../include/pto/comm/a5/async/); [a3_a5_differences.md §1, §12.10](a3_a5_differences.md).
4. Platform — A5 (3510 only).
5. Priority — low.
6. Next action — confirm comm-only with the user.
7. Status — Inferred (no kernel impact).

### 5.6 No auto-mode-safe equivalent of `TPipe` / `TPUSH` / `TPOP` exists today

1. Known — `TPUSH`/`TPOP` use `TASSIGN` internally and are excluded from auto mode by CMake. There is no sanctioned multi-slot pipeline abstraction in auto mode yet.
2. Why — bounds what FA / GEMM-AR pipelines can express in auto mode; ping-pong is out.
3. Evidence — [tests/npu/a2a3/src/st/testcase/CMakeLists.txt:213-220](../tests/npu/a2a3/src/st/testcase/CMakeLists.txt#L213-L220), [tests/npu/a5/src/st/testcase/CMakeLists.txt:243-250](../tests/npu/a5/src/st/testcase/CMakeLists.txt#L243-L250); [auto_mode_bad_patterns.md §2.4, §2.5](auto_mode_bad_patterns.md).
4. Platform — both.
5. Priority — high (kernel design-space constraint).
6. Next action — track whether a successor abstraction is in flight; ask the user.
7. Status — Known.

---

## 6. GEMM / TMATMUL

### 6.1 `SFractalSize = 512` for non-FP16 element types on A5 MX GEMM

1. Question — A5 `tmatmul_mx_kernel.cpp` uses `512` for both A and B Mat tiles regardless of element width (FP4/FP8/FP16). Is that intentional, or a leftover from the FP16 design?
2. Why — wrong fractal block size silently misaligns tile shapes for FP4/FP8 and may degrade or break GEMM-MX.
3. Evidence — [a5/tmatmul_mx_kernel.cpp:95-97](../tests/npu/a5/src/st/testcase/tmatmul_mx/tmatmul_mx_kernel.cpp#L95-L97); [auto_mode_bad_patterns.md §4.4](auto_mode_bad_patterns.md), [a3_a5_differences.md §4.7, §12.5](a3_a5_differences.md).
4. Platform — A5.
5. Priority — high (gates correctness of GEMM-MX skeletons we generate).
6. Next action — ask the user; cross-check with the bisheng-CCE A5 backend docs for the legal `SFractalSize` set per element type.
7. Status — Unknown.

### 6.2 `TileLeft` BLayout split impact on cross-arch GEMM design

See 2.4. Critical for any GEMM skeleton that targets both archs.

### 6.3 A5 `BiasTile` element type = `OutType` (not bias's own type)

1. Known — A5 declares bias in the matmul output's type with valid col = padded `N`; A3 declares bias in its own type with valid col = `validN` ([a3_a5_differences.md §4.3](a3_a5_differences.md)).
2. Why — porting an A3 GEMM verbatim to A5 produces incorrect bias handling.
3. Evidence — [a3 tmatmul:59](../tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp#L59), [a5 tmatmul:60](../tests/npu/a5/src/st/testcase/tmatmul/tmatmul_kernel.cpp#L60).
4. Platform — both (cross-arch port).
5. Priority — high.
6. Next action — capture in any GEMM skeleton template.
7. Status — Known.

### 6.4 GEMM block alignment formula divergence (FP32 case)

1. Known — A3 `blockAlign = C0_SIZE_BYTE / sizeof(U)` gives `8` for FP32; A5 `(sizeof(T)==1) ? 32 : 16` gives `16` for FP32. They disagree for any 4-byte type.
2. Why — copying A3 formula into A5 GEMM produces wrong alignment.
3. Evidence — [a3 tmatmul:128](../tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp#L128), [a5 tmatmul:32](../tests/npu/a5/src/st/testcase/tmatmul/tmatmul_kernel.cpp#L32); [a3_a5_differences.md §4.2](a3_a5_differences.md), [auto_mode_bad_patterns.md §4.3](auto_mode_bad_patterns.md).
4. Platform — both.
5. Priority — high.
6. Next action — document the per-arch formula in the GEMM skeleton template.
7. Status — Known.

---

## 7. Flash Attention / softmax / reductions

### 7.1 `pto_macro_fa_softmax.hpp` is a manual-mode reference, not an auto-mode helper

1. Known — uses unguarded `pipe_barrier(PIPE_V)` and `*_IMPL` calls; designed for manual mode FA only.
2. Why — a kernel author may include it and call from auto-mode kernel code; that breaks library/kernel boundary rules.
3. Evidence — [tests/npu/a2a3/src/st/testcase/tfa/pto_macro_fa_softmax.hpp](../tests/npu/a2a3/src/st/testcase/tfa/pto_macro_fa_softmax.hpp); [auto_mode_bad_patterns.md §6.5](auto_mode_bad_patterns.md).
4. Platform — A3 (the file lives under a2a3); analogous A5 references not surveyed.
5. Priority — high (gates anyone designing an auto-mode FA softmax pipeline).
6. Next action — when designing auto-mode FA softmax, re-derive the math sequence using user-facing instructions only.
7. Status — Known.

### 7.2 A5 high-precision math (`Div754`, `TExp_Custom`, `TLog_Custom`, `TSqrtHp`, `TFmodRemHp`) — auto-mode usable?

See 5.3. A5-only; whether they are user-callable or `_IMPL`-style is the gating question for any A5 attention kernel that needs high-precision math.

### 7.3 A3 cannot use A5 `Hp` helpers — precision floor for A3 attention

1. Inferred — A3 attention kernels must accept the lower-precision `TEXP`/`TLOG`/`TSQRT` baseline because no `*_Custom`/`Hp` analogue exists; software-precision math built from raw CCE intrinsics would conflict with auto mode unless wrapped in a `__tf__` helper.
2. Why — bounds the achievable A3 attention precision.
3. Evidence — [a3_a5_differences.md §9.3](a3_a5_differences.md).
4. Platform — A3.
5. Priority — medium.
6. Next action — confirm with the user whether an A3-side high-precision design is in scope or out of scope.
7. Status — Inferred.

### 7.4 Does the A3 user API expose `TRSQRT` at all?

See 5.1.

### 7.5 Reduction-style kernels may need `TLOAD(dstTile, dstGlobal)` even when `dstTile` is conceptually output-only

1. Question — PR-852's `tcolargmax` adds `TLOAD(dstTile, dstGlobal);` between the src TLOAD and the manual sync block, which looks like the redundant-TLOAD anti-pattern. Is this required because col-reductions accumulate into `dstTile`, or is it a workaround / bug?
2. Why — refines `auto_mode_bad_patterns.md §1.5`; reductions may legitimately need the dst seeded.
3. Evidence — [external_context/pr_852_notes.md §T1](external_context/pr_852_notes.md), [tests/npu/a2a3/src/st/testcase/tcolargmax/tcolargmax_kernel.cpp](../tests/npu/a2a3/src/st/testcase/tcolargmax/tcolargmax_kernel.cpp), [include/pto/npu/a2a3/TColReduceOps.hpp](../include/pto/npu/a2a3/TColReduceOps.hpp).
4. Platform — A3 (and likely both at the semantic level).
5. Priority — medium.
6. Next action — read `TColReduceOps.hpp` to confirm whether `TCOLARGMAX` reads dst.
7. Status — Unknown.

---

## 8. Sparse attention / gather-scatter

### 8.1 A3 must compose multi-table gather/scatter from `TGATHER`/`TSCATTER`; perf trade-off Unknown

1. Inferred — `MGATHER`/`MSCATTER` are A5-only; an A3 sparse-attention kernel must call `TGATHER`/`TSCATTER` repeatedly or fall back to a different sparsity scheme.
2. Why — bounds the A3 sparse-attention design space.
3. Evidence — [a3_a5_differences.md §3.2, §8](a3_a5_differences.md); [auto_mode_bad_patterns.md §4.2](auto_mode_bad_patterns.md).
4. Platform — A3.
5. Priority — medium (sparse attention is later in the kernel progression).
6. Next action — when starting sparse attention design, compare composed A3 path vs A5 native.
7. Status — Inferred.

### 8.2 A5 `MGATHER` / `MSCATTER` API surface — kernel-callable user wrappers?

1. Question — are these exposed as user-facing PTO instructions (`MGATHER`, `MSCATTER`), or only as library-internal `*Impl` calls?
2. Why — determines whether sparse-attention kernels can call them directly from kernel code in auto mode.
3. Evidence — [include/pto/npu/a5/MGather.hpp:164,185](../include/pto/npu/a5/MGather.hpp#L164), [include/pto/npu/a5/MScatter.hpp:274,296,313](../include/pto/npu/a5/MScatter.hpp#L274).
4. Platform — A5.
5. Priority — medium.
6. Next action — read the headers' top-of-file `#define MGATHER ...` mappings (or grep [pto_instr.hpp](../include/pto/common/pto_instr.hpp) / [pto_instr_impl.hpp](../include/pto/common/pto_instr_impl.hpp)).
7. Status — Unknown.

### 8.3 A5 `TSCATTER` semantic divergence from A3

See 5.2. Especially relevant if a sparse-attention kernel mixes `TSCATTER` with `MSCATTER`.

---

## 9. PR-852 follow-up

### 9.1 Full A5 mirror audit of the PR-852 fix targets

1. Question — A5 has independent `TCI`/`TConcat`/`TFillPad`/`TQuant`/`TRowReduce`/`TRowReduceIdx`/`TTrans` headers. Spot-checks negative for `__cce_get_tile_ptr(x+N)` and `PtoSetWaitFlag`-inside-`__tf__`. The full audit is incomplete.
2. Why — without the audit, we cannot assume A5 is auto-mode-clean; identical bugs in A5 won't be fixed by PR-852.
3. Evidence — [external_context/pr_852_notes.md "Open items / Unknown"](external_context/pr_852_notes.md), [a3_a5_differences.md §11, §12.11](a3_a5_differences.md).
4. Platform — A5.
5. Priority — high.
6. Next action — line-by-line read of A5 [TQuant.hpp](../include/pto/npu/a5/TQuant.hpp), [TReshape.hpp](../include/pto/npu/a5/TReshape.hpp), [TTrans.hpp](../include/pto/npu/a5/TTrans.hpp) for: (a) `reinterpret_cast<uintptr_t>(...data())`, (b) `is_tile_data_v` asserts outside `#ifndef __PTO_AUTO__`, (c) single-template-across-mismatched-`TileType` patterns.
7. Status — Unknown.

### 9.2 `tcolargmax` `TLOAD(dstTile, dstGlobal)` addition rationale

See 7.5. Specifically PR-852 [§T1](external_context/pr_852_notes.md).

### 9.3 `TQUANT_IMPL` manual-mode regression risk: `kHasTail && !overlap`

1. Question — PR-852 hoists the dispatch out of `TQuantCvtS32ToFp16`; in manual mode `TQuantBuffersOverlap` may return `false` for `kHasTail` cases that previously hit the deleted `TCVT_IMPL` fallback. Was this case live before, in which case it is a regression?
2. Why — preserve manual-mode behavior per CLAUDE.md ("Preserve existing manual-mode behavior unless explicitly asked otherwise").
3. Evidence — [external_context/pr_852_notes.md §L4c](external_context/pr_852_notes.md).
4. Platform — A3 manual mode.
5. Priority — medium.
6. Next action — ask the PR author / user; otherwise instrument a manual-mode tquant build to confirm.
7. Status — Unknown.

### 9.4 PR-852 truncated diff hunks

1. Question — user paste is truncated mid-`trowargmin`. PR description names additional kernels (`ttrans`, `ttrans_conv`, `tci`, `tquant` test kernel). The diff for those is unseen.
2. Why — there may be additional patterns or new test-kernel idioms worth capturing.
3. Evidence — [external_context/pr_852_notes.md §T8](external_context/pr_852_notes.md).
4. Platform — A3.
5. Priority — medium.
6. Next action — request the missing diff hunks; re-run extraction.
7. Status — Unknown.

### 9.5 Post-merge re-verification of `tquant` ST kernel correctness

1. Question — once PR-852 merges, the `__tf__`-migrated `TQuantCvtS32ToFp16` plus the dispatch hoist may make `tquant_kernel.cpp` numerically correct in auto mode, even with the existing `TASSIGN(...0x0; ...0x0)` overlap.
2. Why — `tquant` could become a clean auto-mode reference instead of "mixed".
3. Evidence — [external_context/pr_852_notes.md "Reconciliation summary"](external_context/pr_852_notes.md) and §L4 chain.
4. Platform — A3.
5. Priority — low (post-merge cleanup).
6. Next action — re-verify after merge.
7. Status — Inferred (post-merge).

---

## 10. Testcase reliability

### 10.1 `tload_gm2mat` (A3) and `tload_shape2d` (A5) — actual auto-mode build status

See 1.4. These are in `ALL_TESTCASES` but use the same `__cce_get_tile_ptr(.data())` chain that excludes `texpands_mat`.

- Affected platform: both. Priority: high. Status: Unknown.

### 10.2 `tcolexpand{add,div,max,mul,sub}` — unguarded `pipe_barrier(PIPE_ALL)`

See 1.3. Kernel-scope `pipe_barrier(PIPE_ALL)` outside the `#ifndef __PTO_AUTO__` block; in `ALL_TESTCASES`.

- Affected platform: A3. Priority: high. Status: Unknown.

### 10.3 `tquant_kernel.cpp` numerical correctness in auto mode

See 3.1. Build-list inclusion is not correctness evidence.

- Affected platform: A3. Priority: high. Status: Inferred risk; Unknown empirically.

### 10.4 `tdequant_kernel.cpp` numerical correctness in auto mode

See 3.2.

- Affected platform: A3. Priority: high. Status: Inferred risk; Unknown empirically.

### 10.5 A5 `textract` direct `aTile.data()` in kernel body

1. Question — `auto &a = aTile.data();` appears in [tests/npu/a5/src/st/testcase/textract/textract_kernel.cpp:281-283](../tests/npu/a5/src/st/testcase/textract/textract_kernel.cpp#L281-L283). Per [Kernel Rules §3.2](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md) this is forbidden in kernel code. Does this kernel actually compile and produce correct results in auto mode?
2. Why — same as 10.1; tests existing in build list ≠ auto-mode-clean.
3. Evidence — [auto_mode_bad_patterns.md §3.3, §6.4](auto_mode_bad_patterns.md), [tile_type_reference.md §13](tile_type_reference.md).
4. Platform — A5.
5. Priority — high.
6. Next action — request build log + numerical diff.
7. Status — Inferred-risk; Unknown empirically.

### 10.6 `tfa` orphan testcase — not in `ALL_TESTCASES`

1. Known — present as a directory but excluded from CI on both archs; uses ungauarded `set_flag`/`wait_flag`, `TPipe`/`TPUSH`/`TPOP`, ping-pong, raw `__cce_get_tile_ptr` patterns. Not exercised under either mode.
2. Why — do not use as a "this works" reference for FA design.
3. Evidence — [auto_mode_bad_patterns.md §6.6](auto_mode_bad_patterns.md), [tests/npu/a2a3/src/st/testcase/tfa/tfa_kernel.cpp](../tests/npu/a2a3/src/st/testcase/tfa/tfa_kernel.cpp).
4. Platform — both.
5. Priority — low (boundary marker).
6. Next action — none; leave as-is.
7. Status — Known.

### 10.7 `tpushpop_*` testcases excluded from auto mode on both archs

1. Known — explicit CMake exclusion list ([tests/npu/a2a3/src/st/testcase/CMakeLists.txt:213-220](../tests/npu/a2a3/src/st/testcase/CMakeLists.txt#L213-L220), [tests/npu/a5/src/st/testcase/CMakeLists.txt:243-250](../tests/npu/a5/src/st/testcase/CMakeLists.txt#L243-L250)). A5 also excludes `texpands_mat`.
2. Why — sets the official "won't compile in auto mode" boundary.
3. Evidence — same as above.
4. Platform — both.
5. Priority — low (boundary marker).
6. Next action — keep referencing as the canonical exclusion list.
7. Status — Known.

---

## 11. Resolved by experiments / build runs

Items here have been promoted from `Inferred` / `Assumption` to `Known` by a
verified build or run. Append (date · platform · evidence). Do not delete;
the audit trail is useful when something later regresses.

### 11.1 Topk-style bisheng-direct CMake harness + `--cce-enable-pto-passes` builds and runs an auto-mode A3 kernel

- **Resolved**: 2026-05-07 · A3 vec · user-reported `bash run.sh -r npu -v Ascend910B1` on
  [kernels/automode/a2a3/add_tile_array/](../kernels/automode/a2a3/add_tile_array/)
  produced `test data success` / `test success`.
- **What this confirms (now Known)**:
  - Static valid region `Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor, 64, 64>` compiles and runs in auto mode on A3 (no `DYNAMIC = -1` / constructor required).
  - An in-kernel serial `for` loop reusing the same Tile across iterations is auto-safe; the auto allocator's pinned-address rule does not break reuse.
  - Reconstructing `GlobalTensor` per iteration with `base + offset` works.
  - `TLOAD → TADD → TSTORE` with no manual sync produces exact output for integer-valued FP32 inputs.
  - Auto mode enabled by **only** adding `--cce-enable-pto-passes` to the kernel target's compile options (no separate `-D__PTO_AUTO__` macro definition needed; `-O2` from the global `add_compile_options(...)` block carries through).
  - The kernel-arch guard `#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)` is **not required** when the CMake target sets `--cce-aicore-arch=dav-c220-vec` directly (matches the topk pattern).
- **Reference entry**: [known_good_kernel_examples.md §A11](known_good_kernel_examples.md).
- **What is still Unknown**: A5 mirror of the same pattern (the project is A3-only); larger / multi-core variants; behavior under `-r sim` (not yet exercised).

### 11.2 Compile errors first observed on this build

Two real errors recorded as Occurrences in [compile_error_logbook.md](compile_error_logbook.md):

- **E8** — `kernel_operator.h` not on the bisheng-direct include path (was inherited from the `demos/auto_mode/baseline/add` pattern which uses a different `ascendc.cmake` harness). Fix: drop the include; use only `<pto/common/constants.hpp>` and `<pto/pto-inst.hpp>`.
- **E9** — `tests/common/test_common.h::ReadFile` second parameter is `size_t &`; cannot bind to `const size_t`. Fix: drop `const` on the `fileSize` local. Not auto-mode-specific.

---

## Cross-references

- [auto_mode_bad_patterns.md](auto_mode_bad_patterns.md) — bad-pattern catalog and the source of most "is this silently broken" entries here.
- [tile_type_reference.md](tile_type_reference.md) — `Tile`/`ConvTile`/`TileDType` open items (§12).
- [a3_a5_differences.md](a3_a5_differences.md) — the §12 "Open assumptions and items to verify" list is the source for Group 2 here.
- [external_context/pr_852_notes.md](external_context/pr_852_notes.md) — the source for Group 9 and several "post-merge" entries.
- [known_good_kernel_examples.md §A11](known_good_kernel_examples.md) — the in-tree confirmed-built reference produced by §11.1.
- [compile_error_logbook.md §E8, §E9](compile_error_logbook.md) — the two real compile-error occurrences from §11.2.
