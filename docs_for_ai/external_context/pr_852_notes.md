# PR 852 Notes — A3 ST testcase fixes for PTO Auto Mode

URL: https://gitcode.com/cann/pto-isa/pull/852

The PR page is JS-rendered and the gitcode API requires a `private-token`, so I
could not retrieve it via tools. **All content here comes from the user-pasted
PR description and diff.** Treat everything below as supporting context, not as
truth grounded in this branch.

## Branch state

**PR 852 is NOT merged into this branch.** Verified by reading the current
source at the lines the diff modifies:

- [include/pto/npu/a2a3/TCI.hpp:59](../../include/pto/npu/a2a3/TCI.hpp#L59) still has the buggy `__cce_get_tile_ptr(tmp + 128)` form.
- [include/pto/npu/a2a3/TQuant.hpp:29-34](../../include/pto/npu/a2a3/TQuant.hpp#L29-L34) still has `TQuantBuffersOverlap` doing `reinterpret_cast<uintptr_t>(a.data())` (pre-fix form).
- [include/pto/npu/a2a3/TQuant.hpp:67](../../include/pto/npu/a2a3/TQuant.hpp#L67) still has `TQuantCvtS32ToFp16(TileDataCvtF16 &src_f16, ...)` (pre-fix; not `__tf__`, takes Tile by reference).
- [include/pto/npu/a2a3/TReshape.hpp:23-28](../../include/pto/npu/a2a3/TReshape.hpp#L23-L28) still has `is_tile_data_v` and `Loc == NewLoc` asserts OUTSIDE the `#ifndef __PTO_AUTO__` guard.
- [tests/npu/a2a3/src/st/testcase/texpands_mat/texpands_mat_kernel.cpp:20](../../tests/npu/a2a3/src/st/testcase/texpands_mat/texpands_mat_kernel.cpp#L20) still has the old helper signature `(GlobalData &dst, TileData &src)`.
- [tests/npu/a2a3/src/st/testcase/texpands_mat/texpands_mat_kernel.cpp:63](../../tests/npu/a2a3/src/st/testcase/texpands_mat/texpands_mat_kernel.cpp#L63) still passes `bufferSizeA` (= byte size) to `ConvTile<...>`.

So the bad-pattern entries in [auto_mode_bad_patterns.md](../auto_mode_bad_patterns.md) and the other `docs_for_ai/` files describe the **current** (pre-PR) source state. The PR is a planned fix.

## Scope

- The PR description explicitly says: *"Some of the ST test cases (for A3) failed for the PTO Auto Mode."* So the fix surface is **A3 (auto mode)**.
- All file paths in the diff are under `include/pto/npu/a2a3/` and `tests/npu/a2a3/`.
- A5 has independent copies of these files (verified): [include/pto/npu/a5/Tci.hpp](../../include/pto/npu/a5/Tci.hpp), [include/pto/npu/a5/TConcat.hpp](../../include/pto/npu/a5/TConcat.hpp), [include/pto/npu/a5/TFillPad.hpp](../../include/pto/npu/a5/TFillPad.hpp), [include/pto/npu/a5/TQuant.hpp](../../include/pto/npu/a5/TQuant.hpp), [include/pto/npu/a5/TRowReduce.hpp](../../include/pto/npu/a5/TRowReduce.hpp), [include/pto/npu/a5/TRowReduceIdx.hpp](../../include/pto/npu/a5/TRowReduceIdx.hpp), [include/pto/npu/a5/TTrans.hpp](../../include/pto/npu/a5/TTrans.hpp). They are **not** auto-#includes of the A3 versions; they are separate sources. **A5 may or may not need the equivalent fixes — Unknown.**
- Quick spot-checks (Inferred):
  - A5 `Tci.hpp` does NOT use the buggy `__cce_get_tile_ptr(x + N)` form; only plain `__cce_get_tile_ptr(dst)` style.
  - A5 `TConcat.hpp` / `TFillPad.hpp` / `TRowReduceIdx.hpp` / `TTrans.hpp` do NOT contain `PtoSetWaitFlag` — they may use a different sync style or already use guarded `set_flag`/`wait_flag`.

---

## Per-change extraction

Format: change → why it matters for auto mode → arch impact → reconciliation with `docs_for_ai/` → confidence.

### L1. [include/pto/npu/a2a3/TCI.hpp](../../include/pto/npu/a2a3/TCI.hpp) — pointer arithmetic before `__cce_get_tile_ptr`

- **What changed** — Eight occurrences of `__cce_get_tile_ptr(tmp + N)` are rewritten as `__cce_get_tile_ptr(tmp) + N` (offsets `+128`, `+256`, `+384`). Five places in `TCI_b32_repeat`/`TCI_b32_normal`/`TCI_b16_repeat`/`TCI_b16_normal`. PR description: *"The test case crashes during the libexpand Pass when it's trying to do a RAUW. Fix: To move the pointer arithmetic after `cce_get_tile_ptr`."* Additionally, several `vadds`/`vmuls`/`vconv_*` calls switch from operating on `dst` (the `TileDType` value) to `dstPtr` (the typed `__ubuf__` pointer extracted at the top of the function).
- **Why for auto mode** — `__cce_get_tile_ptr` operates on the tile's `TileDType`, which in auto mode is a vector type (per [include/pto/common/memory.hpp:29-44](../../include/pto/common/memory.hpp#L29-L44)). Doing `tmp + 128` BEFORE the macro applies pointer arithmetic to a vector, which the libexpand pass cannot rewrite. Doing `__cce_get_tile_ptr(tmp) + 128` first produces a typed `__ubuf__ T *` and then advances it — pointer arithmetic on a real pointer.
- **Arch** — A3 (file is in `a2a3/`).
- **Reconciliation** — Refines [auto_mode_bad_patterns.md §3.2](../auto_mode_bad_patterns.md). Adds a new specific bad sub-pattern: *pointer arithmetic on the `TileDType` argument of `__cce_get_tile_ptr` before extraction.* Also confirms [qualifier_reference.md §3.3](../qualifier_reference.md): `__cce_get_tile_ptr` must wrap the bare `TileDType`.
- **Confidence** — High (the diff is explicit and the rule is the natural reading of the auto-mode `TileDType` contract).

### L2. [include/pto/npu/a2a3/TConcat.hpp](../../include/pto/npu/a2a3/TConcat.hpp) — `PtoSetWaitFlag` is wrong inside `__tf__` in auto mode

- **What changed** — Five `PtoSetWaitFlag<...>()` calls inside `__tf__ PTO_INTERNAL TConcatIdx(...)` are wrapped:
  ```cpp
  #ifndef __PTO_AUTO__
      PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
  #else
      set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
  #endif
  ```
- **Why for auto mode** — `PtoSetWaitFlag` is **no-op in auto mode** ([docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md §3.3](../../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md): *"when compiling in auto mode, this interface is a no-op"*). That is correct at **kernel** level (where auto-sync handles it). But this code is INSIDE a `__tf__` body — and per [docs/auto_mode/Library_Developer_Rules_And_Limitations.md §3](../../docs/auto_mode/Library_Developer_Rules_And_Limitations.md), *"Use `set_flag`, `wait_flag` or `pipe_barrier` explicitly in tile functions and all of their callees. Use `PtoSetWaitFlag` or `TSYNC` anywhere else."* The auto-mode compiler does NOT walk into tile functions, so a no-op here means the function gets no sync at all. Fix: emit raw `set_flag`/`wait_flag` in auto mode too, only inside the `__tf__` body.
- **Arch** — A3 (a2a3/).
- **Reconciliation** — **Refines** [auto_mode_bad_patterns.md §2.1](../auto_mode_bad_patterns.md). The "fix" recommended there ("use `PtoSetWaitFlag`/`TSYNC` instead of `set_flag`/`wait_flag`") is correct only at **kernel** level. Inside a `__tf__` body, the rule is reversed: `set_flag`/`wait_flag`/`pipe_barrier` are required even in auto mode; `PtoSetWaitFlag` is wrong there. The bad-patterns doc should grow a §2.7 *"`PtoSetWaitFlag` inside a tile function silently drops sync in auto mode"*.
- **Confidence** — High.

### L3. [include/pto/npu/a2a3/TFillPad.hpp](../../include/pto/npu/a2a3/TFillPad.hpp) — same fix as L2

- **What changed** — One `PtoSetWaitFlag<PIPE_V, PIPE_S>()` inside `Handle32BAlignedPad_Byte` wrapped with the `#ifndef __PTO_AUTO__ / #else set_flag/wait_flag / #endif` shape.
- **Why / Arch / Reconciliation / Confidence** — Same as L2.

### L4. [include/pto/npu/a2a3/TQuant.hpp](../../include/pto/npu/a2a3/TQuant.hpp) — buffer-overlap check, dispatch, and tile-function migration

Three coupled changes:

- **L4a — `TQuantBuffersOverlap` short-circuits in auto mode.** The body becomes:
  ```cpp
  #ifndef __PTO_AUTO__
      auto aStart = reinterpret_cast<uintptr_t>(a.data());
      ...
      return (aStart < bEnd) && (bStart < aEnd);
  #else
      return true;
  #endif
  ```
- **Why** — In auto mode `Tile::data()` returns a vector type ([memory.hpp:29-44](../../include/pto/common/memory.hpp#L29-L44)), so `reinterpret_cast<uintptr_t>` is meaningless. PR description: *"the Overlap check (`TQuantBuffersOverlap`) was redundant in auto mode since both of the input arguments are reshapes of the same tile."* That matches the existing `TRESHAPE_IMPL` aliasing branch at [TQuant.hpp:108-114](../../include/pto/npu/a2a3/TQuant.hpp#L108-L114): in auto mode both views are `TRESHAPE`s of `src`, so they always overlap.
- **Reconciliation** — Confirms [auto_mode_bad_patterns.md §1.3](../auto_mode_bad_patterns.md): `reinterpret_cast<uintptr_t>(tile.data())` does not work in auto mode and must be guarded out. Confirms [qualifier_reference.md §1](../qualifier_reference.md).
- **Confidence** — High.

- **L4b — `TQuantCvtS32ToFp16` becomes a tile function.**
  - Old: `template<...> PTO_INTERNAL void TQuantCvtS32ToFp16(TileDataCvtF16 &src_f16, TileDataCvtS32 &src_s32, uint32_t validRow)` — takes `Tile&` by reference; calls `src_f16.data()` and `src_s32.data()` to get pointers; not `__tf__`.
  - New: `template<...> __tf__ PTO_INTERNAL void TQuantCvtS32ToFp16(typename TileDataCvtF16::TileDType __out__ src_f16, typename TileDataCvtS32::TileDType __in__ src_s32, uint32_t validRow)` — takes `TileDType` by value with `__in__`/`__out__`; uses `__cce_get_tile_ptr(src_f16)` instead of `.data()`.
- **Why** — The function calls raw CCE intrinsics (`set_deqscale`, `pipe_barrier`, `vconv_*`). Per [Library §6](../../docs/auto_mode/Library_Developer_Rules_And_Limitations.md) such helpers must be tile functions: `__tf__`, `TileDType` parameters by value with `__in__`/`__out__`, and `__cce_get_tile_ptr` to retrieve the raw pointer. Caller updates: `TQuantCvtS32ToFp16<PadColsSrc, TileDataCvtF16, TileDataCvtS32>(src_f16.data(), src_s32.data(), src.GetValidRow())` — passes the `.data()` (vector value) into the tile function.
- **Reconciliation** — Confirms [qualifier_reference.md §3.1, §3.2](../qualifier_reference.md) (canonical migration shape). Confirms the [Library §6 rules](../../docs/auto_mode/Library_Developer_Rules_And_Limitations.md). Adds a concrete diff-grade example to point at when reviewing future `_IMPL`-to-`__tf__` migrations.
- **Confidence** — High.

- **L4c — Dispatch moved out of `TQuantCvtS32ToFp16`; `TQuantCvtS32ToFp16RowByRow` deleted.**
  - The old `TQuantCvtS32ToFp16` chose between row-by-row and `TCVT_IMPL` based on `kHasTail` and `TQuantBuffersOverlap`. After the change there is **only** the row-by-row body (now `__tf__`); the dispatch happens in `TQUANT_IMPL`:
    ```cpp
    if constexpr (kHasTail) {
        if (TQuantBuffersOverlap(src_f16, src_s32)) {
            TQuantCvtS32ToFp16<PadColsSrc, TileDataCvtF16, TileDataCvtS32>(src_f16.data(), src_s32.data(),
                                                                           src.GetValidRow());
        }
    } else {
        TCVT_IMPL(src_f16, src_s32, RoundMode::CAST_RINT);
    }
    ```
- **Why** — Tile functions are black boxes to the auto-mode compiler ([Auto_Mode_Overview.md](../../docs/auto_mode/Auto_Mode_Overview.md)). Keeping the `if constexpr (kHasTail)` / `TQuantBuffersOverlap(...)` decision INSIDE the tile function would either (a) require it to become PTO-instruction-shaped (which `TCVT_IMPL` is) — but that defeats the `__tf__` boundary; or (b) hide the call to `TCVT_IMPL` (a PTO instruction) inside a `__tf__`, where the auto-sync analysis cannot see it. Hoisting the dispatch keeps the PTO instruction (`TCVT_IMPL`) visible to auto-sync and confines the raw-CCE branch to the tile function.
- **Reconciliation** — New pattern: when refactoring a helper to be `__tf__`, hoist any branch that calls a PTO `_IMPL` (i.e., a PTO instruction) up to the caller, so the auto-mode compiler sees the PTO call. Worth adding to a future `library_migration_recipes.md`.
- **Note (mild concern)** — In the new code, the `else` branch of `if constexpr (kHasTail)` calls `TCVT_IMPL` directly, but the if-true branch contains `if (TQuantBuffersOverlap(...))` — and when `kHasTail && !overlap`, neither branch runs. In auto mode `TQuantBuffersOverlap` always returns `true` (per L4a), so this is fine for auto mode. In manual mode there is a possible regression if `kHasTail && !overlap` was previously handled by the deleted `TCVT_IMPL` fallback inside the old dispatcher. **Status: Unknown** (mark with low confidence; only A3 manual mode is at risk; the PR scope says manual still works since A3 manual was passing before).
- **Confidence** — High for the auto-mode side; Medium for the manual-mode side regression risk above.

### L5. [include/pto/npu/a2a3/TReshape.hpp](../../include/pto/npu/a2a3/TReshape.hpp) — guard remaining asserts in auto mode

- **What changed** — Two asserts (`is_tile_data_v<TileDataIn>`, `is_tile_data_v<TileDataOut>`, and the `Loc == NewLoc` check) are moved INSIDE the existing `#ifndef __PTO_AUTO__` block. After the change, **all** static_asserts in `TRESHAPE_IMPL` are manual-mode-only.
- **Why** — Per PR description: *"`TRESHAPE` cannot work with `ConvTile`. There are asserts inside `TRESHAPE` that stops this. Fix: Added macro guards to only run the asserts for manual mode."* So in auto mode `TRESHAPE_IMPL` permits `ConvTile` aliasing, while in manual mode it still enforces `Tile`-only.
- **Why for auto mode** — `ttrans_conv` test (and likely others) needs to alias a `ConvTile` view onto a base tile in auto mode. The is_tile_data_v assertions reject `ConvTile`, so the test fails to compile; auto mode handles aliasing differently anyway (it is a hint, not a runtime instruction — see [Kernel rules §2.4](../../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md)).
- **Arch** — A3.
- **Reconciliation** — Refines [auto_mode_bad_patterns.md §1.1, §1.3](../auto_mode_bad_patterns.md): the recommendation "use `TRESHAPE`/`TSUBVIEW` for aliasing" must be qualified — *for `ConvTile`, the current TReshape rejects in manual mode but PR-852 will allow it in auto mode*. Also new: in the post-PR world, `TRESHAPE` with mismatched `TileType` between dst and src is silently allowed in auto mode (no `Loc == NewLoc` check). Worth flagging as a new soft gotcha.
- **Confidence** — High.

### L6. [include/pto/npu/a2a3/TRowReduceIdxOps.hpp](../../include/pto/npu/a2a3/TRowReduceIdxOps.hpp) — same `PtoSetWaitFlag` wraps as L2

- **What changed** — Six `PtoSetWaitFlag<PIPE_*, PIPE_*>()` calls inside `ProcReduceIdxStage1`, `ProcReduceIdxStage2`, `ExtractValIdxFromTmp` wrapped in the `#ifndef __PTO_AUTO__ / #else / #endif` shape.
- **Why / Arch / Reconciliation / Confidence** — Same as L2.

### L7. [include/pto/npu/a2a3/TTrans.hpp](../../include/pto/npu/a2a3/TTrans.hpp) — decouple template types AND `PtoSetWaitFlag` wraps

Two coupled changes:

- **L7a — Template-type decoupling.** Four helpers are changed from a single `TileData` template parameter to three separate ones: `TileDataDst`, `TileDataSrc`, `TileDataTmp`.
  - Affected: `TTransConvNCHW2NC1HWC0`, `TTransConvNC1HWC02C1HWNC0`, `TTransConvGNCHW2GNC1HWC0`, `TTransConvGNC1HWC02GC1HWNC0`.
  - Inside each: `using T = typename TileData::DType` becomes `using Tdst = typename TileDataDst::DType; using Tsrc = typename TileDataSrc::DType; using Ttmp = typename TileDataTmp::DType;`. Pointer extractions and casts use the per-role type.
  - Callers (`TTransImplConvTile`) updated to pass three template args.
- **Why** — PR description: *"Using the same Tile Type template for three different arguments, two of which had `TileConv` as the type and one had `TileVec`."* When the dst/src/tmp tiles have different `TileType` (Vec vs Conv) or different element types, instantiating one template parameter with all three is invalid. Decoupling fixes this.
- **Reconciliation** — Adds a **new** bad-pattern entry not currently in [auto_mode_bad_patterns.md](../auto_mode_bad_patterns.md): *"Single template parameter shared across tile-function arguments of different `TileType`s."* Belongs in Group 5 (Type and template hazards). Worth adding as §5.5.
- **Arch** — A3.
- **Confidence** — High.

- **L7b — `PtoSetWaitFlag` wraps in `TransTailTiles` and `TTransConvNC1HWC02C1HWNC0`.**
  - Same shape as L2/L3/L6.
- **Confidence** — High.

### T1. [tests/npu/a2a3/src/st/testcase/tcolargmax/tcolargmax_kernel.cpp](../../tests/npu/a2a3/src/st/testcase/tcolargmax/tcolargmax_kernel.cpp)

- **What changed** —
  1. Two `set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(...)` blocks wrapped with `#ifndef __PTO_AUTO__ ... #endif`.
  2. **A new line**: `TLOAD(dstTile, dstGlobal);` is added between `TLOAD(srcTile, srcGlobal);` and the manual sync block.
- **Why for auto mode** — (1) is the standard kernel-level fix for raw set_flag/wait_flag (matches [auto_mode_bad_patterns.md §2.1 fix](../auto_mode_bad_patterns.md)). (2) is **NOT explained by the PR description** — the description for tcolargmax only mentions extending `tmpTile` from 32→64 elements (which is not visible in the diff hunks I have, but that hint suggests the visible 64-element math). The added `TLOAD(dstTile, ...)` looks like the exact "redundant TLOAD on dst" anti-pattern called out in [Kernel rules §3.1](../../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md) and [auto_mode_bad_patterns.md §1.5](../auto_mode_bad_patterns.md). Possible explanations:
  - TCOLARGMAX needs `dstTile` initialized because it accumulates across columns (Inferred — would need to read [TColReduceOps.hpp](../../include/pto/npu/a2a3/TColReduceOps.hpp) to confirm).
  - Or the TLOAD is a workaround to keep `dstTile` alive across the auto-allocator's liveness analysis so it does not get coalesced.
  - Or it is a bug in the PR.
- **Arch** — A3.
- **Reconciliation** — Either **contradicts** or **refines** [auto_mode_bad_patterns.md §1.5](../auto_mode_bad_patterns.md). The cleanest reading is that §1.5 is correct as a default rule but has reduction-style exceptions when the dst is read by the op. Mark as Unknown until confirmed; do NOT generalize the new pattern in fresh kernels.
- **Confidence** — Medium (kernel-side guard fix); Low (the dst-`TLOAD` addition is unexplained).

### T2. [tests/npu/a2a3/src/st/testcase/tconcatidx/tconcatidx_kernel.cpp](../../tests/npu/a2a3/src/st/testcase/tconcatidx/tconcatidx_kernel.cpp)

- **What changed** — Two raw `set_flag`/`wait_flag` blocks wrapped in `#ifndef __PTO_AUTO__`.
- **Why / Arch / Reconciliation** — Standard fix. Confirms [auto_mode_bad_patterns.md §2.1](../auto_mode_bad_patterns.md). Note: the kernel still calls `TCONCAT_IMPL(...)` directly (which my [auto_mode_bad_patterns.md §3.4](../auto_mode_bad_patterns.md) flags as `_IMPL` leaked to kernel code). The PR does NOT fix that. So §3.4 still applies.
- **Confidence** — High.

### T3. [tests/npu/a2a3/src/st/testcase/tcvt/tcvt_kernel.cpp](../../tests/npu/a2a3/src/st/testcase/tcvt/tcvt_kernel.cpp) — dual-mode aliasing pattern

- **What changed** — Across multiple kernels (`runTCVT`, `runTCVT_fp16_to_s4`, `runTCVT_s4_to_fp16`, `runTCVTSaturationTest`, `runTCVTNonSatTorch`):
  1. **Add `TRESHAPE` after the `TASSIGN` pair** that aliases two tiles. Example:
     ```cpp
     TASSIGN(srcTileFull, 0x0 + 0x400 * block_idx);
     TASSIGN(dstTileFull, 0x20000 + 0x400 * block_idx);
     TASSIGN(srcTile, 0x0 + 0x400 * block_idx);
     TASSIGN(dstTile, 0x20000 + 0x400 * block_idx);
     TRESHAPE(srcTile, srcTileFull);   // <-- NEW
     TRESHAPE(dstTile, dstTileFull);   // <-- NEW
     ```
  2. Also adds `TRESHAPE(dstBytesTile, dstS4Tile)` in `runTCVT_fp16_to_s4`, `TRESHAPE(srcS4Tile, srcBytesTile)` in `runTCVT_s4_to_fp16`, and `TRESHAPE(dstTile, dstTileFull); TRESHAPE(srcTile, srcTileFull);` in `runTCVTNonSatTorch`.
  3. Wrap raw `set_flag`/`wait_flag` blocks with `#ifndef __PTO_AUTO__`.
- **Why for auto mode** — In manual mode the two `TASSIGN(..., same_addr)` calls alias the tiles. In auto mode `TASSIGN` is a no-op and the alias is lost — the two tiles get independent addresses and the kernel computes wrong results. The fix is to add `TRESHAPE(b, a)` so the auto-mode compiler binds `b` to the same address as `a`. Manual mode also accepts `TRESHAPE` (it executes the address binding); both modes now do the right thing.
- **Arch** — A3.
- **Reconciliation** — **Confirms and refines** [auto_mode_bad_patterns.md §1.1](../auto_mode_bad_patterns.md). The bad-pattern entry warned against `TASSIGN(a, X); TASSIGN(b, X);` overlapping aliasing without a fix. PR-852 gives the canonical fix shape: keep both `TASSIGN`s (manual-mode aliasing) AND add `TRESHAPE(b, a)` immediately after (auto-mode aliasing hint). This is the **dual-mode aliasing recipe**. Worth promoting to a positive pattern in [known_good_kernel_examples.md](../known_good_kernel_examples.md) once merged.
- **Confidence** — High.

### T4. [tests/npu/a2a3/src/st/testcase/texpands_mat/texpands_mat_kernel.cpp](../../tests/npu/a2a3/src/st/testcase/texpands_mat/texpands_mat_kernel.cpp)

Two coupled changes:

- **T4a — Helper signature: pass `TileDType` by value, not `Tile&`.**
  - Old: `__tf__ AICORE inline void TSTORE_MAT2GM_CONVTILE(GlobalData &dst, TileData &src) { ... __cce_get_tile_ptr(src.data()); dst.data(); ... }`
  - New: `__tf__ AICORE inline void TSTORE_MAT2GM_CONVTILE(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src) { ... __cce_get_tile_ptr(src); dst; ... }`
  - Caller now passes `dstGlobal.data(), MatTile.data()`.
- **Why** — The old form takes `Tile&`, which forces a `.data()` call **inside** the tile function. In auto mode `tile.data()` is a vector value, but the helper was applying `__cce_get_tile_ptr(src.data())` rather than `__cce_get_tile_ptr(src)`. The new form is the canonical `__tf__` shape: `TileDType` by value (with `__in__`/`__out__`), and `__cce_get_tile_ptr(src)` directly. Matches [Library §6](../../docs/auto_mode/Library_Developer_Rules_And_Limitations.md). PR description: *"An issue with the layering of their tile function. Fixed!"*
- **Reconciliation** — **Confirms** [qualifier_reference.md §3.1, §3.3](../qualifier_reference.md) and [auto_mode_bad_patterns.md §3.3](../auto_mode_bad_patterns.md) (do not call `.data()` from kernel/wrapper code; pass `Tile&` to `__tf__` was the wrong fix; the right fix is to pass the `TileDType` value down). Note: the PR moves the `.data()` call to the **caller** (`runTSetValue_ConvTile`), where it is in kernel-level code. This is consistent with how the auto-mode demo and other test kernels do it (e.g., the new `TQuant.hpp` caller).
- **Confidence** — High.

- **T4b — `ConvTile<...>` template parameter: `bufferSizeA` (bytes) → `elementSize` (count).**
  - Old: `using TileData = ConvTile<TileType::Mat, T, bufferSizeA, ...>` where `bufferSizeA = elementSize * sizeof(T)`.
  - New: `using TileData = ConvTile<TileType::Mat, T, elementSize, ...>`.
- **Why** — PR description: *"Allocated tiles were bigger than the UB. ... Fix: Used the correct variable, `numElems`, for ConvTile Construction."* The `BufferSize_` template parameter is misleadingly named — it expects an element count, not bytes. Passing `elementSize * sizeof(T)` allocates `sizeof(T)`× the intended UB. Long-term TODO in PR: rename the template parameter.
- **Arch** — A3.
- **Reconciliation** — Adds a **new** bad-pattern entry not currently in [auto_mode_bad_patterns.md](../auto_mode_bad_patterns.md): *"`ConvTile<Loc, T, BufferSize_, Layout, Shape>` — `BufferSize_` is element count despite the name; passing bytes silently inflates UB allocation."* Belongs in Group 5. The current ConvTile struct definition is at the location quoted in PR text:
  ```cpp
  template <TileType Loc_, typename Element_, const int BufferSize_, Layout Layout_, typename Shape_>
  struct ConvTile { ...; static constexpr int bufferSize = BufferSize_; };
  ```
  Resolving the misleading name belongs in a separate library cleanup, not auto-mode code generation.
- **Confidence** — High (the diff is explicit and the PR description spells it out).

### T5. [tests/npu/a2a3/src/st/testcase/tfillpad/tfillpad_kernel.cpp](../../tests/npu/a2a3/src/st/testcase/tfillpad/tfillpad_kernel.cpp)

- **What changed** — Whitespace fix only (a trailing space removed near a `}     // i` comment in `get_input_golden_case`). Not auto-mode related.
- **Confidence** — High that this hunk is benign.

### T6. [tests/npu/a2a3/src/st/testcase/trowargmax/trowargmax_kernel.cpp](../../tests/npu/a2a3/src/st/testcase/trowargmax/trowargmax_kernel.cpp), T7. [tests/npu/a2a3/src/st/testcase/trowargmin/trowargmin_kernel.cpp](../../tests/npu/a2a3/src/st/testcase/trowargmin/trowargmin_kernel.cpp)

- **What changed** — Raw `set_flag`/`wait_flag` blocks wrapped in `#ifndef __PTO_AUTO__`. Multiple instances per file (single-output and value+index overloads).
- **Why / Arch / Reconciliation** — Standard fix. Confirms [auto_mode_bad_patterns.md §2.1](../auto_mode_bad_patterns.md).
- **Confidence** — High.

### T8. (Truncated in the user's paste — likely additional test files)

The user-pasted text was truncated mid-`trowargmin` and may include additional test kernels (e.g., `ttrans_conv`, `ttrans`, `tquant`, `tci`, `texpands_mat`-side test). The PR description names: *ttrans_conv, ttrans, tci, texpands_mat, trowargmax, trowargmin, tcolargmax, tconcatidx, tfillpad, tquant, tcvt*. Of those, I have visible diff for: tcolargmax (T1), tconcatidx (T2), tcvt (T3), texpands_mat (T4), tfillpad (T5), trowargmax (T6), trowargmin (T7). Missing visibility: ttrans, ttrans_conv, tci, tquant test kernels. **Status: Unknown**.

---

## Reconciliation summary against existing `docs_for_ai/`

| existing entry | PR-852 effect | action when PR merges |
|---|---|---|
| [auto_mode_bad_patterns.md §1.1](../auto_mode_bad_patterns.md) (TASSIGN aliasing) | Confirmed; canonical dual-mode fix shown | Add positive recipe: keep both `TASSIGN(a, X); TASSIGN(b, X);` and **also** add `TRESHAPE(b, a);` |
| [auto_mode_bad_patterns.md §1.3](../auto_mode_bad_patterns.md) (`reinterpret_cast<uintptr_t>(tile.data())`) | Confirmed | Mark resolved-by-PR-852 in `TQuant.hpp` |
| [auto_mode_bad_patterns.md §1.5](../auto_mode_bad_patterns.md) (TLOAD on dst) | Possibly contradicted by tcolargmax — **Unknown** | Investigate `TCOLARGMAX` / `TColReduceIdx32` semantics before generalizing |
| [auto_mode_bad_patterns.md §2.1](../auto_mode_bad_patterns.md) (kernel-level set_flag/wait_flag) | Confirmed | Mark resolved-by-PR-852 for the listed test kernels |
| [auto_mode_bad_patterns.md §2.1 → §2.7 (NEW)](../auto_mode_bad_patterns.md) | Refined: `PtoSetWaitFlag` is wrong **inside** `__tf__` in auto mode | Add new entry |
| [auto_mode_bad_patterns.md §3.2](../auto_mode_bad_patterns.md) (`__cce_get_tile_ptr(tile.data())`) | Refined with new sub-pattern: `__cce_get_tile_ptr(x + N)` is wrong; must be `__cce_get_tile_ptr(x) + N` | Add as §3.2.1 |
| [auto_mode_bad_patterns.md §3.4](../auto_mode_bad_patterns.md) (`_IMPL` from kernel) | NOT addressed by PR — `tconcatidx_kernel.cpp` still calls `TCONCAT_IMPL` | Keep entry |
| [auto_mode_bad_patterns.md §5.x (NEW)](../auto_mode_bad_patterns.md) | Two new template hazards: (a) reusing one TileData template across dst/src/tmp of different `TileType`; (b) `ConvTile`'s `BufferSize_` template param accepts elements, not bytes | Add as §5.5 and §5.6 |
| [qualifier_reference.md §3.1, §3.2, §3.3](../qualifier_reference.md) (`__tf__`, `__in__`/`__out__`, `__cce_get_tile_ptr`) | Confirmed by `TQuantCvtS32ToFp16` and `TSTORE_MAT2GM_CONVTILE` migrations | Cite PR as the canonical "before/after" example |
| [known_good_kernel_examples.md §D2](../known_good_kernel_examples.md) (TQuant library) | Refined — the dispatch pattern around the new `__tf__` is the recommended shape | Update once merged |
| [known_good_kernel_examples.md §D3](../known_good_kernel_examples.md) (tquant ST kernel) | Should be re-verified post-merge — the helper's `__tf__` form may actually make tquant produce correct output in auto mode | Re-verify |

## New patterns introduced by PR-852 (worth adopting)

- **Dual-mode aliasing recipe** (T3): keep `TASSIGN(...)` for manual mode AND add `TRESHAPE(other, base)` immediately after for auto mode. Both modes get correct aliasing.
- **`__tf__` migration template** (L4b, T4a): when a helper contains raw CCE intrinsics, give it `__tf__`, change parameters from `Tile&` to `TileDType` (with `__in__`/`__out__`), and use `__cce_get_tile_ptr(arg)` (no `.data()` inside the body). Caller passes `tile.data()` (vector value).
- **Hoist the dispatch above the tile-function boundary** (L4c): when one branch needs raw CCE and the other needs a PTO instruction (`TCVT_IMPL`), put the `if`/`if constexpr` at the caller, with the raw-CCE branch as a `__tf__` call and the PTO-instruction branch as a direct `_IMPL` call. This keeps PTO calls visible to auto-sync.
- **Pointer arithmetic AFTER `__cce_get_tile_ptr`** (L1): `__cce_get_tile_ptr(x) + N`, never `__cce_get_tile_ptr(x + N)`. The latter applies arithmetic to a vector value, which the libexpand pass cannot rewrite.
- **Inside a `__tf__` body, in auto mode use real `set_flag`/`wait_flag`** (L2/L3/L6/L7b), not `PtoSetWaitFlag` (which is no-op in auto mode and silently drops sync).

## Open items / Unknown

- **A5 mirror status** — A5 has independent `TCI` / `TConcat` / `TFillPad` / `TQuant` / `TRowReduce` / `TRowReduceIdx` / `TTrans` headers. Spot-checks suggest A5 does NOT contain the same `__cce_get_tile_ptr(tmp + N)` or `PtoSetWaitFlag` patterns, but a full audit is **Unknown**.
- **`tcolargmax` `TLOAD(dstTile, dstGlobal)`** — unexplained in the PR description; possible exception, workaround, or bug. Confirm with the PR author or by reading [TColReduceOps.hpp](../../include/pto/npu/a2a3/TColReduceOps.hpp) before generalizing.
- **`TQUANT_IMPL` manual-mode regression** — the new dispatch leaves `kHasTail && !overlap` unhandled in manual mode if `TQuantBuffersOverlap` returns `false` there. Verify this case existed before (in which case it is a regression) or did not (in which case it is a benign cleanup).
- **Truncated diff hunks** — user paste cut off in `trowargmin` and may include `ttrans`, `ttrans_conv`, `tci`, `tquant` test kernel changes. Re-extract once full diff is available.
- **PR title** — not visible in the user paste; only the description and diff.
