# Auto-mode Bad Patterns (A3/A5)

Source-grounded catalog of patterns that are unsafe, misleading, or
incompatible with auto mode for A3/A5 kernels. Companion to
[repo_kernel_map.md](repo_kernel_map.md) and
[known_good_kernel_examples.md](known_good_kernel_examples.md). Nothing here
has been compiled.

For each entry:
1. **Pattern**
2. **Why it is risky**
3. **Where it appears**
4. **Likely fix / safer replacement**
5. **Confidence** — High / Medium / Low
6. **Status** — `Known` / `Inferred` / `Assumption` / `Unknown`

Conventions used below:
- "User wrapper" / "kernel-callable" — the public API symbol like `TADD`, `TMATMUL`, `TASSIGN`, `TQUANT` (as opposed to `_IMPL` / library-only entries).
- "Auto-mode rules" = [docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md) and [docs/auto_mode/Library_Developer_Rules_And_Limitations.md](../docs/auto_mode/Library_Developer_Rules_And_Limitations.md).
- "Auto mode" = compiled with `--cce-enable-pto-passes -O2` and `__PTO_AUTO__` defined.

Several entries below were clarified by **PR-852** (A3 ST testcase fixes), which is merged into the current branch. Cross-references point to [external_context/pr_852_notes.md](external_context/pr_852_notes.md). Keep these entries as review rules and historical debugging context. Do not claim a PR-852-era bug still exists unless current source inspection proves it; if the same symptom appears again, treat it as a regression or a newly introduced instance of the old anti-pattern.

---

## Group 1 — Memory and aliasing

### 1.1 Manual-mode TASSIGN aliasing trick (same address → silent non-aliasing in auto mode)

1. **Pattern** — Two distinct tiles given the same `TASSIGN` literal address to overlap their L1/UB storage. Auto mode no-ops `TASSIGN`, so the compiler allocates each tile independently; the manual-mode aliasing semantics are silently lost.
2. **Why risky** — The kernel may compile but compute different results in auto mode vs manual mode. Memory footprint also changes.
3. **Where**
   - [tests/npu/a2a3/src/st/testcase/tquant/tquant_kernel.cpp:45-47](../tests/npu/a2a3/src/st/testcase/tquant/tquant_kernel.cpp#L45-L47): `TASSIGN(srcTile, 0x0); TASSIGN(dstS8Tile, 0x0); TASSIGN(scaleTile, 0x20100);` — `srcTile` and `dstS8Tile` collide at `0x0`.
   - General pattern noted in [docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md §2.3, §2.4](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md).
4. **Fix** — Auto-only kernels: drop the `TASSIGN`s and use `TRESHAPE(b, a)` / `TSUBVIEW(b, a, row, col)`. Dual-mode kernels: keep the `TASSIGN` pair AND add `TRESHAPE(b, a)` immediately after — full recipe + code block at [known_good_kernel_examples.md §A10](known_good_kernel_examples.md) (also the library-internal analogue in [include/pto/npu/a2a3/TQuant.hpp:108-114](../include/pto/npu/a2a3/TQuant.hpp#L108-L114)).
5. **Confidence** — High.
6. **Status** — Known.

### 1.2 Dynamic / runtime `TASSIGN` (address depends on a loop index or runtime variable)

1. **Pattern** — `TASSIGN(tile, 0x100 * i)` inside a loop, or `TASSIGN(tile, addr_chosen_at_runtime)`.
2. **Why risky** — Auto mode allocates each tile a single constant address based on liveness; it cannot honor a per-iteration override ([docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md §2.3](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md): *"each tile will be allocated memory once and for all"* — *"think of a tile as a C++ reference"*).
3. **Where** — Documented in §2.3 of the rules doc (not a single source path; pattern to scan for in any new kernel). The tfa ST helper [tests/npu/a2a3/src/st/testcase/tfa/tfa_kernel.cpp:80-95](../tests/npu/a2a3/src/st/testcase/tfa/tfa_kernel.cpp#L80-L95) implements `assign_tile_buffers()` that loops `TASSIGN(tiles[idx], tile_offset)` — this is manual-only and is one of the reasons that kernel is not in `ALL_TESTCASES`.
4. **Fix** — Hoist all tile declarations to a single static layout; use `TSUBVIEW` for sub-views (the offsets there are runtime-allowed but the *base* is the parent tile, not a literal address).
5. **Confidence** — High.
6. **Status** — Known.

### 1.3 `TASSIGN(tile, reinterpret_cast<uintptr_t>(other.data()))` aliasing

1. **Pattern** — Copying a tile address by extracting it via `.data()` and re-feeding to `TASSIGN`. Used in pre-auto-mode `_IMPL`s.
2. **Why risky** — In auto mode, `Tile::data()` returns a vector type, not a pointer ([docs/auto_mode/Library_Developer_Rules_And_Limitations.md §1](../docs/auto_mode/Library_Developer_Rules_And_Limitations.md), [include/pto/common/memory.hpp:29-106](../include/pto/common/memory.hpp#L29-L106)), so `reinterpret_cast<uintptr_t>(...)` is meaningless. Also, `TASSIGN` itself is a no-op.
3. **Where** — The canonical "before" of this pattern is preserved in [include/pto/npu/a2a3/TQuant.hpp:108-114](../include/pto/npu/a2a3/TQuant.hpp#L108-L114) under `#ifndef __PTO_AUTO__`. Anywhere outside that guard, this pattern is wrong.
4. **Fix** — Use the `#else` branch from `TQuant.hpp`: `TRESHAPE_IMPL(other_view, base_tile);`. Or, in kernel code, use the user-facing `TRESHAPE`/`TSUBVIEW`.
5. **Confidence** — High.
6. **Status** — Known.

### 1.4 Hard-coded raw memory pointers (`__ca__`, `__cb__`, `__cc__`) cast to literal addresses

1. **Pattern** — `#define L0A_BUF0 ((__ca__ half *)(__ca__ char *)0x0)` and direct use as a tile-storage handle.
2. **Why risky** — In auto mode the `__ca__` (and friends) qualifiers attach to **vector** types, not pointers ([include/pto/common/memory.hpp:46-57](../include/pto/common/memory.hpp#L46-L57)). Casting a literal int to a `__ca__ half *` is meaningless. Also it bypasses the compiler's allocator entirely.
3. **Where** — [tests/npu/a2a3/src/st/testcase/tfa/pto_macro_matmul.hpp:32-37](../tests/npu/a2a3/src/st/testcase/tfa/pto_macro_matmul.hpp#L32-L37) defines `L0A_BUF0/1`, `L0B_BUF0/1`, `L0C_BUF0/1` this way. The same pattern appears in [kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp](../kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp) (manual). Manual-only.
4. **Fix** — Drop the manual address arithmetic. Declare tiles with `Tile<TileType::Left, ...>` etc. and let auto mode allocate.
5. **Confidence** — High.
6. **Status** — Known.

### 1.5 Redundant `TLOAD` on a write-only destination tile

1. **Pattern** — `TLOAD(dstTile, dstGlobal);` where `dstTile` is only ever the OUTPUT of subsequent ops (e.g., `TADD`, `TDEQUANT`).
2. **Why risky** — In auto mode the compiler may coalesce `dstTile` and an unrelated `srcTile` to the same physical address (their liveness does not overlap). Two simultaneous `TLOAD`s into the same physical buffer race ([docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md §3.1](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md): *"DON'T CALL REDUNDANT TLOAD!"*).
3. **Where**
   - [tests/npu/a2a3/src/st/testcase/tdequant/tdequant_kernel.cpp:50](../tests/npu/a2a3/src/st/testcase/tdequant/tdequant_kernel.cpp#L50): `TLOAD(dstTile, dstGlobal);` while `dstTile` is only an OUTPUT of `TDEQUANT(dstTile, srcTile, scaleTile, offsetTile)` at line 58. `tdequant` is in `ALL_TESTCASES`, i.e., currently builds in auto mode.
4. **Fix** — Delete the redundant `TLOAD` on dst. If you really need to seed dst (e.g., RMW), the legitimate analogue is `TAXPY` ([tests/npu/a2a3/src/st/testcase/taxpy/taxpy_kernel.cpp:36](../tests/npu/a2a3/src/st/testcase/taxpy/taxpy_kernel.cpp#L36)).
5. **Confidence** — Medium for "it is silently wrong"; High for "this matches the explicit anti-pattern in the rules doc". (Whether the test passes today is Unknown.)
6. **Status** — Inferred risk; Known anti-pattern.

---

## Group 2 — Synchronization

### 2.1 Unguarded `set_flag` / `wait_flag` in kernel-level code

1. **Pattern** — `set_flag(PIPE_X, PIPE_Y, EVENT_ID0); wait_flag(...)` directly in the kernel (not inside a `__tf__` helper, not inside `#ifndef __PTO_AUTO__`).
2. **Why risky** — Auto mode inserts its own synchronization. Manual `set_flag`/`wait_flag` either conflicts with the auto-sync analysis or fails to compile because the kernel-level usage is a CCE intrinsic ([docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md §3.3](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md): *"If you call set_flag and wait_flag directly in your kernel, you always need to manually guard it using the macro `__PTO_AUTO__`"*).
3. **Where**
   - All of [tests/npu/a2a3/src/st/testcase/tfa/tfa_kernel.cpp](../tests/npu/a2a3/src/st/testcase/tfa/tfa_kernel.cpp) — dozens of `set_flag`/`wait_flag` with no guard (consistent with this file being **excluded** from `ALL_TESTCASES`; Known).
   - Kernels under [kernels/manual/](../kernels/manual/) — e.g., [kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp:40](../kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp#L40) wrappers `SetFlag`/`WaitFlag`. Manual-only by construction.
4. **Fix** — In kernel code: drop them; auto mode synchronizes. If absolutely needed, wrap in `#ifndef __PTO_AUTO__ ... #endif` (canonical pattern in [tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp:91-94](../tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp#L91-L94)). For library-internal sync inside a `__tf__` body, raw `set_flag`/`wait_flag` is allowed per [docs/auto_mode/Library_Developer_Rules_And_Limitations.md §3](../docs/auto_mode/Library_Developer_Rules_And_Limitations.md).
5. **Confidence** — High.
6. **Status** — Known.

### 2.2 Unguarded `pipe_barrier(PIPE_*)` at kernel scope

1. **Pattern** — `pipe_barrier(PIPE_ALL)` or `pipe_barrier(PIPE_V)` directly in the kernel function body, NOT inside a `__tf__` helper, NOT inside `#ifndef __PTO_AUTO__`.
2. **Why risky** — `pipe_barrier` is a CCE intrinsic; per [docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md §3.3](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md) and §3.2, kernel devs should use `PtoSetWaitFlag`/`TSYNC`. Whether the compiler accepts `pipe_barrier` at kernel scope under `--cce-enable-pto-passes` is Unknown.
3. **Where** (intentional cross-mode hits with apparent `#ifndef __PTO_AUTO__` guards omitted)
   - [tests/npu/a2a3/src/st/testcase/tcolexpandmax/tcolexpandmax_kernel.cpp:64](../tests/npu/a2a3/src/st/testcase/tcolexpandmax/tcolexpandmax_kernel.cpp#L64): `pipe_barrier(PIPE_ALL);` is OUTSIDE the preceding `#ifndef __PTO_AUTO__ ... #endif` block. Same shape in [tcolexpandmul](../tests/npu/a2a3/src/st/testcase/tcolexpandmul/tcolexpandmul_kernel.cpp), [tcolexpanddiv](../tests/npu/a2a3/src/st/testcase/tcolexpanddiv/tcolexpanddiv_kernel.cpp), [tcolexpandadd](../tests/npu/a2a3/src/st/testcase/tcolexpandadd/tcolexpandadd_kernel.cpp). All three are in `ALL_TESTCASES`.
   - Inside-the-guard usage (the correct pattern): [tests/npu/a2a3/src/st/testcase/textract/textract_kernel.cpp:62-69](../tests/npu/a2a3/src/st/testcase/textract/textract_kernel.cpp#L62-L69), [tests/npu/a2a3/src/st/testcase/tmrgsort/tmrgsort_kernel.cpp:160](../tests/npu/a2a3/src/st/testcase/tmrgsort/tmrgsort_kernel.cpp#L160).
4. **Fix** — Wrap in `#ifndef __PTO_AUTO__ ... #endif`. Inside a `__tf__` helper, `pipe_barrier` is permitted.
5. **Confidence** — Medium for "this is silently broken"; High for "this conflicts with the rules doc".
6. **Status** — Inferred risk (the testcase still builds); Known anti-pattern.

### 2.3 `Event<Op::A, Op::B>` chains inside a kernel

1. **Pattern** — `Event<Op::TLOAD, Op::TADD> e0; e0 = TLOAD(...); event1 = TADD(..., e0);` style.
2. **Why risky** — Not actually risky for simple kernels — `Event<>` is a no-op in auto mode (Known: [docs/auto_mode/Auto_Mode_Overview.md](../docs/auto_mode/Auto_Mode_Overview.md), [include/pto/common/event.hpp](../include/pto/common/event.hpp)). The risk is **intent**: a kernel writer may believe Events provide ordering in auto mode and rely on them where they should not. They do not.
3. **Where** — Used safely in [tests/npu/a2a3/src/st/testcase/tadd/tadd_kernel.cpp:35-41](../tests/npu/a2a3/src/st/testcase/tadd/tadd_kernel.cpp#L35-L41), [tpow_kernel.cpp:39-45](../tests/npu/a2a3/src/st/testcase/tpow/tpow_kernel.cpp#L39-L45), [taxpy_kernel.cpp:32-38](../tests/npu/a2a3/src/st/testcase/taxpy/taxpy_kernel.cpp#L32-L38). All build in auto mode.
4. **Fix** — Keep them if they aid manual-mode readability; do not add new `Event<>` declarations whose ordering is **only** correct in manual mode (e.g., to permit a write-after-read with only an Event "barrier").
5. **Confidence** — High.
6. **Status** — Known.

### 2.4 FFTS device-side handshakes (`wait_flag_dev` / `TPipe` / `TPUSH` / `TPOP`)

1. **Pattern** — `TPUSH<...>(pipe, slot)`, `TPOP<...>(pipe, slot)`, `wait_flag_dev(BUF0_QK_READY)`, `TPipe<flagId, dirType, slotSize, slotNum, ...>` declarations **via the old `TPipe` API**.
2. **Why risky** — `TPipe`/`TPUSH`/`TPOP` "currently directly use TASSIGN inside their implementations, which doesn't work for auto mode" (Known: [tests/npu/a2a3/src/st/testcase/CMakeLists.txt:213-220](../tests/npu/a2a3/src/st/testcase/CMakeLists.txt#L213-L220), [tests/npu/a5/src/st/testcase/CMakeLists.txt:243-250](../tests/npu/a5/src/st/testcase/CMakeLists.txt#L243-L250), [docs/auto_mode/Library_Developer_Rules_And_Limitations.md §4](../docs/auto_mode/Library_Developer_Rules_And_Limitations.md)). FFTS-managed pipelines depend on user-managed buffer flags — incompatible with the auto-allocator.
3. **Where** — [tests/npu/a2a3/src/st/testcase/tfa/tfa_kernel.cpp](../tests/npu/a2a3/src/st/testcase/tfa/tfa_kernel.cpp), [tests/npu/a2a3/src/st/testcase/tpushpop_*/](../tests/npu/a2a3/src/st/testcase/) (all `tpushpop_*` testcases are explicitly excluded from auto mode), [kernels/manual/common/flash_atten/](../kernels/manual/common/flash_atten/), [include/pto/npu/a2a3/TPush.hpp](../include/pto/npu/a2a3/TPush.hpp), [include/pto/npu/a2a3/TPop.hpp](../include/pto/npu/a2a3/TPop.hpp).
4. **Fix** — Do NOT use the old `TPipe` / bare `TPUSH` / `TPOP` in an auto-mode kernel. Use `TMPipe<FlagId, FIFOType::GM_FIFO, Depth, SyncPeriod, ProdTile, ConsTile, UF, Offset>` (from `<pto/npu/a2a3/custom/TSync_Custom.hpp>`) combined with the setter API:
   ```cpp
   using QKPipe = TMPipe<BUF0_QK_READY, FIFOType::GM_FIFO, FifoDepth, SyncPeriod,
                         TileQKData, TileDataF_T, UF_ENABLE, 0>;
   QKPipe qkPipe(qk_tile_fifo_block_ptr);
   // Producer side:
   qkPipe.prod.setTileId(tile_id, sub_tile);
   qkPipe.prod.setAllocateStatus(should_wait);   // wait for consumer slot
   qkPipe.prod.setRecordStatus(should_record);   // signal data ready
   qkPipe.prod.setEntryOffset(byte_offset);
   TPUSH(accTile, qkPipe);
   // Consumer side:
   qkPipe.cons.setTileId(tile_id, row_slice);
   qkPipe.cons.setWaitStatus(should_wait);       // wait for producer
   qkPipe.cons.setFreeStatus(should_free);       // signal slot reusable
   qkPipe.cons.setEntryOffset(byte_offset);
   TPOP(vecTile, qkPipe);
   ```
   This is hardware-confirmed in [kernels/automode/a2a3/flash_atten/fa_performance_kernel.cpp](../kernels/automode/a2a3/flash_atten/fa_performance_kernel.cpp) (see [known_good_kernel_examples.md §A19](known_good_kernel_examples.md)). The `FifoType::GM_FIFO` form uses a GM intermediate buffer rather than TASSIGN-based UB slots, making it compatible with the auto-allocator. (Inferred — internal implementation of `TMPipe` not independently audited, but hardware produces correct output.)
5. **Confidence** — High for old `TPipe` being unsafe. Inferred for `TMPipe` being safe (hardware-confirmed, mechanism not independently audited).
6. **Status** — Known (old `TPipe` unsafe). Inferred (`TMPipe` safe via hardware run).

### 2.5 Double / multi-buffering (ping-pong) in kernel code

1. **Pattern** — `TileMatA aMatTile[BUFFER_NUM]; TASSIGN(aMatTile[0], 0x0); TASSIGN(aMatTile[1], 0x0 + L0_PINGPONG_BYTES);` plus `mte2DBFlag` / pingpong toggles.
2. **Why risky** — *"It's strongly recommended NOT to use double/multi buffering at the moment"* — [docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md §1.4](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md). Auto-sync cannot reason about the user-managed flag toggles.
3. **Where** — [kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp:195-206](../kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp#L195-L206), [kernels/manual/a2a3/gemm_ar/gemm_compute_kernel.cpp:209-221](../kernels/manual/a2a3/gemm_ar/gemm_compute_kernel.cpp#L209-L221), [kernels/manual/a2a3/conv2d_forward/conv2d_forward_kernel.cpp:225-272](../kernels/manual/a2a3/conv2d_forward/conv2d_forward_kernel.cpp#L225-L272), and the FA kernels.
4. **Fix** — Single-buffer the kernel for the auto-mode skeleton; revisit only when the auto-mode compiler exposes a sanctioned pipeline abstraction.
5. **Confidence** — High.
6. **Status** — Known.

### 2.6 Static state across iterations (e.g., `getPingPong()` static)

1. **Pattern** — `[aicore] inline uint64_t getPingPong(uint32_t flip) { static uint64_t pingpong = 0; ... }` ([tests/npu/a2a3/src/st/testcase/tfa/pto_macro_matmul.hpp:42-49](../tests/npu/a2a3/src/st/testcase/tfa/pto_macro_matmul.hpp#L42-L49)).
2. **Why risky** — Per-call mutable state plus auto-mode liveness analysis is a recipe for surprises; the auto-sync compiler does not model this state.
3. **Where** — `pto_macro_matmul.hpp` (FA helpers).
4. **Fix** — Pass the pingpong index as a function argument, or — better in auto mode — drop the entire ping-pong scheme.
5. **Confidence** — Medium.
6. **Status** — Inferred.

### 2.7 `PtoSetWaitFlag` inside a `__tf__` body silently drops sync in auto mode

1. **Pattern** — A library helper is `__tf__ PTO_INTERNAL` (a tile function) and uses `PtoSetWaitFlag<PIPE_X, PIPE_Y>()` for sync between PTO operations *inside* its body, with no `__PTO_AUTO__` guard.
2. **Why risky** — `PtoSetWaitFlag` is intentionally a no-op in auto mode (it exists so kernel-level code can be written once and let the auto-sync compiler insert real sync). But the auto-sync compiler **does not look inside tile functions** ([docs/auto_mode/Auto_Mode_Overview.md](../docs/auto_mode/Auto_Mode_Overview.md): *"the tile function is a complete black-box to PTO compiler"*). So inside a `__tf__` body, `PtoSetWaitFlag` becomes a no-op AND the compiler does not insert sync to compensate — the function ships with no sync at all. This is **exactly opposite** to the kernel-level rule that prefers `PtoSetWaitFlag` over raw `set_flag`/`wait_flag`. The library spec ([docs/auto_mode/Library_Developer_Rules_And_Limitations.md §3](../docs/auto_mode/Library_Developer_Rules_And_Limitations.md)) says it directly: *"Use `set_flag`, `wait_flag` or `pipe_barrier` explicitly in tile functions and all of their callees. Use `PtoSetWaitFlag` or `TSYNC` anywhere else."*
3. **Where, symptom, fix code, and historical source-evidence list** — [compile_error_logbook.md §E2](compile_error_logbook.md). Merged PR-852 applies the `#ifndef __PTO_AUTO__`-guarded wrap across the affected headers; see [external_context/pr_852_notes.md §L2, §L3, §L6, §L7b](external_context/pr_852_notes.md).
4. **Confidence** — High.
5. **Status** — Historical Known anti-pattern; fixed in the current branch by merged PR-852. Keep as a review rule to avoid reintroducing the bug.

> Cross-cutting note (refines §2.1): the kernel rule "prefer `PtoSetWaitFlag`/`TSYNC` over `set_flag`/`wait_flag`" applies **only at kernel level**. Inside a `__tf__` body the polarity is reversed: real `set_flag`/`wait_flag`/`pipe_barrier` are required, and `PtoSetWaitFlag` is wrong.

### 2.8 Mismatched unit-flag phases between MMAD and L0C readers

1. **Pattern** — A Split-K / Split-F cube accumulation uses phase-less `TMATMUL` / `TMATMUL_ACC` while the L0C result is consumed by a unit-flag-aware reader, or marks a non-final accumulation slice as `AccPhase::Final`, or never marks the true final slice as `AccPhase::Final`.
2. **Why risky** — The MMAD producer and FixPipe/MTE reader coordinate at L0C block granularity through the unit-flag bits. The public source maps `AccPhase::{Unspecified, Partial, Final}` to `{0x0, 0x2, 0x3}` ([include/pto/common/type.hpp:233-239](../include/pto/common/type.hpp#L233-L239)) and passes the value into `mad(...)` ([include/pto/npu/a2a3/TMatmul.hpp:37-52](../include/pto/npu/a2a3/TMatmul.hpp#L37-L52)). The reader side has the same `STPhase` encoding ([type.hpp:224-230](../include/pto/common/type.hpp#L224-L230)) and writes it into `copy_matrix_cc_to_gm`'s `Xt[33:32]` unit-flag control field ([include/pto/npu/a2a3/TStore.hpp:318-335](../include/pto/npu/a2a3/TStore.hpp#L318-L335)). If the producer never publishes `Final`, a `STPhase::Final` reader can wait forever. If the producer publishes `Final` before all K/F slices are accumulated, the reader can observe an incomplete accumulator block.
3. **Where** — Pattern to scan for in any cube kernel that performs Split-K / Split-F and consumes `TileAcc` through `TSTORE` Acc→GM or a UF-aware FIFO path. A known-good selection helper exists in [kernels/automode/a2a3/flash_atten/pto_macro_matmul.hpp:101-123](../kernels/automode/a2a3/flash_atten/pto_macro_matmul.hpp#L101-L123), with phase-specific calls at [pto_macro_matmul.hpp:154-171](../kernels/automode/a2a3/flash_atten/pto_macro_matmul.hpp#L154-L171).
4. **Fix** — Use `TMATMUL<AccPhase::Partial>` for the first non-final slice, `TMATMUL_ACC<AccPhase::Partial>` for middle slices, and `TMATMUL_ACC<AccPhase::Final>` for the final slice. If there is only one slice, use `TMATMUL<AccPhase::Final>`. Pair with the corresponding reader phase, e.g. `TSTORE<STPhase::Final, AccTile, GlobalData, AtomicType::AtomicNone>(dst, accTile)`. See [pto_auto_mode_hw_optimization_guide.md §22.3](pto_auto_mode_hw_optimization_guide.md#223-unit-flag-phases-for-tmatmul--tmatmul_acc-and-l0c-readers).
5. **Confidence** — High for the public API / enum plumbing; Medium for the exact hardware stall semantics because they come from user-provided internal docs rather than repo source.
6. **Status** — Known API rule; hardware behavior source is external/user-provided.

---

## Group 3 — Calls into CCE / library internals from kernel code

### 3.1 Raw CCE intrinsics in the kernel function body (outside any `__tf__` helper)

1. **Pattern** — Calls like `set_mask_norm()`, `set_vector_mask(-1, -1)`, `copy_cbuf_to_gm(...)`, `copy_cbuf_to_ubuf(...)`, `create_cbuf_matrix(...)`, `pipe_barrier(...)`, `set_flag(...)`, `wait_flag(...)` directly in the kernel-entry function.
2. **Why risky** — *"Kernel developers using PTO should call PTO instructions only; they shouldn't call CCE intrinsics directly. ... they take raw pointer as arguments, which won't compile for auto mode (since tile is represented as a vector type instead of pointer type)."* — [docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md §3.2](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md).
3. **Where**
   - [demos/auto_mode/baseline/add/csrc/kernel/add_custom.cpp:30-31](../demos/auto_mode/baseline/add/csrc/kernel/add_custom.cpp#L30-L31): `set_mask_norm(); set_vector_mask(-1, -1);` at the top of `runTAdd`. **This is the auto-mode demo** — Inferred that these specific calls are tolerated as kernel-entry boilerplate, but it is an anomaly versus the general rule. Treat as "do not generalize".
   - [tests/npu/a2a3/src/st/testcase/texpands_mat/texpands_mat_kernel.cpp:34](../tests/npu/a2a3/src/st/testcase/texpands_mat/texpands_mat_kernel.cpp#L34) — wraps `copy_cbuf_to_gm` inside `__tf__ AICORE inline TSTORE_MAT2GM_CONVTILE`, but this is still excluded from auto mode by [tests/npu/a2a3/src/st/testcase/CMakeLists.txt:213-220](../tests/npu/a2a3/src/st/testcase/CMakeLists.txt#L213-L220) ("texpands_mat currently has to directly call CCE intrinsics in the kernel, which won't compile in auto mode; besides the auto-sync won't work with raw CCE intrinsics").
4. **Fix** — Use the user-facing PTO instruction (`TLOAD`, `TSTORE`, `TMOV`, `TSYNC`, `TFILLPAD`, …). If no user-facing equivalent exists, the rules doc says to "submit a request to pto-isa to add a new PTO instruction".
5. **Confidence** — High.
6. **Status** — Known.

### 3.2 `__cce_get_tile_ptr(tile.data())` to get a raw pointer

1. **Pattern** — `__cbuf__ T *p = __cce_get_tile_ptr(tile.data());` then feed `p` to a CCE intrinsic.
2. **Why risky** — In auto mode, `tile.data()` returns a vector type; the macro expects a pointer-like operand and the resulting cast / passthrough may not compile. The `texpands_mat` exclusion comment specifically says "won't compile in auto mode".
3. **Where**
   - [tests/npu/a2a3/src/st/testcase/texpands_mat/texpands_mat_kernel.cpp:23](../tests/npu/a2a3/src/st/testcase/texpands_mat/texpands_mat_kernel.cpp#L23) — excluded from auto mode (CMake).
   - [tests/npu/a2a3/src/st/testcase/tload_gm2mat/tload_gm2mat_kernel.cpp:23,31](../tests/npu/a2a3/src/st/testcase/tload_gm2mat/tload_gm2mat_kernel.cpp#L23) — uses the same pattern but `tload_gm2mat` IS in `ALL_TESTCASES`. Whether it actually compiles under `--cce-enable-pto-passes` is **Unknown**; it may be compiled but functionally broken.
   - [tests/npu/a5/src/st/testcase/tload_shape2d/tload_shape2d_kernel.cpp:20](../tests/npu/a5/src/st/testcase/tload_shape2d/tload_shape2d_kernel.cpp#L20) — same shape on A5.
4. **Fix** — Use the user-facing instruction (`TLOAD` family for L1↔GM, `TMOV` for L1↔L0). Avoid pointer extraction in kernel code entirely.
5. **Confidence** — High for `texpands_mat`; Medium for `tload_gm2mat` / `tload_shape2d` (build-list inclusion conflicts with rules doc).
6. **Status** — Known + Unknown.

### 3.2.1 Pointer arithmetic on the `TileDType` argument BEFORE `__cce_get_tile_ptr`

1. **Pattern** — Inside a `__tf__` helper, computing an offset on the `TileDType` parameter and then passing the result to `__cce_get_tile_ptr`:
   ```cpp
   __ubuf__ float *tmp1 = (__ubuf__ T *)__cce_get_tile_ptr(tmp + 128); // BAD
   ```
   instead of:
   ```cpp
   __ubuf__ float *tmp1 = (__ubuf__ T *)__cce_get_tile_ptr(tmp) + 128; // GOOD
   ```
2. **Why risky** — `tmp` here is the tile's `TileDType` parameter, which in auto mode is a vector type (per [include/pto/common/memory.hpp:29-44](../include/pto/common/memory.hpp#L29-L44)). Doing `tmp + 128` first applies arithmetic to a vector value; the libexpand compiler pass then crashes during RAUW (reported in PR-852). Doing `__cce_get_tile_ptr(tmp) + 128` first lowers `tmp` to a typed `__ubuf__ T *` and then advances that pointer — pointer arithmetic on a real pointer.
3. **Where** (historical pre-merge source)
   - [include/pto/npu/a2a3/TCI.hpp:59,108,147,149,151,203](../include/pto/npu/a2a3/TCI.hpp#L59) — eight historical occurrences across `TCI_b32_repeat`, `TCI_b32_normal`, `TCI_b16_repeat`, `TCI_b16_normal`. Offsets `+128`, `+256`, `+384`.
   - A5 spot-check — [include/pto/npu/a5/Tci.hpp](../include/pto/npu/a5/Tci.hpp) does NOT contain this pattern. **Inferred A3-only.**
4. **Fix** — Always extract first, then offset: `__cce_get_tile_ptr(tmp) + N`. Equivalent corrections for all `(__ubuf__ TmpT *)__cce_get_tile_ptr(tmp + N)` shapes. PR-852 applies this rewrite to all eight TCI sites; in the same diff several `vadds`/`vmuls`/`vconv_*` calls are also corrected to use `dstPtr` (the extracted pointer) rather than the `dst` `TileDType` directly. See [external_context/pr_852_notes.md §L1](external_context/pr_852_notes.md).
5. **Confidence** — High.
6. **Status** — Historical Known anti-pattern; fixed in the current branch by merged PR-852. Keep as a review rule to avoid reintroducing the bug.

### 3.3 Calling `Tile::data()` directly from kernel code

1. **Pattern** — `auto &a = aTile.data();` or `cTile.data()` in kernel code.
2. **Why risky** — *"this is NOT an interface to be called by the kernel developers, it should only be used for library developers instead"* — [docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md §3.2](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md). In auto mode it returns a vector type ([docs/auto_mode/Library_Developer_Rules_And_Limitations.md §1](../docs/auto_mode/Library_Developer_Rules_And_Limitations.md), [include/pto/common/memory.hpp](../include/pto/common/memory.hpp)).
3. **Where**
   - [tests/npu/a5/src/st/testcase/textract/textract_kernel.cpp:281-283, 788](../tests/npu/a5/src/st/testcase/textract/textract_kernel.cpp#L281): `auto &a = aTile.data(); auto &b = bTile.data(); auto &c = cTile.data();` — return-by-reference (matches Library §5 form, but used in **kernel** code which Kernel §3.2 forbids).
   - [tests/npu/a5/src/st/testcase/tmov_ub2l1/tmov_ub2l1_kernel.cpp:96-97](../tests/npu/a5/src/st/testcase/tmov_ub2l1/tmov_ub2l1_kernel.cpp#L96-L97), [tests/npu/a5/src/st/testcase/tload_mx_gmtensor/tload_mx_gmtensor_kernel.cpp:78-129](../tests/npu/a5/src/st/testcase/tload_mx_gmtensor/tload_mx_gmtensor_kernel.cpp#L78) — `tile.data()` passed into `tf_copy_cbuf_to_ubuf<...>(srcTile.data(), aMatTile.data(), ...)`. The helper IS `__tf__`, but the kernel-side `.data()` call is what auto mode disallows.
4. **Fix** — Make the helper a `__tf__` that takes `typename Tile::TileDType` **by value** (not `Tile&`), with `__in__`/`__out__` direction attributes. The caller writes `helper(tile.data(), ...)` — the `.data()` call lives at the kernel/library boundary, not inside the helper. Inside the helper body, use `__cce_get_tile_ptr(param)` directly. **Important**: the older "pass `Tile&` and let the helper call `.data()` internally" shape is itself broken in auto mode and is what PR-852 fixes in `texpands_mat`. See [external_context/pr_852_notes.md §T4a](external_context/pr_852_notes.md) and [qualifier_reference.md §4 item 4](qualifier_reference.md).
5. **Confidence** — High for the rule violation; Medium for "this is the source of compile errors in auto mode" (Unknown until tested).
6. **Status** — Known anti-pattern; Inferred risk.

> Note (Known): `GlobalTensor::data()` is **not** the same thing — it returns a raw GM pointer ([include/pto/common/pto_tile.hpp:549](../include/pto/common/pto_tile.hpp#L549)) and is ubiquitous in test kernels (e.g., `out = dstGlobal.data();`). That usage is fine.

### 3.4 `*_IMPL` calls from kernel code

1. **Pattern** — `TCONCAT_IMPL(dstTile, src0Tile, src1Tile)`, `TCOLEXPAND_IMPL(...)`, `TROWEXPANDSUB_IMPL(...)` in a kernel `.cpp`.
2. **Why risky** — `*_IMPL` is library-only. The user wrapper (e.g., `TCONCAT`) handles `TSYNC` / `Event` plumbing per [include/pto/common/pto_instr.hpp](../include/pto/common/pto_instr.hpp) (the `MAP_INSTR_IMPL` template wrapper). Bypassing it skips that integration point; auto-sync may not see the call.
3. **Where**
   - [tests/npu/a2a3/src/st/testcase/tconcat/tconcat_kernel.cpp:48](../tests/npu/a2a3/src/st/testcase/tconcat/tconcat_kernel.cpp#L48): `TCONCAT_IMPL(dstTile, src0Tile, src1Tile);`
   - [tests/npu/a2a3/src/st/testcase/tconcatidx/tconcatidx_kernel.cpp:65](../tests/npu/a2a3/src/st/testcase/tconcatidx/tconcatidx_kernel.cpp#L65), [tests/npu/a5/src/st/testcase/tconcatidx/tconcatidx_kernel.cpp:65](../tests/npu/a5/src/st/testcase/tconcatidx/tconcatidx_kernel.cpp#L65) — same.
   - [tests/npu/a5/src/st/testcase/tcolexpand/tcolexpand_kernel.cpp:45](../tests/npu/a5/src/st/testcase/tcolexpand/tcolexpand_kernel.cpp#L45): `TCOLEXPAND_IMPL(dstTile, srcTile);`
   - [tests/npu/a2a3/src/st/testcase/tfa/pto_macro_fa_softmax.hpp:56,98,103](../tests/npu/a2a3/src/st/testcase/tfa/pto_macro_fa_softmax.hpp#L56) — `TROWEXPANDSUB_IMPL(...)` in a header that is included by the (excluded) `tfa_kernel.cpp`.
4. **Fix** — Call the user-facing wrapper (`TCONCAT`, `TCOLEXPAND`, `TROWEXPAND` followed by `TSUB`).
5. **Confidence** — Medium for "this breaks auto-sync" (the testcases are in `ALL_TESTCASES`, so they build); High for "this is the wrong public surface".
6. **Status** — Known anti-pattern.

### 3.5 Kernel-scope `TASSIGN_IMPL`, `TRESHAPE_IMPL`, etc.

1. **Pattern** — Same as 3.4 but for memory-shaping IMPLs.
2. **Why risky** — `TASSIGN_IMPL(tile, reinterpret_cast<uintptr_t>(other.data()))` is the manual-mode-only pre-aliasing pattern. In auto mode it must be `TRESHAPE_IMPL` (library) or `TRESHAPE` (kernel) — see 1.3.
3. **Where** — [include/pto/npu/a2a3/TQuant.hpp:108-114](../include/pto/npu/a2a3/TQuant.hpp#L108-L114) is the canonical guarded usage. Anywhere else without the `#ifndef __PTO_AUTO__` branch is risky.
4. **Fix** — `TRESHAPE`/`TSUBVIEW` in kernel; `TRESHAPE_IMPL`/`TSUBVIEW_IMPL` in library `_IMPL`.
5. **Confidence** — High.
6. **Status** — Known.

---

## Group 4 — A3 vs A5 cross-arch hazards

### 4.1 A5-only types accidentally used on A3

1. **Pattern** — Using A5-specific MX FP4/FP8 element types or vector-scalar/SIMT helpers in a kernel meant to also build on A3.
2. **Why risky** — A3 and A5 share the `a2a3/` tree only nominally; per-arch headers diverge. A5 introduces `float4_e1m2x2_t`, `float4_e2m1x2_t`, `float8_e8m0_t`, `vector_f8e5m2`, `vector_f8e4m3`, `vector_f8e8m0` ([include/pto/npu/a5/datatype.hpp:23-39, 135](../include/pto/npu/a5/datatype.hpp#L23-L39)). [include/pto/npu/a5/common.hpp](../include/pto/npu/a5/common.hpp) defines `MaskReg = vector_bool`, `UnalignReg = vector_align`, `AddrReg = vector_address` (A5) — no equivalent in A3 `common.hpp`. Same arch_macro detects A5 via `__NPU_ARCH__ == 3101 || 3510` ([include/pto/common/arch_macro.hpp:16-20](../include/pto/common/arch_macro.hpp#L16-L20)).
3. **Where** — Pattern to scan for; e.g., do NOT include `pto/npu/a5/datatype.hpp` from a kernel intended to build on A3.
4. **Fix** — Gate A5-only types behind `#if defined(PTO_NPU_ARCH_A5)` or template specialization. Prefer the user-facing PTO instruction (`TMATMUL_MX` etc.) which is responsible for arch dispatch via [include/pto/common/pto_instr_impl.hpp](../include/pto/common/pto_instr_impl.hpp).
5. **Confidence** — High.
6. **Status** — Known.

### 4.2 A5-only memory ops (`MGather` / `MScatter`) in A3 paths

1. **Pattern** — Calling `MGATHER` / `MSCATTER` on A3.
2. **Why risky** — `MGather.hpp` and `MScatter.hpp` exist only under [include/pto/npu/a5/](../include/pto/npu/a5/); no A3 counterpart ([include/pto/npu/a2a3/](../include/pto/npu/a2a3/) does not contain these files). The A3-side gather/scatter is `TGATHER`/`TSCATTER` (different shape/contract).
3. **Where** — Pattern to scan for. Tests `mgather/`, `mscatter/` exist only under [tests/npu/a5/src/st/testcase/](../tests/npu/a5/src/st/testcase/).
4. **Fix** — Use `TGATHER`/`TSCATTER` for A3.
5. **Confidence** — High.
6. **Status** — Known.

### 4.3 A3 GEMM alignment assumptions used on A5

1. **Pattern** — Hardcoding `blockAlign = C0_SIZE_BYTE / sizeof(U)` (the A3 form, see [tmatmul A3:128](../tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp#L128)) on an A5 kernel.
2. **Why risky** — A5 uses `(sizeof(AType) == 1) ? 32 : 16` ([tmatmul A5:32](../tests/npu/a5/src/st/testcase/tmatmul/tmatmul_kernel.cpp#L32)). For 8-bit / FP8 / MX-FP4 types the A3 formula gives the wrong block alignment; tile shapes silently misalign.
3. **Where** — Anywhere a developer copies the A3 GEMM skeleton verbatim onto A5 without inspecting A5-side alignment.
4. **Fix** — Use the A5 conditional (or template the alignment by `sizeof(T)`).
5. **Confidence** — High.
6. **Status** — Known.

### 4.4 A2/A3-shared assumption that may not hold on A5: ColMajor `Mat` fractal sizes

1. **Pattern** — `Tile<TileType::Mat, T, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>` literal `512` fractal block size as in [tmatmul A3:51](../tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp#L51).
2. **Why risky** — A5 uses the same `512` in [tmatmul A5:53](../tests/npu/a5/src/st/testcase/tmatmul/tmatmul_kernel.cpp#L53), but MX/FP4/FP8 paths may need a different `SFractalSize`. [include/pto/common/pto_tile.hpp](../include/pto/common/pto_tile.hpp) `TileConfig::fractalABSize` is the documented default. Hard-coding `512` ties the kernel to the FP16 block.
3. **Where** — Both A3 and A5 `tmatmul_kernel.cpp`; OK for FP16/BF16, suspect for FP8/FP4.
4. **Fix** — Use `TileConfig::fractalABSize` or the per-arch helper.
5. **Confidence** — Medium.
6. **Status** — Inferred.

### 4.5 A3 cube/vec macros (`__DAV_C220_VEC__`, `__DAV_C220_CUBE__`) hardcoded

1. **Pattern** — `#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)` as in [demos/auto_mode/baseline/add/csrc/kernel/add_custom.cpp:11](../demos/auto_mode/baseline/add/csrc/kernel/add_custom.cpp#L11).
2. **Why risky** — A5 uses different SoC numbers / cube macros; copying this guard verbatim to A5 produces a kernel that compiles to nothing.
3. **Where** — The auto-mode add demo (A3-only by guard).
4. **Fix** — Replace with the A5 equivalent (e.g., `__CCE_AICORE__ == 310` or a `PTO_NPU_ARCH_A5` check on `arch_macro.hpp`). Confirm the exact A5 SoC macro with the user before writing it.
5. **Confidence** — High for "guard is A3-only"; Medium for "the A5 replacement is `dav-c310-vec`" (Inferred from [docs/auto_mode/Auto_Mode_Overview.md](../docs/auto_mode/Auto_Mode_Overview.md) which mentions `--cce-aicore-arch=dav-c310-vec`).
6. **Status** — Known + Inferred.

---

## Group 5 — Type and template hazards

### 5.1 Default member initializers in tile/struct types

1. **Pattern** — `int shape[ConvTileDetail::MAX_CONVTILE_DIM] = {1};` or any `= ...` default init on a struct/class member.
2. **Why risky** — *"This turns out to cause problems for the SROA pass in the compiler (SROA can't eliminate the AllocaInst...)"* — [docs/auto_mode/Library_Developer_Rules_And_Limitations.md §2](../docs/auto_mode/Library_Developer_Rules_And_Limitations.md). Use a `#ifdef __PTO_AUTO__` branch with an undefaulted member.
3. **Where** — Whenever introducing a new struct/class in library or kernel code.
4. **Fix** — `#ifdef __PTO_AUTO__ /* no default */ #else /* default = {1} */ #endif`.
5. **Confidence** — High.
6. **Status** — Known.

### 5.2 `[aicore]` attribute spelled out instead of the `AICORE` macro

1. **Pattern** — `[aicore] inline uint64_t getPingPong(...)` ([tests/npu/a2a3/src/st/testcase/tfa/pto_macro_matmul.hpp:42, 66](../tests/npu/a2a3/src/st/testcase/tfa/pto_macro_matmul.hpp#L42)).
2. **Why risky** — On CPU-sim and cost-model builds, the `AICORE` macro expands to nothing ([include/pto/common/type.hpp:14-18](../include/pto/common/type.hpp#L14-L18)). The literal `[aicore]` attribute is never defined-away, so cross-build portability fails.
3. **Where** — `pto_macro_matmul.hpp` (FA helpers).
4. **Fix** — Use the `AICORE` macro (or `PTO_INST` / `PTO_INTERNAL`).
5. **Confidence** — Medium.
6. **Status** — Inferred.

### 5.3 `Bias` tile when `__DAV_C220_CUBE__` is undefined

1. **Pattern** — `Tile<TileType::Bias, B, 1, N, ...>` used unconditionally.
2. **Why risky** — [include/pto/common/memory.hpp:73-83](../include/pto/common/memory.hpp#L73-L83) shows `MemoryQualifier<TileType::Bias, DType>::type` is `__biasbuf__ DType` only `#if defined(__DAV_C220_CUBE__)`; otherwise it falls back to `uint64_t`. A kernel that needs the bias tile but is compiled for a vec target gets an effectively-broken type.
3. **Where** — Any GEMM that templates over having/not-having a bias path; e.g., [tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp:53,59](../tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp#L53) declares `BiasTile = Tile<TileType::Bias, ...>` but the test compiles cube-side per `pto_cube_st(tmatmul)`.
4. **Fix** — Gate the bias tile under `#if defined(__DAV_C220_CUBE__)` (or A5 equivalent), or build the kernel with the cube target only.
5. **Confidence** — Medium.
6. **Status** — Known per memory.hpp; Inferred for whether it is currently a problem.

### 5.4 `aclFloat16` in kernel-side template parameters

1. **Pattern** — `runTAdd<aclFloat16, ...>` directly in the kernel function.
2. **Why risky** — `aclFloat16` is a host-side ACL type. Test kernels intentionally cast at the launcher: `if constexpr (std::is_same_v<T, aclFloat16>) runTAdd<half, ...>(...)` (see [tests/npu/a2a3/src/st/testcase/tadd/tadd_kernel.cpp:48-50](../tests/npu/a2a3/src/st/testcase/tadd/tadd_kernel.cpp#L48-L50)).
3. **Where** — Kernels that forget to do the host→device type swap.
4. **Fix** — Always specialize the device kernel on `half` (or `bfloat16_t`) and reinterpret-cast at the launcher boundary.
5. **Confidence** — High.
6. **Status** — Known.

### 5.5 Single template parameter shared across tile-function arguments of different `TileType`s

1. **Pattern** — A `__tf__` helper takes a single `TileData` template parameter and uses it for all of `dst`, `src`, and `tmp` arguments — even though those tiles have different `TileType` (e.g., `Mat` vs `Vec`, or `ConvTile` vs plain `Tile`) or different element types.
2. **Why risky** — Instantiating one template parameter forces all three roles to share the same type. When dst is a `ConvTile<Mat>`, src is a `ConvTile<Mat>`, and tmp is a `Tile<Vec>` (the actual `ttrans_conv` shape per PR-852), there is no type that satisfies all three. Caller-side instantiation produces either a compile error or silently picks the wrong storage class. Auto mode amplifies the risk because `TileDType` (vector vs pointer) depends on `TileType`, so getting the wrong tile type mis-types the `__cce_get_tile_ptr` extractions.
3. **Where** (historical pre-merge source)
   - [include/pto/npu/a2a3/TTrans.hpp](../include/pto/npu/a2a3/TTrans.hpp) — `TTransConvNCHW2NC1HWC0`, `TTransConvNC1HWC02C1HWNC0`, `TTransConvGNCHW2GNC1HWC0`, `TTransConvGNC1HWC02GC1HWNC0` historically used one `TileData` template. Merged PR-852 split this into `TileDataDst`, `TileDataSrc`, `TileDataTmp`.
4. **Fix** — Decouple the template parameters per role: `template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, ...>`. Inside, derive `using Tdst = typename TileDataDst::DType; using Tsrc = typename TileDataSrc::DType; using Ttmp = typename TileDataTmp::DType;`. Use the per-role type for all pointer extractions and casts. Update callers to pass three template arguments. See PR-852 [external_context/pr_852_notes.md §L7a](external_context/pr_852_notes.md).
5. **Confidence** — High.
6. **Status** — Historical Known anti-pattern; fixed in the current branch by merged PR-852. Keep as a review rule to avoid reintroducing the bug.

### 5.6 `ConvTile<Loc, T, BufferSize_, Layout, Shape>` — `BufferSize_` is element count, not bytes

1. **Pattern** — Treating the `BufferSize_` template parameter of `ConvTile` as a byte size:
   ```cpp
   constexpr int elementSize = N * C1 * H * W * C0;
   constexpr int bufferSizeA = elementSize * sizeof(T); // BAD: this is bytes
   using TileData = ConvTile<TileType::Mat, T, bufferSizeA, Layout::NC1HWC0, ...>;
   ```
   despite the parameter being named "BufferSize", `ConvTile` interprets it as **element count** (Inferred from PR-852 description and from the `static constexpr int bufferSize = BufferSize_;` member visible in the PR-quoted struct definition).
2. **Why risky** — Passing `elementSize * sizeof(T)` allocates `sizeof(T)`× the intended UB region. PR-852 description: *"Allocated tiles were bigger than the UB. ... Fix: Used the correct variable, numElems, for ConvTile Construction. Long-term Fix (TODO): Modify the template variable's name, which is misleading."*
3. **Where** (historical pre-merge source)
   - [tests/npu/a2a3/src/st/testcase/texpands_mat/texpands_mat_kernel.cpp:58, 63](../tests/npu/a2a3/src/st/testcase/texpands_mat/texpands_mat_kernel.cpp#L58) historically used `bufferSizeA = elementSize * sizeof(T); ConvTile<TileType::Mat, T, bufferSizeA, ...>`. Current source passes `elementSize`.
   - `ConvTile` is defined at [include/pto/common/pto_tile.hpp:1096](../include/pto/common/pto_tile.hpp#L1096) with `static constexpr int bufferSize = BufferSize_;`.
4. **Fix** — Pass element count: `ConvTile<TileType::Mat, T, elementSize, Layout::NC1HWC0, ...>`. Long-term: rename the template parameter to `NumElems_` (TODO from PR-852). See [external_context/pr_852_notes.md §T4b](external_context/pr_852_notes.md).
5. **Confidence** — High.
6. **Status** — Historical Known anti-pattern; fixed in the current branch by merged PR-852. Keep as a review rule to avoid reintroducing the bug.

### 5.7 Using source-element widths for post-TSORT32 merge logic (packed-width mismatch)

1. **Pattern** — After `TSORT32(packed, src, idx, scratch)`, the destination tile holds (val, idx) packed pairs of width `srcWidth * 2 * TYPE_COEF`. Bug: continuing to use the **source-element** width for any of the subsequent merge / extract steps:
   - the main `TMRGSORT` 4-way self-merge loop bound (`blockLen * 4 <= srcWidth` instead of `<= packedWidth`),
   - the `cols = srcWidth / (blockLen * 4) * (blockLen * 4)` width that drives prefix views,
   - the tail-block guard (`blockLen < srcWidth`),
   - `FillMrgArray<srcWidth>(...)` template arg,
   - the `tmpMrgSortedLen / tmpMrgArray` clip cap (`> topK ? topK` instead of `> packedTopK ? packedTopK`),
   - the TMRGSORT scratch-tile width (one source row vs. one packed row),
   - the final `TGATHER` prefix-view width.
2. **Why risky** — The merge sort then operates only on the first `srcWidth` packed elements (= first half of the TSORT32 output for `TYPE_COEF=1`); the remaining packed run is never seen by the merge. Output is silently truncated — half the source data is dropped before reaching `TGATHER`. The `tmp1Tile` width / template-arg mismatch also makes `TMRGSORT`'s `TmpTile` template arg disagree with `Dst`/`Src` — undefined behavior at best.
3. **Where** — Pattern observation, not a single source path. Caught during the [kernels/automode/a2a3/topk/](../kernels/automode/a2a3/topk/) bring-up. The manual TopK avoids the trap by introducing `dstCols = validCol * 2 * TYPE_COEF` ([kernels/manual/a2a3/topk/topk_kernel.cpp:286-287](../kernels/manual/a2a3/topk/topk_kernel.cpp#L286-L287)) and threading it as the `valid_col` template arg into `MrgsortSingleRow` ([call site:201](../kernels/manual/a2a3/topk/topk_kernel.cpp#L201)) and as `Cols` into `SortTailBlock` ([call site:105](../kernels/manual/a2a3/topk/topk_kernel.cpp#L105)).
4. **Fix** — Define both source-width and packed-width constants in the kernel and use the packed forms throughout the post-TSORT32 path:
   ```cpp
   constexpr int kPackedCols = kCols * 2 * TYPE_COEF;   // ↔ manual `dstCols`
   constexpr int kPackedTopK = kTopK * 2 * TYPE_COEF;   // ↔ manual `dtopk`
   ```
   - Use `kPackedCols` for: main merge loop bound, `cols` width in the loop, tail-block guard, `FillMrgArray<>`, scratch tile width and tile type, post-TSORT32 tile-type capacity.
   - Use `kPackedTopK` for: tail-merge `tmpMrgSortedLen`/`tmpMrgArray` clip cap, final `TGATHER` source prefix view width.
   - All four `TMRGSORT` template roles (`Dst, Tmp, Src0, Src1`) should resolve to the **packed** tile type so the scratch matches the merge format.
5. **Confidence** — High (caught and fixed during the topk bring-up; user-confirmed PASS after fix).
6. **Status** — Known anti-pattern. Resolved-by-experiment in [kernels/automode/a2a3/topk/](../kernels/automode/a2a3/topk/); see [known_good_kernel_examples.md §A12](known_good_kernel_examples.md) and [tile_type_reference.md §11.2](tile_type_reference.md).

---

## Group 6 — Misleading / "builds but suspect" patterns

### 6.1 `tquant` ST kernel — passes auto-mode build but uses manual-mode aliasing

See [known_good_kernel_examples.md §D3](known_good_kernel_examples.md). Combined risks: 1.1 (overlapping `TASSIGN`) + 2.1 guarded properly. Auto-mode functional correctness Unknown.

- Confidence: Medium. Status: Inferred.

### 6.2 `tdequant` ST kernel — passes auto-mode build but `TLOAD`s a write-only dst

See [known_good_kernel_examples.md §D4](known_good_kernel_examples.md). Combined risks: 1.5 (redundant `TLOAD`).

- Confidence: Medium. Status: Inferred.

### 6.3 `tload_gm2mat` / `tload_shape2d` ST kernels — pass auto-mode build but use raw `__cce_get_tile_ptr(.data())`

See 3.2. Listed in `ALL_TESTCASES` for both A3 and A5 but use the same construct that excludes `texpands_mat`. Either there is a vector-aware overload of `__cce_get_tile_ptr` for auto mode, or these tests pass by coincidence. Worth verifying.

- Confidence: Medium. Status: Unknown.

### 6.4 `textract` A5 ST kernel — direct `aTile.data()` in kernel body

See 3.3. `auto &a = aTile.data();` in [tests/npu/a5/src/st/testcase/textract/textract_kernel.cpp:281-283](../tests/npu/a5/src/st/testcase/textract/textract_kernel.cpp#L281-L283). Per Kernel rule §3.2 this should not appear in a kernel.

- Confidence: High for the rule violation; Medium for "this is silently wrong in auto mode".
- Status: Inferred.

### 6.5 `pto_macro_fa_softmax.hpp` looks like a kernel-callable softmax helper, but it isn't

1. **Pattern** — Header named like a reusable softmax helper, used from `tfa_kernel.cpp` only. Calls `pipe_barrier(PIPE_V)` and `*_IMPL` directly.
2. **Why risky** — A new kernel author may include the header and call `pto_macro_fa_softmax(...)` from kernel code. That call is library-internal: it relies on caller-provided `__tf__` context for `pipe_barrier` correctness and on caller-managed live-range invariants.
3. **Where** — [tests/npu/a2a3/src/st/testcase/tfa/pto_macro_fa_softmax.hpp](../tests/npu/a2a3/src/st/testcase/tfa/pto_macro_fa_softmax.hpp).
4. **Fix** — Treat this file as a manual-mode reference for the math sequence only ([known_good_kernel_examples.md §B3](known_good_kernel_examples.md)). Re-derive the softmax sequence in a clean auto-mode helper using user-facing instructions (`TROWMAX`, `TROWEXPAND` + `TSUB`, `TMULS`, `TEXP`, `TROWSUM`, `TROWEXPAND` + `TDIV`).
5. **Confidence** — High.
6. **Status** — Known.

### 6.6 The orphan `tfa` testcase looks like an in-tree FA reference but is not built

See [known_good_kernel_examples.md §C4](known_good_kernel_examples.md). Listed as a directory but absent from `ALL_TESTCASES`. Nothing here is exercised by CI under either mode. Do not use as a "this works" reference.

- Confidence: High. Status: Known.

---

## Group 7 — Auto-sync interactions

### 7.1 Loop-invariant if-statements nested incorrectly

1. **Pattern** — Inner-loop `if` condition that does not depend on the inner induction variable, but is left inside the inner loop.
2. **Why risky** — Auto-sync loop peeling cannot statically remove these; the compiler "may generate the sync operations more conservatively" ([docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md §1.2](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md)).
3. **Where** — Pattern to scan for. Symptom: a kernel works but auto-mode performance is markedly worse than manual.
4. **Fix** — Hoist the `if` outside the inner loop.
5. **Confidence** — High (rule); Medium for spotting it after the fact.
6. **Status** — Known.

### 7.2 Complex compound conditions on PTO-instruction-guarding `if`

1. **Pattern** — `if ((tile.GetValidRow() > 16 || tile.GetValidCol() > 16) && tile.GetKAligned()) { TLOAD(...); }`.
2. **Why risky** — Auto-sync loses precision when the guard expression is not first hoisted into a `bool cond = ...` ([docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md §1.3](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md)).
3. **Where** — Pattern to scan for in any new GEMM/FA-style kernel.
4. **Fix** — Hoist into a `bool cond` first; use `cond` in the `if`.
5. **Confidence** — High. Status — Known.

### 7.3 First/last-iteration guards that are NOT statically peelable

1. **Pattern** — `if (tile_id == 0)` is fine; `if (some_runtime_value == sentinel)` defeats peeling.
2. **Why risky** — Per [docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md §1.1](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md), guards on first/last iteration must be statically evaluable so the compiler can peel them and simplify auto-sync.
3. **Where** — Good example in [tmatmul A3:301-309](../tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp#L301-L309) (`if (i == 0) TMATMUL else TMATMUL_ACC`) — pattern to mirror.
4. **Fix** — Use the loop induction variable directly; avoid wrapping the guard behind a non-constexpr predicate.
5. **Confidence** — High. Status — Known.

### 7.4 Cross-iter auto-sync gap on complex row-loop pipelines — `pipe_barrier(PIPE_ALL)` as a sanctioned escape hatch

1. **Pattern** — A multi-iter in-kernel loop whose body contains a **complex pipeline of PTO ops on multiple distinct tiles** (e.g. `TLOAD srcTile + TLOAD idxTile → TSORT32 → TMRGSORT-loop → SortTailBlock → TGATHER (P0101) → TGATHER (P1010 via TRESHAPE type-pun) → TSTORE × 2`). The whole body is repeated per row, with the SAME tile *types* re-used iteration to iteration. No `set_flag`/`wait_flag`/`pipe_barrier`/`TPipe`/double-buffering is present.
2. **Why risky** — The auto-sync passes (which insert MTE/V ordering) track dependencies **per tile identity, not per UB region**. For a simple body (`TLOAD a → TLOAD b → TADD c → TSTORE c`, like `add_tile_array`) this is sufficient: same-tile WAR/WAW deps across iters are caught. For the TopK-shape body above, the failure mode is:
   - The auto allocator can decide that `outIdxTile` (live only `TGATHER → TSTORE`) and `idxTile` (live only `TLOAD → TSORT32`) have disjoint per-iter lifetimes and place them in the **same UB block**.
   - Auto-sync sees them as different tiles and does not order iter `i`'s `TSTORE outIdxTile` against iter `i+1`'s `TLOAD idxTile`.
   - Iter `i+1`'s TLOAD then clobbers the UB region before iter `i`'s TSTORE retires.
   - Symmetric failure for `outValTile`↔`srcTile`.
3. **Where, symptoms confirmed on Ascend910B1** (kernel: [kernels/automode/a2a3/topk/topk_kernel.cpp](../kernels/automode/a2a3/topk/topk_kernel.cpp), `kRows=4, kCols=1280, kTopK=512`):
   - **No barrier, per-iter TLOAD, tiles inside loop**: rows 0/1/2 indices clobbered to identity (`[0, 1, 2, …]` — the contents `idxTile` held after iter `i+1`'s `TLOAD`), row 3 (last iter, no follower) idx perfect. Values ~25% bad per row.
   - **No barrier, pre-loaded `srcAllTile` + `idxTile` outside the loop**: rows 0/2/3 fully correct, row 1 fully wrong — a different but still cross-iter artifact (Inferred: auto-pipelining at the second iter).
   - **`pipe_barrier(PIPE_ALL)` at the START of each row iter** (either variant): `test value success` + `test index success` + `test success`. The barrier serializes the previous iter's ops before the next iter starts, closing the gap.
4. **Fix — preferred order**:
   1. First, try to remove the alias risk by data flow: pre-load shared inputs ONCE before the loop (e.g. an identity `idxTile` that is row-invariant), and use `TSUBVIEW` to slice per row. This eliminates one alias class without manual sync.
   2. If a hardware run still shows iter-dependent failures (entire rows wrong, only "last iter correct" patterns, identity-shaped output), add `pipe_barrier(PIPE_ALL)` at the **start** of each iter as a serializing barrier.
   3. Treat the barrier as load-bearing: leave an inline comment with the failure symptom + the hardware datum that motivated it. The barrier is a deliberate, narrowly-scoped deviation from §2.2 ("unguarded `pipe_barrier` at kernel scope is an anti-pattern").
5. **When to NOT reach for the barrier**:
   - Single-iter kernels (no loop).
   - Loops with simple bodies that match the `add_tile_array` shape (`TLOAD a, TLOAD b, T<op> c, TSTORE c`) — no evidence the auto-sync gap manifests there.
   - Anywhere correctness is not yet confirmed on hardware — the barrier is a **fix for an observed failure**, not a prophylactic. Adding it speculatively masks bugs and burns auto-mode parallelism unnecessarily.
6. **Confidence** — High that the barrier-at-iter-start recipe works for the TopK 2D row-loop shape (hardware-confirmed). Medium for the underlying mechanism (the auto-sync passes' exact treatment of disjoint-lifetime aliased tiles is not documented in [docs/auto_mode/](../docs/auto_mode/) — diagnosis above is Inferred from hardware-observed symptoms).
7. **Status** — Known (sanctioned exception, hardware-confirmed); refines §2.2.

> **Cross-cutting rule** (refines §2.2): kernel-scope `pipe_barrier(PIPE_ALL)` is *normally* an anti-pattern, BUT it is the sanctioned escape hatch when a complex per-iter pipeline demonstrably fails on hardware due to a cross-iter auto-sync gap. Keep usage at a minimum — one barrier per row iteration at most — and document the failure symptom at the call site.

### 7.5 Loop-unroll pragmas used as temporary auto-sync / pattern-recognition workarounds

1. **Pattern** — Adding `#pragma unroll` or `#pragma unroll(N)` on a loop so the compiler sees a simpler / flatter PTO-instruction sequence. This may be used as a temporary hack when auto-sync does not recognize a complex loop pattern, especially nested loops or a loop body whose PTO dependencies are only obvious after unrolling.
2. **Why risky** — This is a compiler-workaround, not an algorithmic requirement. It can hide an auto-sync pattern-recognition bug, increase code size, change scheduling, and make the kernel look more generally supported than it is. User-provided guidance says this workaround is expected to be fixed later in the compiler, so do not treat it as a clean permanent design rule.
3. **Where** — `#pragma unroll` exists in current repo sources such as [include/pto/npu/a2a3/TCI.hpp](../include/pto/npu/a2a3/TCI.hpp) and `#pragma unroll(N)` exists in A5 gather/scatter helpers such as [include/pto/npu/a5/MGather.hpp](../include/pto/npu/a5/MGather.hpp) / [include/pto/npu/a5/MScatter.hpp](../include/pto/npu/a5/MScatter.hpp). The "use it as an auto-sync workaround for complex / nested loops" part is **user-provided guidance**, not yet tied to a specific checked-in kernel failure in this doc.
4. **Fix / usage rule** — Prefer first simplifying the loop shape (hoist invariant guards, peel first/last iterations, split nested loops, or reduce live tile aliasing). If a hardware/compiler run shows auto-sync misses the pattern and unrolling fixes it, keep the pragma narrowly scoped, add an inline comment naming the auto-sync failure, and mark it as temporary. Remove it once the compiler recognizes the pattern without the pragma.
5. **Confidence** — Medium. The pragma syntax is source-grounded; the auto-sync workaround rationale is user-provided and should be upgraded with a concrete build/run occurrence when available.
6. **Status** — Known workaround; future compiler fix expected (user-provided).

---

## Cross-references

- [docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md) — kernel rules (§1.1, §1.2, §1.3, §1.4, §2.1, §2.2, §2.3, §2.4, §3.1, §3.2, §3.3).
- [docs/auto_mode/Library_Developer_Rules_And_Limitations.md](../docs/auto_mode/Library_Developer_Rules_And_Limitations.md) — library rules (§1, §2, §3, §4, §5, §6).
- [docs/auto_mode/Auto_Mode_Overview.md](../docs/auto_mode/Auto_Mode_Overview.md) — `TASSIGN`/`Event`/`TSYNC` are no-ops; tile-function black-box rule.
- [docs/auto_mode/Examples.md](../docs/auto_mode/Examples.md) — clean auto-mode TADD/TMATMUL skeletons.
- [tests/npu/a2a3/src/st/testcase/CMakeLists.txt:213-220](../tests/npu/a2a3/src/st/testcase/CMakeLists.txt#L213-L220) — official auto-mode exclusion list (A3): `tpushpop_*`.
- [tests/npu/a5/src/st/testcase/CMakeLists.txt:243-250](../tests/npu/a5/src/st/testcase/CMakeLists.txt#L243-L250) — official auto-mode exclusion list (A5): `tpushpop_*`, `texpands_mat`.

---

## Open items / things to verify with the user

- Whether `tload_gm2mat` and `tload_shape2d` actually compile under auto mode (Group 6.3). If they do, there is a vector-aware `__cce_get_tile_ptr` overload we have not located in `include/`.
- Whether the unguarded `pipe_barrier(PIPE_ALL)` in `tcolexpand{add,div,max,mul,sub}_kernel.cpp` compiles or is a benign no-op under auto mode (Group 2.2).
- Whether `tquant_kernel.cpp` and `tdequant_kernel.cpp` produce numerically correct output in auto mode (Groups 1.1 and 1.5; Group 6.1, 6.2).
- The exact A5 SoC cube/vec macros (Group 4.5). The Auto_Mode_Overview.md mentions `dav-c310-vec` as the A5 vec target; the analogous CUBE macro for the `#if` guard pattern is unverified.
- Whether `pto::TileConfig::fractalABSize` differs between A3 and A5 (Group 4.4).
