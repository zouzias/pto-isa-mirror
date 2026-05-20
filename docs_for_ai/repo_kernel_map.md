# Repo Kernel Map (auto-mode A3/A5)

Source-grounded layout map for generating, reviewing, and debugging auto-mode
kernels for A3 and A5. Every claim links to a path inside this repo or is
labeled `Assumption` / `Unknown`. No compilation has been performed.

Confidence labels used below:
- **Known** — directly readable from a file path quoted on the same line.
- **Inferred** — drawn by combining two or more files; noted where used.
- **Assumption** — not present in source, flagged for the user to confirm.
- **Unknown** — referenced by source but not yet inspected; needs follow-up.

---

## 1. Where kernels live

- [kernels/](../kernels/) — top-level kernel projects. (Known: [kernels/README.md](../kernels/README.md))
  - [kernels/manual/](../kernels/manual/) — hand-tuned (manual-mode) kernels with explicit `TASSIGN` and event/sync. (Known: [kernels/manual/README.md](../kernels/manual/README.md))
    - [kernels/manual/a2a3/](../kernels/manual/a2a3/) — A2/A3 manual kernels.
    - [kernels/manual/a5/](../kernels/manual/a5/) — A5 manual kernels.
    - [kernels/manual/common/](../kernels/manual/common/) — cross-platform manual kernels (currently `flash_atten`).
  - [kernels/custom/](../kernels/custom/) — custom-operator scaffolding. Only example: [kernels/custom/fused_add_relu_mul/](../kernels/custom/fused_add_relu_mul/). (Known)
- [tests/npu/a2a3/src/st/testcase/](../tests/npu/a2a3/src/st/testcase/) and [tests/npu/a5/src/st/testcase/](../tests/npu/a5/src/st/testcase/) — small per-op kernels (`<op>_kernel.cpp`) that double as auto-mode build targets. (Known: per-testcase `CMakeLists.txt` files use `pto_vec_st`, `pto_cube_st`, or `pto_mix_st`.)
- [tests/cpu/st/testcase/](../tests/cpu/st/testcase/) — CPU-sim kernels. Useful as functional reference; the build system here uses different flags than NPU auto-mode. (Known)
- [demos/auto_mode/](../demos/auto_mode/) — only auto-mode demo currently committed: `add` (baseline + torch_jit). Confirms the `--cce-enable-pto-passes -O2` recipe for auto mode. (Known: [demos/auto_mode/baseline/add/CMakeLists.txt](../demos/auto_mode/baseline/add/CMakeLists.txt) lines 65–67.)
- [demos/baseline/](../demos/baseline/) — manual-mode baseline demos for `add`, `flash_atten`, `gemm_basic`, `allgather_async`. (Known)
- [kernels/automode/](../kernels/automode/) — auto-mode kernel projects (each self-contained with `run.sh`, `CMakeLists.txt`, `scripts/gen_data.py`). (Known)
  - [kernels/automode/a2a3/](../kernels/automode/a2a3/) — A3 auto-mode kernels confirmed to build and run on Ascend910B1. Current entries: `add_tile_array`, `topk`, `moe_top1_permute`, `moe_top1_gather_precomp`, `moe_top1_unpermute`, `moe_segmented_identity`, `moe_segmented_gemm_one_layer`, `moe_segmented_gemm_relu`, `moe_segmented_ffn_top1`, `flash_atten`. (Known)
  - [kernels/automode/a2a3/flash_atten/](../kernels/automode/a2a3/flash_atten/) — optimized auto-mode Flash Attention 2.0 for A3 with cube+vector SPMD pipeline, `TMPipe` FIFOs, `MultiBuffered`/`MultiStaged` helpers, `TEXTRACT`, `AccPhase::Final/Partial`, and `get_subblockid()` SPMD. Hardware-confirmed (README has onboard TFLOPS table). See [known_good_kernel_examples.md §A19](known_good_kernel_examples.md) for full pattern catalog. (Known)

> ~~Inferred: Larger, optimized auto-mode kernels (GEMM, FA, sparse attention) are **not** present in the repo.~~ **OUTDATED.** As of this branch, `kernels/automode/a2a3/flash_atten/` contains a hardware-confirmed auto-mode FA kernel. Auto-mode MoE GEMM kernels also exist (see `moe_segmented_gemm_one_layer`, `moe_segmented_ffn_top1`). The orphaned `tests/npu/a2a3/src/st/testcase/tfa/` is still NOT in `ALL_TESTCASES`; use `kernels/automode/a2a3/flash_atten/` instead.

---

## 2. Where A3-specific code lives

A3 and A2 share one tree (`a2a3`). (Known: [include/pto/common/arch_macro.hpp:14-15](../include/pto/common/arch_macro.hpp#L14-L15) maps `__NPU_ARCH__ == 2201` → `PTO_NPU_ARCH_A2A3`.)

- [include/pto/npu/a2a3/](../include/pto/npu/a2a3/) — per-instruction headers (`TLoad.hpp`, `TStore.hpp`, `TMatmul.hpp`, `TAdd.hpp`, …) that select between manual and auto-mode bodies via `#ifdef __PTO_AUTO__`. (Known)
- [include/pto/npu/a2a3/common.hpp](../include/pto/npu/a2a3/common.hpp) — A3-side helpers (e.g., `GetCastPreQuantMode`). (Known)
- [include/pto/npu/a2a3/custom/](../include/pto/npu/a2a3/custom/) — A3-specific custom helpers (`TSync_Custom.hpp`, `TSyncCVID.hpp`). (Known)
- [include/pto/npu/a2a3/TAlias.hpp](../include/pto/npu/a2a3/TAlias.hpp) — auto-mode-only aliasing shim; comment explicitly says it is a temporary auto-mode hack. (Known: header guarded by `#ifdef __PTO_AUTO__`.)
- [kernels/manual/a2a3/](../kernels/manual/a2a3/) — A3 hand-tuned reference kernels: `gemm_performance/`, `flash_atten/`, `conv2d_forward/`, `topk/`, `allgather_gemm/`, `gemm_ar/`, `tget_bandwidth/`. (Known)
- [tests/npu/a2a3/](../tests/npu/a2a3/) — A3 system tests; cube/vec compile flags `--cce-aicore-arch=dav-c220-vec` and `--cce-aicore-arch=dav-c220-cube` are set in [tests/npu/a2a3/src/st/testcase/CMakeLists.txt](../tests/npu/a2a3/src/st/testcase/CMakeLists.txt). (Known)
- [include/pto/costmodel/a2a3/](../include/pto/costmodel/a2a3/) — A3 cost-model info. (Known)
- [include/pto/comm/a2a3/](../include/pto/comm/a2a3/) — A3 communication-instruction headers. (Known)

> Assumption: any "A3 only" auto-mode behavior beyond what `arch_macro.hpp` distinguishes (A2 vs A3 explicitly) lives inside the same `a2a3/` tree gated by `__DAV_C220_*__` macros. Confirm with the user before special-casing A3 vs A2.

---

## 3. Where A5-specific code lives

(Known: [include/pto/common/arch_macro.hpp:16-20](../include/pto/common/arch_macro.hpp#L16-L20) maps `__NPU_ARCH__ == 3101` or `3510` → `PTO_NPU_ARCH_A5`; `3510` additionally defines `PTO_URMA_SUPPORTED`.)

- [include/pto/npu/a5/](../include/pto/npu/a5/) — A5 per-instruction headers, including A5-specific entries: [datatype.hpp](../include/pto/npu/a5/datatype.hpp), [MGather.hpp](../include/pto/npu/a5/MGather.hpp), [MScatter.hpp](../include/pto/npu/a5/MScatter.hpp), [TGetScaleAddr.hpp](../include/pto/npu/a5/TGetScaleAddr.hpp), [THistogram.hpp](../include/pto/npu/a5/THistogram.hpp). (Known)
- [include/pto/npu/a5/common.hpp](../include/pto/npu/a5/common.hpp) — A5 helpers (`GetByteSize`, `MaskReg`/`UnalignReg`/`AddrReg` aliases, `CreatePredicate`). Uses MX FP types (`float4_e1m2x2_t`, `float4_e2m1x2_t`). (Known)
- [include/pto/npu/a5/datatype.hpp](../include/pto/npu/a5/datatype.hpp) — A5 vector type mapping (`bfloat16_t → vector_bf16`, `float8_e5m2_t → vector_f8e5m2`, etc.) gated on `__DAV_VEC__`. (Known)
- [include/pto/npu/a5/custom/](../include/pto/npu/a5/custom/) — A5-only kernel helpers: `Div754.hpp`, `TExp_Custom.hpp`, `TFmodRemHp.hpp`, `TLog_Custom.hpp`, `TSqrtHp.hpp`, `TSync_Custom.hpp`, `TSyncCVID.hpp`. (Known)
- [include/pto/npu/a5/TAlias.hpp](../include/pto/npu/a5/TAlias.hpp) — A5 alias shim; just `#include "pto/npu/a2a3/TAlias.hpp"` under `__PTO_AUTO__`. (Known)
- [kernels/manual/a5/](../kernels/manual/a5/) — A5 manual kernels: `flash_atten/`, `matmul_mxfp4_performance/`, `matmul_mxfp8_performance/`, `gemm_ar/`, `allgather_gemm/`, `engram_simt/`. (Known)
- [tests/npu/a5/src/st/testcase/](../tests/npu/a5/src/st/testcase/) — A5 ST tests; includes `mgather/`, `mscatter/`, `tmatmul/`, `tmatmul_mx/`, plus all elementwise ops. (Known)

---

## 4. Where auto-mode code lives

Auto mode is selected by the compiler macro `__PTO_AUTO__` and the build flag `--cce-enable-pto-passes`. (Known.)

- Build wiring:
  - [build.sh](../build.sh) — `--auto_mode` CLI flag passed through to test scripts. (Known: lines 77, 79, 141–143, 189–190, 244–245.)
  - [tests/script/build_st.py](../tests/script/build_st.py) — `-DAUTO_MODE=ON` set when `--auto-mode-enable` is passed. (Known: lines 53–54, 103.)
  - [tests/npu/a2a3/src/st/CMakeLists.txt:88-91](../tests/npu/a2a3/src/st/CMakeLists.txt#L88-L91) and [tests/npu/a5/src/st/CMakeLists.txt:88-91](../tests/npu/a5/src/st/CMakeLists.txt#L88-L91) — append `--cce-enable-pto-passes` to `CMAKE_CCE_COMPILE_OPTIONS` when `AUTO_MODE` is on. (Known)
  - [tests/npu/a2a3/src/st/testcase/CMakeLists.txt:213-220](../tests/npu/a2a3/src/st/testcase/CMakeLists.txt#L213-L220) and [tests/npu/a5/src/st/testcase/CMakeLists.txt:243-250](../tests/npu/a5/src/st/testcase/CMakeLists.txt#L243-L250) — exclude testcases not yet supported in auto mode (`tpushpop_*`; A5 also excludes `texpands_mat`). The comments call out *why*: `TPUSH`/`TPOP` use `TASSIGN` internally and `texpands_mat` calls raw CCE intrinsics. (Known)
  - [demos/auto_mode/baseline/add/CMakeLists.txt:65-67](../demos/auto_mode/baseline/add/CMakeLists.txt#L65-L67) — confirms `--cce-enable-pto-passes -O2`; `auto mode only works with -O2`. (Known)

- Library-side auto-mode branches (`#ifdef __PTO_AUTO__`):
  - [include/pto/common/memory.hpp](../include/pto/common/memory.hpp) — `MemoryQualifier<TileType, DType>::type` is `__ubuf__/__cbuf__/__ca__/__cb__/__cc__/__fbuf__ DType` (vector type) in auto mode and `... DType *` in manual mode. (Known: lines 29–106.)
  - [include/pto/common/pto_tile.hpp](../include/pto/common/pto_tile.hpp) — `Shape`, `Stride`, `Tile` definitions; flagged by Library_Developer rules as the place where `.data()` returns vector vs pointer. (Known)
  - [include/pto/common/event.hpp](../include/pto/common/event.hpp) — auto-mode no-ops for event model. (Known: matches grep for `__PTO_AUTO__`.)
  - [include/pto/common/pto_instr.hpp](../include/pto/common/pto_instr.hpp) — public PTO instruction templates (`TADD`, `TMATMUL`, `TASSIGN`, `TSYNC`, …). (Known)
  - [include/pto/common/pto_instr_impl.hpp](../include/pto/common/pto_instr_impl.hpp) — selects per-arch (A2A3/A5/Kirin) implementation headers via `PTO_NPU_ARCH_*`. (Known)
  - [include/pto/npu/a2a3/TAlias.hpp](../include/pto/npu/a2a3/TAlias.hpp), [include/pto/npu/a5/TAlias.hpp](../include/pto/npu/a5/TAlias.hpp) — auto-mode-only aliasing helpers. (Known)
  - [include/pto/npu/a2a3/TSync.hpp](../include/pto/npu/a2a3/TSync.hpp) — `TSYNC_IMPL` body is guarded by `#ifndef __PTO_AUTO__`, i.e. it is a no-op in auto mode. Same pattern is documented in [docs/auto_mode/Library_Developer_Rules_And_Limitations.md](../docs/auto_mode/Library_Developer_Rules_And_Limitations.md). (Known)

- Documentation:
  - [docs/auto_mode/Auto_Mode_Overview.md](../docs/auto_mode/Auto_Mode_Overview.md) — what auto mode is, what it does (auto-sync, auto memory allocation), and the compile recipe. (Known)
  - [docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md) — kernel-developer rules: control-flow rules, `TRESHAPE`/`TSUBVIEW` instead of `TASSIGN`, no double buffering, no redundant `TLOAD`, no raw CCE intrinsics, prefer `PtoSetWaitFlag`/`TSYNC` over `set_flag`/`wait_flag`. (Known)
  - [docs/auto_mode/Library_Developer_Rules_And_Limitations.md](../docs/auto_mode/Library_Developer_Rules_And_Limitations.md) — library-developer rules: `.data()` returns vector type in auto mode, avoid default member initializers, `TSYNC`/`PtoSetWaitFlag` is no-op, avoid `TASSIGN` in `*_IMPL`, `PTO_INTERNAL` rules. (Known)
  - [docs/auto_mode/Examples.md](../docs/auto_mode/Examples.md) — side-by-side `TADD`/`TMATMUL` auto vs manual examples. (Known)

- Per-kernel auto-mode toggles:
  - Test kernels conditionally drop manual-only code with `#ifndef __PTO_AUTO__`. Canonical example: [tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp](../tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp) — every `TASSIGN`-following `set_flag(...) / wait_flag(...)` block is wrapped in `#ifndef __PTO_AUTO__`. (Known)

> **`__tf__` / `__in__` / `__out__` / `__cce_get_tile_ptr` are bisheng-CCE compiler-provided keywords/builtins on A3/A5** — see [qualifier_reference.md](qualifier_reference.md) for the full evidence chain. Add `__tf__` on any A3/A5 helper that needs to be a tile-function boundary (i.e., that calls CCE intrinsics). Do not add it to user-facing kernel entries.

---

## 5. Where manual-mode code lives

Manual mode is the default (no `--cce-enable-pto-passes`, no `__PTO_AUTO__`). It uses explicit `TASSIGN` for tile addresses and `set_flag`/`wait_flag` or `Event<>` for sync.

- [kernels/manual/](../kernels/manual/) — full hand-tuned reference kernels with comments and pipeline diagrams (`fa_pipeline*.svg`). (Known)
  - [kernels/manual/a2a3/gemm_performance/](../kernels/manual/a2a3/gemm_performance/) — high-perf GEMM A3 reference.
  - [kernels/manual/a2a3/flash_atten/](../kernels/manual/a2a3/flash_atten/) — A3 Flash Attention with `pto_macro_fa_gu.hpp`, `pto_macro_fa_softmax.hpp`, `pto_macro_matmul.hpp`.
  - [kernels/manual/a5/flash_atten/](../kernels/manual/a5/flash_atten/) — A5 Flash Attention; adds dual-network variants (`fa_performance_dn_kernel.cpp`, `pto_macro_fa_dn_*.hpp`).
  - [kernels/manual/common/flash_atten/](../kernels/manual/common/flash_atten/) — cross-platform FA reference (per [kernels/manual/README.md](../kernels/manual/README.md): A2/A3/A5).
  - [kernels/manual/a5/matmul_mxfp4_performance/](../kernels/manual/a5/matmul_mxfp8_performance/), [kernels/manual/a5/matmul_mxfp8_performance/](../kernels/manual/a5/matmul_mxfp8_performance/) — A5 MX FP4/FP8 GEMM.

- The non-`__PTO_AUTO__` branches inside per-instruction headers and ST kernel files are the manual-mode bodies of the same code that gets enabled in auto mode. Examples:
  - [tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp](../tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp) — both modes coexist.
  - [tests/npu/a5/src/st/testcase/tmatmul/tmatmul_kernel.cpp](../tests/npu/a5/src/st/testcase/tmatmul/tmatmul_kernel.cpp) — same pattern on A5.

> Per CLAUDE.md, do not assume manual-mode code can be copied to auto mode — manual kernels under `kernels/manual/` make heavy use of `TASSIGN`, double-buffering, `set_flag`/`wait_flag`, and (for FA) macro-based pipelines, all of which conflict with the auto-mode rules in [docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md).

---

## 6. Where compiler/build-related docs and tests live

Docs (developer-facing):
- [docs/auto_mode/Auto_Mode_Overview.md](../docs/auto_mode/Auto_Mode_Overview.md) — auto-mode concepts and the `--cce-enable-pto-passes` recipe; calls out `--cce-aicore-arch=dav-c310-vec` style flags. (Known)
- [docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md) — kernel-developer constraints (control flow, memory allocation, redundant `TLOAD`, double buffering, `set_flag`/`wait_flag` guidance). (Known)
- [docs/auto_mode/Library_Developer_Rules_And_Limitations.md](../docs/auto_mode/Library_Developer_Rules_And_Limitations.md) — library-developer constraints (`.data()` vector vs pointer, `*_IMPL` rules, `PTO_INTERNAL`). (Known)
- [docs/auto_mode/Examples.md](../docs/auto_mode/Examples.md) — TADD/TMATMUL auto vs manual side-by-side. (Known)
- [docs/coding/compilation-process.md](../docs/coding/compilation-process.md) — high-level build flow and the public `pto/pto-inst.hpp` entry. (Known)
- [docs/coding/Tile.md](../docs/coding/Tile.md) — `pto::Tile<...>` type contract (TileType, BLayout/SLayout, valid region). (Known)
- [docs/coding/tutorials/gemm.md](../docs/coding/tutorials/gemm.md), [docs/coding/tutorials/row-softmax.md](../docs/coding/tutorials/row-softmax.md), [docs/coding/tutorials/vec-add.md](../docs/coding/tutorials/vec-add.md) — single-tile sketches useful as starting points for auto-mode skeletons. (Known)
- [docs/coding/Event.md](../docs/coding/Event.md), [docs/coding/multi-core-programming.md](../docs/coding/multi-core-programming.md), [docs/coding/pipeline-parallel.md](../docs/coding/pipeline-parallel.md), [docs/coding/operator-fusion.md](../docs/coding/operator-fusion.md) — referenced for manual-mode background. (Known)
- [docs/PTO-Virtual-ISA-Manual.md](../docs/PTO-Virtual-ISA-Manual.md) — pointer to the chaptered ISA reference under `mkdocs/src/manual/`. (Known)
- [docs/isa/](../docs/isa/) — per-instruction ISA specs (one `.md` per `T*` op, with `_zh` siblings); authoritative for instruction semantics when source is ambiguous. (Known: 200+ `.md` files.)

Build entry points and CMake plumbing:
- [build.sh](../build.sh) — top-level driver; passes `--auto_mode` through. (Known)
- [CMakeLists.txt](../CMakeLists.txt), [cmake/func.cmake](../cmake/func.cmake) — top-level project files. (Known: structure only, not yet inspected for AUTO_MODE plumbing.) **Unknown** — whether the top-level adds further auto-mode flags beyond what the test CMakeLists do.
- [tests/script/build_st.py](../tests/script/build_st.py) — drives ST builds; the `-a/--auto-mode-enable` CLI flag flips `-DAUTO_MODE=ON`. (Known)
- [tests/script/run_st.py](../tests/script/run_st.py) — ST runner. (Unknown — not inspected for auto-mode semantics.)
- [tests/npu/a2a3/src/st/CMakeLists.txt](../tests/npu/a2a3/src/st/CMakeLists.txt), [tests/npu/a5/src/st/CMakeLists.txt](../tests/npu/a5/src/st/CMakeLists.txt) — set `--cce-enable-pto-passes` when `AUTO_MODE` is ON. (Known)
- [tests/npu/a2a3/src/st/testcase/CMakeLists.txt](../tests/npu/a2a3/src/st/testcase/CMakeLists.txt), [tests/npu/a5/src/st/testcase/CMakeLists.txt](../tests/npu/a5/src/st/testcase/CMakeLists.txt) — define `pto_vec_st`, `pto_cube_st`, `pto_mix_st` and the `ALL_TESTCASES` list, with auto-mode exclusions. (Known)

Tests:
- [tests/npu/a2a3/src/st/testcase/](../tests/npu/a2a3/src/st/testcase/) — A3 ST kernels (119 directories per `ls`). Auto-mode-eligible kernels are exactly the entries in `ALL_TESTCASES`; `tpushpop_*` are the only explicit exclusions. (Known)
- [tests/npu/a5/src/st/testcase/](../tests/npu/a5/src/st/testcase/) — A5 ST kernels (143 directories). `tpushpop_*` and `texpands_mat` are excluded for auto mode. (Known)
- [tests/cpu/st/](../tests/cpu/st/) — CPU-sim version; useful for functional validation but uses different defines (`__CPU_SIM`). (Known)
- [tests/npu/a2a3/comm/](../tests/npu/a2a3/comm/), [tests/npu/a5/comm/](../tests/npu/a5/comm/) — communication ST. (Known: directory existence only.)

> **Unknown** — exact list of auto-mode flags emitted by `cmake/func.cmake` (e.g., whether it forces `-O2`); whether `tests/script/build_st.py` carries any extra flags besides `-DAUTO_MODE=ON`. Confirm by reading those files before claiming any auto-mode build behavior beyond `--cce-enable-pto-passes`.

---

## 7. Top 20 files for generating auto-mode A3/A5 kernels

Ordered by likely day-to-day value when writing or reviewing an auto-mode A3/A5 kernel. Each entry: path, why it matters, and `Known | Inferred | Unknown` for auto mode specifically.

1. [include/pto/pto-inst.hpp](../include/pto/pto-inst.hpp) — public include entry. The single header users `#include` from a kernel. **Known** for auto mode (also the entry from [docs/auto_mode/Examples.md](../docs/auto_mode/Examples.md)).
2. [include/pto/common/pto_instr.hpp](../include/pto/common/pto_instr.hpp) — the full template surface (`TADD`, `TLOAD`, `TSTORE`, `TMATMUL`, `TASSIGN`, `TSYNC`, `TSUBVIEW`, …). Source of truth for argument order and overloads. **Known**.
3. [include/pto/common/pto_instr_impl.hpp](../include/pto/common/pto_instr_impl.hpp) — selects A2A3 vs A5 vs Kirin headers by `PTO_NPU_ARCH_*` macros. Tells you which per-arch `T*.hpp` actually backs each `T*` call. **Known**.
4. [include/pto/common/memory.hpp](../include/pto/common/memory.hpp) — `MemoryQualifier<TileType, DType>::type` decides whether `.data()` is a vector or pointer. The single most-cited gotcha for auto mode in [docs/auto_mode/Library_Developer_Rules_And_Limitations.md](../docs/auto_mode/Library_Developer_Rules_And_Limitations.md). **Known**.
5. [include/pto/common/arch_macro.hpp](../include/pto/common/arch_macro.hpp) — maps `__NPU_ARCH__` to `PTO_NPU_ARCH_A2A3` / `PTO_NPU_ARCH_A5` / `PTO_NPU_ARCH_KIRIN9030` / `PTO_NPU_ARCH_KIRINX90`; `#define`s `__tf__`/`__in__`/`__out__`/`__cce_get_tile_ptr` **as empty** only for kirin (so the same source compiles when those tokens are not compiler builtins on that arch). On A3/A5 they are bisheng-CCE keywords/builtins. See [qualifier_reference.md](qualifier_reference.md). **Known**.
6. [include/pto/common/pto_tile.hpp](../include/pto/common/pto_tile.hpp) — `Shape`, `Stride`, `Tile`, `GlobalTensor`, valid-region semantics. Required reading before declaring any tile in a new kernel. **Known**.
7. [include/pto/common/type.hpp](../include/pto/common/type.hpp) — `AICORE`, `PTO_INST`, `PTO_INTERNAL`, `OP_NAME`, `OP_TYPE` macros. Defines what qualifiers go on a kernel function signature. **Known**.
8. [include/pto/common/event.hpp](../include/pto/common/event.hpp) — `Event<>`, `RecordEvent`, `WaitAllEvents`. In auto mode these are intended to be no-ops (per [docs/auto_mode/Auto_Mode_Overview.md](../docs/auto_mode/Auto_Mode_Overview.md)). **Known** for the auto-mode no-op contract; **Unknown** for the precise auto-mode definitions until the file is read end-to-end.
9. [include/pto/npu/a2a3/TAlias.hpp](../include/pto/npu/a2a3/TAlias.hpp) — auto-mode aliasing shim used in place of `TASSIGN`-based aliasing. The header itself states it is a temporary auto-mode hack. **Known** as auto-mode-only.
10. [include/pto/npu/a2a3/TMatmul.hpp](../include/pto/npu/a2a3/TMatmul.hpp) and [include/pto/npu/a5/TMatmul.hpp](../include/pto/npu/a5/TMatmul.hpp) — the `TMATMUL`/`TMATMUL_BIAS`/`TMATMUL_ACC` implementations per arch. Required reading before any GEMM/FA work. **Known** for manual-mode body; **Inferred** that auto-mode behavior is determined by upstream `pto_instr.hpp` template wrapping and per-arch `*_IMPL`. Read the file before assuming auto-mode signatures match.
11. [include/pto/npu/a2a3/TLoad.hpp](../include/pto/npu/a2a3/TLoad.hpp) and [include/pto/npu/a5/TLoad.hpp](../include/pto/npu/a5/TLoad.hpp) — `TLOAD` overloads (40 KB on A3). Defines what GM→tile shapes are supported. **Known**.
12. [include/pto/npu/a2a3/TStore.hpp](../include/pto/npu/a2a3/TStore.hpp) and [include/pto/npu/a5/TStore.hpp](../include/pto/npu/a5/TStore.hpp) — `TSTORE` overloads. Companion to `TLoad`. **Known**.
13. [include/pto/npu/a2a3/TSync.hpp](../include/pto/npu/a2a3/TSync.hpp) — `TSYNC_IMPL` body wrapped in `#ifndef __PTO_AUTO__`; canonical pattern for "manual sync only" code. **Known**.
14. [include/pto/npu/a2a3/TSubView.hpp](../include/pto/npu/a2a3/TSubView.hpp) — `TSUBVIEW`; the auto-mode-correct way to express a tile as an offset of another tile (per [docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md) §2.2). **Known**.
15. [tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp](../tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp) — the closest existing dual-mode (manual + auto) GEMV/GEMM kernel for A3. Every manual-only block is `#ifndef __PTO_AUTO__`. Best small-kernel template for auto-mode GEMM-shaped work. **Known**.
16. [tests/npu/a5/src/st/testcase/tmatmul/tmatmul_kernel.cpp](../tests/npu/a5/src/st/testcase/tmatmul/tmatmul_kernel.cpp) and [tests/npu/a5/src/st/testcase/tmatmul_mx/tmatmul_mx_kernel.cpp](../tests/npu/a5/src/st/testcase/tmatmul_mx/tmatmul_mx_kernel.cpp) — A5 dual-mode matmul references, including MX FP4/FP8. **Known**.
17. [demos/auto_mode/baseline/add/csrc/kernel/add_custom.cpp](../demos/auto_mode/baseline/add/csrc/kernel/add_custom.cpp) and [demos/auto_mode/baseline/add/CMakeLists.txt](../demos/auto_mode/baseline/add/CMakeLists.txt) — the only end-to-end auto-mode demo committed; confirms `AICORE`/`__gm__` signature, `block_idx` usage, and `--cce-enable-pto-passes -O2`. **Known** as auto-mode-confirmed.
18. [docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md) — kernel-side rules. Treat as auto-mode authoritative. **Known**.
19. [docs/auto_mode/Library_Developer_Rules_And_Limitations.md](../docs/auto_mode/Library_Developer_Rules_And_Limitations.md) — library-side rules; clarifies `.data()` semantics, `PTO_INTERNAL`, `*_IMPL` constraints, `TASSIGN` ban. **Known**.
20. [docs/auto_mode/Examples.md](../docs/auto_mode/Examples.md) — minimal auto vs manual examples for `TADD` and `TMATMUL`. Use as the canonical "what does an auto-mode kernel look like" reference. **Known**.

Honourable mentions (still relevant, just below the top 20):
- [include/pto/common/constants.hpp](../include/pto/common/constants.hpp) — `C0_SIZE_BYTE`, alignment constants used in tile sizing. (Known)
- [include/pto/common/buffer_limits.hpp](../include/pto/common/buffer_limits.hpp) — UB/L1/L0 buffer sizes per arch. (Inferred — name + `include/pto/common/` location.)
- [include/pto/npu/a2a3/TPush.hpp](../include/pto/npu/a2a3/TPush.hpp), [include/pto/npu/a2a3/TPop.hpp](../include/pto/npu/a2a3/TPop.hpp) — explicitly **not** safe in auto mode today (see [tests/npu/a2a3/src/st/testcase/CMakeLists.txt:213-220](../tests/npu/a2a3/src/st/testcase/CMakeLists.txt#L213-L220) and [docs/auto_mode/Library_Developer_Rules_And_Limitations.md](../docs/auto_mode/Library_Developer_Rules_And_Limitations.md) §4). Worth knowing as a no-go area.
- [kernels/manual/a2a3/flash_atten/](../kernels/manual/a2a3/flash_atten/) and [kernels/manual/common/flash_atten/](../kernels/manual/common/flash_atten/) — manual FA references. **Inferred** that direct copy to auto mode will fail (uses `TASSIGN`, manual events, double-buffering macros), so use as a *semantic* source only.
- [tests/npu/a2a3/src/st/testcase/CMakeLists.txt](../tests/npu/a2a3/src/st/testcase/CMakeLists.txt) and [tests/npu/a5/src/st/testcase/CMakeLists.txt](../tests/npu/a5/src/st/testcase/CMakeLists.txt) — list of `ALL_TESTCASES` and the auto-mode exclusions; quickest way to see "what is currently expected to compile in auto mode". (Known)

---

## Open items / assumptions to verify

These are hooks for the user to confirm before any auto-mode kernel work:
- A3 vs A2 split inside `a2a3/` — are there any A3-only `__DAV_C220_*__` branches that should be considered separately for auto mode? (**Assumption**: shared by default.)
- `__tf__` IS used on A3/A5 — it is a bisheng-CCE keyword on those archs; the repo only `#define`s it (as empty) for kirin / CPU-sim / cost-model. Apply it to helper functions that contain raw CCE intrinsics. (Resolved; see [qualifier_reference.md](qualifier_reference.md).)
- Top-level CMake auto-mode wiring — [CMakeLists.txt](../CMakeLists.txt) and [cmake/func.cmake](../cmake/func.cmake) not yet read end-to-end. (**Unknown**.)
- Auto-mode coverage of `TQUANT`, `TPOW`, sparse-attention primitives — referenced as priorities in CLAUDE.md but not yet inventoried. (**Unknown**.)
- Whether the `tfa` directory under [tests/npu/a2a3/src/st/testcase/tfa/](../tests/npu/a2a3/src/st/testcase/tfa/) is intentionally orphaned (not in `ALL_TESTCASES`) or simply not yet wired up. (**Unknown**.)
