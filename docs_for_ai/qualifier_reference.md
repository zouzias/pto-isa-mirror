# Qualifier / Attribute Reference (PTO, A3/A5)

Reconciles the conflicting earlier interpretations of `__tf__`, `__in__`,
`__out__`, and `__cce_get_tile_ptr`. This is the single source of truth for
those tokens; if other docs in `docs_for_ai/` disagree, prefer this file.

Companion to [repo_kernel_map.md](repo_kernel_map.md),
[known_good_kernel_examples.md](known_good_kernel_examples.md), and
[auto_mode_bad_patterns.md](auto_mode_bad_patterns.md).

Confidence labels:
- **Known** — directly readable from a quoted file path.
- **Inferred** — derived by combining file evidence; reasoning given.
- **Assumption** — not present in source, flagged for confirmation.
- **Unknown** — referenced but not yet confirmable.

---

## Summary table

| Token | A3/A5 device | Kirin device | CPU sim | Cost model |
|---|---|---|---|---|
| `__tf__` | bisheng-CCE keyword (Inferred) | empty `#define` | empty `#define` | empty `#define` |
| `__in__` | bisheng-CCE keyword (Inferred) | empty `#define` | empty `#define` | empty `#define` |
| `__out__` | bisheng-CCE keyword (Inferred) | empty `#define` | empty `#define` | empty `#define` |
| `__cce_get_tile_ptr` | bisheng-CCE intrinsic (Inferred) | object-like macro `__cce_get_tile_ptr` (so `__cce_get_tile_ptr(x)` → `(x)`) | function-like macro `__cce_get_tile_ptr(x) x` | function-like macro `__cce_get_tile_ptr(x) x` |
| `__gm__` / `__ubuf__` / `__cbuf__` / `__ca__` / `__cb__` / `__cc__` / `__fbuf__` / `__biasbuf__` | bisheng-CCE memory-space qualifiers (Inferred) | empty `#define` (kirin gets these from compiler too — Unknown) | empty `#define` ([cpu_stub.hpp:25-33](../include/pto/common/cpu_stub.hpp#L25-L33)) | empty `#define` ([qualifiers.hpp:18-28](../include/pto/costmodel/common/qualifiers.hpp#L18-L28)) |
| `AICORE` | `[aicore]` (device) | `[aicore]` (device) | empty | empty |
| `__aicore__` | bisheng-CCE keyword | empty `#define` ([cpu_stub.hpp:24](../include/pto/common/cpu_stub.hpp#L24)) | empty | empty |
| `__global__` | bisheng-CCE attribute | empty `#define` ([cpu_stub.hpp:22](../include/pto/common/cpu_stub.hpp#L22)) | empty | empty |
| `PTO_INST` | `AICORE PTO_INLINE __attribute__((visibility("default")))` ([type.hpp:22](../include/pto/common/type.hpp#L22)) | same | reduces to `inline ...` | same |
| `PTO_INTERNAL` | `AICORE PTO_INLINE` ([type.hpp:24](../include/pto/common/type.hpp#L24)) | same | reduces to `inline ...` | same |

---

## 1. Where these tokens are defined in the repo

There are **three** places — and only three — that `#define` `__tf__`, `__in__`, `__out__`, and `__cce_get_tile_ptr`:

1. **Kirin (kirin9030 / kirinX90) device builds** — [include/pto/common/arch_macro.hpp:29-34](../include/pto/common/arch_macro.hpp#L29-L34):
   ```cpp
   #if defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRINX90)
   #define __tf__
   #define __in__
   #define __out__
   #define __cce_get_tile_ptr
   #endif
   ```
   All four are **empty object-like macros**. `__cce_get_tile_ptr(x)` therefore expands to `(x)` — a parenthesized expression — on kirin.

2. **CPU-sim builds** — [include/pto/common/cpu_stub.hpp:22-34, 80-84](../include/pto/common/cpu_stub.hpp#L22-L34):
   ```cpp
   #define __global__
   #define AICORE
   #define __aicore__
   #define __gm__
   #define __out__
   #define __in__
   #define __ubuf__
   #define __cbuf__
   #define __ca__
   #define __cb__
   #define __cc__
   #define __fbuf__
   #define __tf__
   ...
   #define __cce_get_tile_ptr(x) x
   #define set_mask_norm(...)
   #define set_vector_mask(...)
   ```
   All memory-space qualifiers, `__tf__`, `__in__`, `__out__` are empty. `__cce_get_tile_ptr(x)` is a function-like macro that returns `x`. CCE intrinsics (`set_flag`, `wait_flag`, `pipe_barrier`, `set_mask_norm`, `set_vector_mask`, …) are stubbed empty so host CPU compilation succeeds.

3. **Cost-model builds** — [include/pto/costmodel/common/qualifiers.hpp:15-28, 65-66](../include/pto/costmodel/common/qualifiers.hpp#L15-L28):
   ```cpp
   #define __global__
   #define AICORE
   #define __aicore__
   #define __gm__
   #define __out__
   #define __in__
   ...
   #define __tf__
   #define __biasbuf__
   ...
   #define __cce_get_tile_ptr(x) x
   ```
   Same shape as cpu_stub.hpp.

There is **no other definition** of `__tf__`, `__in__`, `__out__`, or `__cce_get_tile_ptr` anywhere in the repo. (Verified by `grep -rn '#define[[:space:]]\+__\(tf\|in\|out\|cce_get_tile_ptr\)' include docs kernels tests demos scripts cmake`.)

The non-`AICORE` macro `__aicore__` is also `#define`d as empty on cpu_stub.hpp; it is not defined on A3/A5 device by any repo header. Same logic applies.

---

## 2. Why this means the tokens are real on A3/A5

Two pieces of evidence force the conclusion:

a) **They are USED on A3/A5 in the repo without any redefinition.** Distribution of `__tf__` uses (Known):
   - [include/pto/npu/a2a3/](../include/pto/npu/a2a3/): ~71 files (e.g., [TAddS.hpp:32](../include/pto/npu/a2a3/TAddS.hpp#L32), [TPrint.hpp:108](../include/pto/npu/a2a3/TPrint.hpp#L108), [TPartAdd.hpp](../include/pto/npu/a2a3/TPartAdd.hpp), [TRowExpand.hpp](../include/pto/npu/a2a3/TRowExpand.hpp), [TRowMax.hpp](../include/pto/npu/a2a3/TRowMax.hpp), [TDequant.hpp:89-90](../include/pto/npu/a2a3/TDequant.hpp#L89-L90), [TDivS.hpp:190](../include/pto/npu/a2a3/TDivS.hpp#L190), [TSort32.hpp:70,103](../include/pto/npu/a2a3/TSort32.hpp#L70), …).
   - [include/pto/npu/a5/](../include/pto/npu/a5/): ~80 files (e.g., [MGather.hpp:164,185](../include/pto/npu/a5/MGather.hpp#L164), [MScatter.hpp:274,296,313](../include/pto/npu/a5/MScatter.hpp#L274)).
   - [include/pto/npu/kirin9030/](../include/pto/npu/kirin9030/) and [include/pto/npu/kirinX90/](../include/pto/npu/kirinX90/): ~13 files (the kirin-side mirrors many headers from a2a3/a5 — e.g., [include/pto/npu/kirin9030/TAdd.hpp](../include/pto/npu/kirin9030/TAdd.hpp) `#include`s `pto/npu/a5/TAdd.hpp`).
   - Some test kernel files: e.g., [tests/npu/a2a3/src/st/testcase/texpands_mat/texpands_mat_kernel.cpp:22](../tests/npu/a2a3/src/st/testcase/texpands_mat/texpands_mat_kernel.cpp#L22), [tests/npu/a2a3/src/st/testcase/tload_gm2mat/tload_gm2mat_kernel.cpp:19](../tests/npu/a2a3/src/st/testcase/tload_gm2mat/tload_gm2mat_kernel.cpp#L19), [tests/npu/a5/src/st/testcase/tload_shape2d/tload_shape2d_kernel.cpp:18,24](../tests/npu/a5/src/st/testcase/tload_shape2d/tload_shape2d_kernel.cpp#L18).

b) **The library spec describes them as required tile-function attributes.** [docs/auto_mode/Library_Developer_Rules_And_Limitations.md §6](../docs/auto_mode/Library_Developer_Rules_And_Limitations.md):
   > Some general rules for tile functions
   > - Ensure to use `typename <...>::TileDType` instead of `typename <...>::DType *` for tile types on tile function parameters
   > - Ensure these `typename <...>::TileDType` parameters are pass-by-value, not by pointer or reference
   > - Ensure `__in__` or `__out__` attributes are properly attached to these `typename <...>::TileDType` parameters
   > - Always call `__cce_get_tile_ptr` on these `typename <...>::TileDType` arguments to get a tile's underlying buffer pointer
   > - The return type should always be `void`. Otherwise the compiler's assumption about TF interface is broken and it's an undefined behavior.

   This document is part of the auto-mode contract and specifically applies to A3/A5 (the platforms auto mode targets). For these statements to be meaningful, the tokens must be real on A3/A5.

c) **The repo never defines them on A3/A5.** Verified by exhaustive `#define` search above. If the tokens were repo macros for A3/A5, there would be a `#define` somewhere; there is not.

The only consistent reading is: these tokens are **bisheng-CCE compiler-provided keywords/builtins** on A3/A5 device builds. The repo `#define`s them as empty in the three non-A3/A5 environments so the same source compiles there. (Inferred — no compiler documentation in this repo confirms this directly, but the trio of evidence makes it the only viable interpretation.)

---

## 3. What each token does

### 3.1 `__tf__` — tile-function attribute

- **Purpose** — Marks a function as a "tile function". The auto-mode compiler treats every `__tf__`-attributed function as a black box: it does NOT analyze, rewrite, or insert sync inside it ([docs/auto_mode/Auto_Mode_Overview.md](../docs/auto_mode/Auto_Mode_Overview.md): *"the whole auto mode compiler works on the tile function level, meaning that everything inside tile function is a complete black box to auto-mode"*).
- **Placement** — Function attribute, before the return type:
  ```cpp
  __tf__ PTO_INTERNAL void TAddS(typename TileDataDst::TileDType __out__ dstData, ...)
  ```
  See [include/pto/npu/a2a3/TAddS.hpp:32](../include/pto/npu/a2a3/TAddS.hpp#L32) for canonical placement.
- **When to use** — On a helper that contains raw CCE intrinsics (`set_flag`, `wait_flag`, `pipe_barrier`, `copy_*_to_*`, `__cce_get_tile_ptr`, etc.). The auto-mode rules forbid those in kernel-level code ([Kernel rules §3.2](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md)) but explicitly allow them inside tile functions ([Library rules §3](../docs/auto_mode/Library_Developer_Rules_And_Limitations.md)).
- **When NOT to use** — On the user-facing kernel entry (`__global__ AICORE void runKernel(__gm__ T *out, ...)`), which is a kernel function, not a tile function. Examples: [demos/auto_mode/baseline/add/csrc/kernel/add_custom.cpp:25](../demos/auto_mode/baseline/add/csrc/kernel/add_custom.cpp#L25) (`AICORE void runTAdd(...)` — no `__tf__`), every `*_kernel.cpp` under `tests/npu/{a2a3,a5}/src/st/testcase/`.
- **Confidence** — Inferred (strong, based on §2 above).

### 3.2 `__in__` / `__out__` — tile-function parameter attributes

- **Purpose** — Direction hints on `TileDType` parameters of tile functions. Likely consumed by the bisheng-CCE register allocator and by the auto-sync analysis to know which tiles a tile function reads vs writes.
- **Placement** — Before a `TileDType` parameter:
  ```cpp
  __tf__ PTO_INTERNAL void TAddS(
      typename TileDataDst::TileDType __out__ dstData,
      typename TileDataSrc0::TileDType __in__ src0Data, ...)
  ```
  ([include/pto/npu/a2a3/TAddS.hpp:32-34](../include/pto/npu/a2a3/TAddS.hpp#L32-L34) and the same shape in 70+ other headers under `a2a3/` and `a5/`.)
- **Note on kernel arguments** — `__in__`/`__out__` also appear on the GM pointer arguments of kernel-entry functions, e.g., [tests/npu/a2a3/src/st/testcase/tadd/tadd_kernel.cpp:18](../tests/npu/a2a3/src/st/testcase/tadd/tadd_kernel.cpp#L18) (`__gm__ T __out__ *out, __gm__ T __in__ *src0, ...`). On A3/A5 this is meaningful (compiler hint); on kirin/CPU-sim/cost-model it expands to nothing. Preserve the pattern when copying entry signatures.
- **Confidence** — Inferred (strong).

### 3.3 `__cce_get_tile_ptr(x)` — extract raw buffer pointer from `TileDType`

- **Purpose** — Inside an `__tf__` body, converts a `TileDType` value (which is a vector type in auto mode, a pointer in manual mode) to a typed raw pointer suitable for raw CCE intrinsics like `copy_cbuf_to_gm`.
- **Placement** — Around a `TileDType` parameter or a `tile.data()` value, inside an `__tf__` helper. Canonical examples:
  ```cpp
  __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
  ```
  ([include/pto/npu/a2a3/TPartAdd.hpp:46](../include/pto/npu/a2a3/TPartAdd.hpp#L46), [TRowExpand.hpp:28](../include/pto/npu/a2a3/TRowExpand.hpp#L28), [TRowMax.hpp:60-62](../include/pto/npu/a2a3/TRowMax.hpp#L60-L62), [TDequant.hpp:89-90, 105-106](../include/pto/npu/a2a3/TDequant.hpp#L89-L90), [TPrefetch.hpp](../include/pto/npu/a2a3/TPrefetch.hpp), and ~50 other a2a3 headers.)
- **What it expands to** —
  - On A3/A5 device: a real CCE intrinsic that returns a typed memory-space pointer (Inferred).
  - On kirin device: `__cce_get_tile_ptr(x)` → `(x)` (because `__cce_get_tile_ptr` is an empty object-like macro). The cast to `(__ubuf__ T *)` then completes the pointer extraction.
  - On CPU-sim / cost-model: `__cce_get_tile_ptr(x)` → `x` (function-like macro).
- **Constraint** — Only call inside an `__tf__` helper. From kernel code, this is forbidden (Kernel rules §3.2; see [auto_mode_bad_patterns.md §3.2](auto_mode_bad_patterns.md)).
- **Pointer arithmetic gotcha** — `__cce_get_tile_ptr(x + N)` is wrong (applies arithmetic to a vector value before extraction). Always extract first, then offset: `__cce_get_tile_ptr(x) + N`. See [auto_mode_bad_patterns.md §3.2.1](auto_mode_bad_patterns.md) and PR-852 [external_context/pr_852_notes.md §L1](external_context/pr_852_notes.md).
- **Confidence** — Inferred (strong).

---

## 4. Practical guidance for auto-mode A3/A5 kernels

When writing or reviewing a new auto-mode A3/A5 kernel:

1. **Kernel entry** (the `__global__ AICORE void run...(__gm__ T *out, ...)`) — do NOT add `__tf__`. May include `__in__`/`__out__` on GM pointer parameters (decorative on host but meaningful on device); copy from existing test kernels.
2. **Helpers that compose only PTO instructions** (`TLOAD`, `TADD`, `TROWMAX`, …) — do NOT add `__tf__`. They are not tile functions; they are just regular `AICORE` (or `PTO_INTERNAL`) inlines that the auto-mode compiler analyzes.
3. **Helpers that contain raw CCE intrinsics** (`copy_cbuf_to_gm`, `pipe_barrier`, `set_flag`/`wait_flag` of a custom shape, `__cce_get_tile_ptr`, etc.) — DO add `__tf__`. Make all `TileDType` parameters pass-by-value with `__in__`/`__out__` and `void` return. Inside the body, use `__cce_get_tile_ptr(arg)` (no `.data()`) to get the raw pointer. Canonical examples (post-PR-852, not yet merged): the new `TQuantCvtS32ToFp16` migration in `TQuant.hpp` and the new `TSTORE_MAT2GM_CONVTILE` shape in `texpands_mat_kernel.cpp` — see [external_context/pr_852_notes.md §L4b, §T4a](external_context/pr_852_notes.md).
4. **`Tile::data()` in kernel code** — do not call. Instead: pass `TileDType` **by value** into a `__tf__` helper (using `__in__`/`__out__`); the caller writes `helper(tile.data(), ...)` (or with explicit type, `helper<...>(tile.data(), ...)`). Inside the helper, use `__cce_get_tile_ptr(param)` directly, NOT `__cce_get_tile_ptr(param.data())`. Earlier guidance in this file said "pass `Tile&` to a `__tf__` helper and let it call `.data()` internally" — that older form was the bug texpands_mat hit; PR-852 corrects it.
5. **Sync inside a `__tf__` body** — auto-sync does NOT walk into tile functions. Use raw `set_flag`/`wait_flag`/`pipe_barrier` (or guarded `PtoSetWaitFlag` in the `#ifndef __PTO_AUTO__` branch with raw flags in `#else`). `PtoSetWaitFlag` ALONE inside `__tf__` silently drops sync in auto mode. See [auto_mode_bad_patterns.md §2.7](auto_mode_bad_patterns.md).
6. **Library `_IMPL` headers** ([include/pto/npu/a2a3/T*.hpp](../include/pto/npu/a2a3/), [include/pto/npu/a5/T*.hpp](../include/pto/npu/a5/)) — already follow this pattern. Use them as a copy-quality reference for any new `__tf__` helper signature.

---

## 5. Open / Unknown items

- **Exact compiler documentation for these CCE keywords** — not present in this repo. We have not found a header that declares them as compiler builtins explicitly; the inference rests on the absence of any A3/A5 `#define` plus their pervasive use plus the spec in `Library_Developer_Rules_And_Limitations.md`. Confirm with the bisheng-CCE language reference (Unknown — outside this repo).
- **Whether `__tf__` is enforced (e.g., compile error) when missing** — Unknown. Auto-mode rules say it should be present; not yet observed what the compiler actually does without it.
- **Whether CCE provides a vector-aware `__cce_get_tile_ptr` overload that accepts the auto-mode vector types** — Unknown. This bears on whether [tests/npu/a2a3/src/st/testcase/tload_gm2mat/](../tests/npu/a2a3/src/st/testcase/tload_gm2mat/) and [tests/npu/a5/src/st/testcase/tload_shape2d/](../tests/npu/a5/src/st/testcase/tload_shape2d/) actually compile under `--cce-enable-pto-passes`. See [auto_mode_bad_patterns.md §3.2, §6.3](auto_mode_bad_patterns.md).
