# Compile Error Logbook (auto-mode A3/A5)

A running record of compiler errors (and suspected compiler errors) encountered
when building auto-mode A3/A5 kernels. The goal is to make the next occurrence
of the same error trivially debuggable: name → symptom → cause → fix → evidence.

This is a debugging aid, not an implementation reference. For "what is the
right shape," see [qualifier_reference.md](qualifier_reference.md),
[tile_type_reference.md](tile_type_reference.md),
[auto_mode_bad_patterns.md](auto_mode_bad_patterns.md),
[a3_a5_differences.md](a3_a5_differences.md),
[known_good_kernel_examples.md](known_good_kernel_examples.md), and
[external_context/pr_852_notes.md](external_context/pr_852_notes.md).

Scope: A3 (`PTO_NPU_ARCH_A2A3`, `__NPU_ARCH__ == 2201`) and A5 (`PTO_NPU_ARCH_A5`,
`__NPU_ARCH__ == 3101 || 3510`) only. CPU-sim, cost-model, Kirin entries should
be marked `non-target reference` and not used to drive A3/A5 conclusions.

---

## 1. Purpose

- Capture concrete compiler errors with enough context to recognize them again.
- Connect each error to a known anti-pattern or open assumption when applicable.
- Avoid re-deriving the same fix from first principles after every recurrence.
- Keep a record of errors whose root cause is **not yet understood**, so the
  next occurrence has a starting point.
- Surface errors that contradict our current understanding (i.e., trigger a
  refinement of `auto_mode_bad_patterns.md` or `qualifier_reference.md`).

This logbook is **not** a place to reproduce or paraphrase the auto-mode rules
docs, the bisheng-CCE compiler manual, or the kernel-rules / library-rules
documents. Cite, don't restate.

---

## 2. How to add a new entry

When the user provides compiler output:

1. Focus on the **first meaningful error** first (the rest are often
   cascades).
2. Search this logbook for an existing entry that matches the symptom (the
   "Symptom" field is the search index).
3. If a match exists, add a new occurrence under "Occurrences" with the
   minimum reproducer (file:line, the offending construct, and the affected
   platform). Update "Status" / "Confidence" if the new occurrence
   strengthens or weakens the previous read.
4. If no match exists, create a new entry under §8 using the §3 field set.
5. If the error contradicts something in `auto_mode_bad_patterns.md`,
   `qualifier_reference.md`, `tile_type_reference.md`, or
   `a3_a5_differences.md`, leave a `> Refines:` note pointing at the doc/section
   that needs updating.
6. Do **not** invent or rephrase compiler error messages. If the user's log
   is unavailable, mark the symptom field `Unknown` and rely on inferred
   cause + suggested-next-action instead.
7. Do **not** claim a fix compiles unless the user explicitly confirms it
   does (per CLAUDE.md hard constraints).

---

## 3. Required fields

Each entry must include:

- **Error name** — short, recognizable phrase. Use the offending construct
  (e.g., `__cce_get_tile_ptr(x + N)`) when possible.
- **Likely symptom** — what the compiler/linker prints, paraphrased only when
  the exact text is not yet available; mark `Unknown` in that case.
- **Likely cause** — the underlying auto-mode contract that is being
  violated, expressed in terms of repo evidence.
- **Affected platform** — `A3` / `A5` / `both` / `unknown`.
- **Source evidence** — file paths and line numbers from the repo, plus any
  PR/MR/issue references.
- **Fix pattern** — minimal change that resolves the error, described as a
  shape (not a full diff).
- **Status** — `Known` / `Inferred` / `Assumption` / `Unknown`.
- **Confidence** — `High` / `Medium` / `Low`.
- **Related docs** — links into `docs_for_ai/`.
- **Notes for future verification** — what would upgrade `Inferred` →
  `Known`, or what we still need to test/confirm.
- **Occurrences** — running list of (date, build target, file:line,
  reporter). Empty for entries seeded from the existing knowledge base only.

---

## 4. Severity / priority guide

The "priority" of a logbook entry reflects how often we expect to see it AND
how much damage a wrong fix would do.

| Priority | When to use |
|---|---|
| **High** | The error blocks a build that's currently expected to compile, OR a wrong fix could silently change correctness (e.g., aliasing, sync inside `__tf__`, write-only dst aliasing). Seed entries E1, E2, E3, E4, E5, E6, E7 fall here. |
| **Medium** | The error is rare or specific to one kernel family (conv, MX), OR the wrong fix only affects performance, not correctness. |
| **Low** | The error is a benign warning, a CPU-sim/cost-model build issue with no A3/A5 impact, or a one-off that we do not expect to recur. |

A priority is not "how loud the compiler is." Even a single-line syntax error
is High if its root cause is a contract violation we keep hitting (e.g.,
`Tile::data()` returning a vector type in auto mode).

---

## 5. Classification guide

Tag each entry with one or more of these classes. Use the same names in the
entry header so future searches across the logbook are stable.

| Class | What it covers | Cross-ref |
|---|---|---|
| `qualifier` | `__tf__`, `__in__`/`__out__`, `__cce_get_tile_ptr`, `__gm__`/`__ubuf__`/`__cbuf__`/`__ca__`/`__cb__`/`__cc__`/`__fbuf__` misuse | [qualifier_reference.md](qualifier_reference.md) |
| `tiledtype` | Anything around `Tile::data()` returning vector vs pointer, `tile_size(N)`, `TileDType` shape | [tile_type_reference.md §1, §7](tile_type_reference.md) |
| `tile-shape` | `Tile`/`ConvTile` template-parameter mistakes (e.g., `BufferSize_` as bytes), `BLayout`/`SLayout`/`SFractalSize` mismatch | [tile_type_reference.md §1.2, §10.2](tile_type_reference.md) |
| `aliasing` | `TASSIGN`/`TRESHAPE`/`TSUBVIEW` shape errors, dst/src overlap, `Tile<Bias>` fallback | [auto_mode_bad_patterns.md §1](auto_mode_bad_patterns.md) |
| `sync` | `set_flag`/`wait_flag`/`pipe_barrier`/`PtoSetWaitFlag`/`Event<>` placement | [auto_mode_bad_patterns.md §2](auto_mode_bad_patterns.md) |
| `kernel-vs-library` | `_IMPL` from kernel code, `Tile::data()` from kernel code, raw CCE intrinsics outside `__tf__` | [auto_mode_bad_patterns.md §3](auto_mode_bad_patterns.md) |
| `arch-mismatch` | A3-only construct on A5 or vice versa | [a3_a5_differences.md](a3_a5_differences.md) |
| `template` | Template-parameter coupling, `aclFloat16` in device kernel, default member initializers | [auto_mode_bad_patterns.md §5](auto_mode_bad_patterns.md) |
| `auto-sync` | Auto-sync over-conservatism, peeling failures, complex compound conditions | [auto_mode_bad_patterns.md §7](auto_mode_bad_patterns.md) |
| `cmake-build` | Build-list / target-flag / arch-macro issues distinct from source-level errors | [a3_a5_differences.md §1](a3_a5_differences.md) |

An entry can carry more than one tag. The first tag is the dominant one; later
tags refine.

---

## 6. Linked docs_for_ai files

Always reach for these when triaging a compile error:

- [auto_mode_bad_patterns.md](auto_mode_bad_patterns.md) — anti-pattern catalog. Most starter entries below cite a section here.
- [qualifier_reference.md](qualifier_reference.md) — `__tf__`, `__in__`/`__out__`, `__cce_get_tile_ptr` semantics.
- [tile_type_reference.md](tile_type_reference.md) — `Tile`/`ConvTile`/`GlobalTensor` definitions, `TileDType` shape per mode, `TileConfig` constants.
- [a3_a5_differences.md](a3_a5_differences.md) — when an error appears on one arch but not the other.
- [known_good_kernel_examples.md](known_good_kernel_examples.md) — clean references to copy syntax from.
- [repo_kernel_map.md](repo_kernel_map.md) — which kernels exist and which build mode they target.
- [assumptions_to_verify.md](assumptions_to_verify.md) — open questions; check whether a new error is already on the watchlist.
- [external_context/pr_852_notes.md](external_context/pr_852_notes.md) — A3 ST testcase fixes (NOT yet merged into this branch).

---

## 7. Caveats

- **PR-852 is not merged.** The current source still has the bugs the PR
  fixes. Entries E1–E5 below describe the **current** broken state; their
  "Fix pattern" describes the PR-852 recipe but does not imply the fix is
  in tree. ([external_context/pr_852_notes.md "Branch state"](external_context/pr_852_notes.md))
- **Inclusion in `ALL_TESTCASES` is build-coverage evidence, not correctness
  evidence.** Several seed entries below cite testcases that are in the
  build list but still carry the offending pattern. The mere fact that they
  were once observed to build is not proof that the pattern is auto-mode
  safe today.
- **No compiler messages reproduced verbatim.** Of the seed entries below,
  only E1 has any explicit compiler-stage description from PR-852
  ("libexpand Pass crash during RAUW"). All other "Likely symptom" lines
  are `Inferred` or `Unknown` and should be replaced with real text the
  first time the error is actually observed.

---

## 8. Starter entries

All entries below are seeded from existing `docs_for_ai/` evidence; none has
been observed in a build during this work. They exist so the first real
occurrence has somewhere to land.

---

### E1. `__cce_get_tile_ptr(x + N)` — pointer arithmetic on a vector value

- **Class** — `qualifier`, `tiledtype`.
- **Likely symptom** — Compiler crash during the `libexpand` pass while
  attempting RAUW (Replace-All-Uses-With), specifically on a `__tf__`
  helper whose body offsets a `TileDType` parameter before extraction.
  Exact diagnostic text — `Inferred`. (PR-852 description: *"The test case
  crashes during the libexpand Pass when it's trying to do a RAUW."*)
- **Likely cause** — In auto mode, `Tile::TileDType` is a vector value
  (e.g., `__ubuf__ T tile_size(N)`), not a pointer. `tmp + N` applies
  arithmetic to the vector before `__cce_get_tile_ptr` lowers it to a
  typed pointer; the libexpand pass cannot rewrite that.
- **Affected platform** — A3 (file is in `a2a3/`). A5 spot-check negative;
  full audit pending ([a3_a5_differences.md §11](a3_a5_differences.md)).
- **Source evidence** — eight occurrences in
  [include/pto/npu/a2a3/TCI.hpp:59,108,147,149,151,203](../include/pto/npu/a2a3/TCI.hpp#L59)
  inside `TCI_b32_repeat`/`TCI_b32_normal`/`TCI_b16_repeat`/`TCI_b16_normal`.
  PR-852 [§L1](external_context/pr_852_notes.md) rewrites them.
  `TileDType` shape source: [include/pto/common/pto_tile.hpp:1530-1540, 1166-1170](../include/pto/common/pto_tile.hpp#L1530-L1540).
- **Fix pattern** — Always extract first, then offset:
  ```cpp
  // BAD
  __ubuf__ T *p = (__ubuf__ T *)__cce_get_tile_ptr(tmp + 128);
  // GOOD
  __ubuf__ T *p = (__ubuf__ T *)__cce_get_tile_ptr(tmp) + 128;
  ```
  Use the extracted pointer (`dstPtr` style) for downstream `vadds` /
  `vmuls` / `vconv_*` calls instead of feeding the `TileDType` value.
- **Status** — `Known` anti-pattern; resolved-by-PR-852 (not yet merged).
- **Confidence** — High.
- **Related docs** —
  [auto_mode_bad_patterns.md §3.2.1](auto_mode_bad_patterns.md),
  [tile_type_reference.md §7](tile_type_reference.md),
  [external_context/pr_852_notes.md §L1](external_context/pr_852_notes.md),
  [assumptions_to_verify.md §4.2](assumptions_to_verify.md).
- **Notes for future verification** — capture the full compiler stack trace
  on the next observed crash; confirm whether the same pass crashes when the
  arithmetic is on `TileDType` produced by `__cce_get_tile_ptr` followed by
  another vector op (theoretically fine, but unverified).
- **Occurrences** — none recorded.

---

### E2. `PtoSetWaitFlag` inside a `__tf__` body — silently no sync in auto mode

- **Class** — `sync`, `kernel-vs-library`.
- **Likely symptom** — No compiler error. **Symptom is runtime-level**: the
  helper produces wrong results because no sync was emitted between
  pipelines. Possible secondary symptom: an auto-sync warning or a
  cross-pipeline race detected only at runtime. Compiler-level diagnostic —
  `Unknown`.
- **Likely cause** — `PtoSetWaitFlag<...>()` is intentionally a no-op in
  auto mode (so kernel-level code can be written once and let the auto-sync
  pass insert real sync). But the auto-sync pass treats `__tf__` as a black
  box and **does not look inside tile functions** ([Auto_Mode_Overview.md](../docs/auto_mode/Auto_Mode_Overview.md)).
  Inside a `__tf__` body the no-op therefore drops sync entirely. The
  authoritative rule is opposite to the kernel-level rule:
  [Library Rules §3](../docs/auto_mode/Library_Developer_Rules_And_Limitations.md):
  *"Use `set_flag`, `wait_flag` or `pipe_barrier` explicitly in tile
  functions and all of their callees. Use `PtoSetWaitFlag` or `TSYNC`
  anywhere else."*
- **Affected platform** — A3. A5 spot-check found no `PtoSetWaitFlag` inside
  tile-function bodies in the corresponding A5 headers; full audit
  pending. ([a3_a5_differences.md §11](a3_a5_differences.md))
- **Source evidence** — pre-PR-852 sites:
  [include/pto/npu/a2a3/TConcat.hpp:124,137,153,157,158](../include/pto/npu/a2a3/TConcat.hpp#L124),
  [include/pto/npu/a2a3/TFillPad.hpp:56](../include/pto/npu/a2a3/TFillPad.hpp#L56),
  [include/pto/npu/a2a3/TRowReduceIdxOps.hpp](../include/pto/npu/a2a3/TRowReduceIdxOps.hpp)
  (six places),
  [include/pto/npu/a2a3/TTrans.hpp](../include/pto/npu/a2a3/TTrans.hpp)
  (`TransTailTiles`, `TTransConvNC1HWC02C1HWNC0`).
- **Fix pattern** — Wrap each call:
  ```cpp
  #ifndef __PTO_AUTO__
      PtoSetWaitFlag<PIPE_V, PIPE_S>();
  #else
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  #endif
  ```
  Manual mode keeps the `PtoSetWaitFlag` shape (it expands to real
  `set_flag`/`wait_flag`); auto mode emits the raw CCE intrinsics directly.
- **Status** — `Known` anti-pattern; resolved-by-PR-852 (not yet merged).
- **Confidence** — High for the rule; the runtime symptom shape is `Inferred`.
- **Related docs** —
  [auto_mode_bad_patterns.md §2.7, §2.1 cross-cutting note](auto_mode_bad_patterns.md),
  [external_context/pr_852_notes.md §L2, §L3, §L6, §L7b](external_context/pr_852_notes.md),
  [assumptions_to_verify.md §4.4](assumptions_to_verify.md).
- **Notes for future verification** — when the next sync-related test
  failure shows up, check whether the helper containing the failing path is
  `__tf__` and uses `PtoSetWaitFlag`. The presence of any compiler-level
  warning here would be useful to record (we have not seen one).
- **Occurrences** — none recorded.

---

### E3. `ConvTile<..., bytes, ...>` — `BufferSize_` passed in bytes, not element count

- **Class** — `tile-shape`, `tiledtype`.
- **Likely symptom** — Likely no compiler error. Symptoms are **runtime**:
  UB allocation `sizeof(T)`× larger than intended, leading to UB
  exhaustion, allocation failure, or downstream tile-coalescing surprises.
  PR-852 description: *"Allocated tiles were bigger than the UB."* Exact
  compiler/runtime diagnostic — `Unknown`.
- **Likely cause** — The `BufferSize_` template parameter of `ConvTile` is
  fed straight into `tile_size(bufferSize)`, which (Inferred from the
  auto-mode contract) consumes an **element count**. The parameter is
  named "BufferSize" which suggests bytes; passing
  `elementSize * sizeof(T)` inflates the allocation by `sizeof(T)`.
- **Affected platform** — A3 visible site is the texpands_mat test
  (excluded from A3 auto-mode by CMake, see
  [tests/npu/a2a3/src/st/testcase/CMakeLists.txt:213-220](../tests/npu/a2a3/src/st/testcase/CMakeLists.txt#L213-L220)).
  `ConvTile` definition is shared, so the hazard exists on both archs.
- **Source evidence** —
  [tests/npu/a2a3/src/st/testcase/texpands_mat/texpands_mat_kernel.cpp:58, 63](../tests/npu/a2a3/src/st/testcase/texpands_mat/texpands_mat_kernel.cpp#L58):
  `bufferSizeA = elementSize * sizeof(T); ConvTile<TileType::Mat, T, bufferSizeA, ...>`.
  `ConvTile` template at [include/pto/common/pto_tile.hpp:1096, 1102, 1167](../include/pto/common/pto_tile.hpp#L1096).
  PR-852 [§T4b](external_context/pr_852_notes.md) fixes the call site
  (long-term TODO: rename the template parameter to `NumElems_`).
- **Fix pattern** — Pass the element count:
  ```cpp
  using TileData = ConvTile<TileType::Mat, T, elementSize,
                            Layout::NC1HWC0,
                            pto::ConvTileShape<N, C1, H, W, C0>>;
  ```
- **Status** — `Known` anti-pattern; resolved-by-PR-852 in `texpands_mat`
  only. Other call sites — `Unknown`.
- **Confidence** — High for the rule; element-count semantics of
  `tile_size(N)` is `Inferred` (see
  [assumptions_to_verify.md §1.1, §4.1](assumptions_to_verify.md)).
- **Related docs** —
  [auto_mode_bad_patterns.md §5.6](auto_mode_bad_patterns.md),
  [tile_type_reference.md §1.2, §10.2](tile_type_reference.md),
  [external_context/pr_852_notes.md §T4b](external_context/pr_852_notes.md).
- **Notes for future verification** — grep for any other site passing
  `* sizeof(T)` to a `ConvTile` template parameter; confirm element-count
  semantics with the bisheng-CCE compiler docs.
- **Occurrences** — none recorded.

---

### E4. `TRESHAPE` static_assert rejects `ConvTile` (pre-PR-852)

- **Class** — `aliasing`, `tile-shape`, `template`.
- **Likely symptom** — Compile-time `static_assert` failure of the form
  *"is_tile_data_v<TileDataIn> == true"* (or the matching `TileDataOut`
  message), or the `Loc == NewLoc` assert, when the caller passes a
  `ConvTile<...>` to `TRESHAPE`/`TRESHAPE_IMPL`. Exact assert message —
  `Inferred` from the source asserts.
- **Likely cause** — `TRESHAPE_IMPL` requires both arguments satisfy
  `is_tile_data_v` (true for `Tile`, false for `ConvTile`; `ConvTile` uses
  `is_conv_tile_v`). The asserts sit **outside** the `#ifndef __PTO_AUTO__`
  guard, so they fire even in auto mode where aliasing is a hint, not a
  runtime instruction.
- **Affected platform** — A3 (current source state, pre-PR-852). A5
  [TReshape.hpp](../include/pto/npu/a5/TReshape.hpp) status — `Unknown`,
  not yet inspected end-to-end. ([a3_a5_differences.md §11](a3_a5_differences.md))
- **Source evidence** —
  [include/pto/npu/a2a3/TReshape.hpp:23-28](../include/pto/npu/a2a3/TReshape.hpp#L23-L28).
  PR-852 [§L5](external_context/pr_852_notes.md) moves the asserts inside
  the `#ifndef __PTO_AUTO__` block.
  `is_tile_data_v` vs `is_conv_tile_v` traits at
  [pto_tile.hpp:1738, 1758-1762](../include/pto/common/pto_tile.hpp#L1738).
- **Fix pattern (manual workaround until PR-852 merges)** — Avoid
  `TRESHAPE` on `ConvTile` in this branch. Restructure the kernel so the
  alias is between two `Tile` views, or wait for the PR. Post-merge, the
  call becomes valid in auto mode (and remains rejected in manual mode).
- **Status** — `Known` anti-pattern in current source; resolved-by-PR-852
  (not yet merged).
- **Confidence** — High.
- **Related docs** —
  [auto_mode_bad_patterns.md §1.1](auto_mode_bad_patterns.md),
  [tile_type_reference.md §10.1](tile_type_reference.md),
  [external_context/pr_852_notes.md §L5](external_context/pr_852_notes.md),
  [assumptions_to_verify.md §3.5](assumptions_to_verify.md) (post-merge
  soft gotcha: `Loc == NewLoc` check is also dropped in auto mode).
- **Notes for future verification** — capture the exact `static_assert`
  message text the first time this is hit; confirm A5 `TReshape.hpp`
  carries the same shape.
- **Occurrences** — none recorded.

---

### E5. `Tile::data()` returning a vector value misused as a pointer

- **Class** — `tiledtype`, `qualifier`, `kernel-vs-library`.
- **Likely symptom** — One of (Inferred):
  - `reinterpret_cast<uintptr_t>(tile.data())` rejected by the compiler
    because the operand is a vector type, not a pointer.
  - Implicit conversion failure passing `tile.data()` where a `__ubuf__ T *`
    (or `__ca__`/`__cb__`/`__cc__`/`__cbuf__`/`__fbuf__` `T *`) is
    expected.
  - Type-mismatch when feeding `tile.data()` to a CCE intrinsic that
    expects a raw pointer.
  Exact diagnostic — `Unknown`.
- **Likely cause** — In auto mode, `Tile::data()` returns
  `TileDType &` where `TileDType = MemoryQualifier<...>::type tile_size(N)`
  ([pto_tile.hpp:1526-1561](../include/pto/common/pto_tile.hpp#L1526-L1561)).
  That is a sized vector reference, not a pointer. Manual mode and CPU-sim
  return pointers; the same source compiles in those modes and breaks in
  auto.
- **Affected platform** — both. Distinct A3 and A5 testcase sites listed below.
- **Source evidence** —
  - Pre-auto pattern preserved under guard:
    [include/pto/npu/a2a3/TQuant.hpp:108-114](../include/pto/npu/a2a3/TQuant.hpp#L108-L114)
    (`#ifndef __PTO_AUTO__` branch using `reinterpret_cast<uintptr_t>(...)`).
  - Pre-PR-852 broken site:
    [include/pto/npu/a2a3/TQuant.hpp:29-34](../include/pto/npu/a2a3/TQuant.hpp#L29-L34)
    (`TQuantBuffersOverlap` does the cast outside any guard).
  - Kernel-side violations (in `ALL_TESTCASES`, status under auto mode
    `Unknown`):
    [tests/npu/a5/src/st/testcase/textract/textract_kernel.cpp:281-283, 788](../tests/npu/a5/src/st/testcase/textract/textract_kernel.cpp#L281-L283),
    [tests/npu/a5/src/st/testcase/tmov_ub2l1/tmov_ub2l1_kernel.cpp:96-97](../tests/npu/a5/src/st/testcase/tmov_ub2l1/tmov_ub2l1_kernel.cpp#L96-L97),
    [tests/npu/a5/src/st/testcase/tload_mx_gmtensor/tload_mx_gmtensor_kernel.cpp:78-129](../tests/npu/a5/src/st/testcase/tload_mx_gmtensor/tload_mx_gmtensor_kernel.cpp#L78).
  - Excluded testcase that uses the same construct:
    [tests/npu/a2a3/src/st/testcase/texpands_mat/texpands_mat_kernel.cpp:23](../tests/npu/a2a3/src/st/testcase/texpands_mat/texpands_mat_kernel.cpp#L23)
    (auto-mode-excluded by CMake).
  - In-`ALL_TESTCASES` testcases with the same construct that nevertheless
    build (status — see
    [assumptions_to_verify.md §1.4](assumptions_to_verify.md)):
    [tests/npu/a2a3/src/st/testcase/tload_gm2mat/tload_gm2mat_kernel.cpp:23,31](../tests/npu/a2a3/src/st/testcase/tload_gm2mat/tload_gm2mat_kernel.cpp#L23),
    [tests/npu/a5/src/st/testcase/tload_shape2d/tload_shape2d_kernel.cpp:20](../tests/npu/a5/src/st/testcase/tload_shape2d/tload_shape2d_kernel.cpp#L20).
- **Fix pattern** —
  - Library: refactor the helper to be `__tf__`, take parameters typed
    `typename TileData::TileDType` **by value** with `__in__`/`__out__`,
    and use `__cce_get_tile_ptr(arg)` directly inside the body. Caller
    passes `tile.data()` at the kernel/library boundary. PR-852's
    `TQuantCvtS32ToFp16` and `TSTORE_MAT2GM_CONVTILE` migrations are the
    canonical examples
    ([external_context/pr_852_notes.md §L4b, §T4a](external_context/pr_852_notes.md)).
  - Library overlap-check pattern: short-circuit the
    `reinterpret_cast<uintptr_t>(...)` path under `__PTO_AUTO__`:
    ```cpp
    #ifndef __PTO_AUTO__
        // raw-pointer overlap check
    #else
        return true;
    #endif
    ```
  - Kernel: do not call `tile.data()` from kernel code at all. Use
    user-facing PTO instructions (`TLOAD`/`TSTORE`/`TMOV`/...). Note:
    `GlobalTensor::data()` is a different method and IS safe to call from
    kernel code (returns a raw `__gm__ T *` in all modes,
    [pto_tile.hpp:549](../include/pto/common/pto_tile.hpp#L549)).
- **Status** — `Known` anti-pattern; the in-`ALL_TESTCASES` survival
  of `tload_gm2mat`/`tload_shape2d`/`textract` is `Unknown` (see
  [auto_mode_bad_patterns.md §6.3, §6.4](auto_mode_bad_patterns.md)).
- **Confidence** — High for the rule; Medium for the symptom shape.
- **Related docs** —
  [auto_mode_bad_patterns.md §1.3, §3.2, §3.3](auto_mode_bad_patterns.md),
  [tile_type_reference.md §7](tile_type_reference.md),
  [qualifier_reference.md §3.1, §3.3](qualifier_reference.md),
  [external_context/pr_852_notes.md §L4a, §L4b, §T4a](external_context/pr_852_notes.md),
  [assumptions_to_verify.md §1.4, §4.3](assumptions_to_verify.md).
- **Notes for future verification** — capture the exact diagnostic; if a
  vector-aware `__cce_get_tile_ptr` overload exists, locate it (would
  partly soften this rule). Whether `tload_gm2mat` and `tload_shape2d`
  actually compile under `--cce-enable-pto-passes` is the key open
  question.
- **Occurrences** — none recorded.

---

### E6. Redundant `TLOAD(dstTile, dstGlobal)` on a write-only destination

- **Class** — `aliasing`, `auto-sync`.
- **Likely symptom** — No compile error expected. Symptoms are
  **runtime**: data race or wrong numerical result when the auto-mode
  allocator coalesces `dstTile` with another tile whose liveness does not
  overlap, so two simultaneous `TLOAD`s land in the same physical buffer.
  Compiler-level diagnostic — `Unknown`. (Possible auto-sync warning —
  `Unknown`.)
- **Likely cause** — In auto mode the compiler may coalesce two tiles to
  the same physical address when their liveness does not overlap. A
  `TLOAD` on a tile that is otherwise write-only (e.g., a `dstTile` that
  is the OUTPUT of `TADD`/`TDEQUANT`) creates a bogus read that races with
  another tile's load.
  ([Kernel Rules §3.1](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md): *"DON'T CALL REDUNDANT TLOAD!"*)
- **Affected platform** — both at the contract level. Concrete A3 example
  cited.
- **Source evidence** —
  [tests/npu/a2a3/src/st/testcase/tdequant/tdequant_kernel.cpp:50](../tests/npu/a2a3/src/st/testcase/tdequant/tdequant_kernel.cpp#L50):
  `TLOAD(dstTile, dstGlobal);` while `dstTile` is only the OUTPUT of
  `TDEQUANT(...)` at line 58. `tdequant` is in `ALL_TESTCASES`.
- **Fix pattern** — Delete the `TLOAD` on dst. If the op genuinely needs
  the dst seeded (read-modify-write), the legitimate analogue is `TAXPY`
  ([tests/npu/a2a3/src/st/testcase/taxpy/taxpy_kernel.cpp:36](../tests/npu/a2a3/src/st/testcase/taxpy/taxpy_kernel.cpp#L36)).
- **Status** — `Known` anti-pattern; whether it currently produces wrong
  numbers in `tdequant` — `Unknown` empirically.
- **Confidence** — High for the rule; Medium for "it is silently wrong
  today."
- **Related docs** —
  [auto_mode_bad_patterns.md §1.5, §6.2](auto_mode_bad_patterns.md),
  [assumptions_to_verify.md §3.2, §10.4](assumptions_to_verify.md).
- **Notes for future verification** — request a numerical-diff log of
  `tdequant` under auto vs manual; **also** investigate the apparent
  counter-example in PR-852 [§T1](external_context/pr_852_notes.md) where
  `tcolargmax_kernel.cpp` adds a `TLOAD(dstTile, dstGlobal)` between the
  src TLOAD and the manual sync block — possibly required for col
  reductions. ([assumptions_to_verify.md §7.5](assumptions_to_verify.md))
- **Occurrences** — none recorded.

---

### E7. Unguarded kernel-scope `set_flag` / `wait_flag` / `pipe_barrier`

- **Class** — `sync`, `kernel-vs-library`.
- **Likely symptom** — One of (Inferred):
  - Compile error because the raw CCE intrinsic is not visible at kernel
    scope under `--cce-enable-pto-passes` (Kernel Rules §3.2: raw CCE
    intrinsics "take raw pointer as arguments, which won't compile for
    auto mode").
  - No compile error but auto-sync conflicts with the manual flag,
    producing wrong runtime results.
  Exact diagnostic — `Unknown`. The rule explicitly requires the
  `#ifndef __PTO_AUTO__` guard at kernel scope ([Kernel Rules §3.3](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md)).
- **Likely cause** — Auto mode inserts its own sync. Manual `set_flag` /
  `wait_flag` / `pipe_barrier` at kernel scope either fails to compile
  (raw-pointer CCE intrinsic) or conflicts with the auto-sync analysis.
  Inside a `__tf__` body the rule reverses (E2).
- **Affected platform** — both. Most existing instances are A3.
- **Source evidence** —
  - Already-correct guarded shape (the canonical pattern):
    [tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp:91-94](../tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp#L91-L94),
    [tests/npu/a2a3/src/st/testcase/textract/textract_kernel.cpp:62-69](../tests/npu/a2a3/src/st/testcase/textract/textract_kernel.cpp#L62-L69),
    [tests/npu/a2a3/src/st/testcase/tmrgsort/tmrgsort_kernel.cpp:160](../tests/npu/a2a3/src/st/testcase/tmrgsort/tmrgsort_kernel.cpp#L160).
  - Currently broken (unguarded `pipe_barrier(PIPE_ALL)` at kernel scope,
    in `ALL_TESTCASES`):
    [tests/npu/a2a3/src/st/testcase/tcolexpandmax/tcolexpandmax_kernel.cpp:64](../tests/npu/a2a3/src/st/testcase/tcolexpandmax/tcolexpandmax_kernel.cpp#L64)
    and the sibling `tcolexpand{add,div,mul,sub}` testcases.
  - Manual-mode-only kernels (excluded from auto by design):
    [tests/npu/a2a3/src/st/testcase/tfa/tfa_kernel.cpp](../tests/npu/a2a3/src/st/testcase/tfa/tfa_kernel.cpp),
    [kernels/manual/a2a3/gemm_performance/](../kernels/manual/a2a3/gemm_performance/).
  - Standard guard fix in PR-852 testcases:
    [external_context/pr_852_notes.md §T1, §T2, §T6, §T7](external_context/pr_852_notes.md).
- **Fix pattern** — Wrap each call:
  ```cpp
  #ifndef __PTO_AUTO__
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  #endif
  ```
  In auto mode, drop the manual sync; the compiler synchronizes. Inside a
  `__tf__` body, see E2 — the rule is opposite.
- **Status** — `Known` anti-pattern at kernel scope.
- **Confidence** — High for the rule; Medium for "the unguarded
  `pipe_barrier(PIPE_ALL)` in `tcolexpand*` is silently broken" (testcases
  are in `ALL_TESTCASES` so they at least build —
  [assumptions_to_verify.md §1.3, §10.2](assumptions_to_verify.md)).
- **Related docs** —
  [auto_mode_bad_patterns.md §2.1, §2.2](auto_mode_bad_patterns.md),
  [external_context/pr_852_notes.md §T1, §T2, §T6, §T7](external_context/pr_852_notes.md),
  [assumptions_to_verify.md §10.2](assumptions_to_verify.md).
- **Notes for future verification** — when the next set_flag/wait_flag
  failure is observed, capture whether it is at kernel scope or inside a
  `__tf__` body — those have opposite fixes (E2 vs E7). Also confirm
  whether `pipe_barrier(PIPE_ALL)` is silently no-op'd in auto mode or
  rejected outright; the answer affects how strict E7 should be.
- **Occurrences** — none recorded.

---

### E8. `kernel_operator.h: No such file or directory` (bisheng-direct build)

- **Class** — `cmake-build`.
- **Likely symptom** — Compiler error: *fatal error: 'kernel_operator.h' file
  not found* on the kernel `.cpp` during the bisheng-CCE compile step.
- **Likely cause** — `kernel_operator.h` lives under the AscendC kernel
  framework include tree (`${ASCEND_HOME_PATH}/.../tikcpp/...`). It is on
  the include path when the project uses the `ascendc.cmake` macros
  (`ascendc_library` / `ascendc_compile_options`, see
  [demos/auto_mode/baseline/add/CMakeLists.txt](../demos/auto_mode/baseline/add/CMakeLists.txt)).
  It is **not** on the include path when the project uses the bisheng-
  direct toolchain pattern from
  [kernels/manual/a2a3/topk/CMakeLists.txt](../kernels/manual/a2a3/topk/CMakeLists.txt)
  (which only adds `${ASCEND_HOME_PATH}/include` plus
  `${ASCEND_DRIVER_PATH}/kernel/inc` plus `${PROJECT_SOURCE_DIR}/.../include/`).
- **Affected platform** — both at the contract level; first observed on A3.
- **Source evidence** — observed in
  [kernels/automode/a2a3/add_tile_array/](../kernels/automode/a2a3/add_tile_array/)
  on first build attempt. The kernel was originally adapted from
  [demos/auto_mode/baseline/add/csrc/kernel/add_custom.cpp:13](../demos/auto_mode/baseline/add/csrc/kernel/add_custom.cpp#L13)
  which `#include "kernel_operator.h"`. The reference patterns under the
  bisheng-direct path —
  [kernels/manual/a2a3/topk/topk_kernel.cpp](../kernels/manual/a2a3/topk/topk_kernel.cpp)
  and
  [tests/npu/a2a3/src/st/testcase/tadd/tadd_kernel.cpp](../tests/npu/a2a3/src/st/testcase/tadd/tadd_kernel.cpp)
  — do **not** include it.
- **Fix pattern** — drop the include. Use only:
  ```cpp
  #include <pto/common/constants.hpp>
  #include <pto/pto-inst.hpp>
  ```
  `__global__` and `AICORE` are bisheng-CCE compiler-provided keywords; no
  framework header is needed for them. If a kernel genuinely needs `Ascend`
  C framework helpers (queue/pipe/buffer abstractions), switch the project
  to the `ascendc.cmake` build harness instead, but be aware that the
  topk-style auto-mode flow does not use them.
- **Status** — `Known`.
- **Confidence** — High.
- **Related docs** —
  [known_good_kernel_examples.md §A11](known_good_kernel_examples.md)
  (the project that produced this error and the canonical fix),
  [auto_mode_bad_patterns.md §3.1](auto_mode_bad_patterns.md)
  (note about `set_mask_norm()` etc. in the demo, which travels with
  `kernel_operator.h`).
- **Notes for future verification** — when starting any new auto-mode
  kernel under `kernels/automode/`, copy the include block from
  [kernels/manual/a2a3/topk/topk_kernel.cpp:11-13](../kernels/manual/a2a3/topk/topk_kernel.cpp#L11-L13),
  not from the auto-mode add demo.
- **Occurrences** —
  - 2026-05-07 · A3 vec build (`bash run.sh -r npu -v Ascend910B1`) ·
    [kernels/automode/a2a3/add_tile_array/add_tile_array_kernel.cpp](../kernels/automode/a2a3/add_tile_array/add_tile_array_kernel.cpp)
    on first build attempt. Resolved by the fix pattern above.

---

### E9. `no matching function for call to 'ReadFile'`

- **Class** — `cmake-build` (host-side; not auto-mode-specific).
- **Likely symptom** — C++ compiler error from `main.cpp`: *no matching
  function for call to 'ReadFile(...)'* with the candidate signature
  `bool ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize)`.
- **Likely cause** — the second parameter is a **non-const lvalue
  reference** (`size_t &fileSize`). Passing a `const size_t` (or any
  rvalue) to it cannot bind. The function writes the actual on-disk size
  back through the reference (`tests/common/test_common.h:98`:
  `fileSize = size;`).
- **Affected platform** — both at the contract level; first observed on A3.
- **Source evidence** —
  [tests/common/test_common.h:64-101](../tests/common/test_common.h#L64-L101).
  All in-tree examples use a non-const local
  (e.g.,
  [kernels/manual/a2a3/topk/main.cpp:63-64](../kernels/manual/a2a3/topk/main.cpp#L63-L64):
  `size_t inFileSize = rows * cols * sizeof(T);`).
- **Fix pattern** — declare the size variable as `size_t` (not `const
  size_t`). It is fine to reuse the same variable across multiple
  `ReadFile` calls when both files are the same size — `ReadFile`
  overwrites it with the actual size each time, but with identical sizes
  the second call sees the correct value.
- **Status** — `Known`.
- **Confidence** — High.
- **Related docs** —
  [known_good_kernel_examples.md §A11](known_good_kernel_examples.md).
- **Notes for future verification** — none; this is a `test_common.h`
  API quirk, not an auto-mode-specific gotcha.
- **Occurrences** —
  - 2026-05-07 · A3 vec build (`bash run.sh -r npu -v Ascend910B1`) ·
    [kernels/automode/a2a3/add_tile_array/main.cpp](../kernels/automode/a2a3/add_tile_array/main.cpp)
    on second build attempt (after E8 was fixed). Resolved by removing
    the `const` qualifier on the `fileSize` local.

---

## 9. Reserved for new entries

Append new entries below using the §3 field set. Keep entries focused on
**one** root cause; if a single build emits multiple unrelated errors, file
each separately and cross-reference them in "Notes for future verification."
