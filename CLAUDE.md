# PTO-ISA Auto-Mode Kernel Development Instructions

We are working on PTO-ISA kernel development for **A3 and A5 only**. This file is an index and workflow guide; technical detail lives in `docs_for_ai/`. Read the relevant doc before generating or modifying code; do not duplicate its content here.

## Hard constraints

- No compiler access; no test execution.
- Do not claim code compiles or runs unless the user provides output proving it.
- Do not invent APIs, types, intrinsics, qualifiers, macros, or file paths. Every nontrivial claim must be grounded in a repo file path, a provided document, or user-provided PR/MR/issue text — or explicitly marked `Assumption`.
- Mark uncertainty as `Known` / `Inferred` / `Assumption` / `Unknown`.
- Prefer small, reviewable patches. Preserve manual-mode behavior unless explicitly asked otherwise.
- For large kernels (GEMM, Flash Attention, sparse attention), produce a design / skeleton first, not a full optimized implementation.
- If you made a kernel it should be able run `bash run.sh -r npu -v Ascend910B1` and give\
.0  1 you the outcome of comparing with the python script.

## Target platform scope

**A3** (`PTO_NPU_ARCH_A2A3`, `__NPU_ARCH__ == 2201`) and **A5** (`PTO_NPU_ARCH_A5`, `__NPU_ARCH__ == 3101 || 3510`) only.

CPU-sim, cost-model, Kirin, and other non-A3/A5 paths are **secondary references only**. They explain `#define`-as-empty fallbacks and arch-macro logic but do not drive A3/A5 conclusions. When citing them, label as `non-target reference`. If a token is empty in CPU-sim / Kirin / cost-model, do not conclude it is empty on A3/A5.

## Testcase caution

Inclusion in `ALL_TESTCASES` is build-coverage evidence, not clean-style or correctness evidence. A testcase may build in auto mode and still carry manual-mode idioms (`TASSIGN` aliasing, raw `set_flag`/`wait_flag`, `TPipe`/`TPUSH`/`TPOP`, raw CCE intrinsics, `Tile::data()` from kernel code, `*_IMPL` from kernel code, double buffering, `pipe_barrier`).

Before treating a testcase as a clean reference, cross-check `docs_for_ai/auto_mode_bad_patterns.md` and `docs_for_ai/known_good_kernel_examples.md`. Prefer files explicitly listed there as clean.

## docs_for_ai source map

Always read the relevant file(s) before generating or modifying code:

- `docs_for_ai/repo_kernel_map.md` — which kernels exist, where, and which build mode they target.
- `docs_for_ai/known_good_kernel_examples.md` — clean references to copy syntax from.
- `docs_for_ai/auto_mode_bad_patterns.md` — anti-pattern catalog (memory/aliasing, sync, kernel-vs-library, arch hazards, template hazards, auto-sync).
- `docs_for_ai/qualifier_reference.md` — `__tf__`, `__in__` / `__out__`, `__cce_get_tile_ptr`, memory-space qualifiers.
- `docs_for_ai/tile_type_reference.md` — `Tile` / `ConvTile` / `GlobalTensor`, `TileDType` shape per mode, `TileConfig` constants.
- `docs_for_ai/a3_a5_differences.md` — for any cross-arch question (`TileLeft` BLayout split, `BiasTile` divergence, `TMATMUL_MX` / `MGATHER` / `MSCATTER` / `*_Custom`/`Hp` A5-only, etc.).
- `docs_for_ai/assumptions_to_verify.md` — open questions; check before assuming.
- `docs_for_ai/compile_error_logbook.md` — structured log of compiler errors with likely causes and fix patterns.
- `docs_for_ai/external_context/pr_852_notes.md` — historical PR-852 notes; useful for why several auto-mode review rules exist.

When updating these files: include exact file paths for source-grounded claims; keep `Known` / `Inferred` / `Assumption` / `Unknown` labels; remove duplicated or stale notes; do not summarize the whole repo.

## Source priority

1. Current repo source code.
2. Current repo docs / tests.
3. Provided official / internal docs.
4. Relevant PR / MR / issue discussions.
5. Inference from similar code.
6. Assumptions.

If source code and PR/MR discussion conflict, trust the source unless the user says otherwise.

## PR-852 handling

PR-852 has been merged into the current branch. Treat `docs_for_ai/external_context/pr_852_notes.md` as historical context explaining why certain auto-mode rules exist, not as evidence that the current source is still broken.

The PR-852 patterns remain important review rules:
- do pointer arithmetic after `__cce_get_tile_ptr(...)`, not before;
- do not use `PtoSetWaitFlag` inside `__tf__` bodies in auto mode;
- migrate raw-CCE helpers to `__tf__` with `TileDType __in__` / `__out__`;
- avoid `Tile::data()` pointer casts in auto mode;
- use the fixed `TRESHAPE` / `ConvTile` behavior as current source behavior.

Before claiming a PR-852-related bug still exists, inspect the current source. If the fixed pattern is present, mark the issue as `Resolved` / `Known fixed`. If the same symptom appears again, treat it as a regression or a newly introduced instance of the old anti-pattern.

## Cube-layout warning

For GEMM/cube kernels, always reason in terms of 16x16 fractal blocks in L0A/L0B/L0C. A larger matrix tile is a grid of 16x16 fractals. Do not treat `TileLeft`, `TileRight`, or `TileAcc` as flat row-major matrices.

## Memory-budget-first kernel planning

Before generating any nontrivial kernel, especially GEMM, FA, MoE, TopK, reduction, or fused kernels, produce a memory-budget and operation-granularity plan before writing code.

Required pre-code output:

```text
Memory budgets:
- L1 custom budget:
- L0A custom budget:
- L0B custom budget:
- L0C custom budget:
- UB custom budget:

Live tiles by memory level:
- L1:
- L0A:
- L0B:
- L0C:
- UB:

Smallest hardware operation:
- cube operation shape, if any:
- vector operation shape, if any:

Loop tiling plan:
- tile-and-loop dimensions:
- inferred tile sizes:
- compile-time unroll/peel strategy, if any:
- tail handling only where needed:

Test-shape plan:
- tiny debug shape:
- medium tiling shape:
- model-inspired realistic shape:
- tail shape, if supported:
```

Use `docs_for_ai/pto_auto_mode_hw_optimization_guide.md` as the detailed source. Do not write the kernel first and reason about memory later.

## Workflow before generating or modifying code

Use this format for implementation tasks:

```
Files inspected:
- ...

Known:
- ...

Inferred:
- ...

Assumptions:
- ...

Risks:
- ...

Patch plan:
1. ...

Manual checklist:
1. files changed
2. commands the user should run manually if known
3. expected compile / test target if known
4. assumptions to verify
5. likely first failure points
```

When producing code:

- Prefer unified diffs and minimal changes; do not refactor unrelated code.
- Do not silently change manual-mode behavior.
- For every nontrivial API / type / macro / qualifier used, cite the existing repo or doc location of the same syntax. If no source exists, do not use it.
- Writing code does not mean the task is complete; do not claim compile or runtime success.

For risk auditing (memory/aliasing, sync, qualifier, A3-vs-A5, template hazards, auto-sync), defer to `docs_for_ai/auto_mode_bad_patterns.md` and `docs_for_ai/a3_a5_differences.md` rather than restating the checklist here.

For large kernels, defer the staged progression and design-first checklist to a future `docs_for_ai/kernel_generation_playbook.md`. Until then: design / skeleton before implementation, smallest compileable unit first.

## Compiler error workflow

When the user provides compiler output:

1. Focus on the **first meaningful error** first.
2. Search `docs_for_ai/compile_error_logbook.md` for an existing entry that matches the symptom. If matched, add an Occurrence; update Status / Confidence as warranted.
3. If unmatched, create a new entry per the field set in that file (§3).
4. If the error contradicts `auto_mode_bad_patterns.md`, `qualifier_reference.md`, `tile_type_reference.md`, or `a3_a5_differences.md`, leave a `> Refines:` note in the new logbook entry pointing at the doc/section that needs updating.
5. Produce the smallest possible fix, grounded in repo evidence.

## Context management

- Do **not** run `/compact` unless the user explicitly asks.
- Do not summarize prior turns to "save space"; the system handles compression.
- When unsure about prior context, ask before acting.

## Output style

Be direct and practical. State uncertainty clearly. Cite repo paths for nontrivial claims. Avoid restating `docs_for_ai/` content in chat — link to the specific section instead.

## Confirmed baseline auto-mode A3 kernel project

`kernels/automode/a2a3/add_tile_array/` is the first in-tree auto-mode A3 kernel project that has been confirmed to build and run correctly on real hardware (`bash run.sh -r npu -v Ascend910B1` → `test data success`). It is the **simplest form of auto mode**: single AICORE, in-kernel serial loop over fixed-size Vec tiles, static valid region, `TLOAD → TADD → TSTORE`, no manual sync, no double buffering, no `block_idx` work split.

When starting a new auto-mode A3 prototype, copy this project's structure and the `pto_example_vec_auto` CMake function from it. See `docs_for_ai/known_good_kernel_examples.md §A11` for the full pattern catalog and the resolved-Known list.

Future iterations will introduce additional optimization techniques on top of this baseline (multi-core `block_idx` partitioning, larger / dynamic shapes, sanctioned pipeline abstractions if/when they land, A5 mirror, etc.). When the user introduces a new technique:

1. Capture the new pattern as an entry in `docs_for_ai/known_good_kernel_examples.md` only after a build / run confirmation.
2. If a compile error was observed in the process, log it in `docs_for_ai/compile_error_logbook.md` per its §2 / §3.
3. Move resolved items in `docs_for_ai/assumptions_to_verify.md` into §11 (Resolved by experiments / build runs) — do not delete; keep the audit trail.


## Prototype project requirements

When creating a new prototype kernel project under `kernels/automode/`:

- Include a minimal test or run path whenever possible.
- Include a Python reference script for expected output when the kernel computes a numerical result.
- Include clear input/output shape assumptions.
- Include a README with:
  - what the kernel does,
  - target platform,
  - auto-mode constraints,
  - how to build,
  - how to run,
  - how to compare against the Python reference,
  - known limitations.
- Do not claim the project builds or passes tests unless the user provides compiler/runtime output.
- If the repo’s test harness pattern is unclear, create a conservative local `run.sh` and Python reference script, and document what the user should adapt on the compiler server.
