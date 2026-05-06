# PTO-ISA Auto-Mode Kernel Development Instructions

We are working on PTO-ISA kernel development for **A3 and A5 only**. This file is an index and workflow guide; technical detail lives in `docs_for_ai/`. Read the relevant doc before generating or modifying code; do not duplicate its content here.

## Hard constraints

- No compiler access; no test execution.
- Do not claim code compiles or runs unless the user provides output proving it.
- Do not invent APIs, types, intrinsics, qualifiers, macros, or file paths. Every nontrivial claim must be grounded in a repo file path, a provided document, or user-provided PR/MR/issue text — or explicitly marked `Assumption`.
- Mark uncertainty as `Known` / `Inferred` / `Assumption` / `Unknown`.
- Prefer small, reviewable patches. Preserve manual-mode behavior unless explicitly asked otherwise.
- For large kernels (GEMM, Flash Attention, sparse attention), produce a design / skeleton first, not a full optimized implementation.

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
- `docs_for_ai/external_context/pr_852_notes.md` — PR-852 notes (NOT merged into this branch).

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

PR-852 is **not merged** into this branch (verified at the source lines it modifies — see `external_context/pr_852_notes.md` "Branch state"). Treat its recipes as supporting context and forward-looking guidance. The current source still has the bugs. Cite `external_context/pr_852_notes.md`; do not claim a PR-852 fix is in tree.

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
