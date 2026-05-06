# PTO-ISA Auto-Mode Kernel Development Instructions

We are working on PTO-ISA kernel development, especially auto-mode kernels for A3 and A5.

Auto mode is underdeveloped, especially for larger kernels such as GEMM, Flash Attention, and sparse attention. Because of that, do not assume manual-mode kernels can be copied directly into auto mode.

## Hard constraints

- You do not have compiler access.
- You cannot run tests.
- You must work statically from repository files, provided documentation, and user-provided compiler logs.
- Do not claim that code compiles unless the user explicitly provides compiler output proving it.
- Do not invent APIs, types, intrinsics, qualifiers, macros, or file paths.
- Every important claim must be grounded in:
  - an existing repo file path,
  - a provided document,
  - a PR/MR/issue link or text provided by the user,
  - or explicitly marked as an assumption.
- Mark uncertainty clearly using:
  - `Known`
  - `Inferred`
  - `Assumption`
  - `Unknown`
- Prefer small, reviewable patches.
- Preserve existing manual-mode behavior unless explicitly asked otherwise.
- For large kernels such as GEMM, Flash Attention, or sparse attention, produce a design/skeleton first, not a full optimized implementation.

## Main goal

First, help create a source-grounded knowledge base under:

```text
docs_for_ai/
```

Do not start by generating kernels. First read the source code and create focused documentation that will later help generate, review, and debug auto-mode A3/A5 kernels.

Useful files may include:

```text
docs_for_ai/repo_kernel_map.md
docs_for_ai/known_good_kernel_examples.md
docs_for_ai/auto_mode_bad_patterns.md
docs_for_ai/qualifier_reference.md
docs_for_ai/tile_type_reference.md
docs_for_ai/a3_a5_differences.md
docs_for_ai/assumptions_to_verify.md
docs_for_ai/compile_error_logbook.md
docs_for_ai/external_context/pr_index.md
docs_for_ai/external_context/pr_pattern_notes.md
```

## Knowledge base rules

When creating or updating files in `docs_for_ai/`:

- Do not summarize every file in the repo.
- Include only information useful for generating, reviewing, or debugging auto-mode A3/A5 kernels.
- Prefer concrete implementation patterns over generic explanations.
- Include exact file paths for all source-grounded claims.
- Include small code snippets only when they clarify a reusable pattern.
- Clearly label whether each point is `Known`, `Inferred`, `Assumption`, or `Unknown`.
- Keep assumptions separate from confirmed facts.
- If using PRs/MRs/issues, treat them as supporting context, not as the main source of truth.
- Prefer current source code over old PR/MR discussion if they conflict.
- Remove duplicated, outdated, or irrelevant notes when updating docs.

## Source hierarchy

Use this priority order:

1. Current repo source code.
2. Current repo docs/tests.
3. Provided official/internal docs.
4. Relevant PR/MR/issue discussions.
5. Inference from similar code.
6. Assumptions.

If source code and PR/MR discussion conflict, trust the current source code unless the user says otherwise.

## Code generation rules

Before generating or modifying code, always produce:

1. files inspected,
2. relevant patterns found,
3. current behavior,
4. proposed design,
5. assumptions needing confirmation,
6. risks,
7. minimal patch plan.

When producing code:

- Prefer unified diffs.
- Keep changes minimal.
- Do not refactor unrelated code.
- Do not silently change manual-mode behavior.
- Copy syntax from existing files where possible.
- For every nontrivial API/type/macro/qualifier used, cite where the same syntax appears in the repo or docs.
- If no source exists for an API/type/macro, do not use it.
- Do not mark the task complete just because code was written.
- Do not claim compilation or runtime success.

After producing code, always provide a manual checklist:

1. files changed,
2. commands the user should run manually if known,
3. expected compile/test target if known,
4. assumptions to verify,
5. likely first failure points.

## Auto-mode risk checklist

Before proposing auto-mode code, check for:

- manual-mode-only pointer access,
- missing `__tf__` on helper functions,
- missing or incorrect `AICORE` usage,
- unsafe `__gm__` / `__ubuf__` usage,
- helpers that incorrectly return false for auto mode,
- A5-only assumptions accidentally used for A3,
- unsupported template/type patterns,
- memory movement assumptions,
- tile shape/layout assumptions,
- intrinsics that may not be supported in auto mode,
- assumptions copied from manual-mode kernels without auto-mode evidence.

## Large kernel workflow

For GEMM, Flash Attention, sparse attention, or other large kernels:

Do not start with a full implementation.

First produce:

1. operation semantics,
2. expected tensor shapes,
3. data movement plan,
4. tiling strategy,
5. loop structure,
6. required primitives,
7. nearest existing repo examples,
8. missing information,
9. assumptions needing confirmation,
10. smallest compileable skeleton plan.

Recommended progression:

```text
small helper/qualifier fix
↓
simple elementwise or copy kernel
↓
simple tile/repeat-loop kernel
↓
reduction-like kernel
↓
GEMM skeleton
↓
GEMM simple implementation
↓
softmax/reduction pieces
↓
Flash Attention decomposition
↓
Flash Attention pieces
↓
sparse attention
```

## PR/MR and issue usage

PRs/MRs/issues can be used to extract historical design decisions and pitfalls, but do not read a large number blindly.

When asked to use PRs/MRs/issues:

1. First create an index of relevant PRs/MRs/issues.
2. Prioritize items related to:
   - auto mode,
   - A3,
   - A5,
   - kernels,
   - tile types,
   - `__tf__`,
   - `AICORE`,
   - GEMM,
   - matmul,
   - attention,
   - sparse attention,
   - TQuant,
   - TPOW,
   - compiler restrictions,
   - manual mode vs auto mode.
3. Read only high-relevance items in detail.
4. Extract only information useful for auto-mode A3/A5 kernel generation.
5. Link each extracted point to its PR/MR/issue source.
6. Mark each point as `Known from PR/MR`, `Inferred from PR/MR`, or `Assumption`.

## Compiler error workflow

When the user provides compiler errors:

- Focus on the first meaningful error first.
- Explain the likely cause.
- Identify whether the issue was already listed as an assumption or bad pattern.
- Produce the smallest possible fix.
- Update or propose an entry for `docs_for_ai/compile_error_logbook.md`.

A compile error log entry should include:

```md
## Error: short descriptive name

Symptom:
...

Likely cause:
...

Fix pattern:
...

Example file/path:
...

Confidence:
High / Medium / Low
```

## Output style

Be direct and practical.

When uncertain, say so. Do not make confident claims without repo/docs evidence.

Prefer this format for implementation tasks:

```text
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
2. ...

Manual checklist:
1. ...
2. ...
```
