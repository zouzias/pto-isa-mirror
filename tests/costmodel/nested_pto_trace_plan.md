# Nested PTO Trace Support

## Question

Can the current costmodel trace framework represent nested PTO instructions, where one PTO instruction contains another PTO instruction?

Today the answer is only partially yes:

- Nested PTO execution is supported for cycle accounting.
- Nested PTO structure is not preserved in the trace.
- Inner PTO helper calls are collapsed into the active top-level PTO record.

The goal is to support hierarchical trace visibility with minimal code churn.

## Current Behavior

The current implementation in `include/pto/costmodel/trace.hpp` and `include/pto/costmodel/pto_instr.hpp` uses `PtoInstrScope` to mark PTO instruction boundaries.

When a PTO API runs inside another active PTO API:

- `BeginPtoInstr(...)` does not create a new `PtoInstrRecord`.
- It pushes the current top-level record index again onto `active_pto_stack`.
- `AppendCceCall(...)` appends all CCE calls to that same top-level record.
- `total_cycles` are accumulated on the same flat record.
- Pipe tail flushing happens at the outermost PTO scope boundary.

As a result:

- `trace.executed_pto.back().name` is only the top-level PTO name.
- `trace.executed_pto.back().cce_calls` is a flat CCE list.
- `trace.executed_pto.back().total_cycles` is still correct for the whole composed execution.
- Inner PTO names are not preserved as child nodes.

This means the current framework is correct for accounting, but not structural for nested PTO debugging.

## Motivation

Flat trace collapsing is sufficient for:

- total cycle validation
- existing `GetLastPtoInstrCycles()` behavior
- current `EXPECT_CYCLE_NEAR(...)` checks

It is insufficient for:

- debugging composite PTO helpers that call other PTO APIs internally
- understanding lowering shape when an outer PTO delegates to one or more inner PTO steps
- making `log_level=2` output explain the PTO hierarchy instead of only showing a flat CCE stream

The desired improvement is structural visibility, not a change to cycle semantics.

## Minimal-Change Plan

Use an additive design that keeps the current flat trace contract intact.

### 1. Keep the flat compatibility view unchanged

Do not change the current meaning of:

- `TraceState.executed_pto`
- `TraceState.active_pto_stack`
- `GetLastPtoInstrCycles()`
- `GetCurrentPtoInstrCycles()`
- `trace.executed_pto.back().cce_calls`

Existing cycle accounting and flat trace consumers should keep working exactly as they do now.

### 2. Add a parallel nested PTO tree

In `include/pto/costmodel/trace.hpp`, add a new nested node type, for example:

```cpp
struct PtoTraceNode {
    std::string name;
    std::vector<CceCallRecord> cce_calls;
    uint64_t total_cycles = 0;
    std::vector<PtoTraceNode> children;
};
```

Extend `TraceState` with:

- `std::vector<PtoTraceNode> executed_pto_tree`
- `std::vector<PtoTraceNode*> active_pto_tree_stack`

This makes the nested tree a debug structure alongside the existing flat view, not a replacement for it.

### 3. Build the nested tree in PTO scope entry/exit

In `BeginPtoInstr(...)`:

- keep current flat behavior exactly as-is
- additionally create nested tree nodes
- if there is no active nested PTO, create a new root node in `executed_pto_tree`
- otherwise create a child node under the current nested PTO
- push the active nested node pointer onto `active_pto_tree_stack`

In `EndPtoInstr()`:

- keep current flat pop/flush behavior exactly as-is
- also pop `active_pto_tree_stack`

### 4. Record CCE calls into both views

In `AppendCceCall(...)`:

- keep appending to the current flat top-level record
- also append the call to the current nested leaf node

For cycle accumulation:

- keep flat accumulation exactly as-is
- also add the CCE call cycles to every active nested PTO node in `active_pto_tree_stack`

This makes parent nested PTO totals inclusive of descendant work.

### 5. Mirror pipe tail accounting into the nested tree

In `FlushPendingTail(...)`:

- keep current flat tail handling unchanged
- also add the flushed tail cycles to every active nested PTO node

This preserves consistency between flat totals and nested totals.

### 6. Prefer the nested tree in trace printing

In `tests/costmodel/st/common/cost_check.hpp`:

- keep cycle output unchanged
- update `log_level=2` trace printing to prefer the nested tree when present
- preserve the current text-based block format
- print nested PTO children recursively with indentation

Suggested output shape:

```text
[TRACE] MyTest.case
  pto: OUTER
  total_cycles: 100
  cce_calls: 0
  child_pto[0]:
    pto: INNER
    total_cycles: 100
    cce_calls: 1
      [0] name=vadd cycles=78 args=[...]
```

If no nested children exist, the formatter should continue to emit the current single-node block shape.

## Compatibility Notes

This plan is intentionally additive.

It does not change:

- cycle semantics
- flat trace semantics
- existing callers that read `trace.executed_pto.back()`
- existing checks based on `GetLastPtoInstrCycles()`

The nested PTO tree is only a structural debug view.

The current flat trace remains the compatibility view for all existing consumers.

## Validation

Validation after implementation should include:

- one focused costmodel testcase that intentionally nests PTO scopes
- one existing non-nested testcase such as `tadd` to confirm no regression
- verification that `GetLastPtoInstrCycles()` remains unchanged
- verification that `log_level=2` shows parent/child PTO hierarchy when nesting exists
- verification that flat trace consumers still see the current top-level collapsed view

Relevant implementation anchors:

- `include/pto/costmodel/trace.hpp`
- `include/pto/costmodel/pto_instr.hpp`
- `tests/costmodel/st/common/cost_check.hpp`
