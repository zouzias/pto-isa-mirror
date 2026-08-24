# pto.mem_bar

`pto.mem_bar` is part of the [Pipeline Sync](../../pipeline-sync.md) instruction set.

## Summary

Order aliased Unified Buffer memory accesses inside a vector-visible execution scope.

## Mechanism

`pto.mem_bar` is a memory-ordering primitive for vector-side UB traffic. It does not name a producer and consumer pipeline explicitly; instead it constrains the ordering of aliased load/store classes such as store-then-load or load-then-store inside `pto.vecscope` execution.

## Syntax

### PTO Assembly Form

```text
mem_bar "BARRIER_TYPE"
```

### AS Level 1 (SSA)

```mlir
pto.mem_bar "BARRIER_TYPE"
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| `BARRIER_TYPE` | enum | Barrier class such as `VV_ALL`, `VST_VLD`, or `VLD_VST` that selects which aliasing pattern must be ordered |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | This form does not return SSA values; it establishes a memory-ordering point for the selected UB access class. |

## Side Effects

Orders the selected classes of UB memory traffic so later aliased accesses observe the intended program order inside the vector-visible scope.

## Constraints

!!! warning "Constraints"
    - `pto.mem_bar` is meaningful only for UB aliasing hazards inside a vector-visible execution scope.
    - The selected barrier type MUST be valid for the chosen target profile.
    - This barrier does not replace cross-pipeline event ordering when data moves between MTE and vector pipelines.

## Exceptions

!!! danger "Exceptions"
    - The verifier rejects illegal operand shapes, unsupported pipe or event identifiers, and attribute combinations that are not valid for the selected instruction set or target profile.
    - Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - CPU simulation preserves the visible ordering contract but may not expose every hardware hazard that motivates the barrier.
    - A2/A3 and A5 may support different barrier classes or stronger/weaker default ordering.

## Examples

```mlir
pto.vsts %v0, %ub[%c0] : !pto.vreg<64xf32>, !pto.ptr<f32, ub>
pto.mem_bar "VST_VLD"
%v1 = pto.vlds %ub[%c0] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Pipeline Sync](../../pipeline-sync.md)
- Previous op in instruction set: [pto.pipe_barrier](./pipe-barrier.md)
- Next op in instruction set: (none)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
