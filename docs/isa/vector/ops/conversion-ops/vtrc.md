# pto.vtrc

`pto.vtrc` is part of the [Conversion Ops](../../conversion-ops.md) instruction set.

## Summary

Round or truncate each lane while keeping the vector element type unchanged.

## Mechanism

`pto.vtrc` applies a target-selected rounding mode to each active lane of the source vector. Unlike `pto.vcvt`, it does not change the destination element type; instead it rounds the existing element representation in place, which is useful for operations such as floor, round-to-zero, or round-to-nearest on floating-point vectors.

## Syntax

### PTO Assembly Form

```text
vtrc %dst, %src, "ROUND_MODE"
```

### AS Level 1 (SSA)

```mlir
%result = pto.vtrc %input, "ROUND_MODE" : !pto.vreg<NxT> -> !pto.vreg<NxT>
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| %input | `!pto.vreg<NxT>` | Source vector register |
| `ROUND_MODE` | enum | Rounding selector such as round-to-zero, floor, ceil, or round-to-nearest as supported by the target profile |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| %result | `!pto.vreg<NxT>` | Rounded or truncated vector result with the same element type as the source |

## Side Effects

This operation has no architectural side effect beyond producing its destination vector register. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

!!! warning "Constraints"
    - `pto.vtrc` preserves the vector width and element type of the source.
    - The selected `ROUND_MODE` MUST be supported by the chosen target profile.
    - Lowering MUST preserve the lane-wise rounding semantics documented by the selected form.

## Exceptions

!!! danger "Exceptions"
    - The verifier rejects illegal operand shapes, unsupported element types, and unsupported rounding-mode attributes.
    - Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
    - Code that depends on a specific rounding mode should treat that dependency as target-profile-specific unless the manual states cross-target portability explicitly.

## Examples

```mlir
%rounded = pto.vtrc %input, "ROUND_R" : !pto.vreg<64xf32> -> !pto.vreg<64xf32>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Conversion Ops](../../conversion-ops.md)
- Previous op in instruction set: [pto.vcvt](./vcvt.md)
- Rounding-related conversion: [pto.vcvt](./vcvt.md)
