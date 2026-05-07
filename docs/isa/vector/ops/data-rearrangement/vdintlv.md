# pto.vdintlv

`pto.vdintlv` is part of the [Data Rearrangement](../../data-rearrangement.md) instruction set.

## Summary

Deinterleave an interleaved source stream into even/odd result vectors.

## Mechanism

`pto.vdintlv` separates an interleaved source stream into two result vectors. The low result receives the even-position elements and the high result receives the odd-position elements from the logical interleaved stream carried by `%lhs` and `%rhs`.

## Syntax

### PTO Assembly Form

```text
vdintlv %low, %high, %lhs, %rhs
```

### AS Level 1 (SSA)

```mlir
%low, %high = pto.vdintlv %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>, !pto.vreg<NxT>
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| %lhs | `!pto.vreg<NxT>` | First half of the interleaved source stream |
| %rhs | `!pto.vreg<NxT>` | Second half of the interleaved source stream |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| %low | `!pto.vreg<NxT>` | Even-position elements recovered from the interleaved stream |
| %high | `!pto.vreg<NxT>` | Odd-position elements recovered from the interleaved stream |

## Side Effects

This operation has no architectural side effect beyond producing its destination values. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

!!! warning "Constraints"
    - `%lhs`, `%rhs`, `%low`, and `%high` MUST have the same element type and vector width.
    - The result pair ordering is architectural and MUST be preserved by lowering.

## Exceptions

!!! danger "Exceptions"
    - The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
    - Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
    - Code that depends on an instruction-set-specific packing, selector, or permutation mode should treat that dependency as target-profile-specific unless the manual states cross-target portability explicitly.

## Examples

```c
// low  = {src[0], src[2], src[4], ...}
// high = {src[1], src[3], src[5], ...}
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Data Rearrangement](../../data-rearrangement.md)
- Previous op in instruction set: [pto.vintlv](./vintlv.md)
- Next op in instruction set: [pto.vslide](./vslide.md)
