# pto.vshift

`pto.vshift` is part of the [Data Rearrangement](../../data-rearrangement.md) instruction set.

## Summary

Single-source shift that inserts zero-fill at the vacated lanes.

## Mechanism

`pto.vshift` is the single-source sibling of `pto.vslide`. It shifts the source vector by `%amt` lanes and fills newly uncovered lanes with zero according to the selected element type.

## Syntax

### PTO Assembly Form

```text
vshift %dst, %src, %amt
```

### AS Level 1 (SSA)

```mlir
%result = pto.vshift %src, %amt : !pto.vreg<NxT>, i16 -> !pto.vreg<NxT>
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| %src | `!pto.vreg<NxT>` | Source vector |
| %amt | `i16` | Shift amount in lanes |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| %result | `!pto.vreg<NxT>` | Shifted vector with zero-filled vacated lanes |

## Side Effects

This operation has no architectural side effect beyond producing its destination values. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

!!! warning "Constraints"
    - `%src` and `%result` MUST have the same element type and vector width.
    - The shift amount MUST satisfy the range supported by the selected target profile.
    - Zero-fill versus any alternative fill behavior MUST match the selected form.

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
for (int i = 0; i < N; i++)
    dst[i] = (i >= amt) ? src[i - amt] : 0;
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Data Rearrangement](../../data-rearrangement.md)
- Previous op in instruction set: [pto.vslide](./vslide.md)
- Next op in instruction set: [pto.vsqz](./vsqz.md)
