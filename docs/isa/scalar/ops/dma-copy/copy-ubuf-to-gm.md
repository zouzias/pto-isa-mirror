# pto.copy_ubuf_to_gm

`pto.copy_ubuf_to_gm` is part of the [DMA Copy](../../dma-copy.md) instruction set.

## Summary

Execute a DMA transfer from Unified Buffer into Global Memory using the current UB→GM loop and stride configuration.

## Mechanism

The DMA engine reads `%n_burst` rows from `%ub_src` and writes them to `%gm_dst`. `%len_burst` controls the contiguous byte count copied per row, so padded bytes in the UB row are not written back unless they are part of the burst length. `%src_stride` and `%dst_stride` specify the row-to-row start offsets for this copy invocation.

## Syntax

### PTO Assembly Form

```text
copy_ubuf_to_gm %ub_src, %gm_dst, %sid, %n_burst, %len_burst, %reserved, %dst_stride, %src_stride
```

### AS Level 1 (SSA)

```mlir
pto.copy_ubuf_to_gm %ub_src, %gm_dst, %sid, %n_burst, %len_burst, %reserved, %dst_stride, %src_stride : !pto.ptr<T, ub>, !pto.ptr<T, gm>, i64, i64, i64, i64, i64, i64
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| %ub_src | `!pto.ptr<T, ub>` | UB source pointer |
| %gm_dst | `!pto.ptr<T, gm>` | GM destination pointer |
| %sid | `i64` | DMA stream identifier |
| %n_burst | `i64` | Number of burst rows to transfer |
| %len_burst | `i64` | Contiguous byte count transferred per row |
| %reserved | `i64` | Reserved field; portable code should pass zero unless a target profile documents another meaning |
| %dst_stride | `i64` | GM row-to-row start offset in bytes |
| %src_stride | `i64` | UB row-to-row start offset in bytes |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | This form does not return SSA values; it writes data into Global Memory. |

## Side Effects

Reads UB-visible storage, writes GM-visible storage, and consumes the active UB→GM loop and stride configuration.

## Constraints

!!! warning "Constraints"
    - `%ub_src` MUST satisfy the UB alignment requirements of the selected target profile.
    - `%len_burst` MUST fit within the configured row stride and DMA limits of the selected target profile.
    - Only the requested burst bytes are copied from each UB row; padded tail bytes remain local to UB.

## Exceptions

!!! danger "Exceptions"
    - The verifier rejects illegal operand shapes, unsupported pipe or event identifiers, and attribute combinations that are not valid for the selected instruction set or target profile.
    - Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - CPU simulation preserves the visible copy contract but may not expose all DMA overlap hazards.
    - A2/A3 and A5 may narrow supported element sizes, row widths, or cache-control semantics.

## Examples

```mlir
pto.copy_ubuf_to_gm %ub_src, %gm_dst, %sid, %n_burst, %len_burst, %reserved, %dst_stride, %src_stride : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64
```

## Related Ops / Instruction Set Links

- Instruction set overview: [DMA Copy](../../dma-copy.md)
- Previous op in instruction set: [pto.copy_gm_to_ubuf](./copy-gm-to-ubuf.md)
- Next op in instruction set: [pto.copy_ubuf_to_ubuf](./copy-ubuf-to-ubuf.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
