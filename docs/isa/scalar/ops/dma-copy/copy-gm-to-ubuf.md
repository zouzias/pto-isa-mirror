# pto.copy_gm_to_ubuf

`pto.copy_gm_to_ubuf` is part of the [DMA Copy](../../dma-copy.md) instruction set.

## Summary

Execute a DMA transfer from Global Memory into Unified Buffer using the current GM→UB loop and stride configuration.

## Mechanism

The DMA engine reads `%n_burst` rows from `%gm_src` and writes them to `%ub_dst`. `%len_burst` controls the contiguous byte count copied per row. `%left_padding`, `%right_padding`, and `%data_select_bit` control whether the destination row is padded beyond the copied byte range. `%src_stride` and `%dst_stride` specify the row-to-row start offsets for this copy invocation.

## Syntax

### PTO Assembly Form

```text
copy_gm_to_ubuf %gm_src, %ub_dst, %sid, %n_burst, %len_burst, %left_padding, %right_padding, %data_select_bit, %l2_cache_ctl, %src_stride, %dst_stride
```

### AS Level 1 (SSA)

```mlir
pto.copy_gm_to_ubuf %gm_src, %ub_dst, %sid, %n_burst, %len_burst, %left_padding, %right_padding, %data_select_bit, %l2_cache_ctl, %src_stride, %dst_stride : !pto.ptr<T, gm>, !pto.ptr<T, ub>, i64, i64, i64, i64, i64, i1, i64, i64, i64
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| %gm_src | `!pto.ptr<T, gm>` | GM source pointer |
| %ub_dst | `!pto.ptr<T, ub>` | UB destination pointer |
| %sid | `i64` | DMA stream identifier |
| %n_burst | `i64` | Number of burst rows to transfer |
| %len_burst | `i64` | Contiguous byte count transferred per row |
| %left_padding | `i64` | Left padding byte count applied in the destination row |
| %right_padding | `i64` | Right padding byte count applied in the destination row |
| %data_select_bit | `i1` | Controls whether padding bytes are materialized according to the configured pad behavior |
| %l2_cache_ctl | `i64` | Target-specific L2 cache allocation hint |
| %src_stride | `i64` | GM row-to-row start offset in bytes |
| %dst_stride | `i64` | UB row-to-row start offset in bytes |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | This form does not return SSA values; it writes data into Unified Buffer memory. |

## Side Effects

Reads GM-visible storage, writes UB-visible storage, and consumes the active GM→UB loop and stride configuration.

## Constraints

!!! warning "Constraints"
    - `%ub_dst` MUST satisfy the UB alignment requirements of the selected target profile.
    - `%len_burst` MUST fit within the configured row stride and DMA limits of the selected target profile.
    - If padding is enabled, the padded destination footprint MUST still fit in the destination UB region.

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
pto.copy_gm_to_ubuf %gm_src, %ub_dst, %sid, %n_burst, %len_burst, %left_padding, %right_padding, %data_select_bit, %l2_cache_ctl, %src_stride, %dst_stride : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i1, i64, i64, i64
```

## Related Ops / Instruction Set Links

- Instruction set overview: [DMA Copy](../../dma-copy.md)
- Previous op in instruction set: [pto.set_loop1_stride_ubtoout](./set-loop1-stride-ubtoout.md)
- Next op in instruction set: [pto.copy_ubuf_to_gm](./copy-ubuf-to-gm.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
