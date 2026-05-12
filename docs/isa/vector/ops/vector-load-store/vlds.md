# pto.vlds

`pto.vlds` is part of the [Vector Load Store](../../vector-load-store.md) instruction set.

## Summary

Load a vector register from the Unified Buffer (UB). The data layout within the vector is determined by the distribution mode attribute.

## Mechanism

`pto.vlds` reads data from UB into a vector register. The effective address is `UB[base + offset]`. The distribution mode controls how bytes map into destination lanes.

For each lane `i` in the vector register:

$$ \mathrm{dst}_i = \mathrm{UB}[\mathrm{base} + \mathrm{offset} + \mathrm{dispatch}(i, \mathrm{mode})] $$

where `dispatch` selects the memory address per lane based on the distribution mode.

## Syntax

### PTO Assembly Form

```text
vlds %result, %source[%offset] {dist = "DIST"}
```

### AS Level 1 (SSA)

```mlir
%result = pto.vlds %source[%offset] {dist = "DIST"} : !pto.ptr<T, ub> -> !pto.vreg<NxT>
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%source` | `!pto.ptr<T, ub>` | UB base address |
| `%offset` | index | Load displacement |
| `dist` | string attribute | Distribution mode |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%result` | `!pto.vreg<NxT>` | Loaded vector register value |

## Side Effects

Reads from UB-visible storage. Does not allocate buffers, signal events, or establish fences.

## Constraints

!!! warning "Constraints"
    The effective address must satisfy the alignment requirement of the selected distribution mode. Each mode has specific alignment constraints documented below.

    Inactive lanes (masked-off blocks) do not issue memory requests unless the operation explicitly documents otherwise.

## Distribution Modes

| Mode | Contiguous Bytes | Description | C Semantics | Alignment |
|------|:---------:|-------------|-------------|-----------|
| `NORM` | 256 B | Contiguous load | `dst[i] = UB[base + offset + i * sizeof(T)]` | 32 B |
| `BRC_B8` | 32 B | Broadcast 1 byte | `dst[i] = UB[base]` | 32 B |
| `BRC_B16` | 32 B | Broadcast 2 bytes | `dst[i] = UB[base]` | 32 B |
| `BRC_B32` | 32 B | Broadcast 4 bytes | `dst[i] = UB[base]` | 32 B |
| `US_B8` | 128 B | Upsample: duplicate each byte | `dst[2*i] = dst[2*i+1] = UB[base + i]` | 32 B |
| `US_B16` | 256 B | Upsample: duplicate each half | `dst[2*i] = dst[2*i+1] = UB[base + i]` | 32 B |
| `DS_B8` | 128 B | Downsample: every 2nd byte | `dst[i] = UB[base + 2*i]` | 32 B |
| `DS_B16` | 256 B | Downsample: every 2nd half | `dst[i] = UB[base + 2*i]` | 32 B |
| `UNPK_B8` | 64 B | Unpack: zero-extend byte to word | `dst_i32[i] = (uint32_t)UB_i8[base + i]` | 32 B |
| `UNPK_B16` | 128 B | Unpack: zero-extend half to word | `dst_i32[i] = (uint32_t)UB_i16[base + 2*i]` | 32 B |
| `UNPK_B32` | 256 B | Unpack: zero-extend word to dword | `dst_i32[i] = (uint32_t)UB_i32[base + 4*i]` | 32 B |
| `SPLT4CHN_B8` | 128 B | Split 4-channel (RGBA → R plane) | `dst[i] = UB[base + 4*i]` | 32 B |
| `SPLT2CHN_B8` | 64 B | Split 2-channel | `dst[i] = UB[base + 2*i]` | 32 B |
| `SPLT2CHN_B16` | 128 B | Split 2-channel | `dst[i] = UB[base + 2*i]` | 32 B |
| `DINTLV_B32` | 256 B | Deinterleave 32-bit | `dst[i] = UB[base + 8*i]` (even elements) | 32 B |
| `BLK` | varies | Block load | Blocked access pattern | varies |

## Exceptions

!!! danger "Exceptions"
    - Using an address outside UB-visible space is illegal.
    - Violating the alignment or distribution contract of the selected form is illegal.
    - Masked-off lanes do not validate the address; an illegal address is still illegal even if no active lane touches it.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    A5 is the primary concrete profile for vector instructions. CPU simulation and A2/A3-class targets emulate the behavior while preserving the PTO contract, but may support narrower subsets.

    Code that depends on a specific distribution mode list or timing should treat that dependency as target-profile-specific.

## Performance

The current public PTO timing material does not publish numeric latency or steady-state throughput for `pto.vlds`. If software scheduling or performance modeling depends on the exact cost, measure it on the concrete backend rather than inferring a constant.

## Examples

Contiguous load of f32 vector:

```mlir
%v = pto.vlds %ub[%offset] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

Broadcast scalar to all lanes:

```mlir
%v = pto.vlds %ub[%c0] {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

Deinterleave even elements from interleaved data:

```mlir
%v = pto.vlds %ub[%offset] {dist = "DINTLV_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Vector Load Store](../../vector-load-store.md)
- Next op in instruction set: [pto.vldas](./vldas.md)
