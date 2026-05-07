# pto.vci

`pto.vci` is part of the [Conversion Ops](../../conversion-ops.md) instruction set.

## Summary

Standalone contract page for `pto.vci`.

## Mechanism

`pto.vci` is an index-generation operation. It produces a vector of indices starting from the scalar seed `%index` and incrementing or decrementing by 1 per lane. The generated indices are used to support indexed access patterns (gather/scatter) and argsort preparation.

## Syntax

### PTO Assembly Form

```asm
vci %index, %mask {order = "ORDER"} : !pto.vreg<Nxi32> -> !pto.vreg<Nxi32>
```

### AS Level 1 (SSA)

```mlir
%indices = pto.vci %index {order = "ASC"} : i32 -> !pto.vreg<64xi32>
```

### AS Level 2 (DPS)

```mlir
pto.vci ins(%index : i32) outs(%indices : !pto.vreg<64xi32>) {order = "ASC"}
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%index` | scalar `i32` | Scalar seed or base index for index generation |
| `%mask` | `!pto.mask<G>` | Predication mask (optional in some forms); inactive lanes may produce zero or preserve existing values |

**Attributes:**

| Attribute | Values | Description |
|-----------|--------|-------------|
| `order` | `"ASC"` / `"DESC"` | Sort order for index generation; `ASC` generates increasing indices, `DESC` generates decreasing |

## Expected Outputs

| Operand | Type | Description |
|---------|------|-------------|
| `%result` | `!pto.vreg<Nxi32>` | Generated index vector |

### C Semantics

```c
// ASC order: indices = base, base+1, base+2, ..., base+N-1
// DESC order: indices = base, base-1, base-2, ..., base-(N-1)
```

The `%index` scalar is the starting value; each lane `i` produces `base + i` (ASC) or `base - i` (DESC).

This is an index-generation family, not a numeric conversion. `ORDER` and the result element type together determine how indices are generated.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

!!! warning "Constraints"
    - `%result` uses an integer element type (`i32` in the common form).
    - The scalar `%index` type matches the result element type.
    - The `order` attribute is required when using sorted index generation.
    - For the standard form, `N` (lane count) is derived from the result type.

## Exceptions

!!! danger "Exceptions"
    - The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
    - Illegal `order` values are rejected by the verifier.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
    - Under the current documented A5 profile contract, `pto.vci` maps to hardware trace with no vector `RV_*` in sampled `veccore0` trace.

## Performance

### Execution Model

`pto.vci` is an index-generation operation executed within a `pto.vecscope` region. It produces lane-wise index values without invoking the Vector Core's main ALU — the latency is dominated by mask setup and predicate generation rather than compute units.

### A5 Execution

On A5, `pto.vci` maps to hardware trace with no sampled `RV_*` in the `veccore0` trace — it is implemented in the predicate/materialization layer, not as a standard vector compute instruction.

### A2/A3 Throughput

`vci` does not map to a direct CCE vector instruction in the A2/A3 cost model. It is compiled as a scalar index-generation loop within the vecscope:

| Metric | Value | Notes |
|--------|-------|-------|
| Startup | ~10 cycles | mask setup + loop overhead |
| Per-element | O(1) | simple arithmetic per lane |
| Complexity | O(N) | one operation per output lane |

The actual throughput depends on the surrounding loop structure and the number of iterations in the vecscope.

### Execution Note

`vci` is commonly used to initialize index buffers for gather/scatter operations and argsort:

```mlir
// Initialize ascending index buffer: [0, 1, 2, 3, ..., 63] for 64-element gather
%base_idx = pto.vci %c0 {order = "ASC"} : i32 -> !pto.vreg<64xi32>
// Generates: [0, 1, 2, 3, ..., 63] in lane 0
```

---

## Examples

### Generate ascending indices (common use for gather/scatter)

```mlir
%indices = pto.vci %c0 {order = "ASC"} : i32 -> !pto.vreg<64xi32>
// Result: [0, 1, 2, 3, ..., 63]
```

### Generate descending indices

```mlir
%indices = pto.vci %c63 {order = "DESC"} : i32 -> !pto.vreg<64xi32>
// Result: [63, 62, 61, 60, ..., 0]
```

### Use with gather (indexed load)

```mlir
// Generate indices, then use for indexed load
%idx = pto.vci %c0 {order = "ASC"} : i32 -> !pto.vreg<64xi32>
%data = pto.vgather2 %ub_table[%c0], %idx {dist = "DIST"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

### Use with vsort32 (argsort)

```mlir
// Generate ascending indices as sort keys
%indices = pto.vci %c0 {order = "ASC"} : i32 -> !pto.vreg<64xi32>
pto.vsort32 %sorted_indices, %indices, %config : !pto.ptr<i32, ub>, !pto.ptr<i32, ub>, i64
```

## Detailed Notes

`pto.vci` generates lane indices from a scalar seed. The two primary use cases are:

1. **Indexed access**: Generate indices for `vgather2` / `vscatter` operations to access arbitrary elements.
2. **Argsort preparation**: Generate sequential indices before sorting, then rearrange data based on sorted indices.

The generated indices are stable across invocations for the same `%index` seed, making them suitable as sort keys for indirect sort operations.

## Related Ops / Instruction Set Links

- Instruction set overview: [Conversion Ops](../../conversion-ops.md)
- Previous op in instruction set: [pto.vci](./vci.md) (self-referential; see also [pto.vcvt](./vcvt.md))
- Next op in instruction set: [pto.vcvt](./vcvt.md)
- Related index-generation: [pto.vsort32](../sfu-and-dsa-ops/vsort32.md) — argsort using index vectors
- Related gather/scatter: [pto.vgather2](../vector-load-store/vgather2.md), [pto.vscatter](../vector-load-store/vscatter.md)
