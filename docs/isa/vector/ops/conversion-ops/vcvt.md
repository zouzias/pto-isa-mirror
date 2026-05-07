# pto.vcvt

`pto.vcvt` is part of the [Conversion Ops](../../conversion-ops.md) instruction set.

## Summary

`pto.vcvt` performs type conversion between floating-point and integer types, between floating-point types of different widths, and between integer types of different widths. It supports optional rounding mode, saturation mode, and lane-placement (`part`) control via instruction attributes. It produces a destination vector register of a different type from the source vector register — width-changing conversions may reduce or increase the lane count accordingly.

## Mechanism

`pto.vcvt` changes vector element interpretation, width, rounding, or saturation without leaving the vector-register model. The single `pto.vcvt` surface covers four conversion families:

- **Float-to-Int**: converts floating-point elements to integer elements with optional rounding and saturation
- **Float-to-Float**: converts between floating-point types of different widths (e.g., f32 to f16/bf16)
- **Int-to-Float**: converts integer elements to floating-point elements
- **Int-to-Int**: converts integer elements to integer elements of different widths with optional saturation

### Predicate and Zero-Merge Behavior

Inactive lanes (as determined by the `%mask` operand) do not participate in the conversion and produce zero in the destination lane:

```c
for (int i = 0; i < min(N, M); i++) {
    if (mask[i])
        dst[i] = convert(src[i], T0, T1, rnd, sat);
    else
        dst[i] = 0;  // zero-merge for inactive lanes
}
```

## Syntax

### PTO Assembly Form

```assembly
PTO.vcvt  vd, vs, vmask, rnd, sat, part
```

Where:
- `vd` is the destination vector register
- `vs` is the source vector register
- `vmask` is the predicate register
- `rnd` is the rounding mode (optional)
- `sat` is the saturation mode (optional)
- `part` is the part mode for width-changing conversions (optional)

### MLIR SSA Form

```mlir
%result = pto.vcvt %input, %mask {rnd = "RND", sat = "SAT", part = "PART"}
    : !pto.vreg<NxT0>, !pto.mask<G> -> !pto.vreg<MxT1>
```

### AS Level 1 (SSA)

```mlir
%result = pto.vcvt %input, %mask {rnd = "R", sat = "SAT"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xi32>
```

### AS Level 2 (DPS)

```mlir
PTO.vcvt  v0, v1, vmask, R, SAT, NONE
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%input` | `!pto.vreg<NxT0>` | Source vector register. `T0` is the source element type. `N` is the lane count. |
| `%mask` | `!pto.mask<G>` | Execution mask. Gates which lanes participate in the conversion. Mask granularity `G` must match the source vector family: `b32` for f32/i32 families, `b16` for f16/bf16/i16 families, `b8` for i8 families. |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%result` | `!pto.vreg<MxT1>` | Destination vector register. `T1` is the destination element type. `M` may differ from `N` for width-changing conversions. Inactive lanes produce zero. |

## Side Effects

`pto.vcvt` has no architectural side effects beyond producing its SSA result. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

!!! warning "Constraints"
    ### Type Constraints

    - Only the source/destination type pairs listed in the [Supported Type Matrix](#supported-type-matrix) are legal.
    - The source and destination element types must be distinct unless the conversion is explicitly documented as identity (no-op) for that type pair.
    - Width-changing conversions automatically adjust the lane count: `M * bitwidth(T1) = N * bitwidth(T0) = 2048`.

    ### Mask Constraints

    - The execution mask must use the typed-mask granularity that matches the source vector family.
    - There is no `!pto.mask<b64>` form in VPTO.

    ### Attribute Constraints

    - `rnd`: Only the rounding modes listed in the [Rounding Modes](#rounding-modes) table are valid. Default is `"R"` (round to nearest, ties to even) when omitted.
    - `sat`: Use `"SAT"` to enable saturation on overflow; `"NOSAT"` (default) wraps or produces undefined results on overflow.
    - `part`: Only valid for width-changing conversions. Use `"EVEN"` to write to even-indexed lanes of each lane group, `"ODD"` for odd-indexed lanes.

## Exceptions

!!! danger "Exceptions"
    - The verifier rejects illegal operand shapes, unsupported element type pairs, mismatched mask granularity, and invalid attribute combinations.
    - Conversions that overflow the destination range and have `sat = "NOSAT"` produce target-defined results (may wrap or be undefined).

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - **A5**: Full `pto.vcvt` surface is supported on Ascend 950. See the [Supported Type Matrix](#supported-type-matrix) and [Latency and Throughput](#latency-and-throughput-a5) sections.
    - **CPU simulation**: Behavior is preserved with floating-point semantics matching IEEE 754 where applicable. Latency values are target-defined.
    - **A2/A3**: Subset support only; the specific type pairs and attribute combinations available on A2/A3-class targets are target-defined. Code that depends on a specific conversion pair should treat it as target-profile-specific.

## Rounding Modes

| Mode | Name | Description |
|------|------|-------------|
| `R` | Round to nearest, ties to even | Default. Rounds to the nearest representable value; when exactly halfway, rounds to the even alternative. |
| `A` | Round away from zero | Rounds toward the larger-magnitude representable value. |
| `F` | Floor | Rounds toward negative infinity. |
| `C` | Ceil | Rounds toward positive infinity. |
| `Z` | Truncate | Rounds toward zero. |
| `O` | Round to odd | Rounds to the nearest odd representable value. |

## Saturation Modes

| Mode | Description |
|------|-------------|
| `SAT` | On overflow, saturate to the maximum (for positive) or minimum (for negative) representable value of the destination type. |
| `NOSAT` | Default. On overflow, wrap (integer) or produce undefined behavior (float-to-int without saturation). |

## Part Modes

Use `part` when a width-changing conversion writes only one half of each wider destination lane group. This is typically used in even/odd placement forms.

| Mode | Description |
|------|-------------|
| `EVEN` | Output to even-indexed lanes of each lane group. |
| `ODD` | Output to odd-indexed lanes of each lane group. |

## Supported Type Conversions

### Float to Int

| Form | Rounding | Saturation | Notes |
|------|----------|------------|-------|
| `!pto.vreg<64xf32>` → `!pto.vreg<32xsi64>` | Yes | Yes | f32 → i64 (narrowing, 2:1 reduction) |
| `!pto.vreg<64xf32>` → `!pto.vreg<64xsi32>` | Yes | Yes | f32 → i32 (same lane count, different type) |
| `!pto.vreg<64xf32>` → `!pto.vreg<128xsi16>` | Yes | Yes | f32 → i16 (narrowing, 2:1 reduction) |
| `!pto.vreg<128xf16>` → `!pto.vreg<64xsi32>` | Yes | Yes (optional) | f16 → i32 |
| `!pto.vreg<128xf16>` → `!pto.vreg<128xsi16>` | Yes | Yes (optional) | f16 → i16 |
| `!pto.vreg<128xf16>` → `!pto.vreg<256xsi8>` | Yes | Yes | f16 → i8 (narrowing, 2:1 reduction) |
| `!pto.vreg<128xf16>` → `!pto.vreg<256xui8>` | Yes | Yes | f16 → ui8 (narrowing, 2:1 reduction) |
| `!pto.vreg<128xbf16>` → `!pto.vreg<64xsi32>` | Yes | Yes | bf16 → i32 |

### Float to Float

| Form | Part | Notes |
|------|------|-------|
| `!pto.vreg<64xf32>` → `!pto.vreg<128xf16>` | Yes | f32 → f16 (narrowing, 2:1 reduction) |
| `!pto.vreg<64xf32>` → `!pto.vreg<128xbf16>` | Yes | f32 → bf16 (narrowing, 2:1 reduction) |
| `!pto.vreg<128xf16>` → `!pto.vreg<64xf32>` | Yes | f16 → f32 (widening, 1:2 expansion) |
| `!pto.vreg<128xbf16>` → `!pto.vreg<64xf32>` | Yes | bf16 → f32 (widening, 1:2 expansion) |

### Int to Float

| Form | Rounding | Notes |
|------|----------|-------|
| `!pto.vreg<256xui8>` → `!pto.vreg<128xf16>` | No | ui8 → f16 (widening, 2:1 expansion) |
| `!pto.vreg<256xsi8>` → `!pto.vreg<128xf16>` | No | si8 → f16 (widening, 2:1 expansion) |
| `!pto.vreg<128xsi16>` → `!pto.vreg<128xf16>` | Yes | i16 → f16 |
| `!pto.vreg<128xsi16>` → `!pto.vreg<64xf32>` | Yes | i16 → f32 |
| `!pto.vreg<64xsi32>` → `!pto.vreg<64xf32>` | Yes | i32 → f32 |
| `!pto.vreg<64xui32>` → `!pto.vreg<64xf32>` | Yes | ui32 → f32 |

### Int to Int

| Form | Saturation | Part | Notes |
|------|------------|------|-------|
| `!pto.vreg<256xui8>` → `!pto.vreg<128xui16>` | No | Yes | ui8 → ui16 |
| `!pto.vreg<256xui8>` → `!pto.vreg<64xui32>` | No | Yes | ui8 → ui32 |
| `!pto.vreg<256xsi8>` → `!pto.vreg<128xsi16>` | No | Yes | si8 → si16 |
| `!pto.vreg<256xsi8>` → `!pto.vreg<64xsi32>` | No | Yes | si8 → si32 |
| `!pto.vreg<128xui16>` → `!pto.vreg<256xui8>` | Yes | Yes | ui16 → ui8 (narrowing) |
| `!pto.vreg<128xui16>` → `!pto.vreg<64xui32>` | No | Yes | ui16 → ui32 |
| `!pto.vreg<128xsi16>` → `!pto.vreg<256xui8>` | Yes | Yes | si16 → ui8 (narrowing with saturation) |
| `!pto.vreg<128xsi16>` → `!pto.vreg<64xui32>` | No | Yes | si16 → ui32 |
| `!pto.vreg<128xsi16>` → `!pto.vreg<64xsi32>` | No | Yes | si16 → si32 |
| `!pto.vreg<64xui32>` → `!pto.vreg<256xui8>` | Yes | Yes | ui32 → ui8 (narrowing) |
| `!pto.vreg<64xui32>` → `!pto.vreg<128xui16>` | Yes | Yes | ui32 → ui16 (narrowing) |
| `!pto.vreg<64xui32>` → `!pto.vreg<128xsi16>` | Yes | Yes | ui32 → si16 (narrowing) |
| `!pto.vreg<64xsi32>` → `!pto.vreg<256xui8>` | Yes | Yes | si32 → ui8 (narrowing) |
| `!pto.vreg<64xsi32>` → `!pto.vreg<128xui16>` | Yes | Yes | si32 → ui16 (narrowing) |
| `!pto.vreg<64xsi32>` → `!pto.vreg<128xsi16>` | Yes | Yes | si32 → si16 (narrowing) |
| `!pto.vreg<64xsi32>` → `!pto.vreg<32xsi64>` | No | Yes | si32 → si64 (widening) |

## Supported Type Matrix

The table below is a summary overview. For exact attribute combinations, use the per-form entries in the [Supported Type Conversions](#supported-type-conversions) section as the source of truth.

| `src \ dst` | `ui8` | `si8` | `ui16` | `si16` | `ui32` | `si32` | `si64` | `f16` | `f32` | `bf16` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ui8` | — | — | Y | — | Y | — | — | Y | — | — |
| `si8` | — | — | — | Y | — | Y | — | Y | — | — |
| `ui16` | Y | — | — | — | Y | — | — | — | — | — |
| `si16` | Y | — | — | — | Y | Y | — | Y | Y | — |
| `ui32` | Y | — | Y | Y | — | — | — | — | — | — |
| `si32` | Y | — | Y | Y | — | — | Y | — | Y | — |
| `si64` | — | — | — | — | — | — | — | — | — | — |
| `f16` | Y | Y | — | Y | — | Y | — | — | Y | — |
| `f32` | — | — | — | Y | — | Y | Y | Y | — | Y |
| `bf16` | — | — | — | — | — | Y | — | — | Y | — |

## Width-Changing Conversion Pattern

For conversions that change lane width (e.g., f32 → f16), use even/odd parts and combine:

```mlir
// Convert two f32 vectors to one f16 vector using even/odd placement
%even = pto.vcvt %in0, %mask {rnd = "R", sat = "SAT", part = "EVEN"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>
%odd  = pto.vcvt %in1, %mask {rnd = "R", sat = "SAT", part = "ODD"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>
%result = pto.vor %even, %odd, %mask
    : !pto.vreg<128xf16>, !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<128xf16>
```

## Latency and Throughput (A5)

| PTO op | RV (A5) | Note | Latency |
|--------|----------|------|---------|
| `pto.vcvt` | `RV_VCVT_F2F` | f32 → f16 | **7** cycles |
| `pto.vcvt` | — | Other conversion pairs | Target-defined |

!!! note "Note:"
    Only representative traces are listed. Other `pto.vcvt` conversion pairs depend on the RV lowering in the trace. CPU simulation and A2/A3 throughput data is target-defined.

## Examples

### C Code

```c
// Float to Int with round-to-nearest and saturation
float src[64];
int dst[64];
int mask[64];

for (int i = 0; i < 64; i++) {
    if (mask[i]) {
        dst[i] = (int)roundf(src[i]);  // R mode, with SAT
    } else {
        dst[i] = 0;  // zero-merge for inactive lanes
    }
}
```

### MLIR SSA Form

```mlir
// Convert f32 to i32 with saturation
%result = pto.vcvt %input_f32, %mask {rnd = "R", sat = "SAT"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xi32>

// Convert f32 to f16 using even/odd parts
%even = pto.vcvt %in0, %mask {rnd = "R", sat = "SAT", part = "EVEN"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>
%odd  = pto.vcvt %in1, %mask {rnd = "R", sat = "SAT", part = "ODD"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>

// Convert i32 to f32
%float_result = pto.vcvt %input_i32, %mask
    : !pto.vreg<64xi32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```

### Typical Usage: Quantization

```mlir
// Quantize f32 activations to i8 with scale
// Scale factor applied first
%scaled = pto.vmul %input, %scale, %mask
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// Round and saturate to i32
%quantized = pto.vcvt %scaled, %mask {rnd = "R", sat = "SAT"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xi32>

// Narrow i32 → i8 via pack ops
// (use pto.vpack or similar width-reduction sequence)
```

### Typical Usage: Mixed Precision

```mlir
// bf16 → f32 for high-precision accumulation
%f32_vec = pto.vcvt %bf16_input, %mask {part = "EVEN"}
    : !pto.vreg<128xbf16>, !pto.mask<b16> -> !pto.vreg<64xf32>

// f32 → bf16 for storage
%bf16_out = pto.vcvt %f32_result, %mask {rnd = "R", part = "EVEN"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xbf16>
```

## Detailed Notes

### Rounding Mode Selection Guide

| Use case | Recommended mode |
|----------|-----------------|
| General neural network inference | `"R"` (round to nearest, ties to even) |
| Floor division | `"F"` (floor) |
| Ceiling division | `"C"` (ceil) |
| Truncate to integer | `"Z"` (truncate toward zero) |
| Deterministic rounding | `"O"` (round to odd) |

### Attribute Guidance

- **`rnd`**: Use when the conversion needs an explicit rounding rule, especially for float-to-int, float-to-float narrowing, or integer-to-float forms that do not map exactly. Defaults to `"R"` when omitted.
- **`mask`**: Use to select which source lanes participate in the conversion. In width-changing conversions, `mask` works together with `part` to determine which logical lane positions are produced.
- **`sat`**: Use when the conversion may overflow the destination range and hardware exposes a saturating form. Defaults to `"NOSAT"` when omitted.
- **`part`**: Use for width-changing conversions that select the even or odd half of the destination packing layout. Only valid when the destination lane count differs from the source lane count.

## Related Ops / Instruction Set Links

- Instruction set overview: [Conversion Ops](../../conversion-ops.md)
- Previous op in instruction set: [pto.vci](./vci.md)
- Next op in instruction set: [pto.vtrc](./vtrc.md)
- Rounding instruction: [pto.vtrc](./vtrc.md) — float-to-integer-valued-float truncation
