# pto.tadd

`pto.tadd` is part of the [Elementwise Tile Tile](../../elementwise-tile-tile.md) instruction set.

## Summary

`TADD` performs lane-wise addition of two source tiles into a destination tile.

For every element inside the destination tile valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} + \mathrm{src1}_{i,j} $$

The iteration domain is defined by the destination tile valid region.

## Semantics

`TADD` reads corresponding lanes from `src0` and `src1`, adds them, and writes the result into `dst` for all `(i, j)` covered by `dst.GetValidRow()` × `dst.GetValidCol()`.

Backend implementations require the source valid regions to match the destination valid region. In other words, `src0`, `src1`, and `dst` must have the same runtime valid shape for a legal `TADD`.

## Syntax

### Assembly Form (PTO-AS)

```text
%dst = tadd %src0, %src1 : !pto.tile<...>
```

### AS Level 1 — SSA Form

```mlir
%dst = pto.tadd %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 — DPS Form

```mlir
pto.tadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>)
          outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TADD(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events);
```

## Inputs

| Operand | Role | Description |
|---------|------|-------------|
| `%src0` | Left tile | First source tile, read lane-wise over the destination valid region |
| `%src1` | Right tile | Second source tile, read lane-wise over the destination valid region |
| `WaitEvents...` | Optional synchronisation | `RecordEvent` tokens waited on before issuing the operation |

## Output

| Result | Type | Description |
|--------|------|-------------|
| `%dst` | `!pto.tile<...>` | Destination tile containing lane-wise sums over its valid region |

## Side Effects

None beyond writing the destination tile.

## Constraints

!!! warning "Constraints"
    - **Type match**: `dst`, `src0`, and `src1` must have the same element type.
    - **Valid-shape match**: `src0.GetValidRow()/GetValidCol()` and `src1.GetValidRow()/GetValidCol()` must equal `dst.GetValidRow()/GetValidCol()`.
    - **Layout**:
        - A2/A3 requires RowMajor operands.
        - A5 requires RowMajor operands.
        - CPU simulator supports both RowMajor and non-RowMajor layouts implemented by tile offset mapping.
    - **Tile kind**: legality still depends on the chosen tile kinds and backend support.

## Exceptions

!!! danger "Exceptions"
    - Type mismatches are rejected.
    - Unsupported layouts or data types are rejected by the backend.
    - Runtime valid-shape mismatches trigger backend assertions on supported checking paths.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    | Element type | CPU Simulator | A2/A3 | A5 |
    |---|---|---|---|
    | `f32` | Supported | Supported | Supported |
    | `f16` | Supported | Supported | Supported |
    | `bf16` | Supported | No | Supported |
    | `i32` | Supported | Supported | Supported |
    | `i16` | Supported | Supported | Supported |
    | `u16` | Supported | No | Supported |
    | `i8` | Supported | No | Supported |
    | `u8` | Supported | No | Supported |
    | `u32` | Supported | No | Supported |

    Notes:

    - A2/A3 accepts `int32`, `int16`, `half`/`float16_t`, and `float`/`float32_t` in the current implementation.
    - A5 accepts `int32`, `uint32`, `float`, `int16`, `uint16`, `half`, `bfloat16_t`, `uint8`, and `int8` in the current implementation.
    - This instruction page only lists data types evidenced by the current backend checks.

## Performance

### A2/A3 Throughput Model

The A2/A3 cost model uses the binary-vector-op timing constants in `include/pto/costmodel/pto_isa_costmodel.hpp`:

| Metric | Value | Constant |
|--------|-------|----------|
| Startup latency | 14 cycles | `A2A3_STARTUP_BINARY` |
| Completion latency | 19 cycles for FP, 17 cycles for INT | `A2A3_COMPL_FP_BINOP` / `A2A3_COMPL_INT_BINOP` |
| Per-repeat throughput | 2 cycles | `A2A3_RPT_2` |
| Pipeline interval | 18 cycles | `A2A3_INTERVAL` |

The exact total depends on the backend-selected binary instruction path and repeat geometry.

### Implementation Notes

- A2/A3 lowers `TADD` to `vadd` through the shared binary-op helper in `include/pto/npu/a2a3/TBinOp.hpp`.
- A5 lowers `TADD` to register-tensor `vadd` through the shared binary-op helper in `include/pto/npu/a5/TBinOp.hpp`.
- CPU simulator performs direct element-wise addition over the destination valid region.

## Examples

### C++ — Auto Mode

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void add_tiles(Tile<Vec, float, 16, 16>& dst,
               Tile<Vec, float, 16, 16>& src0,
               Tile<Vec, float, 16, 16>& src1) {
    TADD(dst, src0, src1);
}
```

### C++ — Manual Mode

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void add_tiles_manual(Tile<Vec, float, 16, 16>& dst,
                      Tile<Vec, float, 16, 16>& src0,
                      Tile<Vec, float, 16, 16>& src1,
                      auto& ga, auto& gb, auto& gc) {
    TASSIGN(src0, 0x1000);
    TASSIGN(src1, 0x2000);
    TASSIGN(dst, 0x3000);
    RecordEvent e0 = TLOAD(src0, ga);
    RecordEvent e1 = TLOAD(src1, gb);
    TADD(dst, src0, src1, e0, e1);
    TSTORE(gc, dst);
}
```

### MLIR — SSA Form

```mlir
%result = pto.tadd %src0, %src1 : (!pto.tile<f32, 16, 16>, !pto.tile<f32, 16, 16>) -> !pto.tile<f32, 16, 16>
```

### MLIR — DPS Form

```mlir
pto.tadd ins(%src0, %src1 : !pto.tile_buf<f32, 16, 16>, !pto.tile_buf<f32, 16, 16>)
          outs(%result : !pto.tile_buf<f32, 16, 16>)
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Elementwise Tile Tile](../../elementwise-tile-tile.md)
- Previous op in instruction set: (none)
- Next op in instruction set: [pto.tabs](./tabs.md)
- Instruction set: [Tile Instructions](../../../instruction-families/tile-families.md)
- Type system: [Type System](../../../state-and-types/type-system.md)
- Valid regions: [Tiles and Valid Regions](../../../programming-model/tiles-and-valid-regions.md)
