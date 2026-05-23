# pto.trem

`pto.trem` is part of the [Elementwise Tile Tile](../../elementwise-tile-tile.md) instruction set.

## Summary

Elementwise remainder of two tiles.

## Mechanism

Elementwise remainder of two tiles. The result has the same sign as the divider.

For each element `(i, j)` in the valid region:

$$\mathrm{dst}_{i,j} = \mathrm{remainder}(\mathrm{src0}_{i,j}, \mathrm{src1}_{i,j}) = \mathrm{src0}_{i,j} - \mathrm{floor}(\frac{\mathrm{src0}_{i,j}}{\mathrm{src1}_{i,j}}) \times \mathrm{src1}_{i,j}$$

The result sign is corrected to match the sign of the divider (`src1`).

**Note**: This differs from `TFMOD` where the result sign follows the dividend (`src0`).

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = trem %src0, %src1 : !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.trem %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.trem ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <auto PrecisionType = RemAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0,
          typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TREM(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp, WaitEvents &... events);
```

## Inputs

| Operand | Role | Description |
|---------|------|-------------|
| `%src0` | Left tile | First source tile; read at `(i, j)` for each `(i, j)` in `dst` valid region |
| `%src1` | Right tile | Second source tile (divisor); read at `(i, j)` for each `(i, j)` in `dst` valid region |
| `%tmp` | Temporary tile | Required temporary working tile for computation |
| `WaitEvents...` | Optional synchronisation | `RecordEvent` tokens to wait on before issuing the operation |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%dst` | `!pto.tile<...>` | Destination tile; all `(i, j)` in its valid region contain `src0[i,j] % src1[i,j]` after the operation |

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

!!! warning "Constraints"
    - **Division by Zero**:
        - Behavior is target-defined; the CPU simulator asserts in debug builds.

    - **Valid Region**:
        - The op uses `dst.GetValidRow()` / `dst.GetValidCol()` as the iteration domain.

## Exceptions

!!! danger "Exceptions"
    - Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
    - Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - **Implementation Checks (A2A3)**:
        - `TileData::DType` must be one of: `float`, `float32_t`, `int32_t`.
        - Tile layout must be row-major (`TileData::isRowMajor`).
        - Tile location must be vector (`TileData::Loc == TileType::Vec`).
        - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.
        - `tmp` tile must have at least 2 rows and `validCols` columns (row 0 for intermediate results, row 1 for comparison mask).

    - **Implementation Checks (A5)**:
        - `TileData::DType` must be one of: `half`, `float`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`.
        - Tile layout must be row-major (`TileData::isRowMajor`).
        - Tile location must be vector (`TileData::Loc == TileType::Vec`).
        - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.
        - Note: `tmp` parameter is accepted but not used on A5.

    - **High Precision Algorithm**:
        - Only available on A5 for `float` type; `PrecisionType` option is ignored on A2A3.

## Performance

### A2/A3 Throughput

`TREM` compiles to CCE vector instructions via the `TBinOp.hpp` performance model. The throughput is identical to `TADD` (binary arithmetic):

| Metric | Value (FP) | Value (INT) |
|--------|-------------|-------------|
| Startup latency | 14 | 14 |
| Completion latency | 19 | 17 |
| Per-repeat throughput | 2 | 2 |
| Pipeline interval | 18 | 18 |

---

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using TmpT = Tile<TileType::Vec, float, 2, 16>;
  TileT dst, src0, src1;
  TmpT tmp;
  TREM(dst, src0, src1, tmp);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.trem %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.trem %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = trem %src0, %src1 : !pto.tile<...>
# AS Level 2 (DPS)
pto.trem ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Elementwise Tile Tile](../../elementwise-tile-tile.md)
- Previous op in instruction set: [pto.tneg](./tneg.md)
- Next op in instruction set: [pto.tfmod](./tfmod.md)
