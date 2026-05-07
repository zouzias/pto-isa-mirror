# pto.tsqrt

`pto.tsqrt` is part of the [Elementwise Tile Tile](../../elementwise-tile-tile.md) instruction set.

## Summary

Elementwise square root.

## Mechanism

Elementwise square root.

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \sqrt{\mathrm{src}_{i,j}} $$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = tsqrt %src : !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tsqrt %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tsqrt ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TSQRT(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

## Inputs

| Operand | Role | Description |
|---------|------|-------------|
| `%src` | Source tile | Source tile; read at `(i, j)` for each `(i, j)` in `dst` valid region |
| `%dst` | Destination tile | Destination tile receiving the result |
| `WaitEvents...` | Optional synchronisation | `RecordEvent` tokens to wait on before issuing the operation |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%dst` | `!pto.tile<...>` | Destination tile; all `(i, j)` in its valid region contain `sqrt(src[i,j])` after the operation |

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

!!! warning "Constraints"
    - **Valid region**:
        - The op uses `dst.GetValidRow()` / `dst.GetValidCol()` as the iteration domain.

    - **Domain / NaN**:
        - Behavior is target-defined (e.g., for negative inputs).

## Exceptions

!!! danger "Exceptions"
    - Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
    - Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - **Implementation checks (NPU)**:
        - `TileData::DType` must be one of: `float` or `half`;
        - Tile location must be vector (`TileData::Loc == TileType::Vec`);
        - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`;
        - Runtime: `src.GetValidRow() == dst.GetValidRow()` and `src.GetValidCol() == dst.GetValidCol()`;
        - Tile layout must be row-major (`TileData::isRowMajor`).

## Performance

### A2/A3 Throughput

`TSQRT` compiles to CCE vector instructions via the `TUnaryOp.hpp` performance model:

| Metric | Value |
|--------|-------|
| Startup latency | 13 |
| Completion latency | 26 (FP transcendental) |
| Per-repeat throughput | 1 |
| Pipeline interval | 18 |

---

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TSQRT(dst, src);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TSQRT(dst, src);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tsqrt %src : !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tsqrt %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tsqrt %src : !pto.tile<...>
# AS Level 2 (DPS)
pto.tsqrt ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Elementwise Tile Tile](../../elementwise-tile-tile.md)
- Previous op in instruction set: [pto.trsqrt](./trsqrt.md)
- Next op in instruction set: [pto.texp](./texp.md)
