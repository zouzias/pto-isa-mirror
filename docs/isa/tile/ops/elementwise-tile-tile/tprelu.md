# pto.tprelu

`pto.tprelu` is part of the [Elementwise Tile Tile](../../elementwise-tile-tile.md) instruction set.

## Summary

Elementwise PReLU (parametric ReLU) with a per-element slope tile.

## Mechanism

Elementwise PReLU (parametric ReLU) with a per-element slope tile.

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = (\mathrm{src0}_{i,j} > 0) ? \mathrm{src0}_{i,j} : (\mathrm{src0}_{i,j} \cdot \mathrm{src1}_{i,j}) $$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = tprelu %src0, %src1 : !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tprelu %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tprelu ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TPRELU(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp, WaitEvents &... events);
```

## Inputs

| Operand | Role | Description |
|---------|------|-------------|
| `%src0` | Left tile | First source tile (value); read at `(i, j)` for each `(i, j)` in `dst` valid region |
| `%src1` | Right tile | Second source tile (slope); read at `(i, j)` for each `(i, j)` in `dst` valid region |
| `%tmp` | Temporary tile | Required temporary working tile for PReLU slope selection (A3) |
| `WaitEvents...` | Optional synchronisation | `RecordEvent` tokens to wait on before issuing the operation |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%dst` | `!pto.tile<...>` | Destination tile; all `(i, j)` in its valid region contain `(src0[i,j] > 0) ? src0[i,j] : (src0[i,j] * src1[i,j])` after the operation |

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

!!! warning "Constraints"
    - The op iterates over `dst.GetValidRow()` / `dst.GetValidCol()`.

    - For A3, 2 source Tile, destination Tile, temporary space must in different memory range without overlapping.

    - For A3, `tmp.GetValidRow()` must be greater than or equal to `dst.GetValidCol() + 1`.

## Performance

### A2/A3 Throughput

`TPRELU` compiles to CCE vector instructions via the `TBinOp.hpp` performance model. The throughput is identical to `TADD` (binary arithmetic):

| Metric | Value (FP) | Value (INT) |
|--------|-------------|-------------|
| Startup latency | 14 | 14 |
| Completion latency | 19 | 17 |
| Per-repeat throughput | 2 | 2 |
| Pipeline interval | 18 | 18 |

---

## Exceptions

!!! danger "Exceptions"
    - Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
    - Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - Temporary space is required by A3 for calculation, while not used by A5.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, slope, out, tmp;
  TPRELU(out, x, slope, tmp);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tprelu %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tprelu %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tprelu %src0, %src1 : !pto.tile<...>
# AS Level 2 (DPS)
pto.tprelu ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Elementwise Tile Tile](../../elementwise-tile-tile.md)
- Previous op in instruction set: [pto.trecip](./trecip.md)
- Next op in instruction set: [pto.taddc](./taddc.md)
