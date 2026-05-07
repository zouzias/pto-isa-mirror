# pto.tsubc

`pto.tsubc` is part of the [Elementwise Tile Tile](../../elementwise-tile-tile.md) instruction set.

## Summary

Elementwise ternary op: `src0 - src1 + src2`.

## Mechanism

Elementwise ternary op: `src0 - src1 + src2`.

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} - \mathrm{src1}_{i,j} + \mathrm{src2}_{i,j} $$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = tsubc %src0, %src1, %src2 : !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tsubc %src0, %src1, %src2 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tsubc ins(%src0, %src1, %src2 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TSUBC(TileData &dst, TileData &src0, TileData &src1, TileData &src2, WaitEvents &... events);
```

## Inputs

| Operand | Role | Description |
|---------|------|-------------|
| `%src0` | First source tile | First source tile; read at `(i, j)` for each `(i, j)` in `dst` valid region |
| `%src1` | Second source tile | Second source tile; read at `(i, j)` for each `(i, j)` in `dst` valid region |
| `%src2` | Third source tile | Third source tile; read at `(i, j)` for each `(i, j)` in `dst` valid region |
| `WaitEvents...` | Optional synchronisation | `RecordEvent` tokens to wait on before issuing the operation |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%dst` | `!pto.tile<...>` | Destination tile; all `(i, j)` in its valid region contain `src0[i,j] - src1[i,j] + src2[i,j]` after the operation |

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

!!! warning "Constraints"
    - The op iterates over `dst.GetValidRow()` / `dst.GetValidCol()`.

## Exceptions

!!! danger "Exceptions"
    - Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
    - Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - `pto.tsubc` preserves PTO-visible semantics across CPU simulation, A2/A3-class targets, and A5-class targets, but concrete support subsets may differ by profile.

    - Portable code must rely only on the documented type, layout, shape, and mode combinations that the selected target profile guarantees.

## Performance

### A2/A3 Throughput

`TSUBC` compiles to CCE vector instructions via the `TBinOp.hpp` performance model. The throughput is identical to `TADD` (binary arithmetic):

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
  TileT a, b, c, out;
  TSUBC(out, a, b, c);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tsubc %src0, %src1, %src2 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tsubc %src0, %src1, %src2 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tsubc %src0, %src1, %src2 : !pto.tile<...>
# AS Level 2 (DPS)
pto.tsubc ins(%src0, %src1, %src2 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Elementwise Tile Tile](../../elementwise-tile-tile.md)
- Previous op in instruction set: [pto.taddc](./taddc.md)
- Next op in instruction set: [pto.tcvt](./tcvt.md)
