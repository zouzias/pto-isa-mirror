# TTRI


## Tile Operation Diagram

![TTRI tile operation](../figures/isa/TTRI.svg)

## Introduction

Generate a (lower/upper) triangular mask tile with ones and zeros. The triangular orientation is controlled by the compile-time template parameter `isUpperOrLower` (0 = lower, 1 = upper).

## Math Interpretation

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Let `d = diagonal`.

Lower-triangular (`isUpperOrLower=0`) conceptually produces:

$$
\mathrm{dst}_{i,j} = \begin{cases}1 & j \le i + d \\\\ 0 & \text{otherwise}\end{cases}
$$

Upper-triangular (`isUpperOrLower=1`) conceptually produces:

$$
\mathrm{dst}_{i,j} = \begin{cases}0 & j < i + d \\\\ 1 & \text{otherwise}\end{cases}
$$

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, int isUpperOrLower, typename... WaitEvents>
PTO_INST RecordEvent TTRI(TileData &dst, int diagonal, WaitEvents &... events);
```

## Constraints

- `isUpperOrLower` must be `0` (lower) or `1` (upper).
- Destination tile must be row-major on some targets (see `include/pto/npu/*/TTri.hpp`).

## Assembly Syntax

### AS Level 1 (SSA)

```text
%dst = pto.ttri %diag : i32 -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.ttri ins(%diag : i32) outs(%dst : !pto.tile_buf<...>)
```
## Examples

See related examples in `docs/isa/` and `docs/coding/tutorials/`.

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.ttri {isUpperOrLower = 0} : i32 -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# pto.tassign %arg0, @tile(0x1000)
%dst = pto.ttri %diag : i32 -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = pto.ttri %diag : i32 -> !pto.tile<...>
# AS Level 2 (DPS)
pto.ttri ins(%diag : i32) outs(%dst : !pto.tile_buf<...>)
```
