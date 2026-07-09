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

## Examples

See related examples in `docs/isa/` and `docs/coding/tutorials/`.
