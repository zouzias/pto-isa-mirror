# TCOLEXPANDMUL

## Tile Operation Diagram

![TCOLEXPANDMUL tile operation](../figures/isa/TCOLEXPANDMUL.svg)

## Introduction

Column-wise broadcast multiply: multiply each element of `src0` by a per-column scalar vector `src1`.

## Math Interpretation

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Let `s_j` be the per-column scalar taken from `src1` (one value per column).

For `0 <= i < R` and `0 <= j < C`:

$$
\mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \cdot s_j
$$

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDMUL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## Constraints

- `TileDataDst::DType`, `TileDataSrc1::DType` must be one of: `half`, `float`, `int16`, `int32` for A2, A3 and A5, `uint16`, `uint32` for A5.
- Tile shape/layout constraint (compile-time): `TileDataDst::isRowMajor`.
- `src1` is expected to provide **one scalar per column** (i.e., its valid shape must cover `C` values).
- Exact layout/fractal constraints are target-specific; see backend headers under `include/pto/npu/*/TColExpand*.hpp`.

## Examples

See related examples in `docs/isa/` and `docs/coding/tutorials/`.
