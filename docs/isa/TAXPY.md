# TAXPY

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-26T03:33:23.616Z pushedAt=2026-08-29T09:05:18.416Z -->

## Introduction

Performs an in-place scaled accumulation (AXPY, $a \cdot x + y$) on a tile: scales `src0` by `scalar` and accumulates the result onto `dst`.

$$ \mathrm{dst}_{i,j} \leftarrow \mathrm{scalar} \cdot \mathrm{src0}_{i,j} + \mathrm{dst}_{i,j} $$

`dst` serves as both the accumulation input ($y$) and the output, and must be initialized before the call; `src0` ($x$) is read-only; `scalar` ($a$) is a scalar.

## Mathematical Semantics

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j}^{\text{new}} = \mathrm{scalar} \cdot \mathrm{src0}_{i,j} + \mathrm{dst}_{i,j}^{\text{old}} $$

- `dst`: read-modify-write (RMW). Reads the old value as the accumulation base $y$, and writes back $\mathrm{scalar} \cdot x + y$.
- `src0`: read-only. Participates in the operation element by element ($x$).
- `scalar`: scalar scaling factor ($a$), of type `TileDataSrc::DType`.

> Unless otherwise specified, the semantics are defined within the valid region, and target-dependent behavior is marked as implementation-defined.

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`.
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TAXPY(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar,
                           WaitEvents &...events);
```

| Parameter | Direction | Meaning |
|------|------|------|
| `dst` | Input/Output | Accumulation base and result tile ($y$), read-modify-write, `Vec` |
| `src0` | Input | Scaling source tile ($x$), read-only, `Vec`, with the same valid shape as `dst` |
| `scalar` | Input | Scalar scaling coefficient ($a$), of type `TileDataSrc::DType` |
| `events...` | Input | Wait events (`WaitEvents`), implicit `TSYNC` before the instruction |

## Tile Size and Data Type

For a valid shape of $M \times N$:

| Tile | Dtype | Valid Shape | TileType | Description |
|------|-------|---------|----------|------|
| `dst` | `half` or `float` | $M \times N$ | `Vec` (UB) | Accumulation base + result (RMW) |
| `src0` | `half` or `float` | $M \times N$ | `Vec` (UB) | Scaling source, element-wise |

> The valid rows and valid columns of `dst` and `src0` must be exactly the same.

## Supported Input Dtypes

| `dst` Dtype | `src0` Dtype | `scalar` Dtype | Description |
|-------------|--------------|----------------|------|
| `half` | `half` | `half` | Same-type path, directly `vaxpy` |
| `float` | `float` | `float` | Same-type path, directly `vaxpy` |
| `float` | `half` | `half` | Difference path: `src0` is widened to FP32 before accumulation |

> `dst` and `src0` must have the same dtype, or `dst` is `float` and `src0` is `half` (allowing half-to-float widened accumulation). The combination where `dst` is `half` and `src0` is `float` is invalid (blocked by `static_assert` in the implementation).

## Implementation Notes

TAXPY executes on the vector pipe (`PIPE_V`) using the built-in `vaxpy` ($a \cdot x + y$) vector:

1. **Same type (`dst` and `src0` have the same dtype)**: loads `src0` and `dst` repeat by repeat, executes `vaxpy(dst, src0, scalar)`, and writes back to `dst`; the columns in the tail that are less than one repeat are masked by the predicate mask.
2. **Different type (`dst`=`float`, `src0`=`half`)**: the half data of `src0` is widened to FP32 before participating in the accumulation (on Ascend 950PR/Ascend 950DT, it is unpacked via `UNPK_B16` and converted via `vcvt`; on Atlas A2/A3 training products/Atlas A2/A3 inference products, it is natively handled by `vaxpy` as 4-block src / 8-block dst).
3. On Atlas A2/A3 training products/Atlas A2/A3 inference products, the implementation selects between count mode and norm mode based on whether the repeat-stride overflows and on the relationship between the number of columns and the number of rows, to cover any valid shape.

## Constraints

| Constraint | Scope | Reason |
|------|---------|------|
| `dst` and `src0` must be `TileType::Vec` | All targets | Executes on UB (vector pipeline) |
| `dst` and `src0` have the same valid shape ($M \times N$) | All targets | Element-wise one-to-one correspondence |
| `dst` dtype ∈ {`half`, `float`} | All targets | Floating-point word width supported by `vaxpy` |
| `dst`/`src0` have the same dtype, or (`float`,`half`) | All targets | Only half→float widened accumulation is allowed |
| `dst` must be initialized before the call | All targets | `dst` is read as the accumulation base $y$ |

## Examples

```cpp
// dst must be initialized first (as the accumulation base y); result: dst = scalar * src0 + dst
TAXPY(dstTile, srcTile, scalar);
```

For complete ST examples, see `tests/npu/a5/src/st/testcase/taxpy/` (A5), `tests/npu/a2a3/src/st/testcase/taxpy/` (A2/A3), `tests/npu/kirin9030/src/st/testcase/taxpy/` (Kirin9030), and `tests/cpu/st/testcase/taxpy/` (CPU reference implementation).
