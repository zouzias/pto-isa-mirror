# pto.trems

`pto.trems` is part of the [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md) instruction set.

## Summary

`TREMS` computes lane-wise remainder between a source tile and a scalar value.

## Semantics

For each active lane, `TREMS` computes a backend-defined remainder operation using the source lane and the scalar divisor.

The exact algorithm differs by backend:

- A2/A3 uses custom helper logic with temporary storage.
- A5 uses backend vector remainder/division helper paths, with an optional high-precision mode for selected floating-point cases.
- CPU simulator uses the scalar-op framework and checks for zero divisors.

## Syntax

### Assembly Form (PTO-AS)

```text
%dst = trems %src, %scalar : !pto.tile<...>, dtype
```

### AS Level 1 (SSA)

```mlir
%dst = pto.trems %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```mlir
pto.trems ins(%src, %scalar : !pto.tile_buf<...>, dtype)
         outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

```cpp
template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TREMS(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar,
                           TileDataTmp &tmp, WaitEvents &... events);
```

## Constraints

!!! warning "Constraints"
    - `dst` and `src` must use the same element type.
    - Scalar zero is illegal on CPU simulator and generally backend-invalid.
    - `tmp` is part of the public API and is required by documented NPU implementations.
    - The documented NPU backends require matching source/destination valid shapes.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    | Property | CPU Simulator | A2/A3 | A5 |
    |---|---|---|---|
    | Supported dtypes | follows CPU scalar-op framework | `float`, `int32_t` | 2-byte or 4-byte dtypes via verified file |
    | RowMajor requirement | not identical to NPU contract | Required | handled through binary-instr path |
    | `tmp` required | API supports it, implementation may ignore it | Required and validated | API requires it; implementation path does not strongly validate use |
    | High precision mode | not documented | No verified special mode | available for selected float path |

    Notes:

    - A2/A3 explicitly validates tmp capacity and uses row-wise helper logic.
    - A5 accepts `sizeof(T) == 2` or `4`; verified implementation covers float/half/integer remainder-style paths via custom helpers.
    - Earlier prose that listed a wider exact A5 dtype table than verified source evidence has been narrowed.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
    using TileT = Tile<TileType::Vec, float, 16, 16>;
    TileT x, out;
    Tile<TileType::Vec, float, 16, 16> tmp;
    TREMS(out, x, 3.0f, tmp);
}
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md)
- Previous op in instruction set: [pto.tfmods](./tfmods.md)
- Next op in instruction set: [pto.tmaxs](./tmaxs.md)
