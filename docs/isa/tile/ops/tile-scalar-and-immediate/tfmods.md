# pto.tfmods

`pto.tfmods` is part of the [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md) instruction set.

## Summary

`TFMODS` computes lane-wise floating remainder in `fmod` style between a source tile and a scalar value.

## Semantics

For each active lane, `TFMODS` computes a backend-defined `fmod(src, scalar)`-style result.

The exact algorithm differs by backend:

- A2/A3 uses helper logic equivalent to `src - trunc(src / scalar) * scalar` for supported float paths.
- A5 uses vector helper logic and can enable a high-precision path for selected floating-point cases.
- CPU simulator uses the scalar-op framework.

## Syntax

### Assembly Form (PTO-AS)

```text
%dst = tfmods %src, %scalar : !pto.tile<...>, dtype
```

### AS Level 1 (SSA)

```mlir
%dst = pto.tfmods %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```mlir
pto.tfmods ins(%src, %scalar : !pto.tile_buf<...>, dtype)
          outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TFMODS(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar,
                            WaitEvents &... events);
```

## Constraints

!!! warning "Constraints"
    - `dst` and `src` must use the same element type.
    - Scalar zero is generally invalid / backend-defined.
    - The documented NPU backends require matching source/destination valid shapes.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    | Property | CPU Simulator | A2/A3 | A5 |
    |---|---|---|---|
    | Supported dtypes | follows CPU scalar-op framework | `float` / `float32_t` only | 2-byte or 4-byte floating-capable paths via verified file |
    | RowMajor requirement | not identical to NPU contract | Required | handled through binary-instr path |
    | High precision mode | not documented | No verified special mode | available for selected float path |

    Notes:

    - A2/A3 explicitly supports float-only verified paths in the inspected file.
    - A5 verified helper accepts 2-byte and 4-byte data-width paths and includes specialized handling for `float` and `half`.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
    using TileT = Tile<TileType::Vec, float, 16, 16>;
    TileT x, out;
    TFMODS(out, x, 3.0f);
}
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md)
- Previous op in instruction set: [pto.tmuls](./tmuls.md)
- Next op in instruction set: [pto.trems](./trems.md)
