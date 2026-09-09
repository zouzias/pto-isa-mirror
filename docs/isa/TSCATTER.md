# TSCATTER

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-26T05:09:38.040Z pushedAt=2026-08-29T09:05:18.468Z -->

## Instruction Diagram

![TSCATTER tile operation](../figures/isa/TSCATTER.svg)

## Introduction

TSCATTER provides two operation modes:

1. **Index-based scatter**: Uses element-wise row indices to scatter the rows of the source tile into the destination tile.
2. **Mask scatter**: Scatters source elements to destination positions according to a mask pattern, interleaving zero values between elements. It supports two modes: row-wise scatter (`SCATTER_ROW`) and column-wise scatter (`SCATTER_COL`).

## Mathematical Semantics

### Index-based Scatter

For each source element `(i, j)`, writes:

$$ \mathrm{dst}_{\mathrm{idx}_{i,j},\ j} = \mathrm{src}_{i,j} $$

If multiple elements map to the same destination position, the final value is implementation-defined (in the current implementation, the last writer wins).

### Mask Scatter

For mask mode `P`, the source elements are scattered and interleaved with zero values. The scatter direction is controlled by `ScatterAxis`:

#### SCATTER_ROW (Default)

Scatters along the column direction and expands the column dimension:

$$ \mathrm{dst}_{i, P \cdot j + \mathrm{pos}_P} = \mathrm{src}_{i,j} $$

$$ \mathrm{dst}_{i, P \cdot j + \mathrm{zeros}_P} = 0 $$

Where:

- `DstTileData::ValidCol` = `SrcTileData::ValidCol` × expansion multiple
- `DstTileData::ValidRow` = `SrcTileData::ValidRow`

#### SCATTER_COL

Scatters along the row direction and expands the row dimension:

$$ \mathrm{dst}_{P \cdot i + \mathrm{pos}_P, j} = \mathrm{src}_{i,j} $$

$$ \mathrm{dst}_{P \cdot i + \mathrm{zeros}_P, j} = 0 $$

Where:

- `DstTileData::ValidRow` = `SrcTileData::ValidRow` × expansion multiple
- `DstTileData::ValidCol` = `SrcTileData::ValidCol`

#### Expansion Multiple

- `P1010` or `P0101`: expansion multiple = 2
- `P0001`, `P0010`, `P0100`, `P1000`: expansion multiple = 4
- `P1111`: expansion multiple = 1 (equivalent to `TMOV`)

## Assembly Syntax

Synchronous form:

```text
%dst = tscatter %src, %idx : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tscatter %src, %idx : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

### Index-based Scatter

```cpp
template <typename TileDataD, typename TileDataS, typename TileDataI, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(TileDataD& dst, TileDataS& src, TileDataI& indexes, WaitEvents&... events);
```

### Mask Scatter

```cpp
template <MaskPattern maskPattern = MaskPattern::P1111, auto ScatterType = ScatterAxis::SCATTER_ROW,
          typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(DstTileData& dst, SrcTileData& src, WaitEvents&... events);
```

### MaskPattern Enum

Defined in `include/pto/common/type.hpp`:

| Value | Mode | Description | Expansion Multiple |
|---|------|------|---------|
| `P0101` | 01010101... | Takes the first of every two elements. | ×2 |
| `P1010` | 10101010... | Takes the second of every two elements. | ×2 |
| `P0001` | 00010001... | Takes the first of every four elements. | ×4 |
| `P0010` | 00100010... | Takes the second of every four elements. | ×4 |
| `P0100` | 01000100... | Takes the third of every four elements. | ×4 |
| `P1000` | 10001000... | Takes the fourth of every four elements. | ×4 |
| `P1111` | 11111111... | Takes all elements (equivalent to TMOV). | ×1 |

### ScatterAxis Enum

Defined in `include/pto/common/type.hpp`:

| Value | Description |
|---|------|
| `SCATTER_ROW` | Scatters along the column direction, and expands the column dimension (default). |
| `SCATTER_COL` | Scatters along the row direction, and expands the row dimension. |

## Constraints

### Index-based Scatter

- **Implementation check (Atlas A2/A3 training products/Atlas A2/A3 inference products)**:
    - `TileDataD::Loc`, `TileDataS::Loc`, and `TileDataI::Loc` must be `TileType::Vec`.
    - `TileDataD::DType` and `TileDataS::DType` must be one of the following: `int32_t`, `int16_t`, `int8_t`, `half`, `float16_t`, `float32_t`, `uint32_t`, `uint16_t`, `uint8_t`, `bfloat16_t`.
    - `TileDataI::DType` must be one of the following: `int16_t`, `int32_t`, `uint16_t`, or `uint32_t`.
    - No boundary check is performed on the `indexes` values.
    - Statically valid boundary: `TileDataD::ValidRow <= TileDataD::Rows`, `TileDataD::ValidCol <= TileDataD::Cols`, `TileDataS::ValidRow <= TileDataS::Rows`, `TileDataS::ValidCol <= TileDataS::Cols`, `TileDataI::ValidRow <= TileDataI::Rows`, `TileDataI::ValidCol <= TileDataI::Cols`.
    - `TileDataD::DType` and `TileDataS::DType` must be the same.
    - When the size of `TileDataD::DType` is 4 bytes, the size of `TileDataI::DType` must be 4 bytes.
    - When the size of `TileDataD::DType` is 2 bytes, the size of `TileDataI::DType` must be 2 bytes.
    - When the size of `TileDataD::DType` is 1 byte, the size of `TileDataI::DType` must be 2 bytes.
- **Implementation check (Ascend 950PR/Ascend 950DT)**:
    - `TileDataD::Loc`, `TileDataS::Loc`, and `TileDataI::Loc` must be `TileType::Vec`.
    - `TileDataD::DType` and `TileDataS::DType` must be one of the following: `int32_t`, `int16_t`, `int8_t`, `half`, `float16_t`, `float32_t`, `uint32_t`, `uint16_t`, `uint8_t`, `bfloat16_t`.
    - `TileDataI::DType` must be one of the following: `int16_t`, `int32_t`, `uint16_t`, or `uint32_t`.
    - No boundary check is performed on the `indexes` values.
    - Statical valid boundary: `TileDataD::ValidRow <= TileDataD::Rows`, `TileDataD::ValidCol <= TileDataD::Cols`, `TileDataS::ValidRow <= TileDataS::Rows`, `TileDataS::ValidCol <= TileDataS::Cols`, `TileDataI::ValidRow <= TileDataI::Rows`, `TileDataI::ValidCol <= TileDataI::Cols`.
    - `TileDataD::DType` and `TileDataS::DType` must be the same.
    - When the size of `TileDataD::DType` is 4 bytes, the size of `TileDataI::DType` must be 4 bytes.
    - When the size of `TileDataD::DType` is 2 bytes, the size of `TileDataI::DType` must be 2 bytes.
    - When the size of `TileDataD::DType` is 1 byte, the size of `TileDataI::DType` must be 2 bytes.

### Mask Scatter

- **Implementation check (Atlas A2/A3 training products/Atlas A2/A3 inference products)**:
    - `DstTileData::Loc` and `SrcTileData::Loc` must be `TileType::Vec`.
    - `DstTileData::DType` and `SrcTileData::DType` must be one of the following: `int32_t`, `int16_t`, `int8_t`, `half`, `float16_t`, `float32_t`, `uint32_t`, `uint16_t`, `uint8_t`, `bfloat16_t`.
    - `DstTileData::DType` and `SrcTileData::DType` must be the same.
    - `maskPattern` must be within the range from `P0101` to `P1111`.
    - Statical valid boundary: `DstTileData::ValidCol <= DstTileData::Cols`, `SrcTileData::ValidCol <= SrcTileData::Cols`, `DstTileData::ValidRow <= DstTileData::Rows`, `SrcTileData::ValidRow <= SrcTileData::Rows`.
    - The `P1111` mode is equivalent to `TMOV`: it requires `validRow` and `validCol` to match respectively, and is implemented internally through `TMOV_IMPL`.
- **Implementation check (Ascend 950PR/Ascend 950DT)**:
    - `DstTileData::Loc` and `SrcTileData::Loc` must be `TileType::Vec`.
    - `DstTileData::DType` and `SrcTileData::DType` must be one of the following: `int32_t`, `int16_t`, `int8_t`, `half`, `float16_t`, `float32_t`, `uint32_t`, `uint16_t`, `uint8_t`, `bfloat16_t`.
    - `DstTileData::DType` and `SrcTileData::DType` must be the same.
    - `maskPattern` must be within the range from `P0101` to `P1111`.
    - Statical valid boundary: `DstTileData::ValidRow <= DstTileData::Rows`, `DstTileData::ValidCol <= DstTileData::Cols`, `SrcTileData::ValidRow <= SrcTileData::Rows`, `SrcTileData::ValidCol <= SrcTileData::Cols`.
    - Runtime assertions for `SCATTER_ROW` mode:
        - `SrcTileData::ValidRow` must be equal to `DstTileData::ValidRow`.
        - `SrcTileData::ValidCol` must be equal to `DstTileData::ValidCol × expansion multiple`, where the expansion multiple depends on the mask mode (1 for P1111, 2 for P1010/P0101, 4 for P0001/P0010/P0100/P1000).
    - Runtime assertions for `SCATTER_COL` mode:
        - `SrcTileData::ValidCol` must be equal to `DstTileData::ValidCol`.
        - `SrcTileData::ValidRow` must be equal to `DstTileData::ValidRow × expansion multiple`, where the expansion multiple depends on the mask mode (1 for P1111, 2 for P1010/P0101, 4 for P0001/P0010/P0100/P1000).

## Important Notes

> **Warning**: Before performing the scatter operation, the destination tile buffer is **completely initialized to 0** (the entire tile size `Rows × Cols`), **not limited by `ValidRow` and `ValidCol`**. This means:
>
> - The entire UB buffer allocated to `dstTile` is written with zero values.
> - Elements outside the `ValidRow`/`ValidCol` range are also zero after the operation.
> - Ensure that the UB buffer of the destination tile does not overlap with other active data.

## Examples

### Index Scatter (Automatic)

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileT src, dst;
  IdxT idx;
  TSCATTER(dst, src, idx);
}
```

### Index Scatter (Manual)

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileT src, dst;
  IdxT idx;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(idx, 0x3000);
  TSCATTER(dst, src, idx);
}
```

### Mask Scatter (Automatic)

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_mask_auto() {
  // P1010: destination size = source size × 2.
  using SrcTileT = Tile<TileType::Vec, half, 16, 64>;
  using DstTileT = Tile<TileType::Vec, half, 16, 128>;
  SrcTileT src;
  DstTileT dst;
  TSCATTER<MaskPattern::P1010>(dst, src);
}

void example_mask_p1000() {
  // P1000: destination size = source size × 4.
  using SrcTileT = Tile<TileType::Vec, float, 16, 64>;
  using DstTileT = Tile<TileType::Vec, float, 16, 256>;
  SrcTileT src;
  DstTileT dst;
  TSCATTER<MaskPattern::P1000>(dst, src);
}

void example_mask_scatter_col() {
  // SCATTER_COL: scatter along the row direction, expand the row dimension.
  // P1010: destination row count = source row count × 2.
  using SrcTileT = Tile<TileType::Vec, half, 64, 16>;
  using DstTileT = Tile<TileType::Vec, half, 128, 16>;
  SrcTileT src;
  DstTileT dst;
  TSCATTER<MaskPattern::P1010, ScatterAxis::SCATTER_COL>(dst, src);
}
```

### Mask Scatter (Manual)

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_mask_manual() {
  using SrcTileT = Tile<TileType::Vec, half, 16, 64>;
  using DstTileT = Tile<TileType::Vec, half, 16, 128>;
  SrcTileT src;
  DstTileT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TSCATTER<MaskPattern::P1010>(dst, src);
}

void example_mask_manual_scatter_col() {
  // SCATTER_COL manual binding mode
  using SrcTileT = Tile<TileType::Vec, half, 64, 16>;
  using DstTileT = Tile<TileType::Vec, half, 128, 16>;
  SrcTileT src;
  DstTileT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TSCATTER<MaskPattern::P1010, ScatterAxis::SCATTER_COL>(dst, src);
}
```

## ASM Examples

### Automatic Mode

```text
# Automatic mode: the compiler/runtime handles resource placement and scheduling.
%dst = pto.tscatter %src, %idx : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: explicitly bind resources first, then issue the instruction.
# Optional (when the instruction contains tile operands):
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tscatter %src, %idx : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tscatter %src, %idx : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
