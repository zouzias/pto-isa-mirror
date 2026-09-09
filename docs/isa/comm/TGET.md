# TGET

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-26T03:14:11.386Z pushedAt=2026-08-29T09:05:18.405Z -->

## Introduction

Remote read operation: reads data from the remote NPU to local memory. Data is transferred through a UB tile as an intermediate staging buffer.

When the GlobalTensor exceeds the UB tile capacity, TGET automatically performs **2D sliding** — tiling along rows (DIM_3) and columns (DIM_4) to fit the tile, and traverses all outer dimensions (DIM_0, DIM_1, DIM_2).

## Mathematical Semantics

For each element `(i, j)` in the valid region:

$$\mathrm{dst}^{\mathrm{local}}_{i,j} = \mathrm{src}^{\mathrm{remote}}_{i,j}$$

Data flow: `srcGlobalData (remote GM)` → `stagingTileData (UB)` → `dstGlobalData (local GM)`

## Assembly Syntax

Synchronous form:

```text
tget %dst_local, %src_remote : (!pto.memref<...>, !pto.memref<...>)
```

During degradation, UB staging tiles are introduced for the GM→UB→GM data path. C++ built-in APIs require explicitly passing the `stagingTileData` (or `pingTile`/`pongTile`) operand.

## C++ Built-in APIs

Declared in `include/pto/comm/pto_comm_inst.hpp`

### Single Tile (Automatic Tiling)

```cpp
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGET(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                          TileData &stagingTileData, WaitEvents&... events);
```

### Ping-Pong Double Buffering

Two staging tiles are used to overlap the TGET and TSTORE of adjacent blocks, hiding the DMA transfer latency.

```cpp
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGET(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                          TileData &pingTile, TileData &pongTile, WaitEvents&... events);
```

## Constraints

- **Type constraints**:
    - `GlobalSrcData::RawDType` must be equal to `GlobalDstData::RawDType`.
    - `TileData::DType` must be equal to `GlobalSrcData::RawDType`.
    - `GlobalSrcData::layout` must be equal to `GlobalDstData::layout`.
- **Memory constraints**:
    - `srcGlobalData` must point to a remote address (the source NPU).
    - `dstGlobalData` must point to a local address (the current NPU).
    - `stagingTileData` / `pingTile` / `pongTile` must be pre-allocated in the Unified Buffer.
- **Valid region**:
    - The transfer size is determined by the shape of the `GlobalTensor` (automatically tiled to fit the tile).
- **Ping-pong constraints**:
    - `pingTile` and `pongTile` must have the same type and dimensions.
    - They must be located at non-overlapping UB offsets.

## Examples

### Basic Usage

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_tget(__gm__ T* local_data, __gm__ T* remote_addr) {
    using TileT   = Tile<TileType::Vec, T, 16, 16>;
    using GShape  = Shape<1, 1, 1, 16, 16>;
    using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
    using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

    GTensor srcG(remote_addr);
    GTensor dstG(local_data);
    TileT stagingTile;
    TASSIGN(stagingTile, 0);

    // Basic remote read.
    comm::TGET(dstG, srcG, stagingTile);
}
```

### Ping-Pong Double Buffering

```cpp
constexpr size_t tileUBBytes = ((64 * 64 * sizeof(float) + 1023) / 1024) * 1024;
TileT pingTile(64, 64);
TileT pongTile(64, 64);
TASSIGN(pingTile, 0);
TASSIGN(pongTile, tileUBBytes);  // Non-overlapping UB regions.

// Overlap the TGET operations of adjacent blocks to improve pipeline utilization.
comm::TGET(dstG, srcG, pingTile, pongTile);
```
