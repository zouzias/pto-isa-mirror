# TPUT

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-26T03:14:17.026Z pushedAt=2026-08-29T09:05:18.406Z -->

## Introduction

Remote write operation: writes local data to the memory of a remote NPU. The data is transferred through a UB tile as an intermediate staging buffer.

When the GlobalTensor exceeds the UB tile capacity, TPUT automatically performs **2D sliding** — tiling along rows (DIM_3) and columns (DIM_4) to fit the tile, and traverses all outer dimensions (DIM_0, DIM_1, DIM_2).

## Mathematical Semantics

For each element `(i, j)` in the valid region:

$$\mathrm{dst}^{\mathrm{remote}}_{i,j} = \mathrm{src}^{\mathrm{local}}_{i,j}$$

Data flow: `srcGlobalData (local GM)` → `stagingTileData (UB)` → `dstGlobalData (remote GM)`

## Assembly Syntax

Synchronous form:

```text
tput %dst_remote, %src_local : (!pto.memref<...>, !pto.memref<...>)
```

During degradation, UB staging tiles are introduced for the GM→UB→GM data path. C++ built-in APIs require explicitly passing the `stagingTileData` (or `pingTile`/`pongTile`) operand.

## C++ Built-in APIs

Declared in `include/pto/comm/pto_comm_inst.hpp`

### Single Tile (Automatic Tiling)

```cpp
template <AtomicType atomicType = AtomicType::AtomicNone,
          typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TPUT(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                          TileData &stagingTileData, WaitEvents&... events);
```

### Ping-Pong Double Buffering

Two temporary tiles are used to overlap the TLOAD and TSTORE of adjacent blocks, hiding the DMA transfer latency.

```cpp
template <AtomicType atomicType = AtomicType::AtomicNone,
          typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TPUT(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                          TileData &pingTile, TileData &pongTile, WaitEvents&... events);
```

### Runtime Atomic Type

```cpp
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TPUT(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                          TileData &stagingTileData, AtomicType atomicType, WaitEvents&... events);
```

## Constraints

- **Type constraints**:
    - `GlobalSrcData::RawDType` must be equal to `GlobalDstData::RawDType`.
    - `TileData::DType` must be equal to `GlobalSrcData::RawDType`.
    - `GlobalSrcData::layout` must be equal to `GlobalDstData::layout`.
- **Memory constraints**:
    - `dstGlobalData` must point to a remote address (the destination NPU).
    - `srcGlobalData` must point to local memory (the current NPU).
    - `stagingTileData` / `pingTile` / `pongTile` must be pre-allocated in the Unified Buffer.
- **Valid region**:
    - The transfer size is determined by the shape of the `GlobalTensor` (automatically tiled to fit the tile).
- **Atomic operations**:
    - `atomicType` supports `AtomicNone` and `AtomicAdd`.
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
void example_tput(__gm__ T* local_data, __gm__ T* remote_addr) {
    using TileT   = Tile<TileType::Vec, T, 16, 16>;
    using GShape  = Shape<1, 1, 1, 16, 16>;
    using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
    using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

    GTensor srcG(local_data);
    GTensor dstG(remote_addr);
    TileT stagingTile;
    TASSIGN(stagingTile, 0);

    // Basic remote write.
    comm::TPUT(dstG, srcG, stagingTile);

    // Remote write with atomic addition.
    comm::TPUT<AtomicType::AtomicAdd>(dstG, srcG, stagingTile);
}
```

### Ping-Pong Double Buffering

```cpp
constexpr size_t tileUBBytes = ((64 * 64 * sizeof(float) + 1023) / 1024) * 1024;
TileT pingTile(64, 64);
TileT pongTile(64, 64);
TASSIGN(pingTile, 0);
TASSIGN(pongTile, tileUBBytes);  // Non-overlapping UB region.

// Overlap TLOAD[i+1] with TSTORE[i] to improve pipeline utilization.
comm::TPUT(dstG, srcG, pingTile, pongTile);
```

### Runtime Atomic Type

```cpp
// Select the atomic type at runtime instead of using a compile-time template parameter.
comm::TPUT(dstG, srcG, stagingTile, AtomicType::AtomicAdd);
```
