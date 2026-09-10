# TBROADCAST

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-21T08:28:36.611Z pushedAt=2026-08-22T02:49:41.506Z -->

## Introduction

Broadcasts data from the current NPU to all ranks in the parallel group. The calling NPU is the root node, and its data is copied to all other NPUs.

Only the root node needs to execute `TBROADCAST`. Non-root nodes only need to ensure that their destination buffers are allocated and writable during the operation. Calling `TBROADCAST` on a non-root node is undefined behavior.

**Large tile support**: When a GlobalTensor exceeds the Unified Buffer (UB) tile capacity in the row and/or column direction, the transfer is automatically tiled through 2D sliding.

## Mathematical Semantics

After the operation completes:

$$ \mathrm{dst}^{(k)}_{i,j} = \mathrm{src}^{(\text{root})}_{i,j} \quad \forall k \in [0, N) $$

Where $N$ is the total number of ranks and `root` is the calling NPU.

## Assembly Syntax

Synchronous form:

```text
tbroadcast %group, %src : (!pto.group<...>, !pto.memref<...>)
```

During degradation, UB staging tiles are introduced for the GM→UB→GM data path. C++ built-in APIs require explicitly passing the `stagingTileData` (or `pingTile`/`pongTile`) operand.

## Template Parameters

- `engine`:
    - `CollEngine::AIV` (default)
    - `CollEngine::CCU` (Ascend 950PR/Ascend 950DT, NPU_ARCH 3510 only)

## C++ Built-in APIs

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// Basic broadcast (single staging tile)
template <CollEngine engine = CollEngine::AIV,
          typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... Args>
PTO_INST RecordEvent TBROADCAST(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                                TileData &stagingTileData, Args&... args);

// Ping-pong broadcast (using two staging tiles for double buffering)
template <CollEngine engine = CollEngine::AIV,
          typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... Args>
PTO_INST RecordEvent TBROADCAST(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                                TileData &pingTile, TileData &pongTile, Args&... args);
```

When `engine == CollEngine::CCU`, the first variable parameter must be a `CcuTriggerContext` that contains the CKE slot virtual address and gate mask. The AIV kernel triggers the CKE gate, and the actual broadcast data path is executed on the CCU engine.

## Constraints

- **Type constraints**:
    - `ParallelGroup::value_type::RawDType` must be equal to `GlobalSrcData::RawDType`.
    - `TileData::DType` must be equal to `GlobalSrcData::RawDType`.
- **Memory constraints**:
    - `srcGlobalData` must point to local memory (the current NPU).
    - `stagingTileData` (or `pingTile` / `pongTile`) must be pre-allocated in the UB.
- **ParallelGroup constraints**:
    - `parallelGroup.tensors[k]` must point to the destination buffer of rank `k` (the remote GM as seen from the root node's perspective).
    - `parallelGroup.GetRootIdx()` identifies the calling NPU as the broadcast root node.
    - All target tensors are assumed to have the same shape and stride.
- **Tiling mode constraints** (when data exceeds a single UB tile):
    - If `TileData` has a static `ValidRow`, `GetShape(DIM_3)` must be divisible by `ValidRow`. To support the case of fewer than one row, use a tile with a `DYNAMIC` ValidRow.
    - If `TileData` has a static `ValidCol`, `GetShape(DIM_4)` must be divisible by `ValidCol`. To support the case of fewer than one column, use a tile with a `DYNAMIC` ValidCol.

> **CCU path**: Unlike the AIV path (where only the root node calls `TBROADCAST`), the CCU path requires all ranks to register and launch the CCU kernel through the host-side `HcclCcuKernelRegister`/`HcclCcuKernelLaunch`. For a complete example, see `tests/npu/a5/comm/st/testcase/tbroadcast_ccu/`.

## Examples

### Basic Broadcast

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void broadcast(__gm__ T* group_addrs[NRANKS], __gm__ T* my_data, int my_rank) {
    // The tile dimensions can differ from the tensor dimensions.
    // The 2D sliding tiling path automatically tiles in both the row and column directions.
    using TileT = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                 BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;

    GTensor tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GTensor(group_addrs[i]);
    }

    comm::ParallelGroup<GTensor> group(tensors, NRANKS, my_rank);
    GTensor srcG(my_data);
    TileT stagingTile(TILE_ROWS, TILE_COLS);

    // The current NPU broadcasts its own data to all other NPUs.
    comm::TBROADCAST(group, srcG, stagingTile);
}
```

### Ping-Pong Broadcast (Double Buffering)

Uses two UB tiles to overlap the TLOAD of the next block with the TSTORE of the current block.

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void broadcast_pingpong(__gm__ T* group_addrs[NRANKS], __gm__ T* my_data, int my_rank) {

    using TileT = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using GPerRank = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                  BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;

    GPerRank tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GPerRank(group_addrs[i]);
    }

    comm::ParallelGroup<GPerRank> group(tensors, NRANKS, my_rank);
    GPerRank srcG(my_data);
    TileT pingTile(TILE_ROWS, TILE_COLS);
    TileT pongTile(TILE_ROWS, TILE_COLS);

    // Ping-pong mode: overlap TLOAD and TSTORE to improve throughput.
    comm::TBROADCAST(group, srcG, pingTile, pongTile);
}
```
