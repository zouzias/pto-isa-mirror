# TGATHER

## Introduction

Gather operation: the calling NPU (root) collects data from all ranks in the parallel group and concatenates the results along **DIM_3** (row dimension) into a local output buffer.

> **Hardware Note**: This instruction may be offloaded to dedicated collective communication hardware.

Only the root needs to execute `TGATHER`. Non-root ranks only need to ensure their source buffers are ready and remain valid for the duration of the operation.

**Large Tile Support**: When the GlobalTensor exceeds the UB tile capacity in rows and/or columns, the transfer is automatically chunked via 2D sliding — the same mechanism used by other PTO-COMM instructions.

## Math Interpretation

Each rank $r$ has source data of shape $(D_0, D_1, D_2, H, W)$. The gather concatenates all $N$ ranks along DIM_3:

$$\mathrm{dst}_{d_0, d_1, d_2,\; r \cdot H + i,\; j} = \mathrm{src}^{(r)}_{d_0, d_1, d_2,\; i,\; j} \quad \forall\, r \in [0, N),\; i \in [0, H),\; j \in [0, W)$$

The destination tensor has shape $(D_0, D_1, D_2, N \times H, W)$.

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Synchronous form:

```text
tgather %group, %dst : (!pto.group<...>, !pto.memref<...>)
```
Lowering introduces UB staging tile(s) for the GM→UB→GM data path; the C++ intrinsic requires explicit `stagingTileData` (or `pingTile` / `pongTile`) operand(s).

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// Basic gather (single staging tile)
template <typename ParallelGroup, typename GlobalDstData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(ParallelGroup &parallelGroup, GlobalDstData &dstGlobalData,
                             TileData &stagingTileData, WaitEvents&... events);

// Ping-pong gather (double buffering with two staging tiles)
template <typename ParallelGroup, typename GlobalDstData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(ParallelGroup &parallelGroup, GlobalDstData &dstGlobalData,
                             TileData &pingTile, TileData &pongTile, WaitEvents&... events);
```

## Constraints

- **Type constraints**:
  - `ParallelGroup::value_type::RawDType` must equal `GlobalDstData::RawDType`.
  - `TileData::DType` must equal `GlobalDstData::RawDType`.
- **Memory constraints**:
  - `dstGlobalData` must point to local memory (current NPU) and be large enough to hold the concatenated result from all ranks. Specifically, `dstGlobalData.GetShape(DIM_3)` must be $\geq N \times H$ where $H$ is each rank's `GetShape(DIM_3)`.
  - `stagingTileData` (or `pingTile` / `pongTile`) must be pre-allocated in UB.
- **ParallelGroup constraints**:
  - `parallelGroup.tensors[r]` must refer to rank `r`'s source buffer (remote GM as seen by the root).
  - `parallelGroup.GetRootIdx()` identifies the calling NPU as the gather root.
  - All source tensors are assumed to have the same shape and strides.
- **Chunked mode constraints** (when source data exceeds a single UB tile):
  - If `TileData` has static `ValidRow`, `GetShape(DIM_3)` of each rank's source must be divisible by `ValidRow`. Use a Tile with `DYNAMIC` ValidRow for partial row support.
  - If `TileData` has static `ValidCol`, `GetShape(DIM_4)` must be divisible by `ValidCol`. Use a Tile with `DYNAMIC` ValidCol for partial column support.

## Examples

### Basic Gather (Single Row Per Rank)

Each rank contributes one row of `CHUNK_SIZE` elements. The root collects them into `NRANKS` rows.

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int CHUNK_SIZE, int NRANKS>
void gather(__gm__ T* group_addrs[NRANKS], __gm__ T* result, int my_rank) {
    using TileT = Tile<TileType::Vec, T, 1, CHUNK_SIZE>;
    using GChunk = GlobalTensor<T, Shape<1,1,1,1,CHUNK_SIZE>, 
                                BaseShape2D<T, 1, CHUNK_SIZE, Layout::ND>, Layout::ND>;
    using GResult = GlobalTensor<T, Shape<1,1,1,NRANKS,CHUNK_SIZE>, 
                                 BaseShape2D<T, NRANKS, CHUNK_SIZE, Layout::ND>, Layout::ND>;

    GChunk tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GChunk(group_addrs[i]);
    }
    
    comm::ParallelGroup<GChunk> group(tensors, NRANKS, my_rank);
    GResult dstG(result);
    TileT stagingTile;
    
    // Root gathers data from all ranks into `result`.
    comm::TGATHER(group, dstG, stagingTile);
}
```

### Large Tile Gather (Multi-Row Per Rank, Auto-Chunked)

Each rank has `ROWS × COLS` data. The result is `NRANKS * ROWS × COLS`, concatenated along DIM_3.
The UB tile only holds `TILE_ROWS × COLS` — the implementation automatically chunks the transfer.

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int ROWS, int COLS, int TILE_ROWS, int NRANKS>
void gather_large(__gm__ T* group_addrs[NRANKS], __gm__ T* result, int my_rank) {
    using TileT = Tile<TileType::Vec, T, TILE_ROWS, COLS, BLayout::RowMajor, -1, -1>;
    using GPerRank = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                  BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;
    using GResult = GlobalTensor<T, Shape<1,1,1,NRANKS*ROWS,COLS>,
                                  BaseShape2D<T, NRANKS*ROWS, COLS, Layout::ND>, Layout::ND>;

    GPerRank tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GPerRank(group_addrs[i]);
    }

    comm::ParallelGroup<GPerRank> group(tensors, NRANKS, my_rank);
    GResult dstG(result);
    TileT stagingTile(TILE_ROWS, COLS);

    // Each rank contributes ROWS rows; implementation auto-chunks in tiles of TILE_ROWS.
    comm::TGATHER(group, dstG, stagingTile);
}
```

### Ping-Pong Gather (Double Buffering)

Uses two UB tiles to overlap TLOAD of the next chunk (MTE2) with TSTORE of the current chunk (MTE3).

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int ROWS, int COLS, int TILE_ROWS, int NRANKS>
void gather_pingpong(__gm__ T* group_addrs[NRANKS], __gm__ T* result, int my_rank) {
    using TileT = Tile<TileType::Vec, T, TILE_ROWS, COLS, BLayout::RowMajor, -1, -1>;
    using GPerRank = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                  BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;
    using GResult = GlobalTensor<T, Shape<1,1,1,NRANKS*ROWS,COLS>,
                                  BaseShape2D<T, NRANKS*ROWS, COLS, Layout::ND>, Layout::ND>;

    GPerRank tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GPerRank(group_addrs[i]);
    }

    comm::ParallelGroup<GPerRank> group(tensors, NRANKS, my_rank);
    GResult dstG(result);
    TileT pingTile(TILE_ROWS, COLS);
    TileT pongTile(TILE_ROWS, COLS);

    // Ping-pong: overlaps TLOAD and TSTORE for better throughput
    comm::TGATHER(group, dstG, pingTile, pongTile);
}
```
