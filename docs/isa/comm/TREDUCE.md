# TREDUCE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-26T03:16:34.348Z pushedAt=2026-08-29T09:05:18.408Z -->

## Introduction

Reduce operation: gathers data from multiple remote NPUs and performs element-wise reduction locally.

Only the root node needs to execute `TREDUCE`. Non-root nodes only need to ensure that their source buffers are ready and remain valid during the operation. Calling `TREDUCE` on a non-root node is undefined behavior.

**Large tile support**: When a GlobalTensor exceeds the UB tile capacity in the row and/or column direction, the reduction operation is automatically tiled through 2D sliding.

## Mathematical Semantics

For each element `(i, j)` in the valid region:

$$\mathrm{dst}^{\mathrm{local}}_{i,j} = \bigoplus_{r=0}^{N-1} \mathrm{src}^{(r)}_{i,j}$$

Where $N$ is the total number of ranks and $\oplus$ is the reduction operation (such as sum, maximum, and minimum).

## Assembly Syntax

Synchronous form:

```text
treduce %group, %dst {op = #pto.reduce_op<Sum>} : (!pto.group<...>, !pto.memref<...>)
treduce %group, %dst {op = #pto.reduce_op<Max>} : (!pto.group<...>, !pto.memref<...>)
```

In the degraded mode, an internal accumulation tile and a receive tile are introduced for the reduce pipeline. The C++ built-in APIs require explicitly passing the `accTileData`, `recvTileData` (or `accTileData`, `pingTileData`, `pongTileData`) operands.

## Template Parameters

- `engine`:
    - `CollEngine::AIV` (default)
    - `CollEngine::CCU` (Ascend 950PR/Ascend 950DT, NPU_ARCH 3510 only)

## C++ Built-in APIs

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// Basic reduce (accumulation tile + receive tile)
template <CollEngine engine = CollEngine::AIV,
          typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... Args>
PTO_INST RecordEvent TREDUCE(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                              TileData &accTileData, TileData &recvTileData, ReduceOp op, Args&... args);

// Ping-pong reduce (accumulate tile + ping/pong tile for double buffering).
template <CollEngine engine = CollEngine::AIV,
          typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... Args>
PTO_INST RecordEvent TREDUCE(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                              TileData &accTileData, TileData &pingTileData, TileData &pongTileData,
                              ReduceOp op, Args&... args);
```

When `engine == CollEngine::CCU`, the first variable parameter must be a `CcuTriggerContext` that contains the CKE slot virtual address and gate mask. The AIV kernel triggers the CKE gate, and the actual reduce data path is executed on the CCU engine.

## Constraints

- **Type constraints**:
    - `ParallelGroup::value_type::RawDType` must be equal to `GlobalDstData::RawDType`.
    - `TileData::DType` must be equal to `GlobalDstData::RawDType`.
- **Memory constraints**:
    - `dstGlobalData` must point to local memory (the current NPU).
    - `accTileData`, `recvTileData` (or `accTileData`, `pingTileData`, `pongTileData`) must be pre-allocated UB tiles.
- **ParallelGroup constraints**:
    - `parallelGroup.tensors[r]` must point to the source buffer of rank `r` (the remote GM as seen from the root node's perspective).
    - `parallelGroup.GetRootIdx()` identifies the calling NPU as the reduce root node.
    - All source tensors are assumed to have the same shape and stride.
- **Tiling mode constraints** (when data exceeds a single UB tile):
    - If `TileData` has a static `ValidRow`, `GetShape(DIM_3)` must be divisible by `ValidRow`. To support the case of fewer than one row, use a tile with a `DYNAMIC` ValidRow.
    - If `TileData` has a static `ValidCol`, `GetShape(DIM_4)` must be divisible by `ValidCol`. To support the case of fewer than one column, use a tile with a `DYNAMIC` ValidCol.

> **CCU path**: Unlike the AIV path (where only the root node calls `TREDUCE`), the CCU path requires all ranks to register and launch the CCU kernel through the host-side `HcclCcuKernelRegister`/`HcclCcuKernelLaunch`. For a complete example, see `tests/npu/a5/comm/st/testcase/treduce_ccu/`.

## Examples

### Basic Sum Reduction

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SIZE, int NRANKS>
void reduce_sum(__gm__ T* group_addrs[NRANKS], __gm__ T* result, int my_rank) {
    using TileT   = Tile<TileType::Vec, T, 1, SIZE>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>,
                                 BaseShape2D<T, 1, SIZE, Layout::ND>, Layout::ND>;

    GTensor tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) tensors[i] = GTensor(group_addrs[i]);

    comm::ParallelGroup<GTensor> group(tensors, NRANKS, my_rank);
    GTensor dstG(result);
    TileT accTile, recvTile;
    comm::TREDUCE(group, dstG, accTile, recvTile, comm::ReduceOp::Sum);
}
```

### Maximum Reduction

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SIZE, int NRANKS>
void reduce_max(__gm__ T* group_addrs[NRANKS], __gm__ T* result, int my_rank) {
    using TileT   = Tile<TileType::Vec, T, 1, SIZE>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>,
                                 BaseShape2D<T, 1, SIZE, Layout::ND>, Layout::ND>;

    GTensor tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) tensors[i] = GTensor(group_addrs[i]);

    comm::ParallelGroup<GTensor> group(tensors, NRANKS, my_rank);
    GTensor dstG(result);
    TileT accTile, recvTile;
    comm::TREDUCE(group, dstG, accTile, recvTile, comm::ReduceOp::Max);
}
```
