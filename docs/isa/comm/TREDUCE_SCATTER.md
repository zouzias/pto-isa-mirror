# TREDUCE_SCATTER

## Introduction

ReduceScatter operation: all ranks collectively reduce the full input, then each rank receives a distinct slice of the reduced result.

All ranks must participate in the operation. After completion, rank `r` holds slice `r` of the element-wise reduced output.

## Math Interpretation

Given `N` ranks and total element count `M = N × sliceSize`:

$$\mathrm{output}^{(r)}_{i} = \bigoplus_{k=0}^{N-1} \mathrm{input}^{(k)}_{r \cdot S + i}, \quad i \in [0, S)$$

where $S$ is the per-rank slice size, $r$ is the local rank index, and $\oplus$ is the reduction operation.

## Template Parameter

- `engine`:
    - `CollEngine::AIV` (default) — each rank directly TLOADs its designated slice from all peers, reduces in UB
    - `CollEngine::CCU` (Ascend950, NPU_ARCH 3510 only) — AIV triggers CKE gate, CCU engine handles data path

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// Basic ReduceScatter (accumulator + receive tile)
template <CollEngine engine = CollEngine::AIV,
          typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... Args>
PTO_INST RecordEvent TREDUCE_SCATTER(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                     TileData &accTileData, TileData &recvTileData, ReduceOp op, Args&... args);

// Ping-pong ReduceScatter (accumulator + ping + pong tiles)
template <CollEngine engine = CollEngine::AIV,
          typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... Args>
PTO_INST RecordEvent TREDUCE_SCATTER(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                     TileData &accTileData, TileData &pingTileData, TileData &pongTileData,
                                     ReduceOp op, Args&... args);
```

When `engine == CollEngine::CCU`, the first variadic argument must be a `CcuTriggerContext`. The AIV kernel triggers the CKE gate; the CCU engine performs the actual ReduceScatter data path (ReadNb from all peers + LocalReduceNb + LocalCopyNb).

## Constraints

- **Type constraints**:
    - `TileData::DType` must equal `GlobalDstData::RawDType`.
- **Memory constraints**:
    - `dstGlobalData` must point to local HBM (current NPU).
    - `accTileData`, `recvTileData` must be pre-allocated UB tiles with valid dimensions initialized.
- **ParallelGroup constraints**:
    - `parallelGroup.tensors[r]` must refer to rank `r`'s input buffer.
- **AIV path**: Each rank TLOADs its designated slice from all peers via ParallelGroup, reduces in UB, then TSTOREs result locally. No host-side CCU kernel registration required. See `tests/npu/a2a3/comm/st/testcase/treduce_scatter/` for a complete example.
- **CCU path**: All ranks must register and launch the CCU kernel via host-side APIs.
    - **CcuTriggerContext**:
        - `inputSource == AivStored`: AIV TSTOREs `accTileData` into `parallelGroup[selfIdx]` before triggering CKE.
        - `inputSource == HostManaged`: Host has already prepared the input HBM; AIV only triggers CKE.

> **CCU path**: All ranks must register and launch the CCU kernel via `HcclCcuKernelRegister` / `HcclCcuKernelLaunch`. See `tests/npu/a5/comm/st/testcase/treduce_scatter_ccu/` for a complete example.

## Examples

### ReduceScatter Sum (AIV path)

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SLICE_SIZE, int NRANKS>
void reduce_scatter_sum(__gm__ T *remoteAddrs[NRANKS], __gm__ T *output, int myRank) {
    using TileT = Tile<TileType::Vec, T, 1, SLICE_SIZE>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,NRANKS,SLICE_SIZE>,
                                 Stride<NRANKS*SLICE_SIZE,NRANKS*SLICE_SIZE,NRANKS*SLICE_SIZE,SLICE_SIZE,1>, Layout::ND>;
    using DstTensor = GlobalTensor<T, Shape<1,1,1,1,SLICE_SIZE>,
                                   Stride<SLICE_SIZE,SLICE_SIZE,SLICE_SIZE,SLICE_SIZE,1>, Layout::ND>;

    GTensor tensors[NRANKS];
    for (int r = 0; r < NRANKS; ++r) tensors[r] = GTensor(remoteAddrs[r]);
    comm::ParallelGroup<GTensor> group(tensors, NRANKS, myRank);

    DstTensor dstG(output);
    TileT accTile, recvTile;

    comm::TREDUCE_SCATTER(group, dstG, accTile, recvTile, comm::ReduceOp::Sum);
}
```

### ReduceScatter Sum (CCU path, A5)

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SLICE_SIZE, int NRANKS>
void reduce_scatter_sum(__gm__ T *inputVa, __gm__ T *outputVa, uint32_t selfIdx,
                        uint64_t ckeVA, uint32_t mask) {
    using TileT = Tile<TileType::Vec, T, 1, SLICE_SIZE, BLayout::RowMajor, -1, -1>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SLICE_SIZE>,
                                 Stride<SLICE_SIZE,SLICE_SIZE,SLICE_SIZE,SLICE_SIZE,1>, Layout::ND>;

    TileT accTile(1, SLICE_SIZE);
    TileT recvTile(1, SLICE_SIZE);
    GTensor inputGm(inputVa);
    GTensor outputGm(outputVa);

    GTensor rankTensors[NRANKS];
    for (int r = 0; r < NRANKS; ++r) rankTensors[r] = GTensor(inputVa);
    comm::ParallelGroup<GTensor> group(rankTensors, NRANKS, 0);

    // Fill accTile with this rank's data
    TEXPANDS(accTile, static_cast<T>(selfIdx + 1));

    comm::CcuTriggerContext ctx{ckeVA, mask, selfIdx, comm::CcuInputSource::AivStored};
    comm::TREDUCE_SCATTER<comm::CollEngine::CCU>(group, outputGm, accTile, recvTile, comm::ReduceOp::Sum, ctx);
}
```
