# TALL_GATHER

## Introduction

AllGather operation: each rank contributes a chunk of data, and after the operation every rank holds the full concatenation of all ranks' contributions.

All ranks must participate. After completion, every rank holds the same full output buffer containing all chunks ordered by rank index.

## Math Interpretation

Given `N` ranks, each contributing `S` elements:

$$\mathrm{output}^{(\text{any rank})}_{r \cdot S + i} = \mathrm{input}^{(r)}_{i}, \quad r \in [0, N),\; i \in [0, S)$$

Every rank's output is identical and contains all ranks' inputs concatenated in rank order.

## Template Parameter

- `engine`:
    - `CollEngine::AIV` (default) — each rank TLOADs all peers' data and assembles the full output
    - `CollEngine::CCU` (Ascend950, NPU_ARCH 3510 only) — AIV triggers CKE gate, CCU engine handles data path

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// AIV path (ParallelGroup-based, A2A3)
template <CollEngine engine = CollEngine::AIV,
          typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... Args>
PTO_INST RecordEvent TALL_GATHER(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                 TileData &stagingTileData, Args&... args);

// CCU path (GlobalSrc + GlobalDst, A5)
template <CollEngine engine = CollEngine::CCU,
          typename GlobalSrcData, typename GlobalDstData, typename TileData, typename... Args>
PTO_INST RecordEvent TALL_GATHER(GlobalSrcData &srcGlobalData, GlobalDstData &dstGlobalData,
                                 TileData &srcTileData, Args&... args);
```

When `engine == CollEngine::AIV`, each rank iterates over the ParallelGroup, TLOADs each peer's source buffer, and TSTOREs it to the corresponding offset in the local output.

When `engine == CollEngine::CCU`, the first variadic argument must be a `CcuTriggerContext`. The AIV kernel triggers the CKE gate; the CCU engine performs the AllGather (WriteNb to each peer's output at the appropriate offset + LocalCopyNb for self).

## Constraints

- **Type constraints**:
    - `TileData::DType` must equal `GlobalSrcData::RawDType`.
- **Memory constraints**:
    - `srcGlobalData` points to this rank's input buffer (local HBM).
    - `dstGlobalData` points to the output buffer (local HBM, will be filled with all ranks' data).
    - `srcTileData` must be a pre-allocated UB tile with valid dimensions initialized.
- **AIV path**: ParallelGroup[r] points to rank r's source buffer. Each rank TLOADs each peer's data and TSTOREs to the corresponding offset in local output. No host-side CCU kernel registration required. See `tests/npu/a2a3/comm/st/testcase/tall_gather/` for a complete example.
- **CcuTriggerContext** (CCU path only):
    - `inputSource == AivStored`: AIV TSTOREs `srcTileData` into `srcGlobalData` before triggering CKE.
    - `inputSource == HostManaged`: Host has already prepared the input HBM; AIV only triggers CKE.

> **CCU path**: All ranks must register and launch the CCU kernel via `HcclCcuKernelRegister` / `HcclCcuKernelLaunch`. See `tests/npu/a5/comm/st/testcase/tall_gather_ccu/` for a complete example.

## Examples

### AllGather (AIV path, A2A3)

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int COUNT, int NRANKS>
void all_gather(__gm__ T *remoteAddrs[NRANKS], __gm__ T *output, int myRank) {
    using TileT = Tile<TileType::Vec, T, 1, COUNT>;
    using SrcTensor = GlobalTensor<T, Shape<1,1,1,1,COUNT>,
                                   Stride<COUNT,COUNT,COUNT,COUNT,1>, Layout::ND>;
    using DstTensor = GlobalTensor<T, Shape<1,1,1,NRANKS,COUNT>,
                                   Stride<NRANKS*COUNT,NRANKS*COUNT,NRANKS*COUNT,COUNT,1>, Layout::ND>;

    SrcTensor tensors[NRANKS];
    for (int r = 0; r < NRANKS; ++r) tensors[r] = SrcTensor(remoteAddrs[r]);
    comm::ParallelGroup<SrcTensor> group(tensors, NRANKS, myRank);

    DstTensor dstG(output);
    TileT stagingTile;

    comm::TALL_GATHER(group, dstG, stagingTile);
}
```

### AllGather (CCU path, A5)

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SLICE_SIZE>
void all_gather(__gm__ T *inputVa, __gm__ T *outputVa, uint32_t selfIdx,
                uint64_t ckeVA, uint32_t mask) {
    using TileT = Tile<TileType::Vec, T, 1, SLICE_SIZE, BLayout::RowMajor, -1, -1>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SLICE_SIZE>,
                                 Stride<SLICE_SIZE,SLICE_SIZE,SLICE_SIZE,SLICE_SIZE,1>, Layout::ND>;

    TileT srcTile(1, SLICE_SIZE);
    GTensor inputGm(inputVa);
    GTensor outputGm(outputVa);

    TEXPANDS(srcTile, static_cast<T>(selfIdx + 1));

    comm::CcuTriggerContext ctx{ckeVA, mask, selfIdx, comm::CcuInputSource::AivStored};
    comm::TALL_GATHER<comm::CollEngine::CCU>(inputGm, outputGm, srcTile, ctx);
}
```
