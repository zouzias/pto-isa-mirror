# TALL_TO_ALL

## Introduction

AllToAll operation (uniform): full data exchange where each rank sends a distinct slice to every other rank.

All ranks must participate. Each rank's input is logically divided into `N` slices (one per rank). After the operation, rank `r`'s output slice `i` contains what rank `i` sent to rank `r`.

## Math Interpretation

Given `N` ranks, each with `N × S` elements of input:

$$\mathrm{output}^{(r)}_{i \cdot S + j} = \mathrm{input}^{(i)}_{r \cdot S + j}, \quad i \in [0, N),\; j \in [0, S)$$

In words: rank `r` receives from rank `i` the slice that rank `i` designated for rank `r`.

## Template Parameter

- `engine`:
    - `CollEngine::AIV` (default) — each rank TLOADs its designated slice from each peer
    - `CollEngine::CCU` (Ascend950, NPU_ARCH 3510 only) — AIV triggers CKE gate, CCU engine handles data path

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// AIV path (ParallelGroup-based, A2A3)
template <CollEngine engine = CollEngine::AIV,
          typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... Args>
PTO_INST RecordEvent TALL_TO_ALL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                 TileData &stagingTileData, Args&... args);

// CCU path (GlobalSrc + GlobalDst, A5)
template <CollEngine engine = CollEngine::CCU,
          typename GlobalSrcData, typename GlobalDstData, typename TileData, typename... Args>
PTO_INST RecordEvent TALL_TO_ALL(GlobalSrcData &srcGlobalData, GlobalDstData &dstGlobalData,
                                 TileData &srcTileData, Args&... args);
```

When `engine == CollEngine::AIV` (default), the AIV engine directly performs the all-to-all exchange: each rank TLOADs its designated slice from every peer via the `ParallelGroup` and TSTOREs to the local output.

When `engine == CollEngine::CCU`, the first variadic argument must be a `CcuTriggerContext`. The AIV kernel triggers the CKE gate; the CCU engine performs the AllToAll data exchange (WriteNb from local input slices to each peer's output at appropriate offsets + LocalCopyNb for self).

## Constraints

- **Type constraints**:
    - `TileData::DType` must equal `GlobalSrcData::RawDType`.
- **AIV path**:
    - `ParallelGroup[r]` points to rank `r`'s full input buffer (N × sliceSize). Each rank TLOADs the selfIdx-th slice from each peer and TSTOREs to the corresponding offset in local output. No host-side CCU kernel registration required. See `tests/npu/a2a3/comm/st/testcase/tall_to_all/` for a complete example.
- **Memory constraints**:
    - `srcGlobalData` points to this rank's input buffer (local HBM, size = N × sliceSize).
    - `dstGlobalData` points to the output buffer (local HBM, size = N × sliceSize).
    - `srcTileData` must be a pre-allocated UB tile with valid dimensions initialized.
- **Data layout**:
    - Input is divided into N contiguous slices of `sliceSize` bytes each.
    - Slice `i` of this rank's input is sent to rank `i`.
    - After the op, slice `i` of this rank's output contains what rank `i` sent to this rank.
- **CcuTriggerContext**:
    - `inputSource == AivStored`: AIV TSTOREs `srcTileData` into `srcGlobalData` before triggering CKE.
    - `inputSource == HostManaged`: Host has already prepared the input HBM; AIV only triggers CKE.

> **CCU path**: All ranks must register and launch the CCU kernel via `HcclCcuKernelRegister` / `HcclCcuKernelLaunch`. See `tests/npu/a5/comm/st/testcase/tall_to_all_ccu/` for a complete example.

## Examples

### AllToAll (AIV path, A2A3)

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SLICE_SIZE, int NRANKS>
void all_to_all(__gm__ T *remoteAddrs[NRANKS], __gm__ T *output, int myRank) {
    using TileT = Tile<TileType::Vec, T, 1, SLICE_SIZE>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,NRANKS,SLICE_SIZE>,
                                 Stride<NRANKS*SLICE_SIZE,NRANKS*SLICE_SIZE,NRANKS*SLICE_SIZE,SLICE_SIZE,1>, Layout::ND>;

    GTensor tensors[NRANKS];
    for (int r = 0; r < NRANKS; ++r) tensors[r] = GTensor(remoteAddrs[r]);
    comm::ParallelGroup<GTensor> group(tensors, NRANKS, myRank);

    GTensor dstG(output);
    TileT stagingTile;

    comm::TALL_TO_ALL(group, dstG, stagingTile);
}
```

### AllToAll (CCU path, A5)

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int TOTAL_SIZE>
void all_to_all(__gm__ T *inputVa, __gm__ T *outputVa, uint32_t selfIdx,
                uint64_t ckeVA, uint32_t mask) {
    using TileT = Tile<TileType::Vec, T, 1, TOTAL_SIZE, BLayout::RowMajor, -1, -1>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,TOTAL_SIZE>,
                                 Stride<TOTAL_SIZE,TOTAL_SIZE,TOTAL_SIZE,TOTAL_SIZE,1>, Layout::ND>;

    TileT srcTile(1, TOTAL_SIZE);
    GTensor inputGm(inputVa);
    GTensor outputGm(outputVa);

    // Fill input: uniform value (selfIdx + 1) — in practice, each slice may differ
    TEXPANDS(srcTile, static_cast<T>(selfIdx + 1));

    comm::CcuTriggerContext ctx{ckeVA, mask, selfIdx, comm::CcuInputSource::AivStored};
    comm::TALL_TO_ALL<comm::CollEngine::CCU>(inputGm, outputGm, srcTile, ctx);
}
```
