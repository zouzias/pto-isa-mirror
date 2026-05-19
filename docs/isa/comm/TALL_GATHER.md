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
    - `CollEngine::CCU` (Ascend950, NPU_ARCH 3510 only)

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// Basic AllGather (single source tile)
template <CollEngine engine = CollEngine::CCU,
          typename GlobalSrcData, typename GlobalDstData, typename TileData, typename... Args>
PTO_INST RecordEvent TALL_GATHER(GlobalSrcData &srcGlobalData, GlobalDstData &dstGlobalData,
                                 TileData &srcTileData, Args&... args);

// Ping-pong AllGather (ping + pong tiles)
template <CollEngine engine = CollEngine::CCU,
          typename GlobalSrcData, typename GlobalDstData, typename TileData, typename... Args>
PTO_INST RecordEvent TALL_GATHER(GlobalSrcData &srcGlobalData, GlobalDstData &dstGlobalData,
                                 TileData &pingTileData, TileData &pongTileData, Args&... args);
```

When `engine == CollEngine::CCU`, the first variadic argument must be a `CcuTriggerContext`. The AIV kernel triggers the CKE gate; the CCU engine performs the AllGather (WriteNb to each peer's output at the appropriate offset + LocalCopyNb for self).

## Constraints

- **Engine**: Only `CollEngine::CCU` is currently supported.
- **Type constraints**:
    - `TileData::DType` must equal `GlobalSrcData::RawDType`.
- **Memory constraints**:
    - `srcGlobalData` points to this rank's input buffer (local HBM).
    - `dstGlobalData` points to the output buffer (local HBM, will be filled with all ranks' data).
    - `srcTileData` must be a pre-allocated UB tile with valid dimensions initialized.
- **CcuTriggerContext**:
    - `inputSource == AivStored`: AIV TSTOREs `srcTileData` into `srcGlobalData` before triggering CKE.
    - `inputSource == HostManaged`: Host has already prepared the input HBM; AIV only triggers CKE.

> **CCU path**: All ranks must register and launch the CCU kernel via `HcclCcuKernelRegister` / `HcclCcuKernelLaunch`. See `tests/npu/a5/comm/st/testcase/tall_gather_ccu/` for a complete example.

## Examples

### AllGather (AIV-fused)

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
