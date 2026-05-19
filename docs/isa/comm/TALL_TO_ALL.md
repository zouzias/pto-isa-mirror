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
    - `CollEngine::CCU` (Ascend950, NPU_ARCH 3510 only)

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// Basic AllToAll (single source tile)
template <CollEngine engine = CollEngine::CCU,
          typename GlobalSrcData, typename GlobalDstData, typename TileData, typename... Args>
PTO_INST RecordEvent TALL_TO_ALL(GlobalSrcData &srcGlobalData, GlobalDstData &dstGlobalData,
                                 TileData &srcTileData, Args&... args);

// Ping-pong AllToAll (ping + pong tiles)
template <CollEngine engine = CollEngine::CCU,
          typename GlobalSrcData, typename GlobalDstData, typename TileData, typename... Args>
PTO_INST RecordEvent TALL_TO_ALL(GlobalSrcData &srcGlobalData, GlobalDstData &dstGlobalData,
                                 TileData &pingTileData, TileData &pongTileData, Args&... args);
```

When `engine == CollEngine::CCU`, the first variadic argument must be a `CcuTriggerContext`. The AIV kernel triggers the CKE gate; the CCU engine performs the AllToAll data exchange (WriteNb from local input slices to each peer's output at appropriate offsets + LocalCopyNb for self).

## Constraints

- **Engine**: Only `CollEngine::CCU` is currently supported.
- **Type constraints**:
    - `TileData::DType` must equal `GlobalSrcData::RawDType`.
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

### AllToAll (AIV-fused)

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
