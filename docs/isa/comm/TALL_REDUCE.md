# TALL_REDUCE

## Introduction

AllReduce operation: all ranks collectively reduce the full input and every rank receives the complete reduced result.

All ranks must participate. After completion, every rank holds the same full element-wise reduced output.

Internally implemented as ReduceScatter + AllGather in a single CCU kernel launch.

## Math Interpretation

For each element `i` in the output (all ranks get the same result):

$$\mathrm{output}_{i} = \bigoplus_{r=0}^{N-1} \mathrm{input}^{(r)}_{i}$$

where $N$ is the number of ranks and $\oplus$ is the reduction operation.

## Template Parameter

- `engine`:
    - `CollEngine::CCU` (Ascend950, NPU_ARCH 3510 only)

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// Basic AllReduce (accumulator + receive tile)
template <CollEngine engine = CollEngine::CCU,
          typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... Args>
PTO_INST RecordEvent TALL_REDUCE(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                 TileData &accTileData, TileData &recvTileData, ReduceOp op, Args&... args);

// Ping-pong AllReduce (accumulator + ping + pong tiles)
template <CollEngine engine = CollEngine::CCU,
          typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... Args>
PTO_INST RecordEvent TALL_REDUCE(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                 TileData &accTileData, TileData &pingTileData, TileData &pongTileData,
                                 ReduceOp op, Args&... args);
```

When `engine == CollEngine::CCU`, the first variadic argument must be a `CcuTriggerContext`. The AIV kernel triggers the CKE gate; the CCU engine performs the AllReduce (ReduceScatter phase via ReadNb + LocalReduceNb, followed by AllGather phase via WriteNb).

## Constraints

- **Engine**: Only `CollEngine::CCU` is currently supported.
- **Type constraints**:
    - `TileData::DType` must equal `GlobalDstData::RawDType`.
- **Memory constraints**:
    - `dstGlobalData` must point to local HBM (current NPU).
    - `accTileData`, `recvTileData` must be pre-allocated UB tiles with valid dimensions initialized.
- **ParallelGroup constraints**:
    - `parallelGroup.tensors[r]` must refer to rank `r`'s input buffer.
    - All ranks must register and launch the CCU kernel via host-side APIs.
- **CcuTriggerContext**:
    - `inputSource == AivStored`: AIV TSTOREs `accTileData` into `parallelGroup[selfIdx]` before triggering CKE.
    - `inputSource == HostManaged`: Host has already prepared the input HBM; AIV only triggers CKE.

> **CCU path**: All ranks must register and launch the CCU kernel via `HcclCcuKernelRegister` / `HcclCcuKernelLaunch`. See `tests/npu/a5/comm/st/testcase/tall_reduce_ccu/` for a complete example.

## Examples

### AllReduce Sum (AIV-fused)

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SIZE, int NRANKS>
void all_reduce_sum(__gm__ T *inputVa, __gm__ T *outputVa, uint32_t selfIdx,
                    uint64_t ckeVA, uint32_t mask) {
    using TileT = Tile<TileType::Vec, T, 1, SIZE, BLayout::RowMajor, -1, -1>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>,
                                 Stride<SIZE,SIZE,SIZE,SIZE,1>, Layout::ND>;

    TileT accTile(1, SIZE);
    TileT recvTile(1, SIZE);
    GTensor inputGm(inputVa);
    GTensor outputGm(outputVa);

    GTensor rankTensors[NRANKS];
    for (int r = 0; r < NRANKS; ++r) rankTensors[r] = GTensor(inputVa);
    comm::ParallelGroup<GTensor> group(rankTensors, NRANKS, 0);

    TEXPANDS(accTile, static_cast<T>(selfIdx + 1));

    comm::CcuTriggerContext ctx{ckeVA, mask, selfIdx, comm::CcuInputSource::AivStored};
    comm::TALL_REDUCE<comm::CollEngine::CCU>(group, outputGm, accTile, recvTile, comm::ReduceOp::Sum, ctx);
}
```
