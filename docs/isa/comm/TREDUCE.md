# TREDUCE

## Introduction

Reduce operation: gather data from multiple remote NPUs and perform element-wise reduction locally. 


Only the root needs to execute `TREDUCE`. Non-root ranks only need to ensure their source buffers are ready and remain valid for the duration of the operation. Calling `TREDUCE` on non-root ranks is undefined behavior.

**Large Tile Support**: When the GlobalTensor exceeds the UB tile capacity in rows and/or columns, the reduction is automatically chunked via 2D sliding.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}^{\mathrm{local}}_{i,j} = \bigoplus_{r=0}^{N-1} \mathrm{src}^{(r)}_{i,j} $$

where $N$ is the number of ranks and $\oplus$ is the reduction operation (sum, max, min, etc.).

## Assembly Syntax

PTO-AS form: see [PTO-AS Specification](../../assembly/PTO-AS.md).

Synchronous form:

```text
treduce %group, %dst {op = #pto.reduce_op<Sum>} : (!pto.group<...>, !pto.memref<...>)
treduce %group, %dst {op = #pto.reduce_op<Max>} : (!pto.group<...>, !pto.memref<...>)
```
Lowering introduces internal accumulator and receive tiles for the reduce pipeline; the C++ intrinsic requires explicit `accTileData`, `recvTileData` (or `accTileData`, `pingTileData`, `pongTileData`) operand(s).

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// Basic reduce (accumulator + receive tile)
template <typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TREDUCE(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData, 
                              TileData &accTileData, TileData &recvTileData, ReduceOp op, WaitEvents&... events);

// Ping-pong reduce (accumulator + ping + pong tiles for double buffering)
template <typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TREDUCE(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                              TileData &accTileData, TileData &pingTileData, TileData &pongTileData,
                              ReduceOp op, WaitEvents&... events);
```

## Constraints

- **Type constraints**:
    - `ParallelGroup::value_type::RawDType` must equal `GlobalDstData::RawDType`.
    - `TileData::DType` must equal `GlobalDstData::RawDType`.
- **Memory constraints**:
    - `dstGlobalData` must point to local address (on current NPU).
    - `accTileData`, `recvTileData` (or `accTileData`, `pingTileData`, `pongTileData`) must be pre-allocated UB tiles.
- **ParallelGroup constraints**:
    - `parallelGroup.tensors[r]` must refer to rank `r`'s source buffer (remote GM as seen by the root).
    - `parallelGroup.GetRootIdx()` identifies the calling NPU as the reduce root.
    - All source tensors are assumed to have the same shape and strides.
- **Chunked mode constraints** (when data exceeds a single UB tile):
    - If `TileData` has static `ValidRow`, `GetShape(DIM_3)` must be divisible by `ValidRow`. Use a Tile with `DYNAMIC` ValidRow for partial row support.
    - If `TileData` has static `ValidCol`, `GetShape(DIM_4)` must be divisible by `ValidCol`. Use a Tile with `DYNAMIC` ValidCol for partial column support.

## Examples

### Basic Reduce Sum

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SIZE, int NRANKS>
void reduce_sum(__gm__ T* group_addrs[NRANKS], __gm__ T* result, int my_rank) {
    using TileT = Tile<TileType::Vec, T, 1, SIZE>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>, 
                                 BaseShape2D<T, 1, SIZE, Layout::ND>, Layout::ND>;

    // Stack-allocated tensors
    GTensor tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GTensor(group_addrs[i]);
    }
    
    comm::ParallelGroup<GTensor> group(tensors, NRANKS, my_rank);
    GTensor dstG(result);
    TileT accTile, recvTile;

    comm::TREDUCE(group, dstG, accTile, recvTile, comm::ReduceOp::Sum);
}
```

### Max Reduce

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SIZE, int NRANKS>
void reduce_max(__gm__ T* group_addrs[NRANKS], __gm__ T* result, int my_rank) {
    using TileT = Tile<TileType::Vec, T, 1, SIZE>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>, 
                                 BaseShape2D<T, 1, SIZE, Layout::ND>, Layout::ND>;

    GTensor tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GTensor(group_addrs[i]);
    }
    
    comm::ParallelGroup<GTensor> group(tensors, NRANKS, my_rank);
    GTensor dstG(result);
    TileT accTile, recvTile;

    comm::TREDUCE(group, dstG, accTile, recvTile, comm::ReduceOp::Max);
}
```

## CCU Backend (Ascend950)

> CCU (Communication Control Unit) is a hardware-accelerated communication engine available on Ascend950. The CCU backend offloads collective operations to CCU microcode built with `hcomm::CcuRep` primitives, while the AIV core only serves as a trigger for the CKE (Communication Kernel Engine) gate.

Unlike the default AIV path which uses `ParallelGroup` + UB tiles on AI Vector cores, the CCU path records microcode via `CcuRep` primitives (`InitResource`, `PreSync`, `DoReduce`, `PostSync`, `GeneArgs`) and is launched by host-side `HcclCcuKernelRegister` / `HcclCcuKernelLaunch`. The AIV kernel's only role is to open the CKE gate via `rtGetDevResAddress` + store.

### CCU-specific Constraints

- **Minimum payload size**: The total payload must be at least **4096 bytes** (e.g., 1024 float elements). Payloads smaller than 4096 bytes cause `hcomm::CcuRep::GetTokenInfo` to fail with `"Ccu api exception: failed to query tokenInfo."`.
- **Shared HcclComm lifecycle**: The `HcclComm`, CCU channels, and streams must be created once and reused across multiple CCU kernel invocations within the same process. Repeated `HcclCommDestroy` / `HcclCommInitRootInfo` cycles leave stale CCU microcode state, causing data corruption or hangs for subsequent invocations with different payload sizes.
- **Channel protocol**: CCU channels use `UBC_CTP` (Unified Buffer Coherent Transport Protocol).
- **All ranks execute**: Unlike the AIV path where only root calls `TREDUCE`, the CCU path requires all ranks to register and launch the CCU kernel.

### ST Reference

See `tests/npu/a5/comm/st/testcase/treduce_ccu/` for a complete host-side test driver demonstrating CCU kernel registration, CKE gate triggering, and result verification.
