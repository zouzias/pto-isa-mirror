# TGATHER

## Introduction

Gather operation across a parallel group. The calling NPU is the root and gathers data from all ranks into a single concatenated output buffer on the root.

Only the root needs to execute `TGATHER`. Non-root ranks only need to ensure their source buffers are ready and remain valid for the duration of the operation.

## Math Interpretation

After the operation (on the calling/root NPU):

$$ \mathrm{dst}^{(\text{my\_rank})}[\text{offset}(r) : \text{offset}(r+1)] = \mathrm{src}^{(r)} \quad \forall r \in [0, N) $$

where $N$ is the number of ranks and $\text{offset}(r)$ is the starting position for rank $r$'s contribution.

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

```text
tgather %group, %dst, %ub_tile
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <typename ParallelGroup, typename GlobalDstData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(ParallelGroup &parallelGroup, GlobalDstData &dstGlobalData, TileData &stagingTileData, WaitEvents&... events);
```

## Constraints

- **Type constraints**:
  - `ParallelGroup::value_type::RawDType` must equal `GlobalDstData::RawDType`.
  - `TileData::DType` must equal `GlobalDstData::RawDType`.
- **Memory constraints**:
  - `dstGlobalData` must point to local memory (current NPU) and be large enough to hold the concatenated result from all ranks.
  - `stagingTileData` must be pre-allocated in UB.
- **ParallelGroup constraints**:
  - `parallelGroup.tensors[r]` must refer to rank `r`'s source buffer (remote GM as seen by the root).
  - `parallelGroup.my_rank` identifies the calling NPU as the gather root.

## Examples

### Basic Gather (Root Collects)

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

    // Stack-allocated tensors (no memory leak)
    GChunk tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GChunk(group_addrs[i]);
    }
    
    comm::ParallelGroup<GChunk> group(tensors, NRANKS, my_rank);
    GResult dstG(result);
    TileT stagingTile;
    
    // The calling NPU (group.my_rank) gathers data from all ranks into `result`.
    comm::TGATHER(group, dstG, stagingTile);
}
```
