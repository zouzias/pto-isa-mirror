# TGATHER

## Introduction

All-gather operation across parallel group. Each NPU contributes its local data, and all NPUs receive the concatenated result from all ranks.

## Math Interpretation

After the operation, each NPU has the concatenated data from all ranks:

$$ \mathrm{dst}^{(k)}[\text{offset}(r) : \text{offset}(r+1)] = \mathrm{src}^{(r)} \quad \forall r \in [0, N), \forall k \in [0, N) $$

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
PTO_INST RecordEvent TGATHER(ParallelGroup &parallelGroup, GlobalDstData &dstGlobal, TileData &ubTile, WaitEvents&... events);
```

## Constraints

- **Type constraints**:
  - `ParallelGroup::value_type::RawDType` must equal `GlobalDstData::RawDType`.
  - `TileData::DType` must equal `GlobalDstData::RawDType`.
- **Memory constraints**:
  - `dstGlobal` must point to memory accessible by all NPUs.
  - `ubTile` must be pre-allocated in UB.
- **ParallelGroup constraints**:
  - Must contain valid pointers to GlobalTensors for all participating ranks.

## Examples

### Basic All-Gather

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
    TileT ubTile;
    
    comm::TGATHER(group, dstG, ubTile);
}
```
