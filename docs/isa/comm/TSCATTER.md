# TSCATTER

## Introduction

Scatter operation: distribute different chunks of data from the calling NPU (root) to multiple ranks in the parallel group. This is the inverse of `TGATHER` (root gather).

> **Hardware Note**: This instruction may be offloaded to dedicated collective communication hardware.

Only the root needs to execute `TSCATTER`. Non-root ranks only need to ensure their destination buffers are allocated and writable for the duration of the operation.

## Math Interpretation

After the operation, each remote NPU receives its portion:

$$ \mathrm{dst}^{(r)}_{i,j} = \mathrm{src}^{\mathrm{local}}[\text{offset}(r) + i, j] \quad \forall r \in [0, N) $$

where $N$ is the number of ranks and $\text{offset}(r)$ is the starting position for rank $r$'s chunk in the source data.

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

```text
tscatter %group, %src, %ub_tile
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <typename ParallelGroup, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(ParallelGroup &parallelGroup, GlobalSrcData &srcGlobalData, TileData &stagingTileData, WaitEvents&... events);
```

## Constraints

- **Type constraints**:
  - `ParallelGroup::value_type::RawDType` must equal `GlobalSrcData::RawDType`.
  - `TileData::DType` must equal `GlobalSrcData::RawDType`.
- **Memory constraints**:
  - `srcGlobalData` must point to local memory (current NPU) and be large enough to hold data for all ranks.
  - `stagingTileData` must be pre-allocated in UB.
- **ParallelGroup constraints**:
  - `parallelGroup.tensors[r]` must refer to rank `r`'s destination buffer (remote GM as seen by the root).
  - `parallelGroup.my_rank` identifies the calling NPU as the scatter root.

## Examples

### Basic Scatter

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int CHUNK_SIZE, int NRANKS>
void scatter(__gm__ T* local_data, __gm__ T* group_addrs[NRANKS], int my_rank) {
    using TileT = Tile<TileType::Vec, T, 1, CHUNK_SIZE>;
    using GChunk = GlobalTensor<T, Shape<1,1,1,1,CHUNK_SIZE>, 
                                BaseShape2D<T, 1, CHUNK_SIZE, Layout::ND>, Layout::ND>;
    using GSource = GlobalTensor<T, Shape<1,1,1,NRANKS,CHUNK_SIZE>, 
                                 BaseShape2D<T, NRANKS, CHUNK_SIZE, Layout::ND>, Layout::ND>;

    // Stack-allocated tensors (no memory leak)
    GChunk tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GChunk(group_addrs[i]);
    }
    
    comm::ParallelGroup<GChunk> group(tensors, NRANKS, my_rank);
    GSource srcG(local_data);
    TileT stagingTile;
    
    comm::TSCATTER(group, srcG, stagingTile);
}
```
