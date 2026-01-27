# TALLGATHER

## Introduction

All-gather operation across parallel group. Each PE contributes its local data, and all PEs receive the concatenated result from all ranks.

## Math Interpretation

After the operation, each PE has the concatenated data from all ranks:

$$ \mathrm{dst}^{(k)}[\text{offset}(r) : \text{offset}(r+1)] = \mathrm{src}^{(r)} \quad \forall r \in [0, N), \forall k \in [0, N) $$

where $N$ is the number of ranks and $\text{offset}(r)$ is the starting position for rank $r$'s contribution.

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Synchronous form:

```text
tallgather %group, %dst, %ub_tile
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <typename ParallelGroup, typename GlobalDstData, typename TileData>
PTO_INST void TALLGATHER(ParallelGroup &parallelGroup, GlobalDstData &dstGlobal, TileData &ubTile);
```

## Constraints

- **Type constraints**:
  - `ParallelGroup::value_type::RawDType` must equal `GlobalDstData::RawDType`.
  - `TileData::DType` must equal `GlobalDstData::RawDType`.
- **Memory constraints**:
  - `dstGlobal` must be large enough to hold data from all ranks.
  - `dstGlobal` must point to symmetric memory accessible by all PEs.
  - `ubTile` must be pre-allocated in UB.
- **ParallelGroup constraints**:
  - Must contain valid pointers to GlobalTensors for all participating ranks.

## Examples

### Basic All-Gather

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int CHUNK_SIZE>
void allgather(__gm__ T** group_tensors, __gm__ T* result, int my_rank, int nranks) {
    using TileT = Tile<TileType::Vec, T, 1, CHUNK_SIZE>;
    using GChunk = GlobalTensor<T, Shape<1,1,1,1,CHUNK_SIZE>, Stride<CHUNK_SIZE,CHUNK_SIZE,CHUNK_SIZE,CHUNK_SIZE,1>, Layout::ND>;
    // Result holds CHUNK_SIZE * nranks elements
    using GResult = GlobalTensor<T, Shape<DYNAMIC,DYNAMIC,DYNAMIC,DYNAMIC,DYNAMIC>, Stride<DYNAMIC,DYNAMIC,DYNAMIC,DYNAMIC,DYNAMIC>, Layout::ND>;
    using Group = comm::ParallelGroup<GChunk>;

    // Create ParallelGroup
    GChunk* tensors[nranks];
    for (int i = 0; i < nranks; ++i) {
        tensors[i] = new GChunk(group_tensors[i]);
    }
    Group group(tensors, nranks, my_rank);

    // Destination tensor (nranks * CHUNK_SIZE elements)
    Shape<DYNAMIC,DYNAMIC,DYNAMIC,DYNAMIC,DYNAMIC> shape(1, 1, 1, 1, CHUNK_SIZE * nranks);
    Stride<DYNAMIC,DYNAMIC,DYNAMIC,DYNAMIC,DYNAMIC> stride(CHUNK_SIZE * nranks, CHUNK_SIZE * nranks, 
                                                           CHUNK_SIZE * nranks, CHUNK_SIZE * nranks, 1);
    GResult dstG(result, shape, stride);

    TileT ubTile;
    comm::TALLGATHER(group, dstG, ubTile);
}
```

### Gather Model Shards

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

// Each worker has a shard of the model; gather all shards
template <typename T>
void gather_model_shards(comm::ParallelGroup<...>& group, GlobalTensor<T, ...>& full_model) {
    using TileT = Tile<TileType::Vec, T, 16, 16>;
    
    TileT ubTile;
    comm::TALLGATHER(group, full_model, ubTile);
    
    // Now full_model contains all shards concatenated
}
```
