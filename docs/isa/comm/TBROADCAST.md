# TBROADCAST

## Introduction

Broadcast data from root rank to all ranks in the parallel group. The root PE's data is copied to all other PEs.

## Math Interpretation

After the operation:

$$ \mathrm{dst}^{(k)}_{i,j} = \mathrm{src}^{(\text{root})}_{i,j} \quad \forall k \in [0, N) $$

where $N$ is the number of ranks and `root` is the broadcasting rank.

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Synchronous form:

```text
tbroadcast %group, %src, %root, %ub_tile
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <typename ParallelGroup, typename GlobalSrcData, typename TileData>
PTO_INST void TBROADCAST(ParallelGroup &parallelGroup, GlobalSrcData &srcGlobal, int root, TileData &ubTile);
```

## Constraints

- **Type constraints**:
  - `ParallelGroup::value_type::RawDType` must equal `GlobalSrcData::RawDType`.
  - `TileData::DType` must equal `GlobalSrcData::RawDType`.
- **Memory constraints**:
  - `srcGlobal` must point to symmetric memory accessible by all PEs.
  - `ubTile` must be pre-allocated in UB.
- **Root constraints**:
  - `root` must be valid: `0 <= root < parallelGroup.nranks`.
- **ParallelGroup constraints**:
  - Must contain valid pointers to GlobalTensors for all participating ranks.

## Examples

### Basic Broadcast

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SIZE>
void broadcast(__gm__ T** group_tensors, __gm__ T* data, int root, int my_rank, int nranks) {
    using TileT = Tile<TileType::Vec, T, 1, SIZE>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>, Stride<SIZE,SIZE,SIZE,SIZE,1>, Layout::ND>;
    using Group = comm::ParallelGroup<GTensor>;

    // Create ParallelGroup
    GTensor* tensors[nranks];
    for (int i = 0; i < nranks; ++i) {
        tensors[i] = new GTensor(group_tensors[i]);
    }
    Group group(tensors, nranks, my_rank);

    // Source tensor (same memory on all PEs, root's data will be broadcast)
    GTensor srcG(data);

    TileT ubTile;
    comm::TBROADCAST(group, srcG, root, ubTile);
    
    // Now all PEs have root's data in their srcG
}
```

### Broadcast Configuration Parameters

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

// Root PE (rank 0) broadcasts configuration to all workers
template <typename T>
void broadcast_config(comm::ParallelGroup<...>& group, GlobalTensor<T, ...>& config) {
    using TileT = Tile<TileType::Vec, T, 1, 64>;
    
    TileT ubTile;
    int root = 0;  // Rank 0 is the root
    
    comm::TBROADCAST(group, config, root, ubTile);
    
    // All workers now have the same configuration
}
```

### Model Weight Distribution

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

// Master worker broadcasts initialized weights to all workers
template <typename T>
void distribute_weights(comm::ParallelGroup<...>& group, GlobalTensor<T, ...>& weights, int master_rank) {
    using TileT = Tile<TileType::Vec, T, 16, 16>;
    
    TileT ubTile;
    comm::TBROADCAST(group, weights, master_rank, ubTile);
}
```

### Conditional Broadcast

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

// Different roots broadcast different data sections
template <typename T>
void rotating_broadcast(comm::ParallelGroup<...>& group, GlobalTensor<T, ...>* data_sections, 
                        int num_sections, int my_rank) {
    using TileT = Tile<TileType::Vec, T, 1, 128>;
    TileT ubTile;

    for (int section = 0; section < num_sections; ++section) {
        int root = section % group.GetSize();
        comm::TBROADCAST(group, data_sections[section], root, ubTile);
    }
}
```
