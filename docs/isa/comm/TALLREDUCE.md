# TALLREDUCE

## Introduction

All-reduce operation across parallel group. Performs element-wise reduction (sum) of data from all PEs and distributes the result to all PEs.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}^{(k)}_{i,j} = \sum_{r=0}^{N-1} \mathrm{src}^{(r)}_{i,j} \quad \forall k \in [0, N) $$

where $N$ is the number of ranks in the parallel group.

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Synchronous form:

```text
tallreduce %group, %dst, %acc_tile, %ping_tile, %pong_tile
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <typename ParallelGroup, typename GlobalDstData, typename TileData>
PTO_INST void TALLREDUCE(ParallelGroup &parallelGroup, GlobalDstData &dstGlobal, 
                         TileData &accTile, TileData &pingTile, TileData &pongTile);
```

## Constraints

- **Type constraints**:
  - `ParallelGroup::value_type::RawDType` must equal `GlobalDstData::RawDType`.
  - `TileData::DType` must equal `GlobalDstData::RawDType`.
- **Memory constraints**:
  - `dstGlobal` must point to symmetric memory accessible by all PEs.
  - `accTile`, `pingTile`, `pongTile` must be pre-allocated UB tiles.
- **ParallelGroup constraints**:
  - Must contain valid pointers to GlobalTensors for all participating ranks.
  - `parallelGroup.my_rank` must be valid.

## Examples

### Basic All-Reduce

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SIZE>
void allreduce(__gm__ T** group_tensors, __gm__ T* result, int my_rank, int nranks) {
    using TileT = Tile<TileType::Vec, T, 1, SIZE>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>, Stride<SIZE,SIZE,SIZE,SIZE,1>, Layout::ND>;
    using Group = comm::ParallelGroup<GTensor>;

    // Create ParallelGroup
    GTensor* tensors[nranks];
    for (int i = 0; i < nranks; ++i) {
        tensors[i] = new GTensor(group_tensors[i]);
    }
    Group group(tensors, nranks, my_rank);

    // Destination tensor
    GTensor dstG(result);

    // UB tiles for double buffering
    TileT accTile, pingTile, pongTile;

    // Perform all-reduce
    comm::TALLREDUCE(group, dstG, accTile, pingTile, pongTile);
}
```

### Gradient Averaging in Distributed Training

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T>
void average_gradients(comm::ParallelGroup<GlobalTensor<T, ...>>& group,
                       GlobalTensor<T, ...>& gradients) {
    using TileT = Tile<TileType::Vec, T, 16, 16>;
    
    TileT accTile, pingTile, pongTile;
    
    // Sum gradients across all workers
    comm::TALLREDUCE(group, gradients, accTile, pingTile, pongTile);
    
    // Note: Division by nranks for averaging should be done separately
}
```
