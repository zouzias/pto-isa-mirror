# TREDUCE

## Introduction

Reduce operation: gather data from multiple remote NPUs and perform element-wise reduction locally. 

> **Hardware Note**: This instruction may be offloaded to dedicated collective communication hardware.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}^{\mathrm{local}}_{i,j} = \bigoplus_{r=0}^{N-1} \mathrm{src}^{(r)}_{i,j} $$

where $N$ is the number of ranks and $\oplus$ is the reduction operation (sum, max, min, etc.).

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

```text
treduce %group, %dst, %acc_tile, %recv_tile {op = #pto.reduce_op<Sum>}
treduce %group, %dst, %acc_tile, %recv_tile {op = #pto.reduce_op<Max>}
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <typename ParallelGroup, typename GlobalDstData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TREDUCE(ParallelGroup &parallelGroup, GlobalDstData &dstGlobal, 
                              TileData &accTile, TileData &recvTile, ReduceOp op, WaitEvents&... events);
```

## Constraints

- **Type constraints**:
  - `ParallelGroup::value_type::RawDType` must equal `GlobalDstData::RawDType`.
  - `TileData::DType` must equal `GlobalDstData::RawDType`.
- **Memory constraints**:
  - `dstGlobal` must point to local address (on current NPU).
  - `accTile`, `recvTile` must be pre-allocated UB tiles.
- **ParallelGroup constraints**:
  - All tensors must point to symmetric addresses across NPUs.

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
