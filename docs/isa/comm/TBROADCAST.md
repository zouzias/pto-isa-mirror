# TBROADCAST

## Introduction

Broadcast data from current NPU to all ranks in the parallel group. The calling NPU is the root and its data is copied to all other NPUs.

> **Hardware Note**: This instruction may be offloaded to dedicated collective communication hardware.

## Math Interpretation

After the operation:

$$ \mathrm{dst}^{(k)}_{i,j} = \mathrm{src}^{(\text{my\_rank})}_{i,j} \quad \forall k \in [0, N) $$

where $N$ is the number of ranks and `my_rank` is the calling NPU (root).

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

```text
tbroadcast %group, %src, %ub_tile
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <typename ParallelGroup, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TBROADCAST(ParallelGroup &parallelGroup, GlobalSrcData &srcGlobal, TileData &ubTile, WaitEvents&... events);
```

## Constraints

- **Type constraints**:
  - `ParallelGroup::value_type::RawDType` must equal `GlobalSrcData::RawDType`.
  - `TileData::DType` must equal `GlobalSrcData::RawDType`.
- **Memory constraints**:
  - `srcGlobal` must point to local memory (current NPU).
  - `ubTile` must be pre-allocated in UB.
- **ParallelGroup constraints**:
  - All tensors must point to symmetric addresses across NPUs.
  - `parallelGroup.my_rank` identifies the calling NPU as the broadcast root.

## Examples

### Basic Broadcast

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SIZE, int NRANKS>
void broadcast(__gm__ T* group_addrs[NRANKS], __gm__ T* my_data, int my_rank) {
    using TileT = Tile<TileType::Vec, T, 1, SIZE>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>, 
                                 BaseShape2D<T, 1, SIZE, Layout::ND>, Layout::ND>;

    // Stack-allocated tensors (no memory leak)
    GTensor tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GTensor(group_addrs[i]);
    }
    
    comm::ParallelGroup<GTensor> group(tensors, NRANKS, my_rank);
    GTensor srcG(my_data);
    TileT ubTile;
    
    // Current NPU broadcasts its data to all others
    comm::TBROADCAST(group, srcG, ubTile);
}
```
