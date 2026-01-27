# TGET

## Introduction

Remote read operation: read remote PE's data to local memory. Data is transferred via a UB tile as intermediate staging buffer.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}^{\mathrm{local}}_{i,j} = \mathrm{src}^{\mathrm{remote}}_{i,j} $$

Data flow: `srcGlobal (remote GM)` → `ubTile (UB)` → `dstGlobal (local GM)`

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Synchronous form:

```text
tget %dst_local, %src_remote, %ub_tile : (!pto.memref<...>, !pto.memref<...>, !pto.tile<...>)
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <typename GlobalDstData, typename GlobalSrcData, typename TileData>
PTO_INST void TGET(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal, TileData &ubTile);
```

## Constraints

- **Type constraints**:
  - `GlobalSrcData::RawDType` must equal `GlobalDstData::RawDType`.
  - `TileData::DType` must equal `GlobalSrcData::RawDType`.
  - `GlobalSrcData::layout` must equal `GlobalDstData::layout`.
- **Memory constraints**:
  - `srcGlobal` must point to remote PE's symmetric memory (obtained via `ShmemPtr`).
  - `dstGlobal` must point to local PE's symmetric memory.
  - `ubTile` must be pre-allocated in Unified Buffer.
- **Valid region**:
  - Transfer size is determined by `ubTile.GetValidRow()` / `ubTile.GetValidCol()`.

## Examples

### Basic Usage

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_tget(__gm__ T* local_data, __gm__ T* remote_data, int remote_pe) {
    using TileT = Tile<TileType::Vec, T, 16, 16>;
    using GShape = Shape<1, 1, 1, 16, 16>;
    using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
    using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

    // Remote source tensor (address obtained via ShmemPtr)
    __gm__ T* remote_addr = ShmemPtr(remote_data, remote_pe);
    GTensor srcG(remote_addr);
    
    // Local destination tensor
    GTensor dstG(local_data);
    
    // UB staging buffer
    TileT ubTile;
    
    // Perform remote read
    comm::TGET(dstG, srcG, ubTile);
}
```

### Gather from Previous Rank

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SIZE>
void gather_from_prev(__gm__ T* send_buf, __gm__ T* recv_buf, int my_rank, int nranks) {
    using TileT = Tile<TileType::Vec, T, 1, SIZE>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>, Stride<SIZE,SIZE,SIZE,SIZE,1>, Layout::ND>;

    int prev_rank = (my_rank + nranks - 1) % nranks;
    
    __gm__ T* remote_send = ShmemPtr(send_buf, prev_rank);
    GTensor srcG(remote_send);
    GTensor dstG(recv_buf);
    
    TileT ubTile;
    comm::TGET(dstG, srcG, ubTile);
}
```
