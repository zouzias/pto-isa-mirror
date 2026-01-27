# TPUT

## Introduction

Remote write operation: write local data to remote PE's memory. Data is transferred via a UB tile as intermediate staging buffer.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}^{\mathrm{remote}}_{i,j} = \mathrm{src}^{\mathrm{local}}_{i,j} $$

Data flow: `srcGlobal (local GM)` → `ubTile (UB)` → `dstGlobal (remote GM)`

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Synchronous form:

```text
tput %dst_remote, %src_local, %ub_tile : (!pto.memref<...>, !pto.memref<...>, !pto.tile<...>)
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <typename GlobalDstData, typename GlobalSrcData, typename TileData>
PTO_INST void TPUT(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal, TileData &ubTile);
```

## Constraints

- **Type constraints**:
  - `GlobalSrcData::RawDType` must equal `GlobalDstData::RawDType`.
  - `TileData::DType` must equal `GlobalSrcData::RawDType`.
  - `GlobalSrcData::layout` must equal `GlobalDstData::layout`.
- **Memory constraints**:
  - `dstGlobal` must point to remote PE's symmetric memory (obtained via `ShmemPtr`).
  - `srcGlobal` must point to local PE's symmetric memory.
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
void example_tput(__gm__ T* local_data, __gm__ T* remote_data, int remote_pe) {
    using TileT = Tile<TileType::Vec, T, 16, 16>;
    using GShape = Shape<1, 1, 1, 16, 16>;
    using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
    using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

    // Local source tensor
    GTensor srcG(local_data);
    
    // Remote destination tensor (address obtained via ShmemPtr)
    __gm__ T* remote_addr = ShmemPtr(remote_data, remote_pe);
    GTensor dstG(remote_addr);
    
    // UB staging buffer
    TileT ubTile;
    
    // Perform remote write
    comm::TPUT(dstG, srcG, ubTile);
}
```

### Ring Communication Pattern

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SIZE>
void ring_put(__gm__ T* send_buf, __gm__ T* recv_buf, int my_rank, int nranks) {
    using TileT = Tile<TileType::Vec, T, 1, SIZE>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>, Stride<SIZE,SIZE,SIZE,SIZE,1>, Layout::ND>;

    int next_rank = (my_rank + 1) % nranks;
    
    GTensor sendG(send_buf);
    __gm__ T* remote_recv = ShmemPtr(recv_buf, next_rank);
    GTensor recvG(remote_recv);
    
    TileT ubTile;
    comm::TPUT(recvG, sendG, ubTile);
}
```
