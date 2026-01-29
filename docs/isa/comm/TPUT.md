# TPUT

## Introduction

Remote write operation: write local data to remote NPU's memory. Data is transferred via a UB tile as intermediate staging buffer.

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
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TPUT(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal, TileData &ubTile, WaitEvents&... events);
```

## Constraints

- **Type constraints**:
  - `GlobalSrcData::RawDType` must equal `GlobalDstData::RawDType`.
  - `TileData::DType` must equal `GlobalSrcData::RawDType`.
  - `GlobalSrcData::layout` must equal `GlobalDstData::layout`.
- **Memory constraints**:
  - `dstGlobal` must point to remote address (on target NPU).
  - `srcGlobal` must point to local address (on current NPU).
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
void example_tput(__gm__ T* local_data, __gm__ T* remote_addr) {
    using TileT = Tile<TileType::Vec, T, 16, 16>;
    using GShape = Shape<1, 1, 1, 16, 16>;
    using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
    using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

    // Local source tensor
    GTensor srcG(local_data);
    
    // Remote destination tensor
    GTensor dstG(remote_addr);
    
    // UB staging buffer
    TileT ubTile;
    
    // Perform remote write
    comm::TPUT(dstG, srcG, ubTile);
}
```