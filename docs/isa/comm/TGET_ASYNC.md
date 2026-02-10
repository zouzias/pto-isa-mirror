# TGET_ASYNC

## Introduction

Asynchronous remote read operation using configurable DMA engine. Directly transfers data from remote NPU's GM to local GM. The operation returns immediately with an event handle.

## Template Parameters

- `engine`: DMA engine selection
  - `DmaEngine::SDMA` (default) - System DMA
  - `DmaEngine::URMA` - A5 URMA (UB Remote Memory Access) based on Unified Bus

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}^{\mathrm{local}}_{i,j} = \mathrm{src}^{\mathrm{remote}}_{i,j} $$

Data flow: `srcGlobalData (remote GM)` → `DMA Engine` → `dstGlobalData (local GM)`

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Asynchronous form:

```text
%event = tget_async<sdma> %dst_local, %src_remote : (!pto.memref<...>, !pto.memref<...>) -> !pto.event
%event = tget_async<urma> %dst_local, %src_remote : (!pto.memref<...>, !pto.memref<...>) -> !pto.event
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// Asynchronous GET with template-specified DMA engine
template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TGET_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData, WaitEvents&... events);
```

## Constraints

- **Type constraints**:
  - `GlobalSrcData::RawDType` must equal `GlobalDstData::RawDType`.
  - `GlobalSrcData::layout` must equal `GlobalDstData::layout`.
  - Element size must be 1, 2, 4, or 8 bytes.
- **Memory constraints**:
  - `srcGlobalData` must point to remote address (on target NPU).
  - `dstGlobalData` must point to local address (on current NPU).
  - Both addresses should be naturally aligned to element size; 32-byte alignment is recommended for best performance.
- **DMA constraints**:
  - SDMA: 1D transfer
  - URMA: 1D transfer
  - DMA channel availability is limited; implementations may serialize requests when channels are exhausted.
- **Valid region**:
  - Transfer size is determined by GlobalTensor shape or explicit parameters.

## Completion Semantics

After `TSYNC(event)` returns, all loads into `dstGlobalData` performed by the asynchronous transfer are complete and visible to subsequent operations on the current NPU.

## Examples

### Basic Asynchronous GET with SDMA (default)

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T, int SIZE>
void example_tget_async(__gm__ T* local_dst, __gm__ T* remote_src) {
    using GShape = Shape<1, 1, 1, 1, SIZE>;
    using GStride = Stride<SIZE, SIZE, SIZE, SIZE, 1>;
    using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

    // Remote source tensor
    GTensor srcG(remote_src);
    
    // Local destination tensor
    GTensor dstG(local_dst);
    
    // Initiate asynchronous transfer using SDMA (default)
    auto event = comm::TGET_ASYNC(dstG, srcG);
    
    // Do other computation while transfer is in progress
    // ...
    
    // Wait for transfer completion
    TSYNC(event);
}
```

### Using URMA 

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SIZE>
void low_latency_get(__gm__ T* recv_buf, __gm__ T* remote_send_addr) {
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>, Stride<SIZE,SIZE,SIZE,SIZE,1>, Layout::ND>;

    GTensor dstG(recv_buf);
    GTensor srcG(remote_send_addr);
    
    // Use URMA for low-latency small transfer
    auto event = comm::TGET_ASYNC<comm::DmaEngine::URMA>(dstG, srcG);
    
    // Wait for completion
    TSYNC(event);
}
```
# TGET_ASYNC

## Introduction

Asynchronous remote read operation using configurable DMA engine. Directly transfers data from remote NPU's GM to local GM without UB staging. The operation returns immediately with an event handle, allowing computation-communication overlap.

## Template Parameters

- `engine`: DMA engine selection
  - `DmaEngine::SDMA` (default) - System DMA
  - `DmaEngine::URMA` - A5 URMA (UB Remote Memory Access) based on Unified Bus

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}^{\mathrm{local}}_{i,j} = \mathrm{src}^{\mathrm{remote}}_{i,j} $$

Data flow: `srcGlobalData (remote GM)` → `DMA Engine` → `dstGlobalData (local GM)`

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Asynchronous form:

```text
%event = tget_async<sdma> %dst_local, %src_remote : (!pto.memref<...>, !pto.memref<...>) -> !pto.event
%event = tget_async<urma> %dst_local, %src_remote : (!pto.memref<...>, !pto.memref<...>) -> !pto.event
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// Asynchronous GET with template-specified DMA engine
template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TGET_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData, WaitEvents&... events);
```

## Constraints

- **Type constraints**:
  - `GlobalSrcData::RawDType` must equal `GlobalDstData::RawDType`.
  - `GlobalSrcData::layout` must equal `GlobalDstData::layout`.
  - Element size must be 1, 2, 4, or 8 bytes.
- **Memory constraints**:
  - `srcGlobalData` must point to remote address (on source NPU).
  - `dstGlobalData` must point to local address (on current NPU).
  - Both addresses should be naturally aligned to element size; 32-byte alignment is recommended for best performance.
- **DMA constraints**:
  - SDMA: Supports 2D transfer. 
  - URMA: Supports 1D transfer.
  - DMA channel availability is limited; implementations may serialize requests when channels are exhausted.
- **Valid region**:
  - Transfer size is determined by GlobalTensor shape or explicit parameters.

## Completion Semantics

After `TSYNC(event)` returns, all writes to `dstGlobalData` performed by the asynchronous transfer are complete and visible to subsequent operations on the current NPU.

## Examples

### Basic Asynchronous GET with SDMA (default)

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_tget_async(__gm__ T* local_data, __gm__ T* remote_addr) {
    using GShape = Shape<1, 1, 1, 64, 256>;
    using GStride = BaseShape2D<T, 64, 256, Layout::ND>;
    using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

    // Remote source tensor
    GTensor srcG(remote_addr);
    
    // Local destination tensor
    GTensor dstG(local_data);
    
    // Initiate asynchronous transfer using SDMA (default)
    auto event = comm::TGET_ASYNC(dstG, srcG);
    
    // Do other computation while transfer is in progress
    // ...
    
    // Wait for transfer completion before using the data
    TSYNC(event);
}
```

### URMA 

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SIZE>
void low_latency_get(__gm__ T* recv_buf, __gm__ T* remote_send_addr) {
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>, Stride<SIZE,SIZE,SIZE,SIZE,1>, Layout::ND>;

    GTensor srcG(remote_send_addr);
    GTensor dstG(recv_buf);
    
    // Use URMA for low-latency small transfer
    auto event = comm::TGET_ASYNC<comm::DmaEngine::URMA>(dstG, srcG);
    
    // Wait for completion
    TSYNC(event);
}
```

### Prefetching Remote Data

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SIZE>
void prefetch_and_compute(__gm__ T* local_buf, __gm__ T* remote_addr, 
                          __gm__ T* compute_buf) {
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>, Stride<SIZE,SIZE,SIZE,SIZE,1>, Layout::ND>;
    using TileT = Tile<TileType::Vec, T, 1, SIZE>;

    // Start prefetching remote data
    GTensor remoteSrcG(remote_addr);
    GTensor localDstG(local_buf);
    
    auto prefetch_event = comm::TGET_ASYNC(localDstG, remoteSrcG);
    
    // Compute on local data while prefetch is in progress
    GTensor computeG(compute_buf);
    TileT computeTile;
    TLOAD(computeTile, computeG);
    // ... compute on computeTile ...
    TSTORE(computeG, computeTile);
    
    // Wait for prefetch to complete
    TSYNC(prefetch_event);
    
    // Now local_buf contains the prefetched data
    TLOAD(computeTile, localDstG);
    // ... use prefetched data ...
}
```
