# TPUT_ASYNC

## Introduction

Asynchronous remote write operation using configurable DMA engine. Directly transfers data from local GM to remote NPU's GM. The operation returns immediately with an event handle.

## Template Parameters

- `engine`: DMA engine selection
  - `DmaEngine::SDMA` (default) - System DMA
  - `DmaEngine::URMA` - A5 URMA (UB Remote Memory Access) based on Unified Bus

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}^{\mathrm{remote}}_{i,j} = \mathrm{src}^{\mathrm{local}}_{i,j} $$

Data flow: `srcGlobal (local GM)` → `DMA Engine` → `dstGlobal (remote GM)`

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Asynchronous form:

```text
%event = tput_async<sdma> %dst_remote, %src_local : (!pto.memref<...>, !pto.memref<...>) -> !pto.event
%event = tput_async<urma> %dst_remote, %src_local : (!pto.memref<...>, !pto.memref<...>) -> !pto.event
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// Asynchronous PUT with template-specified DMA engine
template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal, WaitEvents&... events);

// With explicit size specification
template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal, 
                                uint32_t numRows, uint32_t numCols, WaitEvents&... events);
```

## Constraints

- **Type constraints**:
  - `GlobalSrcData::RawDType` must equal `GlobalDstData::RawDType`.
  - `GlobalSrcData::layout` must equal `GlobalDstData::layout`.
  - Element size must be 1, 2, 4, or 8 bytes.
- **Memory constraints**:
  - `dstGlobal` must point to remote address (on target NPU).
  - `srcGlobal` must point to local address (on current NPU).
  - Both addresses must be 32-byte aligned for optimal performance.
- **DMA constraints**:
  - SDMA: Allows 2D transfer
  - URMA: 1D transfer
  - DMA channel must be available (limited concurrent operations).
- **Valid region**:
  - Transfer size is determined by GlobalTensor shape or explicit parameters.

## Comparison with TPUT

| Feature | TPUT | TPUT_ASYNC |
|---------|------|-----------|
| Execution | Synchronous | Asynchronous |
| Data path | GM → UB → GM | GM → GM (direct) |
| UB required | Yes | No |
| Overlap | No | Yes (with computation) |
| Latency | Lower for small transfers | Lower for large transfers |
| Throughput | Limited by UB size | Higher for bulk transfers |

## Examples

### Basic Asynchronous PUT with SDMA (default)

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_tput_async(__gm__ T* local_data, __gm__ T* remote_addr, int target_npu) {
    using GShape = Shape<1, 1, 1, 64, 256>;
    using GStride = BaseShape2D<T, 64, 256, Layout::ND>;
    using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

    // Local source tensor
    GTensor srcG(local_data);
    
    // Remote destination tensor
    GTensor dstG(remote_addr);
    
    // Initiate asynchronous transfer using SDMA (default)
    auto event = comm::TPUT_ASYNC(dstG, srcG);
    
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
void low_latency_put(__gm__ T* send_buf, __gm__ T* remote_recv_addr) {
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>, Stride<SIZE,SIZE,SIZE,SIZE,1>, Layout::ND>;

    GTensor sendG(send_buf);
    GTensor recvG(remote_recv_addr);
    
    // Use URMA for low-latency small transfer
    auto event = comm::TPUT_ASYNC<comm::DmaEngine::URMA>(recvG, sendG);
    
    // Wait for completion
    TSYNC(event);
}
```

### Overlapping Communication and Computation

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SIZE>
void overlap_comm_compute(__gm__ T* send_buf, __gm__ T* remote_recv_addr, 
                          __gm__ T* compute_buf, int my_rank, int nranks) {
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>, Stride<SIZE,SIZE,SIZE,SIZE,1>, Layout::ND>;
    using TileT = Tile<TileType::Vec, T, 1, SIZE>;

    GTensor sendG(send_buf);
    GTensor recvG(remote_recv_addr);
    
    // Start asynchronous data transfer with SDMA
    auto put_event = comm::TPUT_ASYNC<comm::DmaEngine::SDMA>(recvG, sendG);
    
    // Perform local computation while transfer is in progress
    GTensor computeG(compute_buf);
    TileT tile;
    TLOAD(tile, computeG);
    // ... compute on tile ...
    TSTORE(computeG, tile);
    
    // Wait for transfer to complete before using the data
    TSYNC(put_event);
}
```

### Pipeline with Multiple Transfers

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T>
void pipelined_transfer(__gm__ T* local_buffers[], __gm__ T* remote_buffers[], 
                        int num_buffers, int target_npu) {
    using GTensor = GlobalTensor<T, Shape<1,1,1,64,64>, Stride<4096,4096,4096,64,1>, Layout::ND>;
    
    comm::AsyncEvent events[num_buffers];
    
    // Initiate all transfers
    for (int i = 0; i < num_buffers; ++i) {
        GTensor srcG(local_buffers[i]);
        GTensor dstG(remote_buffers[i]);
        
        events[i] = comm::TPUT_ASYNC(dstG, srcG);
    }
    
    // Wait for all transfers to complete
    TSYNC(events, num_buffers);
}
```
