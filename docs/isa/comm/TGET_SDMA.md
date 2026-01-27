# TGET_SDMA

## Introduction

Asynchronous remote read operation using SDMA (System DMA) engine. Directly transfers data from remote PE's GM to local GM without UB staging. The operation returns immediately and completes in the background, allowing computation-communication overlap.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}^{\mathrm{local}}_{i,j} = \mathrm{src}^{\mathrm{remote}}_{i,j} $$

Data flow: `srcGlobal (remote GM)` → `SDMA` → `dstGlobal (local GM)`

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Asynchronous form:

```text
%event = tget_sdma %dst_local, %src_remote : (!pto.memref<...>, !pto.memref<...>) -> !pto.event
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// Asynchronous SDMA GET - returns event for synchronization
template <typename GlobalDstData, typename GlobalSrcData>
PTO_INST SdmaEvent TGET_SDMA(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal);

// With explicit size specification
template <typename GlobalDstData, typename GlobalSrcData>
PTO_INST SdmaEvent TGET_SDMA(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal,
                              uint32_t numRows, uint32_t numCols);
```

## Constraints

- **Type constraints**:
  - `GlobalSrcData::RawDType` must equal `GlobalDstData::RawDType`.
  - `GlobalSrcData::layout` must equal `GlobalDstData::layout`.
  - Element size must be 1, 2, 4, or 8 bytes.
- **Memory constraints**:
  - `srcGlobal` must point to remote PE's symmetric memory (obtained via `ShmemPtr`).
  - `dstGlobal` must point to local PE's symmetric memory.
  - Both addresses must be 32-byte aligned for optimal performance.
- **SDMA constraints**:
  - Maximum transfer size per operation: implementation-defined (typically 64MB).
  - SDMA channel must be available (limited concurrent operations).
- **Valid region**:
  - Transfer size is determined by GlobalTensor shape or explicit parameters.

## Comparison with TGET

| Feature | TGET | TGET_SDMA |
|---------|------|-----------|
| Execution | Synchronous | Asynchronous |
| Data path | GM → UB → GM | GM → GM (direct) |
| UB required | Yes | No |
| Overlap | No | Yes (with computation) |
| Latency | Lower for small transfers | Lower for large transfers |
| Throughput | Limited by UB size | Higher for bulk transfers |

## Examples

### Basic Asynchronous GET

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_tget_sdma(__gm__ T* local_data, __gm__ T* remote_data, int remote_pe) {
    using GShape = Shape<1, 1, 1, 64, 256>;
    using GStride = BaseShape2D<T, 64, 256, Layout::ND>;
    using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

    // Remote source tensor (address obtained via ShmemPtr)
    __gm__ T* remote_addr = ShmemPtr(remote_data, remote_pe);
    GTensor srcG(remote_addr);
    
    // Local destination tensor
    GTensor dstG(local_data);
    
    // Initiate asynchronous transfer
    auto event = comm::TGET_SDMA(dstG, srcG);
    
    // Do other computation while transfer is in progress
    // ...
    
    // Wait for transfer completion before using the data
    comm::TWAIT_SDMA(event);
}
```

### Prefetching Remote Data

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SIZE>
void prefetch_and_compute(__gm__ T* local_buf, __gm__ T* remote_buf, 
                          __gm__ T* compute_buf, int remote_pe) {
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>, Stride<SIZE,SIZE,SIZE,SIZE,1>, Layout::ND>;
    using TileT = Tile<TileType::Vec, T, 1, SIZE>;

    // Start prefetching remote data
    __gm__ T* remote_addr = ShmemPtr(remote_buf, remote_pe);
    GTensor remoteSrcG(remote_addr);
    GTensor localDstG(local_buf);
    
    auto prefetch_event = comm::TGET_SDMA(localDstG, remoteSrcG);
    
    // Compute on local data while prefetch is in progress
    GTensor computeG(compute_buf);
    TileT tile;
    TLOAD(tile, computeG);
    // ... compute on tile ...
    TSTORE(computeG, tile);
    
    // Wait for prefetch to complete
    comm::TWAIT_SDMA(prefetch_event);
    
    // Now local_buf contains the prefetched data
    TLOAD(tile, localDstG);
    // ... use prefetched data ...
}
```

### Double Buffering Pattern

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SIZE>
void double_buffer_processing(__gm__ T* buffer_a, __gm__ T* buffer_b,
                              __gm__ T* remote_data[], int num_chunks, int remote_pe) {
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>, Stride<SIZE,SIZE,SIZE,SIZE,1>, Layout::ND>;
    using TileT = Tile<TileType::Vec, T, 1, SIZE>;

    GTensor bufA(buffer_a);
    GTensor bufB(buffer_b);
    
    // Initial fetch into buffer A
    __gm__ T* remote0 = ShmemPtr(remote_data[0], remote_pe);
    GTensor remoteSrc0(remote0);
    auto event_a = comm::TGET_SDMA(bufA, remoteSrc0);
    comm::TWAIT_SDMA(event_a);
    
    for (int i = 1; i < num_chunks; ++i) {
        // Determine current and next buffers
        GTensor& curr_buf = (i % 2 == 1) ? bufA : bufB;
        GTensor& next_buf = (i % 2 == 1) ? bufB : bufA;
        
        // Start fetching next chunk into next buffer
        __gm__ T* remote_i = ShmemPtr(remote_data[i], remote_pe);
        GTensor remoteSrcI(remote_i);
        auto fetch_event = comm::TGET_SDMA(next_buf, remoteSrcI);
        
        // Process current buffer while fetching
        TileT tile;
        TLOAD(tile, curr_buf);
        // ... process tile ...
        TSTORE(curr_buf, tile);
        
        // Wait for fetch before next iteration
        comm::TWAIT_SDMA(fetch_event);
    }
    
    // Process final chunk
    GTensor& final_buf = (num_chunks % 2 == 1) ? bufA : bufB;
    TileT tile;
    TLOAD(tile, final_buf);
    // ... process final tile ...
}
```

### Gather from Multiple PEs

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int CHUNK_SIZE>
void gather_from_all(__gm__ T* local_result, __gm__ T* remote_chunks[], 
                     int my_rank, int nranks) {
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,CHUNK_SIZE>, 
                                  Stride<CHUNK_SIZE,CHUNK_SIZE,CHUNK_SIZE,CHUNK_SIZE,1>, Layout::ND>;

    SdmaEvent events[nranks];
    
    // Initiate all GET operations
    for (int pe = 0; pe < nranks; ++pe) {
        if (pe == my_rank) continue;  // Skip self
        
        __gm__ T* remote_addr = ShmemPtr(remote_chunks[pe], pe);
        GTensor remoteSrcG(remote_addr);
        
        // Each PE's data goes to different offset in local_result
        __gm__ T* local_offset = local_result + pe * CHUNK_SIZE;
        GTensor localDstG(local_offset);
        
        events[pe] = comm::TGET_SDMA(localDstG, remoteSrcG);
    }
    
    // Wait for all transfers to complete
    for (int pe = 0; pe < nranks; ++pe) {
        if (pe == my_rank) continue;
        comm::TWAIT_SDMA(events[pe]);
    }
}
```

### Bidirectional Exchange

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SIZE>
void bidirectional_exchange(__gm__ T* send_buf, __gm__ T* recv_buf,
                            int partner_pe) {
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>, Stride<SIZE,SIZE,SIZE,SIZE,1>, Layout::ND>;

    // Send to partner (PUT)
    GTensor sendG(send_buf);
    __gm__ T* remote_recv = ShmemPtr(recv_buf, partner_pe);
    GTensor remoteDstG(remote_recv);
    auto put_event = comm::TPUT_SDMA(remoteDstG, sendG);
    
    // Receive from partner (GET)
    __gm__ T* remote_send = ShmemPtr(send_buf, partner_pe);
    GTensor remoteSrcG(remote_send);
    GTensor localRecvG(recv_buf);
    auto get_event = comm::TGET_SDMA(localRecvG, remoteSrcG);
    
    // Wait for both operations to complete
    comm::TWAIT_SDMA(put_event);
    comm::TWAIT_SDMA(get_event);
}
```
