# SDMA Class Usage Guide

## Overview

The `SDMA` class provides a high-level interface for SDMA (System DMA) operations in PTO communication. It encapsulates the initialization and data transfer operations, making it easy to use SDMA for asynchronous GM-to-GM transfers.

## File Structure

```
include/pto/comm/sdma/
├── sdma.hpp              # Main SDMA class definition with public interface
├── sdma_impl.hpp         # Implementation of SDMA class methods (init, wait, test)
├── sdma_host_init.h      # Host-side initialization function declarations and data structures
├── sdma_host_init.cpp    # Host-side initialization function implementations
├── sdma_device_impl.hpp  # Device-side SDMA put/get implementation
├── sdma_types.hpp        # SDMA data type definitions
└── README.md             # This document
```

## Architecture

### Call Flow

```
User Code
  ↓
TPUT_SDMA(dstGlobal, srcGlobal)
  ↓
TPUT_SDMA_IMPL() [TPut_sdma.hpp]
  ↓
sdma::SDMA::put() [sdma.hpp]
  ↓
detail::put() [sdma_device_impl.hpp]
  ↓
sdma_post_send() [sdma_device_impl.hpp]
  ↓
Hardware executes SDMA transfer
```

### Initialization Flow

```
Host Code
  ↓
sdma::SDMA::init(attributes) [sdma_impl.hpp]
  ↓
pto_sdma_init(attributes) [sdma_host_init.cpp]
  ↓
1. Create AICPU stream
  ↓
2. create_sdma_streams(): Create 40 SDMA streams
  ↓
3. Allocate 16KB shared workspace memory
  ↓
4. Copy resource info to device memory (H2D)
  ↓
5. run_aicpu_kernel(): Run AICPU kernel to init device-side mapping
  ↓
Update global state for device-side access
```

## Initialization

Before using SDMA operations, you must initialize the SDMA engine:

```cpp
#include "pto/comm/sdma/sdma.hpp"
#include "pto/comm/sdma/sdma_impl.hpp"

// Initialize SDMA (typically called once at program startup)
pto_comm_init_attr_t init_attr = {};
init_attr.my_pe = 0;           // Local PE rank
init_attr.n_pes = 8;           // Total number of PEs
init_attr.local_mem_size = 1024 * 1024 * 1024;  // 1GB local memory

bool success = pto::comm::sdma::SDMA::init(&init_attr);
if (!success) {
    // Handle initialization error
}
```

The `init()` function internally calls `pto_tput_sdma_init()` from `sdma_host_init.cpp` to set up SDMA resources including:
- 40 SDMA streams for data transfer
- 16KB shared workspace memory for AICPU and AIV communication
- Device-side resource mapping via AICPU kernel

## PUT Operations

PUT performs asynchronous remote write operations (local GM → remote GM).

### Using GlobalTensor Shape

```cpp
using namespace pto;
using GTensor = GlobalTensor<float, Shape<1,1,1,64,256>, ...>;

GTensor localSrcG(local_data);   // Source data on local PE
GTensor remoteDstG(remote_data); // Destination on remote PE

// Asynchronous transfer: local -> remote
auto event = comm::sdma::SDMA::put(remoteDstG, localSrcG);

// Do other computation while transfer is in progress...

// Wait for transfer completion
comm::sdma::SDMA::wait(event);
```

### Using Explicit Size

```cpp
// Transfer specific number of bytes
uint64_t transfer_size = numRows * numCols * sizeof(float);
auto event = comm::sdma::SDMA::put(remoteDstG, localSrcG, transfer_size);
comm::sdma::SDMA::wait(event);
```

## GET Operations

GET performs asynchronous remote read operations (remote GM → local GM).

### Using GlobalTensor Shape

```cpp
GTensor localDstG(local_data);   // Destination on local PE
GTensor remoteSrcG(remote_data); // Source data on remote PE

// Asynchronous transfer: remote -> local
auto event = comm::sdma::SDMA::get(localDstG, remoteSrcG);

// Do other computation while transfer is in progress...

// Wait for transfer completion
comm::sdma::SDMA::wait(event);
```

### Using Explicit Size

```cpp
// Transfer specific number of bytes
uint64_t transfer_size = numRows * numCols * sizeof(float);
auto event = comm::sdma::SDMA::get(localDstG, remoteSrcG, transfer_size);
comm::sdma::SDMA::wait(event);
```

## Synchronization

### Wait for Completion

```cpp
auto event = comm::sdma::SDMA::put(remoteDstG, localSrcG);
comm::sdma::SDMA::wait(event);  // Blocks until transfer completes
```

### Test Completion (Non-blocking)

```cpp
auto event = comm::sdma::SDMA::put(remoteDstG, localSrcG);
while (!comm::sdma::SDMA::test(event)) {
    // Do other work while waiting
    // ...
}
```

## Integration with TPUT_SDMA/TGET_SDMA Instructions

The `TPUT_SDMA` and `TGET_SDMA` instructions delegate to the `SDMA` class:

```cpp
// In TPut_sdma.hpp
template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL SdmaEvent TPUT_SDMA_IMPL(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal)
{
    return sdma::SDMA::put(dstGlobal, srcGlobal);
}

// In TGet_sdma.hpp
template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL SdmaEvent TGET_SDMA_IMPL(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal)
{
    return sdma::SDMA::get(dstGlobal, srcGlobal);
}
```

This allows users to use either:
- The high-level `SDMA` class API directly (`SDMA::put`, `SDMA::get`)
- The instruction interface (`TPUT_SDMA`, `TGET_SDMA`)

## Constraints

- Element types must match between source and destination
- Layouts must match between source and destination
- Element size must be 1, 2, 4, or 8 bytes
- Addresses should be 32-byte aligned for optimal performance
- SDMA must be initialized before use

## Error Handling

- If SDMA resources are unavailable, operations return an invalid `SdmaEvent` (event_id == 0)
- Invalid events can be checked by testing `event.event_id == 0`

## Data Structures

### sdma_types.hpp

| Structure | Description |
|-----------|-------------|
| `sdma_config_t` | SDMA configuration parameters |
| `workspace_layout_t` | Workspace memory layout |
| `batch_write_flag_info_t` | Flag synchronization information |
| `batch_write_channel_info_t` | Channel information |
| `batch_write_item_t` | SQE (Submission Queue Entry) structure |

### sdma_host_init.h

| Structure/Function | Description |
|--------------------|-------------|
| `pto_sdma_op_res_info_t` | SDMA operation resource information |
| `pto_host_stream_info_t` | Host stream information |
| `pto_sdma_init()` | Host-side initialization function |
| `pto_sdma_finalize()` | Release SDMA resources |

## Device-side Implementation

The `sdma_device_impl.hpp` implements the complete SDMA put functionality:

| Function | Description |
|----------|-------------|
| `sdma_post_send()` | Main function coordinating the entire SDMA transfer flow |
| `add_one_memcpy_sqe()` | Build SQE (Submission Queue Entry) |
| `init_sdma_config()` | Initialize configuration parameters |
| `prepare_workspace()` | Prepare workspace memory layout |
| `submit_data_transfer_sqes()` | Submit data transfer SQEs |
| `submit_flag_transfer_sqes()` | Submit flag synchronization SQEs |
| `flush_cache_and_ring_doorbell()` | Flush cache and ring doorbell |
| `poll_for_completion()` | Poll for transfer completion |
| `put()` | Public put interface (in detail namespace) |
| `get()` | Public get interface (in detail namespace) |

## Important Notes

1. **Host/Device Separation**:
   - `init()` method should be called on host side
   - `put()`/`get()` methods are called on device side

2. **Initialization Order**:
   - Must call `SDMA::init()` to initialize SDMA first
   - Then can use `SDMA::put()`/`SDMA::get()` for transfers

3. **Namespaces**:
   - Device-side implementation is in `pto::comm::sdma::detail` namespace
   - SDMA class is in `pto::comm::sdma` namespace

## Build Instructions

`sdma_host_init.cpp` requires linking the following libraries:

- `libruntime.so`: For `rtsStreamCreate` and `rtGetDeviceInfo`
- `libopapi.so`: For `aclnnSdmaMap` AICPU kernel
- ACL runtime library: For stream and memory management

Add the following link options when compiling:

```bash
-ldl -lascendcl
```

## Implementation Notes

- The SDMA class uses a static flag to track initialization status
- Channel selection uses a simple round-robin algorithm based on address
- Actual SDMA API calls are marked with TODO comments and need to be implemented based on Ascend SDK
