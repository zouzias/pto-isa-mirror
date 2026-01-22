# pto::comm Communication Extension

## Overview

`pto::comm` extends the PTO tile library with inter-tile communication capabilities for Ascend NPUs, designed to support distributed computing frameworks like Triton-Distributed and IRIS.

### Key Features

- **Point-to-Point Communication**: `TPUT`, `TGET` for direct data transfer between PEs
- **Synchronization Primitives**: `TQUIET`, `TBARRIER`, `TNOTIFY`, `TWAIT`, `TTEST`
- **Collective Operations**: `TALLREDUCE`, `TALLGATHER`, `TBROADCAST`
- **Multi-Backend Support**: Currently supports SHMEM backend (CANN-SHMEM / ASCEND-SHMEM)
- **Unified Context Management**: Symmetric heap allocation, rank management

## Instruction Reference

### Point-to-Point Instructions

| Instruction | Description | Signature |
|-------------|-------------|-----------|
| `TPUT` | Non-blocking put to remote PE | `TPUT(dstGlobal, srcGlobal)` |
| `TGET` | Non-blocking get from remote PE | `TGET(dstGlobal, srcGlobal)` |
| `TQUIET` | Wait for all outstanding put/get to complete | `TQUIET()` |

### Synchronization Instructions

| Instruction | Description | Signature |
|-------------|-------------|-----------|
| `TBARRIER` | Global barrier synchronization | `TBARRIER()` |
| `TNOTIFY` | Send flag notification to remote PE | `TNOTIFY<op>(signal, value)` |
| `TWAIT` | Blocking wait until signal meets condition | `TWAIT<cmp>(signal, cmpValue)` |
| `TTEST` | Non-blocking test if signal meets condition | `TTEST<cmp>(signal, cmpValue) -> bool` |

### Collective Instructions

| Instruction | Description | Signature |
|-------------|-------------|-----------|
| `TALLREDUCE` | Reduce across all PEs, result on all PEs | `TALLREDUCE(parallelGroup, dstGlobal)` |
| `TALLGATHER` | Gather data from all PEs to all PEs | `TALLGATHER(parallelGroup, dstGlobal)` |
| `TBROADCAST` | Broadcast from root PE to all PEs | `TBROADCAST(parallelGroup, srcGlobal, root)` |

## Type Definitions

### NotifyOp (Notification Operation)

```cpp
enum class NotifyOp : uint8_t {
    AtomicAdd = 0,  // Atomic add operation
    Set = 1,        // Direct set operation
};
```

### WaitCmp (Comparison Operators)

```cpp
enum class WaitCmp : uint8_t {
    EQ = 0,  // Equal (==)
    NE = 1,  // Not equal (!=)
    GT = 2,  // Greater than (>)
    GE = 3,  // Greater than or equal (>=)
    LT = 4,  // Less than (<)
    LE = 5,  // Less than or equal (<=)
};
```

### ParallelGroup

A lightweight wrapper for collective operations:

```cpp
template <typename GlobalData>
struct ParallelGroup {
    GlobalData **tensors;  // Array of GlobalTensor pointers
    int nranks;            // Number of participating ranks
    int my_rank;           // This PE's rank in the group
};
```

## File Structure

```
include/pto/comm/
├── pto_comm_inst.hpp      # Public API entry point
├── pto_comm_instr_impl.hpp# Implementation includes
├── comm_types.hpp         # Type definitions (enums, Copy2DParams, ParallelGroup)
├── context_manager.hpp    # Context initialization and memory management
├── TPut.hpp               # TPUT implementation
├── TGet.hpp               # TGET implementation
├── TQuiet.hpp             # TQUIET implementation
├── TBarrier.hpp           # TBARRIER implementation
├── TNotify.hpp            # TNOTIFY implementation
├── TWait.hpp              # TWAIT implementation
├── TTest.hpp              # TTEST implementation
├── TAllReduce.hpp         # TALLREDUCE implementation
├── TAllGather.hpp         # TALLGATHER implementation
├── TBroadCast.hpp         # TBROADCAST implementation
└── backend/
    ├── backend_selector.hpp
    └── shmem/
        └── shmem_backend.hpp  # SHMEM backend implementation
```

## Usage Examples

### Host-side Initialization

```cpp
#include "pto/comm/context_manager.hpp"

int world_rank = /* from launcher */;
int world_size = /* from launcher */;

pto::comm::InitOptions opts;
opts.backend = pto::comm::BackendKind::Shmem;
opts.rank = world_rank;
opts.size = world_size;
opts.symmetricHeapBytes = 256 * 1024 * 1024;  // 256MB symmetric heap
opts.ipPort = "tcp://127.0.0.1:8765";

int ret = pto::comm::ContextManager::Init(opts);

// Allocate symmetric memory
void* ptr = pto::comm::ContextManager::SymmetricAlloc(bytes);
// ... use ptr ...
pto::comm::ContextManager::SymmetricFree(ptr);

// Cleanup before exit
pto::comm::ContextManager::Finalize();
```

### Device-side Put/Get

```cpp
#include "pto/comm/pto_comm_inst.hpp"
#include "pto/common/pto_tile.hpp"

__global__ AICORE void MyKernel(__gm__ float* srcPtr, __gm__ float* dstPtr) {
    using GData = pto::GlobalTensor<float, MyShape, MyStride, pto::Layout::ND>;
    
    GData src(srcPtr, shape, stride);
    GData dst(dstPtr, shape, stride);
    
    int peer = (shmem_my_pe() + 1) % shmem_n_pes();
    dst.SetRank(peer);
    
    pto::comm::TPUT(dst, src);  // Non-blocking put
    pto::comm::TQUIET();        // Wait for completion
    pto::comm::TBARRIER();      // Global sync
}
```

### Signal-based Synchronization (TNOTIFY + TWAIT)

```cpp
__global__ AICORE void ProducerConsumerKernel(__gm__ int32_t* signal, __gm__ float* data) {
    using GSignal = pto::GlobalTensor<int32_t, Shape1D, Stride1D, pto::Layout::ND>;
    
    int my_rank = shmem_my_pe();
    GSignal sig(signal, shape, stride);
    
    if (my_rank == 0) {  // Producer
        // ... produce data ...
        
        // Notify consumer that data is ready
        sig.SetRank(1);
        pto::comm::TNOTIFY<pto::comm::NotifyOp::Set>(sig, 1);
        pto::comm::TQUIET();
    } else {  // Consumer
        sig.SetRank(my_rank);
        
        // Wait until signal == 1
        pto::comm::TWAIT<pto::comm::WaitCmp::EQ>(sig, 1);
        
        // ... consume data ...
    }
}
```

### Non-blocking Test (TTEST) with Polling

```cpp
__global__ AICORE void PollingKernel(__gm__ int32_t* signal) {
    using GSignal = pto::GlobalTensor<int32_t, Shape1D, Stride1D, pto::Layout::ND>;
    
    GSignal sig(signal, shape, stride);
    sig.SetRank(shmem_my_pe());
    
    int polls = 0;
    const int maxPolls = 10000;
    
    while (polls < maxPolls) {
        polls++;
        // Non-blocking check
        if (pto::comm::TTEST<pto::comm::WaitCmp::GE>(sig, 100)) {
            break;  // Condition met
        }
        // ... do other work while waiting ...
    }
}
```

### Collective Operations

```cpp
__global__ AICORE void CollectiveKernel(__gm__ float* tensors[], __gm__ float* result) {
    using GData = pto::GlobalTensor<float, MyShape, MyStride, pto::Layout::ND>;
    
    int nranks = shmem_n_pes();
    int my_rank = shmem_my_pe();
    
    // Setup parallel group
    GData* tensorPtrs[MAX_RANKS];
    for (int i = 0; i < nranks; i++) {
        tensorPtrs[i] = new GData(tensors[i], shape, stride);
        tensorPtrs[i]->SetRank(i);
    }
    
    pto::comm::ParallelGroup<GData> pg(tensorPtrs, nranks, my_rank);
    GData dst(result, shape, stride);
    
    // All-reduce sum
    pto::comm::TALLREDUCE(pg, dst);
}
```

## Test Cases

Test cases are located in `tests/comm/st/testcase/`:

| Directory | Instructions Tested | Description |
|-----------|---------------------|-------------|
| `tput/` | TPUT, TQUIET | Point-to-point put operations |
| `tget/` | TGET, TQUIET | Point-to-point get operations |
| `tnotify/` | TNOTIFY, TBARRIER | Flag notification (AtomicAdd, Set, Scoreboard) |
| `twait/` | TWAIT, TNOTIFY | Blocking wait with various comparisons |
| `ttest/` | TTEST | Non-blocking test with all comparison operators |
| `tallreduce/` | TALLREDUCE | All-reduce collective |
| `tallgather/` | TALLGATHER | All-gather collective |
| `tbroadcast/` | TBROADCAST | Broadcast collective |
| `shmem_init/` | - | SHMEM initialization test |
| `shmem_malloc/` | - | Symmetric memory allocation test |

### Running Tests

```bash
# Set environment
source set_env.sh

# Run specific test
python3 tests/script/run_st.py -r comm -v a3 -t tput
python3 tests/script/run_st.py -r comm -v a3 -t tnotify
python3 tests/script/run_st.py -r comm -v a3 -t twait
python3 tests/script/run_st.py -r comm -v a3 -t ttest
```

## SHMEM Backend Configuration

### Dependencies

- CANN-SHMEM library: https://gitee.com/ascend/shmem
- Required headers: `shmem.h` or `shmem_api.h`

### Runtime Configuration

| Option | Description |
|--------|-------------|
| `symmetricHeapBytes` | Symmetric heap size (default: 8MB) |
| `rank` | This PE's rank ID |
| `size` | Total number of PEs |
| `ipPort` | Out-of-band coordination address (e.g., `tcp://127.0.0.1:8765`) |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Initialization failed | Check `rank`, `size`, `ipPort` parameters; verify SHMEM library installation |
| Transfer errors | Ensure src/dst GlobalTensor have matching layout and element types |
| Out of memory | Increase `symmetricHeapBytes` |
| Race conditions | Add `TQUIET()` after TPUT/TGET; use `TBARRIER()` for global sync |
| Signal not received | Ensure `SetRank()` is correctly set on GlobalTensor; verify TNOTIFY/TWAIT pairing |

## Data Layout Support

- Supported layouts: `ND`, `DN`, `NZ`
- Source and destination must have matching layouts
- Element types must match (supported: `float`, `half`, `bfloat16`, `int32_t`, etc.)
