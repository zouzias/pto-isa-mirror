<p align="center">
  <img src="../figures/pto_logo.svg" alt="PTO Tile Lib" width="180" />
</p>

# PTO Communication ISA Reference

This directory contains the per-instruction reference for the PTO Communication ISA.

- Source of truth (C++ intrinsics): `include/pto/comm/pto_comm_inst.hpp`
- Type definitions: `include/pto/comm/comm_types.hpp`

## Point-to-Point Communication (Synchronous)
- `TPUT`: `docs/isa/comm/TPUT.md` - Remote write (GM → UB → GM)
- `TGET`: `docs/isa/comm/TGET.md` - Remote read (GM → UB → GM)

## Point-to-Point Communication (Asynchronous)
- `TPUT_ASYNC`: `docs/isa/comm/TPUT_ASYNC.md` - Async remote write (GM → GM direct)
- `TGET_ASYNC`: `docs/isa/comm/TGET_ASYNC.md` - Async remote read (GM → GM direct)

## Signal-Based Synchronization
- `TNOTIFY`: `docs/isa/comm/TNOTIFY.md` - Send notification to remote NPU
- `TWAIT`: `docs/isa/comm/TWAIT.md` - Blocking wait for signal condition
- `TTEST`: `docs/isa/comm/TTEST.md` - Non-blocking test signal condition

## Collective Communication

> **Note**: Collective communication instructions may be offloaded to dedicated collective communication hardware (e.g., CCU) in future implementations. The ISA interface remains stable regardless of the underlying execution engine.

- `TGATHER`: `docs/isa/comm/TGATHER.md` - Gather data from all ranks
- `TSCATTER`: `docs/isa/comm/TSCATTER.md` - Scatter data to all ranks
- `TREDUCE`: `docs/isa/comm/TREDUCE.md` - Reduce data from all ranks to local
- `TBROADCAST`: `docs/isa/comm/TBROADCAST.md` - Broadcast from current NPU to all ranks

## Type Definitions

### DmaEngine

DMA engine selection for `TPUT_ASYNC` / `TGET_ASYNC`:

| Value | Description |
|-------|-------------|
| `DmaEngine::SDMA` | System DMA - high bandwidth, for large transfers (>4KB) |
| `DmaEngine::URMA` | User-space RMA - low latency, for small transfers (<4KB) |

### AsyncEvent

Event handle returned by asynchronous operations. Use `TSYNC(event)` to wait:

```cpp
struct AsyncEvent {
    uint64_t handle;
    DmaEngine engine;
    bool valid() const;
};

// Usage:
AsyncEvent event = comm::TPUT_ASYNC<DmaEngine::SDMA>(dst, src);
TSYNC(event);  // Wait for completion
```

### NotifyOp

Operation type for `TNOTIFY`:

| Value | Description |
|-------|-------------|
| `NotifyOp::Set` | Direct set (`signal = value`) |
| `NotifyOp::AtomicAdd` | Atomic add (`signal += value`) |

### WaitCmp

Comparison operators for `TWAIT` and `TTEST`:

| Value | Description |
|-------|-------------|
| `WaitCmp::EQ` | Equal (`==`) |
| `WaitCmp::NE` | Not equal (`!=`) |
| `WaitCmp::GT` | Greater than (`>`) |
| `WaitCmp::GE` | Greater or equal (`>=`) |
| `WaitCmp::LT` | Less than (`<`) |
| `WaitCmp::LE` | Less or equal (`<=`) |

```cpp
// Usage (unified runtime parameter style):
comm::TNOTIFY(signal, 1, comm::NotifyOp::Set);
comm::TWAIT(signal, 1, comm::WaitCmp::EQ);
comm::TTEST(signal, 1, comm::WaitCmp::GE);
```

### ReduceOp

Reduction operators for `TREDUCE`:

| Value | Description |
|-------|-------------|
| `ReduceOp::Sum` | Element-wise sum |
| `ReduceOp::Max` | Element-wise maximum |
| `ReduceOp::Min` | Element-wise minimum |

### ParallelGroup

Wrapper for collective communication across multiple NPUs:

```cpp
template <typename GlobalData>
struct ParallelGroup {
    GlobalData *tensors;   // Array of GlobalTensors (not pointers)
    int nranks;            // Number of ranks
    int my_rank;           // Current NPU's rank
    
    // Factory function (recommended)
    static ParallelGroup Create(__gm__ T** addrs, int nranks, int my_rank);
};
```
