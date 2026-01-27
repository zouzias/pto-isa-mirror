<p align="center">
  <img src="../figures/pto_logo.svg" alt="PTO Tile Lib" width="180" />
</p>

# PTO Communication ISA Reference

This directory contains the per-instruction reference for the PTO Communication ISA.

- Source of truth (C++ intrinsics): `include/pto/comm/pto_comm_inst.hpp`
- Type definitions: `include/pto/comm/comm_types.hpp`

## Point-to-Point Communication (Synchronous, MTE-based)
- `TPUT`: `docs/isa/comm/TPUT.md` - Remote write operation (GM → UB → GM)
- `TGET`: `docs/isa/comm/TGET.md` - Remote read operation (GM → UB → GM)

## Point-to-Point Communication (Asynchronous, SDMA-based)
- `TPUT_SDMA`: `docs/isa/comm/TPUT_SDMA.md` - Asynchronous remote write (GM → GM direct)
- `TGET_SDMA`: `docs/isa/comm/TGET_SDMA.md` - Asynchronous remote read (GM → GM direct)

## Signal-Based Synchronization
- `TNOTIFY`: `docs/isa/comm/TNOTIFY.md` - Send flag notification to remote PE
- `TWAIT`: `docs/isa/comm/TWAIT.md` - Wait until signal meets condition
- `TTEST`: `docs/isa/comm/TTEST.md` - Non-blocking test if signal meets condition

## Collective Communication
- `TBARRIER`: `docs/isa/comm/TBARRIER.md` - Global barrier synchronization
- `TALLREDUCE`: `docs/isa/comm/TALLREDUCE.md` - All-reduce operation
- `TALLGATHER`: `docs/isa/comm/TALLGATHER.md` - All-gather operation
- `TBROADCAST`: `docs/isa/comm/TBROADCAST.md` - Broadcast from root rank

## Type Definitions

### NotifyOp

Operation type for `TNOTIFY`:

| Value | Description |
|-------|-------------|
| `NotifyOp::AtomicAdd` | Atomic add operation |
| `NotifyOp::Set` | Direct set operation |

### WaitCmp

Comparison operators for `TWAIT` and `TTEST`:

| Value | Description |
|-------|-------------|
| `WaitCmp::EQ` | Equal (`==`) |
| `WaitCmp::NE` | Not equal (`!=`) |
| `WaitCmp::GT` | Greater than (`>`) |
| `WaitCmp::GE` | Greater than or equal (`>=`) |
| `WaitCmp::LT` | Less than (`<`) |
| `WaitCmp::LE` | Less than or equal (`<=`) |

### ParallelGroup

Lightweight wrapper for collective communication across multiple PEs:

```cpp
template <typename GlobalData>
struct ParallelGroup {
    GlobalData **tensors;  // Array of GlobalTensor pointers
    int nranks;            // Number of ranks in the group
    int my_rank;           // Current PE's rank
};
```
