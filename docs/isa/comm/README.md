# PTO Communication ISA Reference

This directory contains the per-instruction reference for the PTO Communication ISA.

- Source of truth (C++ intrinsics): `include/pto/comm/pto_comm_inst.hpp`
- Type definitions: `include/pto/comm/comm_types.hpp`

## Point-to-Point Communication (Synchronous)
- [**TPUT**](TPUT.md): Remote write (GM → UB → GM)
- [**TGET**](TGET.md): Remote read (GM → UB → GM)

## Point-to-Point Communication (Asynchronous)
- [**TPUT_ASYNC**](TPUT_ASYNC.md): Asynchronous remote write (GM → DMA engine → GM)
- [**TPUT_ASYNC_NOTIFY**](TPUT_ASYNC_NOTIFY.md): Remote write followed by a remote `int32_t` signal update
- [**TGET_ASYNC**](TGET_ASYNC.md): Asynchronous remote read (GM → DMA engine → GM)

## Signal-Based Synchronization
- [**TNOTIFY**](TNOTIFY.md): Send notification to remote NPU
- [**TWAIT**](TWAIT.md): Blocking wait for signal condition
- [**TTEST**](TTEST.md): Non-blocking test signal condition

## Collective Communication

- [**TGATHER**](TGATHER.md): Gather data from all ranks
- [**TSCATTER**](TSCATTER.md): Scatter data to all ranks
- [**TREDUCE**](TREDUCE.md): Reduce data from all ranks to local
- [**TBROADCAST**](TBROADCAST.md): Broadcast from current NPU to all ranks

## Type Definitions

### NotifyOp

Operation type for `TNOTIFY`:

| Value | Description |
|-------|-------------|
| `NotifyOp::AtomicAdd` | Atomic add (`signal += value`) |
| `NotifyOp::Set` | Direct set (`signal = value`) |

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

### AtomicType

Atomic operation type for `TPUT` (defined in `include/pto/common/constants.hpp`):

| Value | Description |
|-------|-------------|
| `AtomicType::AtomicNone` | No atomic operation (default) |
| `AtomicType::AtomicAdd` | Atomic add operation |

### DmaEngine

DMA backend selection for `TPUT_ASYNC`, `TPUT_ASYNC_NOTIFY`, and `TGET_ASYNC`:

| Value | Description |
|-------|-------------|
| `DmaEngine::SDMA` | SDMA API (A2/A3 uses SDMA; on A5, TGET uses SDMA while TPUT/TPUT_ASYNC_NOTIFY currently use an MTE fallback) |
| `DmaEngine::URMA` | URMA engine for `TPUT_ASYNC`/`TGET_ASYNC` (Ascend950 / NPU_ARCH 3510 only; requires CANN >= 9.1.0); `TPUT_ASYNC_NOTIFY<URMA>` is not implemented |
| `DmaEngine::ROCE` | Reserved for future A5 RoCE backends; current async communication instructions reject it at compile time |

### AsyncEvent

Returned by `TPUT_ASYNC` / `TPUT_ASYNC_NOTIFY` / `TGET_ASYNC`. Use to synchronize completion:

```cpp
struct AsyncEvent {
    uint64_t handle;
    DmaEngine engine;

    bool valid() const;                        // true if handle != 0
    bool Wait(const AsyncSession &session) const; // block until transfer completes
    bool Test(const AsyncSession &session) const; // non-blocking completion check
};
```

Note: A5 `TPUT_ASYNC_NOTIFY<SDMA>` currently completes synchronously and returns `handle == 0`, so its
successful event also has `valid() == false`. See [TPUT_ASYNC_NOTIFY](TPUT_ASYNC_NOTIFY.md) for the required
handling.

### AsyncSession

Engine-agnostic session for async DMA operations. Build once, pass to all async calls:

```cpp
comm::AsyncSession session;
comm::BuildAsyncSession<comm::DmaEngine::SDMA>(scratchTile, workspace, session);
```

Defined in `include/pto/comm/async_common/async_types.hpp`. See [TPUT_ASYNC](TPUT_ASYNC.md) for construction details and parameters.
For A2/A3 `TPUT_ASYNC_NOTIFY`, explicitly set `channelGroupIdx = 0` instead of using the default shown above;
see [TPUT_ASYNC_NOTIFY](TPUT_ASYNC_NOTIFY.md).

### ParallelGroup

Wrapper for collective communication across multiple NPUs:

```cpp
template <typename GlobalData>
struct ParallelGroup {
    // Pointer to an array of `GlobalData` objects (each wraps a GM address).
    // The array itself is local metadata; the wrapped addresses may refer to local or remote GM,
    // depending on the collective instruction.
    GlobalData *tensors;
    int nranks;   // Number of ranks
    int rootIdx;  // Root NPU's rank index

    // Factory function (recommended): build from an existing tensor array.
    static ParallelGroup Create(GlobalData *tensorArray, int size, int rank_id);
};
```
