# PTO Communication ISA Reference

This directory contains the per-instruction reference for the PTO Communication ISA.

- Source of truth (C++ intrinsics): `include/pto/comm/pto_comm_inst.hpp`
- Type definitions: `include/pto/comm/comm_types.hpp`

## Point-to-Point Communication (Synchronous)
- [**TPUT**](TPUT.md): Remote write (GM → UB → GM)
- [**TGET**](TGET.md): Remote read (GM → UB → GM)

## Point-to-Point Communication (Asynchronous)
- [**TPUT_ASYNC**](TPUT_ASYNC.md): Asynchronous remote write (GM → DMA engine → GM)
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

DMA backend selection for `TPUT_ASYNC` and `TGET_ASYNC`:

| Value | Description |
|-------|-------------|
| `DmaEngine::SDMA` | SDMA engine. Current async path supports flat contiguous logical 1D tensors only. |
| `DmaEngine::URMA` | URMA engine (User-level RDMA Memory Access). Supports flat contiguous logical 1D tensors only. Available on Ascend950 (NPU_ARCH 3510) only, and requires CANN Toolkit **>= 9.1.0**. |

### CollEngine

Backend engine selection for collective instructions (`TGATHER`, `TSCATTER`, `TREDUCE`, `TBROADCAST`):

| Value | Description |
|-------|-------------|
| `CollEngine::AIV` | Default. Tile-based path using `TLOAD` + compute + `TSTORE` on AI Vector. |
| `CollEngine::CCU` | AIV triggers the CKE gate; the CCU hardware performs the collective. Available on Ascend950 (NPU_ARCH 3510) only. When selected, the caller must pass a `CcuTriggerContext` as the first variadic argument. |

### CcuTriggerContext

Opaque host-to-kernel context for `CollEngine::CCU`. Host fills via `ccu::TryGet()` + `rtGetDevResAddress()` before kernel launch.

```cpp
struct CcuTriggerContext {
    uint64_t ckeSlotVA;  // from rtGetDevResAddress(dieId, ckeId)
    uint32_t mask;       // 16-bit CKE trigger mask
};
```

### AsyncEvent

Returned by `TPUT_ASYNC` / `TGET_ASYNC`:

```cpp
struct AsyncEvent {
    uint64_t handle;
    DmaEngine engine;

    bool valid() const;                           // true if handle != 0
    bool Wait(const AsyncSession &session) const; // quiet-semantics wait (see TPUT_ASYNC)
    bool Test(const AsyncSession &session) const; // non-blocking completion check
};
```

### AsyncSession

Engine-agnostic session for async DMA operations. Build once, pass to all async calls:

```cpp
comm::AsyncSession session;
comm::BuildAsyncSession<comm::DmaEngine::SDMA>(scratchTile, workspace, session);
```

Defined in `include/pto/comm/async_common/async_types.hpp`. See [TPUT_ASYNC](TPUT_ASYNC.md) for construction details and parameters.

### Signal / Signal2D / GlobalSignal

`GlobalTensor` aliases for signal-based synchronization (`TNOTIFY` / `TWAIT` / `TTEST`):

```cpp
using Signal = GlobalTensor<int32_t, Shape<1,1,1,1,1>, Stride<1,1,1,1,1>, Layout::ND>;

// Dense: stride = Cols; strided ctor takes a custom DIM_3 stride for sub-region views.
template <int Rows, int Cols> struct Signal2D;

// Generic alias for custom shape/stride cases.
template <typename E, typename S, typename St, Layout L = Layout::ND>
using GlobalSignal = GlobalTensor<E, S, St, L>;
```

### ParallelGroup

Wrapper for multi-NPU collectives (`TGATHER` / `TSCATTER` / `TREDUCE` / `TBROADCAST`):

```cpp
template <typename GlobalData>
struct ParallelGroup {
    GlobalData *tensors;   // per-rank GlobalTensor views (local or remote GM)
    int nranks;
    int rootIdx;           // all ranks must pass the same value

    static ParallelGroup Create(GlobalData *tensorArray, int size, int rootIdx);

    int  GetRootIdx() const;
    int  GetSize()    const;
    bool empty()      const;
    GlobalData&       operator[](int teamRank);
    const GlobalData& operator[](int teamRank) const;
};
```
