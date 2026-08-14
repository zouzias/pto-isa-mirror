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

DMA engine selection for `TPUT_ASYNC` and `TGET_ASYNC`:

| Value | Description |
|-------|-------------|
| `DmaEngine::SDMA` | SDMA engine (supports 1D transfer) |
| `DmaEngine::URMA` | URMA engine (supports 1D transfer, Ascend950 / NPU_ARCH 3510 only; requires CANN >= 9.1.0) |
| `DmaEngine::RDMA` | RDMA engine (supports 1D transfer, Ascend950 / NPU_ARCH 3510 only). The current NIC backend is HNS1825 and must be enabled at configure time. |

### AsyncEvent

Returned by `TPUT_ASYNC` / `TGET_ASYNC`. Use to synchronize completion:

```cpp
struct AsyncEvent {
    uint64_t handle;
    DmaEngine engine;

    bool valid() const;                        // true if handle != 0
    bool Wait(const AsyncSession &session) const; // block until transfer completes
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

### RDMA Backend and Host Control Plane

`DmaEngine::RDMA` selects RDMA communication, and `RdmaBackend` identifies the NIC implementation compiled into the
binary. One binary can contain at most one RDMA backend; the only supported value is currently
`RdmaBackend::HNS_1825`.

For the Ascend950 / NPU_ARCH 3510 communication ST build, select the backend before the first CMake configure:

```bash
export PTO_RDMA_BACKEND=HNS_1825
python3 tests/script/run_st.py -r npu -v a5 -t comm/tput_async_rdma -d -n 2
```

`PTO_RDMA_BACKEND` is a configure-time input. CMake translates it into the same host and device compile definitions;
the generated binaries do not read it at runtime. Unset, empty, and unsupported values produce a build without an
RDMA backend. Reconfigure the build directory after changing the value.

Host code includes `pto/comm/async/rdma/rdma_workspace_manager.hpp` and follows this lifecycle:

```cpp
pto::comm::rdma::RdmaWorkspaceManager manager;
if (manager.Preflight() != pto::comm::rdma::WorkspaceInitResult::READY) {
    // Disabled, or the selected backend is unsupported on this architecture.
}

pto::comm::rdma::WorkspaceConfig config;
// Fill rankId/rankCount, local and peer RDMA NIC information, and the
// per-rank registered communication-buffer addresses.
if (manager.Init(config) != pto::comm::rdma::WorkspaceInitResult::READY) {
    // Initialization failed.
}
void *rdmaWorkspace = manager.GetWorkspaceAddr();

// Launch kernels that build DmaEngine::RDMA sessions with rdmaWorkspace.

bool finalized = manager.Finalize();
```

The application is responsible for exchanging the following information consistently across ranks before `Init`:

- rank id and rank count;
- local physical device id and RDMA NIC IPv4 address;
- every rank's physical device id and RDMA NIC IPv4 address;
- every rank's device virtual address for the registered communication buffer;
- a common base port.

The manager does not perform MPI/HCCL bootstrap. It creates the HCOMM endpoint, registers the local communication
buffer, creates one RoCE channel per peer, waits for channel readiness, and publishes an RDMA device workspace with
the HNS1825 queue and MR metadata. Call `Finalize()` before freeing the registered buffer; it destroys channels,
unregisters memory, destroys the endpoint, and frees the device workspace.

Current HNS1825 constraints:

- The HNS1825 backend is limited to Ascend950 / NPU_ARCH 3510; `Preflight()` returns `ERROR` on unsupported targets.
- Both local and remote operand ranges must be fully contained in the communication buffers registered during
  `Init`. "Symmetric" here means that every rank registers the same logical region; equal device virtual addresses
  are not required because peer base addresses are exchanged explicitly.
- One transfer is limited to `0x7fffffff` bytes.
- The RDMA `scratchTile` must expose at least 64 bytes of UB storage, and `syncId` must be in `[0, 7]`.
- At most 16 ranks may share one RDMA NIC IPv4 address under the current port mapping.
- Host control-plane code depends on HCOMM. If the HNS1825 verbs provider is not installed in a default provider
  path, the deployment may also need `IBV_EXTEND_DRIVERS` to point to `libhrn5-rdmav34.so`.

`HCCL_RDMA_TC` (default `132`) and `HCCL_RDMA_SL` (default `4`) configure RoCE traffic class and service level.
`PTO_ROCE_VERBOSE=1` enables host control-plane progress logs. Root-info and MPI convenience variables used only by
the repository ST harness are documented in [tests/README.md](../../../tests/README.md); they are not runtime
selection inputs for the PTO library.

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
