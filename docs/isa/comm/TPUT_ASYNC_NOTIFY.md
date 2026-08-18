# TPUT_ASYNC_NOTIFY

## Introduction

`TPUT_ASYNC_NOTIFY` is an asynchronous remote write with notification. It transfers a non-empty payload from local
GM to remote GM and then updates one remote 32-bit signal. The returned `AsyncEvent` represents completion of both
parts of the operation.

Data flow:

```text
srcGlobalData (local GM) -> DMA engine -> dstGlobalData (remote GM)
                                             |
                                      after the payload
                                             v
                                  dstSignalData (remote GM)
```

The caller supplies already-translated remote addresses for both the payload destination and the signal. This
instruction does not allocate signal memory or translate a rank into a remote address. Use `TNOTIFY` instead when
only a signal update is required.

## Template Parameter and Backend Support

- `engine` selects the DMA backend at compile time:

| Engine | Platform or backend restriction | Supported `NotifyOp` |
|---|---|---|
| `DmaEngine::SDMA` (default) | A2/A3 and A5 | `Set`, `AtomicAdd` |
| `DmaEngine::URMA` | A5 (Ascend950), NPU_ARCH 3510 only; requires CANN Toolkit >= 9.1.0 | `Set`, `AtomicAdd` |
| `DmaEngine::RDMA` | A5 (Ascend950), NPU_ARCH 3510 only; HNS1825 RoCE NIC platform | `Set` only; `AtomicAdd` is unsupported |

The engine is selected at compile time. RDMA builds must set `PTO_RDMA_BACKEND=HNS_1825` when configuring CMake;
this option is not a runtime engine selector.

## C++ Intrinsic

The public intrinsic is declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData,
          typename GlobalSrcData,
          typename GlobalSignalData,
          typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC_NOTIFY(GlobalDstData &dstGlobalData,
                                      GlobalSrcData &srcGlobalData,
                                      GlobalSignalData &dstSignalData,
                                      int32_t signalValue,
                                      NotifyOp notifyOp,
                                      const AsyncSession &session,
                                      WaitEvents &... events);
```

On A5, an overload with an explicit destination peer is also provided:

```cpp
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData,
          typename GlobalSrcData,
          typename GlobalSignalData,
          typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC_NOTIFY(GlobalDstData &dstGlobalData,
                                      GlobalSrcData &srcGlobalData,
                                      GlobalSignalData &dstSignalData,
                                      int32_t signalValue,
                                      NotifyOp notifyOp,
                                      const AsyncSession &session,
                                      uint32_t peer,
                                      WaitEvents &... events);
```

For URMA and RDMA, `peer` selects the per-peer queue and remote-memory metadata. Both `dstGlobalData` and
`dstSignalData` must refer to that peer. For SDMA, `peer` is ignored because the remote virtual addresses are carried
by the global tensors.

The optional `events` parameter pack accepts zero or more PTO pipeline events that provide a zero-argument `Wait()`.
The operation waits for all of them before issuing the payload transfer. `AsyncEvent` itself requires
`Wait(session)`, so it cannot be passed directly in this pack; wait for a preceding `AsyncEvent` explicitly with the
session. Passing no dependency event is valid.

Compact syntax may show the tail as `session [, peer] [, events...]`. Square brackets mean that the enclosed
argument is optional; the brackets are documentation notation and are not written in C++ code.

## Parameters

| Parameter | Description |
|---|---|
| `dstGlobalData` | Destination of the payload in remote GM. |
| `srcGlobalData` | Source of the payload in local GM. |
| `dstSignalData` | One `int32_t` signal in remote GM. `comm::Signal` is the recommended type. It wraps a caller-provided address and does not allocate memory. |
| `signalValue` | Value assigned by `Set`, or addend used by `AtomicAdd`. |
| `notifyOp` | Signal update operation: `NotifyOp::Set` or `NotifyOp::AtomicAdd`. |
| `session` | Existing async session built for the selected `engine`. |
| `peer` | Optional explicit destination rank on A5. Recommended for URMA and RDMA multi-peer use. |
| `events` | Zero or more prerequisite PTO pipeline events with a zero-argument `Wait()`. |

The return value is an `AsyncEvent` whose completion covers both the payload transfer and the signal update.

### Signal and `signalValue`

`comm::Signal` is an alias for a one-element `int32_t` global tensor:

```cpp
using Signal = GlobalTensor<int32_t,
                            Shape<1, 1, 1, 1, 1>,
                            Stride<1, 1, 1, 1, 1>,
                            Layout::ND>;
```

Constructing a `Signal` wraps a caller-provided GM address; it does not allocate or initialize the underlying
`int32_t`. The caller initializes the signal according to the application protocol. `signalValue` is the value
published by `Set`, or the signed increment applied by `AtomicAdd`. A receiver normally observes this value with
`TWAIT` or `TTEST` before consuming the associated payload.

## Operation Semantics

For one invocation, the observable order is:

1. Wait for all supplied dependency events.
2. Transfer the complete payload from `srcGlobalData` to `dstGlobalData`.
3. After the payload has been delivered to remote GM, update `dstSignalData` as selected by `notifyOp`.

For `NotifyOp::Set`:

$$
\mathrm{signal}^{\mathrm{remote}} = \mathrm{signalValue}
$$

For `NotifyOp::AtomicAdd`:

$$
\mathrm{signal}^{\mathrm{remote}} \mathrel{+}= \mathrm{signalValue} \quad (\text{atomic})
$$

`NotifyOp::AtomicAdd` is unsupported by `DmaEngine::RDMA`.

Atomicity applies only to the signal update performed by `AtomicAdd`; it does not make payload writes atomic.
`Set` defines the value written by one invocation but provides no winner or counting guarantee when multiple
producers concurrently update the same signal. Do not mix `Set`, `AtomicAdd`, or ordinary stores concurrently on one
signal unless the application provides an additional synchronization protocol.

The ordering above applies to the payload and signal update of the same invocation. It does not define ordering
between independent operations issued from different sessions or execution flows.

## AsyncSession Construction

Build an `AsyncSession` in the AICore kernel with the overload for the selected engine. Do not populate session
fields manually. Check the Boolean result before issuing the operation, and keep the session and all resources it
references alive until the associated events have completed.

### SDMA Construction (default)

```cpp
template <DmaEngine engine = DmaEngine::SDMA, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(
    ScratchTile &scratchTile,
    __gm__ uint8_t *workspace,
    AsyncSession &session,
    uint32_t syncId = 0,
    const sdma::SdmaBaseConfig &baseConfig = {
        sdma::kDefaultSdmaBlockBytes, 0, 1},
    uint32_t channelGroupIdx = sdma::kAutoChannelGroupIdx);
```

| Parameter | Default | Description |
|---|---|---|
| `scratchTile` | - | UB/Vec scratch tile used for SDMA control and synchronization metadata. |
| `workspace` | - | GM pointer initialized by the host-side `SdmaWorkspaceManager`. |
| `session` | - | Output session. |
| `syncId` | `0` | MTE3/MTE2 synchronization event ID in the range 0-7. Use another ID if the kernel already uses this one. |
| `baseConfig` | `{kDefaultSdmaBlockBytes, 0, 1}` | SDMA block bytes, communication-block offset, and queue count. |
| `channelGroupIdx` | `kAutoChannelGroupIdx` | Channel group. The default derives it from `get_block_idx()`; assign isolated groups to concurrent sessions. |

The host-side manager is declared in `include/pto/comm/async/sdma/sdma_workspace_manager.hpp`.

### URMA Construction (NPU_ARCH 3510 only)

```cpp
#ifdef PTO_URMA_SUPPORTED
template <DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(
    __gm__ uint8_t *workspace,
    AsyncSession &session);

template <DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(
    __gm__ uint8_t *workspace,
    uint32_t destRankId,
    AsyncSession &session);
#endif
```

| Parameter | Description |
|---|---|
| `workspace` | GM pointer initialized by the host-side `UrmaWorkspaceManager`. |
| `destRankId` | Destination rank bound to the compatibility session overload. |
| `session` | Output session. |

For new A5 code, prefer the peer-independent overload without `destRankId`, then pass the destination rank to the
explicit-`peer` `TPUT_ASYNC_NOTIFY` overload. This lets one session communicate with multiple peers. The peer-bound
builder remains available with the common instruction overload for compatibility.

URMA does not require `scratchTile`. It requires CANN Toolkit >= 9.1.0. The symmetric data buffer passed to
`UrmaWorkspaceManager::Init()` must use huge-page backing, allocated with `ACL_MEM_MALLOC_HUGE_ONLY`; a small
allocation made with `ACL_MEM_MALLOC_HUGE_FIRST` may fall back to 4 KB pages and fail memory registration. The
host-side manager is declared in `include/pto/comm/async/urma/urma_workspace_manager.hpp`.

### RDMA Construction (NPU_ARCH 3510 only)

```cpp
#ifdef PTO_RDMA_SUPPORTED
template <DmaEngine engine, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(
    ScratchTile &scratchTile,
    __gm__ uint8_t *workspace,
    uint32_t myPe,
    AsyncSession &session,
    uint32_t syncId = 0);

template <DmaEngine engine, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(
    ScratchTile &scratchTile,
    __gm__ uint8_t *workspace,
    uint32_t destRankId,
    uint32_t myPe,
    AsyncSession &session,
    uint32_t syncId = 0);
#endif
```

| Parameter | Description |
|---|---|
| `scratchTile` | UB/Vec tile used for RDMA control data; at least 64 bytes. |
| `workspace` | GM pointer returned by the host-side RDMA initialization flow. |
| `destRankId` | Destination rank bound to the compatibility session overload. |
| `myPe` | Local rank used to select the registered local-memory metadata. |
| `session` | Output session. |
| `syncId` | MTE/scalar synchronization event ID in the range 0-7. |

For new A5 code, prefer the peer-independent overload without `destRankId`, then use the explicit-`peer` instruction
overload. The peer-bound builder remains available with the common instruction overload for compatibility. The
host-side interface is declared in `include/pto/comm/async/rdma/rdma_workspace_manager.hpp`.

### `scratchTile` Role and Size

`scratchTile` does not hold user payload. SDMA uses it for queue-control and event metadata; RDMA uses it for
WQE/CQE control data. The payload moves directly between GM buffers.

- The tile must be a `pto::Tile` in UB/Vec memory (`ScratchTile::Loc == TileType::Vec`).
- SDMA requires at least 8 available bytes. The recommended type is
  `Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>` (256 bytes).
- RDMA requires at least 64 available bytes.
- Keep the tile alive until all events issued through its session have completed.

## Constraints

- `GlobalSrcData::RawDType == GlobalDstData::RawDType`.
- `GlobalSrcData::layout == GlobalDstData::layout`.
- Source and destination payload tensors must be flat, contiguous logical 1D tensors.
- The destination element capacity must be at least the source element count. The payload byte count is derived from
  the source tensor's logical element count and element width.
- The payload size must be greater than zero. Use `TNOTIFY` for a signal-only operation.
- `dstSignalData` must represent exactly one non-null, 4-byte-aligned `int32_t` in remote GM. The caller must
  allocate and initialize it before use.
- The payload destination range and `dstSignalData` must not overlap.
- For URMA and RDMA, the payload destination and signal must belong to the same remote peer selected by the session
  or explicit `peer` argument.
- For URMA and RDMA, the complete local payload, remote payload, and remote signal ranges must be covered by memory
  regions registered during host initialization.
- One RDMA payload must not exceed `0x7fffffff` bytes.
- RDMA supports only `NotifyOp::Set`; `NotifyOp::AtomicAdd` must not be used.
- `session.valid` must be `true`, and the session engine must match the `engine` template argument.
- Keep the session, workspace, and any scratch tile alive until all associated events have completed.
- Every `events` argument must provide a zero-argument `Wait()` method.

## Completion and Ordering Semantics

The returned event has the same engine recorded by the issuing session. Pass that same session to both methods:

- `event.Wait(session)` blocks and returns `true` when the covered operations complete, or `false` on a detected
  completion failure.
- `event.Test(session)` performs the corresponding non-blocking completion check.
- Successful completion covers the complete payload transfer and the following signal update.

The user-facing completion rule is quiet-style: after operations have been submitted sequentially to one session
and one backend queue, waiting for the latest event also covers earlier pending operations in that ordered queue.
For URMA and RDMA operations sent to different peers, queues complete independently; wait for the last event of
each peer. The peer encoded in a URMA or RDMA event selects the completion queue during `Wait`/`Test`, so the caller
does not pass `peer` again.

After the signal update becomes observable at the receiver, the corresponding payload has been delivered to remote
GM. This guarantee is scoped to one `TPUT_ASYNC_NOTIFY` invocation.

## Session Ownership and Concurrency

- Do not submit concurrently through one session from independent execution flows.
- SDMA operations that share a channel group must share the same session. Concurrent SDMA kernels or independent
  sessions must use isolated channel groups.
- Complete all outstanding events before rebuilding a session, releasing its workspace or scratch tile, or reusing
  its SDMA channel group.
- Backends may apply submission backpressure when their queue or outstanding-event capacity is reached; callers must
  not assume that an unbounded number of operations can remain outstanding.
- Different producers may use `AtomicAdd` for a completion counter only when the engine supports it, their payload
  ranges do not conflict, and their sessions and backend resources are isolated as required by the platform.

## Receiver-Side Payload Visibility

The instruction does not invalidate or refresh a receiver-side cache that may contain an older copy of
`dstGlobalData`. `TWAIT` and `TTEST` observe the signal; they do not add cache maintenance for the payload.

After observing the signal and before consuming the payload, the receiver must ensure payload visibility according
to the target platform and runtime memory rules. Observing the signal means that the payload has reached remote GM;
it does not by itself guarantee that a previously cached payload copy has been refreshed.

## Examples

The sender examples assume that the host communication runtime has already translated the remote payload and signal
addresses and initialized the selected backend workspace.

### SDMA `Set`: Complete Sender Kernel

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
__global__ AICORE void PutAndNotifySdma(__gm__ T *remoteDst,
                                       __gm__ T *localSrc,
                                       __gm__ int32_t *remoteSignal,
                                       __gm__ uint8_t *sdmaWorkspace)
{
    using GShape = Shape<1, 1, 1, 1, 1024>;
    using GStride = Stride<1024, 1024, 1024, 1024, 1>;
    using GT = GlobalTensor<T, GShape, GStride, Layout::ND>;
    using ScratchTile =
        Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    GT dstGlobalData(remoteDst);
    GT srcGlobalData(localSrc);
    comm::Signal remoteReady(remoteSignal);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::SDMA>(
            scratchTile, sdmaWorkspace, session)) {
        return;
    }

    comm::AsyncEvent event =
        comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::SDMA>(
            dstGlobalData, srcGlobalData, remoteReady, 1,
            comm::NotifyOp::Set, session);
    if (!event.Wait(session)) {
        return;
    }
}
```

### URMA `AtomicAdd` with an Explicit Peer

URMA needs no scratch tile. The signal must be initialized before this kernel runs; adding `1` lets the receiver
count completed producers.

```cpp
template <typename T>
__global__ AICORE void PutAndCountUrma(__gm__ T *remoteDst,
                                      __gm__ T *localSrc,
                                      __gm__ int32_t *remoteCounter,
                                      __gm__ uint8_t *urmaWorkspace,
                                      uint32_t peer)
{
    using GShape = Shape<1, 1, 1, 1, 1024>;
    using GStride = Stride<1024, 1024, 1024, 1024, 1>;
    using GT = GlobalTensor<T, GShape, GStride, Layout::ND>;

    GT dstGlobalData(remoteDst);
    GT srcGlobalData(localSrc);
    comm::Signal counter(remoteCounter);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::URMA>(
            urmaWorkspace, session)) {
        return;
    }

    comm::AsyncEvent event =
        comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::URMA>(
            dstGlobalData, srcGlobalData, counter, 1,
            comm::NotifyOp::AtomicAdd, session, peer);
    if (!event.Wait(session)) {
        return;
    }
}
```

### RDMA `Set` with an Explicit Peer

RDMA supports `Set` only. A 256-byte Vec tile is used here and satisfies the 64-byte minimum.

```cpp
template <typename T>
__global__ AICORE void PutAndNotifyRdma(__gm__ T *remoteDst,
                                       __gm__ T *localSrc,
                                       __gm__ int32_t *remoteSignal,
                                       __gm__ uint8_t *rdmaWorkspace,
                                       uint32_t myPe,
                                       uint32_t peer)
{
    using GShape = Shape<1, 1, 1, 1, 1024>;
    using GStride = Stride<1024, 1024, 1024, 1024, 1>;
    using GT = GlobalTensor<T, GShape, GStride, Layout::ND>;
    using ScratchTile =
        Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    GT dstGlobalData(remoteDst);
    GT srcGlobalData(localSrc);
    comm::Signal remoteReady(remoteSignal);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::RDMA>(
            scratchTile, rdmaWorkspace, myPe, session)) {
        return;
    }

    comm::AsyncEvent event =
        comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::RDMA>(
            dstGlobalData, srcGlobalData, remoteReady, 1,
            comm::NotifyOp::Set, session, peer);
    if (!event.Wait(session)) {
        return;
    }
}
```

### Receiver

The receiver waits on the local address corresponding to the remotely updated signal. It then performs any payload
cache maintenance required by its target platform before reading the payload:

```cpp
comm::Signal ready(localSignalPtr);
comm::TWAIT(ready, 1, comm::WaitCmp::EQ);

// Apply the target platform/runtime's required payload visibility operation.
// Consume localPayloadPtr only after that requirement has been satisfied.
```
