# TPUT_ASYNC_NOTIFY

## Introduction

`TPUT_ASYNC_NOTIFY` starts a data transfer from local GM to remote GM, then updates a remote `int32_t` signal
after that transfer completes to notify the remote side that the data is ready. The signal supports `Set` for
state publication and `AtomicAdd` for completion counting.

Data flow:

```text
srcGlobalData (local GM) -> dstGlobalData (remote GM)
                                  |
                                  v
                       dstSignalData (remote int32 signal)
```

The caller must provide already-translated remote addresses for both the payload destination and the signal.
The instruction does not translate an address from a rank and does not allocate signal memory.

## Operation Semantics

For every element in the valid payload region:

$$\mathrm{dst}^{\mathrm{remote}}_i = \mathrm{src}^{\mathrm{local}}_i$$

After the payload operation, `notifyOp` selects the signal update:

- `NotifyOp::Set`

  $$\mathrm{signal}^{\mathrm{remote}} = \mathrm{signalValue}$$

- `NotifyOp::AtomicAdd`

  $$\mathrm{signal}^{\mathrm{remote}} \mathrel{+}= \mathrm{signalValue}$$

The `AtomicAdd` signal update is atomic. It does not make payload writes from different senders atomic.

## Template Parameters

- `engine`: currently only `DmaEngine::SDMA` is supported; it is also the default.
- `GlobalDstData`: remote payload destination tensor type.
- `GlobalSrcData`: local payload source tensor type.
- `GlobalSignalData`: remote signal tensor type; `RawDType` must be `int32_t`.
- `WaitEvents...`: PTO pipeline event types that must complete before this instruction starts and provide a
  zero-argument `Wait()`. The current `AsyncEvent` requires `Wait(session)` and cannot be passed here directly.

The URMA notify path is not implemented on A2/A3 or A5, and the A5 RoCE notify path is also reserved but not
implemented. Explicitly instantiating `TPUT_ASYNC_NOTIFY<DmaEngine::URMA>` or
`TPUT_ASYNC_NOTIFY<DmaEngine::ROCE>` is rejected at compile time.

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`.
The implementation headers reside in `pkg_inc/` and are not public APIs. A source-tree build must include both
`-I<pto-isa>/include` and `-I<pto-isa>/pkg_inc`; an installed CANN package build must include both
`-I${ASCEND_HOME_PATH}/include` and `-I${ASCEND_HOME_PATH}/pkg_inc`.

### Common overload

Available on A2/A3 and A5:

```cpp
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData,
          typename GlobalSrcData,
          typename GlobalSignalData,
          typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC_NOTIFY(
    GlobalDstData &dstGlobalData,
    GlobalSrcData &srcGlobalData,
    GlobalSignalData &dstSignalData,
    int32_t signalValue,
    NotifyOp notifyOp,
    const AsyncSession &session,
    WaitEvents&... events);
```

### A5 explicit-peer overload

A5 also provides an overload consistent with `TPUT_ASYNC`:

```cpp
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData,
          typename GlobalSrcData,
          typename GlobalSignalData,
          typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC_NOTIFY(
    GlobalDstData &dstGlobalData,
    GlobalSrcData &srcGlobalData,
    GlobalSignalData &dstSignalData,
    int32_t signalValue,
    NotifyOp notifyOp,
    const AsyncSession &session,
    uint32_t peer,
    WaitEvents&... events);
```

The current A5 SDMA/MTE path obtains its target from the already-translated virtual addresses in
`dstGlobalData` and `dstSignalData`. The `peer` argument is currently accepted for API consistency and is not
used for addressing.

## Parameters

| Parameter | Address ownership | Description |
|---|---|---|
| `dstGlobalData` | Remote GM | Payload destination tensor containing an already-translated remote address |
| `srcGlobalData` | Local GM | Payload source tensor |
| `dstSignalData` | Remote GM | Signal tensor, typically `comm::Signal` wrapping a remote `int32_t*` |
| `signalValue` | Local scalar | SET value or AtomicAdd increment |
| `notifyOp` | Local enum | `NotifyOp::Set` or `NotifyOp::AtomicAdd` |
| `session` | Local context | Built with `BuildAsyncSession<DmaEngine::SDMA>` |
| `peer` | Rank id | Present only in the A5 overload; ignored by the current SDMA/MTE path |
| `events...` | Local events | Dependencies with zero-argument `Wait()` automatically waited before this instruction starts |

`comm::Signal` is an alias for a one-element tensor:

```cpp
using Signal =
    GlobalTensor<int32_t,
                 Shape<1, 1, 1, 1, 1>,
                 Stride<1, 1, 1, 1, 1>,
                 Layout::ND>;
```

Constructing a `Signal` only wraps an address. It does not allocate memory or translate a remote address.

## AsyncSession Construction

For public usage, initialize the workspace through the host-side `SdmaWorkspaceManager`, then call
`BuildAsyncSession<DmaEngine::SDMA>` in the AICore kernel. Do not populate `AsyncSession` fields manually.
Keep the session, workspace, and scratch tile alive through the call and event wait.

On A2/A3, explicitly pass `channelGroupIdx = 0`; do not use the default that derives a group from
`get_block_idx()`:

```cpp
comm::sdma::SdmaBaseConfig config{
    comm::sdma::kDefaultSdmaBlockBytes, 0, 1};

bool ok = comm::BuildAsyncSession<comm::DmaEngine::SDMA>(
    scratchTile, sdmaWorkspace, session,
    0,       // syncId
    config,
    0);      // channelGroupIdx
```

This also means the current A2/A3 notify path supports one sending AICore and one active session per rank.
A5 uses the same session construction interface, but this instruction has completed when the A5 call returns.

See [TPUT_ASYNC](TPUT_ASYNC.md) for host workspace setup and the remaining construction parameters.

## Constraints

- **Engine**
  - Only `DmaEngine::SDMA` is currently supported.
- **Payload type and layout**
  - `GlobalSrcData::RawDType` must equal `GlobalDstData::RawDType`.
  - `GlobalSrcData::layout` must equal `GlobalDstData::layout`.
  - Source and destination must be flat contiguous logical 1D tensors.
  - The destination element capacity must be at least the source element count.
  - The payload element count must be greater than zero. An A2/A3 zero-length submission does not update the
    signal.
- **Signal**
  - `GlobalSignalData::RawDType` must be `int32_t`.
  - The signal address must be non-null and 4-byte aligned.
  - The caller must allocate and initialize signal memory before use.
- **Addresses**
  - `srcGlobalData` must refer to local GM on the current NPU.
  - In normal usage, `dstGlobalData` and `dstSignalData` must refer to remote GM on the same target rank.
- **Session**
  - `session.valid` must be `true`, and `session.engine` must be `DmaEngine::SDMA`.
  - A2/A3 requires `channelGroupIdx == 0`, one sending AICore, and one active session per rank.
  - On A5, do not mutate one session concurrently from independent execution flows.

## Completion Semantics

### A2/A3

The returned `AsyncEvent` covers the payload, signal update, and completion marker. A successful submission
has `event.valid() == true`. If `event.valid() == false`, the A2/A3 submission failed; do not treat
`Wait()` alone as proof of success.

After a successful submission:

```cpp
event.Wait(session);
```

returns successfully, the combined operation and earlier operations in the same session covered by the event
have completed.

### A5

The A5 path completes the payload and signal update before returning. Its successful event currently also has
`handle == 0`, so `event.valid() == false` must not be interpreted as failure on A5. `Wait(session)` and
`Test(session)` treat the event as complete.

The A5 path therefore preserves the asynchronous API shape, but currently does not continue executing after
the function returns.
Because a successful event and a default invalid event may both have `handle == 0`, `Wait/Test == true` is not
an independent success diagnostic on A5. The current implementation handles invalid parameters through
assertions.

`TWAIT` and `TTEST` maintain only the signal cache line on the receiver. Before reading payload written by a
remote producer, perform the payload cache maintenance required by the target platform. Do not interpret a
matching signal as an automatic invalidation of all payload cache lines. Protocols that require immediate
receiver-side polling visibility should be validated on the target software and hardware version.

## Synchronization and Concurrency

- Prefer `Set` for publishing state from one producer.
- On A5, multiple producers may use `AtomicAdd` as a completion-counting protocol when each producer uses
  isolated session/resources and non-conflicting payload ranges. Validate this concurrent combination on the
  target environment.
- The current A2/A3 path fixes notify to channel group 0 and supports one sending AICore/active session per
  rank. Do not use it as a same-rank multi-AICore counting interface.
- Multiple producers using `Set` on one signal have overwrite semantics. The final value cannot prove that
  every producer completed.
- Initialize a shared AtomicAdd signal before use, and do not concurrently update it with SET or a normal
  store.
- Signal atomicity does not protect payload data. Different senders must use non-conflicting payload
  destination ranges or provide an additional synchronization protocol.
- A signal proves completion only for payload explicitly associated with it by the application protocol.

## Examples

The examples below assume:

- `remoteDst` and `remoteSignal` have already been translated by the communication runtime;
- the host initialized `sdmaWorkspace` through `SdmaWorkspaceManager`;
- the example kernel launches on one AICore.

### SET: transfer payload and publish readiness

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
__global__ AICORE void put_and_set(__gm__ T *remoteDst,
                                   __gm__ T *localSrc,
                                   __gm__ int32_t *remoteSignal,
                                   __gm__ uint8_t *sdmaWorkspace)
{
    using GShape = Shape<1, 1, 1, 1, 1024>;
    using GStride = Stride<1024, 1024, 1024, 1024, 1>;
    using GT = GlobalTensor<T, GShape, GStride, Layout::ND>;
    using ScratchTile =
        Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    GT dst(remoteDst);
    GT src(localSrc);
    comm::Signal signal(remoteSignal);
    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0);

    comm::AsyncSession session;
    comm::sdma::SdmaBaseConfig config{
        comm::sdma::kDefaultSdmaBlockBytes, 0, 1};
    if (!comm::BuildAsyncSession(
            scratchTile, sdmaWorkspace, session, 0, config, 0)) {
        return;
    }
    comm::AsyncEvent event = comm::TPUT_ASYNC_NOTIFY(
        dst, src, signal, 1, comm::NotifyOp::Set, session);
#ifdef PTO_NPU_ARCH_A2A3
    if (!event.valid()) {
        return;
    }
#endif
    if (!event.Wait(session)) {
        return;
    }
}
```

The receiver waits on its local signal address:

```cpp
__global__ AICORE void wait_until_ready(__gm__ int32_t *localPayload,
                                        __gm__ int32_t *localSignal)
{
    comm::Signal signal(localSignal);
    comm::TWAIT(signal, 1, comm::WaitCmp::EQ);

    // TWAIT only maintains the signal cache line. Invalidate payload cache
    // before consuming data written by the remote producer.
    dcci((__gm__ void *)localPayload, cache_line_t::ENTIRE_DATA_CACHE);
    dsb(DSB_DDR);
    // Consume the payload associated with this signal.
}
```

### AtomicAdd: accumulate completion count

This fragment reuses a valid session built as in the previous example. On A2/A3, use it only from the single
sending AICore.

```cpp
template <typename GT>
PTO_INTERNAL comm::AsyncEvent put_and_count(
    GT &remoteDst,
    GT &localSrc,
    __gm__ int32_t *remoteCounter,
    comm::AsyncSession &session)
{
    comm::Signal counter(remoteCounter);
    return comm::TPUT_ASYNC_NOTIFY(
        remoteDst, localSrc, counter, 1,
        comm::NotifyOp::AtomicAdd, session);
}
```

The receiver waits for `expectedProducers` producers and performs the required payload cache maintenance
before reading their payloads:

```cpp
comm::Signal counter(localCounter);
comm::TWAIT(counter, expectedProducers, comm::WaitCmp::GE);
dcci((__gm__ void *)localPayload, cache_line_t::ENTIRE_DATA_CACHE);
dsb(DSB_DDR);
```

### Wait for an AsyncEvent before notifying

In this fragment, `remoteDst0/1` and `localSrc0/1` are already constructed `GlobalTensor` objects:

```cpp
comm::AsyncEvent previous = comm::TPUT_ASYNC(remoteDst0, localSrc0, session);
comm::Signal signal(remoteSignal);

bool previousReady = previous.Wait(session);
#ifdef PTO_NPU_ARCH_A2A3
previousReady = previous.valid() && previousReady;
#endif
if (previousReady) {
    comm::AsyncEvent notified = comm::TPUT_ASYNC_NOTIFY(
        remoteDst1, localSrc1, signal, 2,
        comm::NotifyOp::Set, session);
#ifdef PTO_NPU_ARCH_A2A3
    if (!notified.valid()) {
        return;
    }
#endif
    if (!notified.Wait(session)) {
        return;
    }
}
```

Wait an `AsyncEvent` explicitly with its session, as above. Only PTO pipeline events with a zero-argument
`Wait()` can be passed directly in `events...`.
