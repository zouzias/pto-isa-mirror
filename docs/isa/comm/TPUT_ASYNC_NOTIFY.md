# TPUT_ASYNC_NOTIFY

## Introduction

`TPUT_ASYNC_NOTIFY` is an asynchronous remote write with notification. It transfers a non-empty payload from local
GM to remote GM and then updates one remote 32-bit signal. The instruction returns the existing `AsyncEvent` used by
`TPUT_ASYNC`; it does not introduce a new event type.

Data flow:

```text
srcGlobalData (local GM) -> DMA engine -> dstGlobalData (remote GM)
                                             |
                                      after the payload
                                             v
                                  dstSignalData (remote GM)
```

Use `TNOTIFY` instead when only a signal update is required.

## Template Parameter and Backend Support

- `engine` selects the DMA backend at compile time:

| Engine | Platform or backend restriction | Supported `NotifyOp` |
|---|---|---|
| `DmaEngine::SDMA` (default) | Platforms supported by `TPUT_ASYNC` | `Set`, `AtomicAdd` |
| `DmaEngine::URMA` | Ascend950, NPU_ARCH 3510 only; requires CANN Toolkit >= 9.1.0 | `Set`, `AtomicAdd` |
| `DmaEngine::RDMA` | Ascend950, NPU_ARCH 3510 only; HNS1825 NIC platform | `Set` only; `AtomicAdd` is unsupported |

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

On A5, an overload with an explicit destination peer is provided, consistent with `TPUT_ASYNC`:

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

The optional `events` parameter pack follows the existing PTO dependency-event convention. The operation waits for
all supplied dependency events before issuing the payload transfer. Passing no dependency event is valid.

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
| `events` | Zero or more prerequisite PTO pipeline events. |

The return value is an `AsyncEvent` whose completion covers both the payload transfer and the signal update.

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

The ordering above applies to the payload and signal update of the same invocation. It does not define ordering
between independent operations issued from different sessions or execution flows.

## AsyncSession Construction

`TPUT_ASYNC_NOTIFY` reuses the same `AsyncSession`, workspace, and `AsyncEvent` contract as `TPUT_ASYNC`. Follow
[TPUT_ASYNC](TPUT_ASYNC.md) for engine-specific construction parameters, workspace requirements, lifetime rules,
and session ownership.

On A5, build one peer-independent URMA or RDMA session and pass the destination rank to the explicit-`peer` overload
when communicating with multiple peers. The session-bound overload remains available for compatibility.

## Constraints

- `GlobalSrcData::RawDType == GlobalDstData::RawDType`.
- `GlobalSrcData::layout == GlobalDstData::layout`.
- Source and destination payload tensors must be flat, contiguous logical 1D tensors.
- The payload size must be greater than zero. Use `TNOTIFY` for a signal-only operation.
- `dstSignalData` must represent exactly one `int32_t` in remote GM and must be 4-byte aligned.
- The payload destination range and `dstSignalData` must not overlap.
- For URMA and RDMA, the payload destination and signal must belong to the same remote peer selected by the session
  or explicit `peer` argument.
- For URMA and RDMA, the complete local payload, remote payload, and remote signal ranges must be covered by memory
  regions registered during host initialization.
- One RDMA payload must not exceed `0x7fffffff` bytes.
- RDMA supports only `NotifyOp::Set`; `NotifyOp::AtomicAdd` must not be used.
- The session engine must match the `engine` template argument. Keep the session and its workspace alive until all
  associated events have completed.

## Completion and Ordering Semantics

The returned event follows the same quiet semantics as `TPUT_ASYNC`:

- `event.Wait(session)` blocks until completion.
- `event.Test(session)` checks completion without blocking.
- Successful completion covers both the payload transfer and the following signal update.
- For RDMA operations targeting different peers, wait for the last event of each peer separately.

After the signal update becomes observable at the receiver, the corresponding payload has been delivered to remote
GM. This guarantee is scoped to one `TPUT_ASYNC_NOTIFY` invocation.

## Receiver-Side Payload Visibility

The instruction does not invalidate or refresh a receiver-side cache that may contain an older copy of
`dstGlobalData`. `TWAIT` and `TTEST` observe the signal; they do not add cache maintenance for the payload.

After observing the signal and before consuming the payload, the receiver must ensure payload visibility according
to the target platform and runtime memory rules. Observing the signal means that the payload has reached remote GM;
it does not by itself guarantee that a previously cached payload copy has been refreshed.

## Examples

### Set Notification

The session and payload tensors below are constructed in the same way as for `TPUT_ASYNC`:

```cpp
comm::Signal remoteReady(remoteSignalPtr);

auto event = comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::SDMA>(
    dstGlobalData, srcGlobalData, remoteReady, 1, comm::NotifyOp::Set, session);
(void)event.Wait(session);
```

The receiver can wait for the signal before consuming the payload:

```cpp
comm::Signal ready(localSignalPtr);
comm::TWAIT(ready, 1, comm::WaitCmp::EQ);

// Ensure platform-specific payload visibility before reading localPayloadPtr.
// Consume the payload only after that requirement has been satisfied.
```

### RDMA with an Explicit Peer

RDMA supports `Set` only:

```cpp
comm::Signal remoteReady(remoteSignalPtr);

auto event = comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::RDMA>(
    dstGlobalData, srcGlobalData, remoteReady, 1, comm::NotifyOp::Set, session, peer);
(void)event.Wait(session);
```
