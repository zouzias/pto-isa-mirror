# TNOTIFY

## Introduction

`TNOTIFY` updates one 32-bit signal for lightweight notification between NPUs. It transfers no payload and returns no
`AsyncEvent`.

The overload with an explicit `peer` is intended for remote notification. The communication runtime selects a directly
addressable GM, URMA, or RDMA path from the reachability of that peer; callers do not select an engine in the
interface. The signal update is complete when the call returns.

The original overload without `peer` remains available for compatibility. It directly accesses the GM address in
`dstSignalData`, does not inspect peer reachability, and cannot automatically switch to URMA or RDMA. This page specifies
only the new explicit-peer interface.

## Mathematical Semantics

For `NotifyOp::Set`:

$$
\mathrm{signal}^{\mathrm{peer}} = \mathrm{value}
$$

For `NotifyOp::AtomicAdd`:

$$
\mathrm{signal}^{\mathrm{peer}} \mathrel{+}= \mathrm{value}
\quad (\text{atomic})
$$

## Assembly Syntax

```text
tnotify %signal_remote, %value, %peer {op = #pto.notify_op<Set>} : (!pto.memref<i32>, i32, i32)
tnotify %signal_remote, %value, %peer {op = #pto.notify_op<AtomicAdd>} : (!pto.memref<i32>, i32, i32)
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`.

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST void TNOTIFY(GlobalSignalData &dstSignalData,
                      int32_t value,
                      NotifyOp op,
                      uint32_t peer,
                      WaitEvents &... events);
```

## Parameters

| Parameter | Description |
|---|---|
| `dstSignalData` | Signal to update. In the explicit-peer form, it must designate the target GM address on `peer`. |
| `value` | Value written by `Set`, or the signed increment used by `AtomicAdd`. |
| `op` | Signal operation: `NotifyOp::Set` or `NotifyOp::AtomicAdd`. |
| `peer` | Target rank id. It selects peer reachability and communication resources; it does not translate a local address into a remote address. |
| `events` | Zero or more predecessor PTO pipeline events. The call waits for them before updating the signal. |

Every object in `events` must provide a parameterless `Wait()`. An `AsyncEvent` requires `Wait(session)` and therefore
cannot be passed directly as an `events` argument.

### Signal Object

`comm::Signal` is the GlobalTensor alias for one `int32_t` signal:

```cpp
using Signal = GlobalTensor<int32_t,
                            Shape<1, 1, 1, 1, 1>,
                            Stride<1, 1, 1, 1, 1>,
                            Layout::ND>;
```

Constructing a `Signal` only wraps a caller-provided GM address. It neither allocates nor initializes storage. The
caller must allocate the signal and establish its initial value according to the communication protocol.

## Routing and Capabilities

- The explicit-peer overload reads peer reachability initialized by the communication runtime and selects a path that
  supports the requested `NotifyOp`. The selected path is an implementation detail and must not be relied upon by the
  caller.
- `dstSignalData` and `peer` must identify the same target. `peer` selects communication resources but does not repair
  an incorrect address.

The current interface capabilities are:

| Reachability path | `NotifyOp::Set` | `NotifyOp::AtomicAdd` | Notes |
|---|---|---|---|
| Directly addressable GM | Supported | Supported | For local or P2P cases in which the signal address is directly accessible. |
| URMA | Supported | Supported | NPU_ARCH 3510 only; URMA resources for the peer must be initialized by the runtime. |
| RDMA | Supported | Not supported | A5 only; the current RDMA backend is HNS1825 RoCE. |

SDMA data-path reachability does not by itself prove that a signal address is directly accessible to Scalar code.
`TNOTIFY` does not submit a standalone SDMA payload transfer; it uses the direct-GM path only when the runtime reports
that the target signal is directly addressable.

## Operation and Concurrency Semantics

- `Set` writes `value` to the signal. The final value is unspecified when multiple execution flows concurrently `Set`
  the same signal.
- `AtomicAdd` makes the signal update itself atomic, so increments from multiple producers do not overwrite one another.
- Do not concurrently mix `Set`, `AtomicAdd`, or ordinary stores on the same signal without application-level
  synchronization.
- The RDMA path does not support `AtomicAdd`; that operation must not be used for a peer reachable only through RDMA.

## Completion, Ordering, and Visibility

One call performs these steps in order:

1. Wait for every object in `events`.
2. Select a path from `peer`.
3. Update the signal and return only after that update completes.

For URMA or RDMA, the implementation must wait for the corresponding remote operation internally. A `void` return does
not mean that the operation was merely posted asynchronously.

`TNOTIFY` covers only its own signal update. It does not implicitly wait for operations previously posted on another
Session or queue. To publish payload written by `TPUT_ASYNC`, first call `Wait(session)` on the PUT event and then call
`TNOTIFY`.

When the receiver waits with `TWAIT`, `TWAIT` refreshes the signal cache line while polling. A receiver that uses ordinary
loads for the signal or related payload must still follow the memory-consistency requirements of the target platform and
runtime.

## Constraints

- `GlobalSignalData::RawDType` must be `int32_t`.
- `dstSignalData.data()` must be non-null and 4-byte aligned.
- `peer` must be less than the rank count initialized by the communication runtime and must have a path that supports
  the requested operation.
- For URMA and RDMA, the complete signal address range must be inside remote memory registered during Host
  initialization.
- The explicit-peer overload requires peer reachability and the selected communication resources to be initialized
  before kernel launch and remain valid for the call.
- RDMA supports `NotifyOp::Set` only.

## Examples

The following examples assume that the Host communication runtime has initialized peer reachability and the required
transport resources, and that the remote signal address belongs to the target peer.

### Explicit-peer Set

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

__global__ AICORE void NotifyPeerSet(__gm__ int32_t *remoteSignalPtr,
                                    uint32_t peer)
{
    comm::Signal remoteSignal(remoteSignalPtr);
    comm::TNOTIFY(remoteSignal, 1, comm::NotifyOp::Set, peer);
}
```

The same call form works for directly addressable GM, URMA, or RDMA targets; the runtime selects the path.

### Explicit-peer AtomicAdd

```cpp
__global__ AICORE void NotifyPeerAdd(__gm__ int32_t *remoteCounterPtr,
                                    uint32_t peer)
{
    comm::Signal remoteCounter(remoteCounterPtr);

    // Use only when this peer has a direct-GM or URMA path that supports AtomicAdd.
    comm::TNOTIFY(remoteCounter, 1, comm::NotifyOp::AtomicAdd, peer);
}
```

### Standalone Notification After an Asynchronous PUT

```cpp
auto event = comm::TPUT_ASYNC<comm::DmaEngine::URMA>(
    remoteDst, localSrc, session, peer);

if (event.Wait(session)) {
    comm::TNOTIFY(remoteSignal, 1, comm::NotifyOp::Set, peer);
}
```

The PUT event must be waited explicitly. Calling `TNOTIFY` immediately after posting the PUT does not ensure that the
asynchronous payload completes first.
