# TPUT_ASYNC_SUBMIT

## Introduction

`TPUT_ASYNC_SUBMIT` publishes all remote writes prepared in the current `AsyncSession` by [`TPUT_ASYNC_DEFER`](TPUT_ASYNC_DEFER.md). It advances the hardware-visible Producer, rings the Send Doorbell or Doorbells required by the backend, and returns one `AsyncEvent` that covers the complete batch.

Call sequence:

`TPUT_ASYNC_DEFER × N (stage)` → `TPUT_ASYNC_SUBMIT (publish)` → `Wait/Test (complete)`

Returning from Submit means that the batch has been made available to hardware, not that its transfers have completed. The caller must use the returned event to determine completion.

## Template Parameter

`engine` selects the DMA backend at compile time:

| Engine | Platform restriction | Description |
|---|---|---|
| `DmaEngine::SDMA` (default) | A2/A3 | Submits the batch through one SDMA Channel Group. |
| `DmaEngine::URMA` | Ascend 950PR/Ascend 950DT, NPU_ARCH 3510 only | Submits the batch through one Jetty. |

Currently, only standard `TPUT_ASYNC` data transfers are supported, and A5 `DmaEngine::SDMA` is not supported.

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <DmaEngine engine = DmaEngine::SDMA>
PTO_INST AsyncEvent TPUT_ASYNC_SUBMIT(
    const AsyncSession& session,
    uint32_t peer,
    uint32_t jettyIndex = 0U);
```

## Parameters

| Parameter | Description |
|---|---|
| `session` | The `AsyncSession` holding a non-empty batch; it must be the session used by all preceding Defer calls. |
| `peer` | Target rank for URMA; it must match every Defer in the batch. Unused by SDMA. |
| `jettyIndex` | Zero-based index within the Jetties available to the current AIV under the URMA workspace configuration; it must match every Defer in the batch. The default is 0. Unused by SDMA. |

The return value is the final `AsyncEvent`. A valid event covers every non-empty write in the batch and does not
provide individual per-write completion. Check `event.valid()` before Wait or Test; an invalid event must not be used
to conclude that the batch completed successfully.

## Batch Submission Semantics

Submit requires the current batch to contain at least one successful non-empty Defer.

- **SDMA:** Advance the Producers for the participating queues, ring the required Send Doorbells, and return one
  event for the complete batch.
- **URMA:** Advance the Producer of the Jetty selected by `jettyIndex`, ring its Send Doorbell, and return one event
  for the complete batch.

After Submit returns a valid event, the session can start constructing the next batch. This does not mean that the
previous batch has completed.

## AsyncSession Construction

Use the engine-specific `BuildAsyncSession` overload.

### SDMA Construction (default)

```cpp
comm::BuildAsyncSession<comm::DmaEngine::SDMA>(
    scratchTile, workspace, session, syncId, baseConfig, channelGroupIdx);
```

| Parameter | Default | Description |
|---|---|---|
| `scratchTile` | — | UB scratch tile for SDMA control data. |
| `workspace` | — | GM workspace initialized by `SdmaWorkspaceManager`. |
| `session` | — | Output `AsyncSession`. |
| `syncId` | `0` | Pipeline synchronization event ID. |
| `baseConfig` | Default SDMA configuration | Block size, communication offset, and queue count. |
| `channelGroupIdx` | Automatic | Selects the SDMA Channel Group referenced by the session. |

`scratchTile` is temporary UB workspace for SDMA control data and must provide at least `sizeof(uint64_t)` (8 bytes). The recommended type is `Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>` (256 bytes). Keep `scratchTile` valid until all events associated with the session complete. `workspace` is initialized by `SdmaWorkspaceManager`.

### URMA Construction (NPU_ARCH 3510 only)

```cpp
comm::BuildAsyncSession<comm::DmaEngine::URMA>(
    workspace, session);
```

| Parameter | Description |
|---|---|
| `workspace` | GM workspace initialized by `UrmaWorkspaceManager`. |
| `session` | Output `AsyncSession`. |

URMA does not require a scratch tile. Its workspace is initialized by `UrmaWorkspaceManager`. Only a successfully
built session may be used by Defer, Submit, and event completion calls.

## Constraints

- An empty batch must not be submitted.
- Submit must use the same session and `engine` as every Defer in the batch.
- URMA must use the same `peer` and `jettyIndex` as every Defer in the batch.
- If Defer or Submit detects an argument, resource, or capacity error before publication, it discards the current
  batch's staged descriptors and batch state, and none of the previously staged content is published. The Submit
  failure path returns an invalid event; the batch is not submitted partially.
- When one Jetty is available to the current AIV, only `jettyIndex == 0` is valid. When multiple Jetties are
  available, any index in that range may be selected.
- `peer` must be less than the URMA workspace rank count, and `jettyIndex < session.qpCount`.
- All source ranges must remain valid and unchanged until the event completes.
- Do not release or rebuild the session, workspace, SDMA scratch tile, SDMA Channel Group, or URMA Jetty before
  event completion.
- Before rebuilding or destroying the session and its communication resources, confirm completion of the last event
  on every active physical queue used by the session. A later event on one physical queue covers earlier submissions
  on that queue, but an event for one peer or Jetty does not cover another independent queue.
- Before releasing URMA communication resources, complete the last event for every active peer and Jetty, and synchronize
  host streams using the communication context.
- Tensor and address ranges, SDMA `commBlockOffset` and SQ capacity, and URMA registered-memory and WQ/CQ capacity
  must satisfy the constraints in [`TPUT_ASYNC_DEFER`](TPUT_ASYNC_DEFER.md).
- Unsupported engines are rejected at compile time. Other arguments and communication resources must satisfy the
  requirements established by the corresponding `BuildAsyncSession`.

## Completion Semantics

- `event.Wait(session)` blocks while waiting for batch completion.
- `event.Test(session)` tests batch completion without blocking.
- Confirm that `event.valid()` is `true` before calling Wait or Test.
- Wait or Test must use the same session that submitted the batch.
- Event completion means that all payload transfers in the batch have completed and their source ranges may be
  reused.
- A batch is not a transaction. A receiver may observe writes incrementally, and completed remote writes are not
  rolled back after an error.
- The public contract does not define completion order between individual writes.
- The final event is a sender-side completion object and does not automatically notify the receiver. Remote
  consumption still requires an application protocol such as `TNOTIFY`/`TWAIT` and receiver-side cache visibility
  handling appropriate for the platform.

Only a `true` result from `Wait` means that the batch has completed. A `false` result means completion has not been
confirmed and does not cancel the submitted transfer. Do not reuse source data or release or rebuild the session,
workspace, or communication resources in that state.

## Concurrency and Session Ownership

- `AsyncSession` is a single-owner, non-thread-safe object. Do not copy a session to create another owner.
- Build/Rebuild, Defer, Submit, Wait, and Test on one session must be called serially by one execution flow.
- Distinct sessions do not imply distinct physical resources. Concurrent A2/A3 sessions sharing one SDMA workspace
  must use consistent `queueNum` configurations and select non-overlapping physical Channel ranges. A session uses
  `[channelGroupIdx * queueNum, channelGroupIdx * queueNum + queueNum)`.
- From the Build that selects a physical resource until any staged batch is submitted and the last event on that
  resource completes, do not Build/Rebuild another session mapped to the same SDMA Channel, URMA WQ, or URMA CQ.
- In URMA `PER_PEER` mode, calls to the same peer access the same physical WQ/CQ and must be serialized; different
  peers use different queues.
- In URMA `SHARED_POOL` mode, the physical Jetty index is `session.qpIdxBase + jettyIndex`. Sessions built by the same
  AIV alias when they select the same relative `jettyIndex`; concurrent calls must select distinct physical Jetties,
  and the same Jetty conflicts even when the peers differ.
- Wait or Test on one physical Channel Group, WQ, or CQ must not run concurrently with Defer, Submit, or another
  Wait/Test on that resource.
- Concurrent remote destination ranges must not overlap.
- After Submit, the same owner may serially stage a new batch, but the caller remains responsible for capacity and
  ordering in the session.

## Examples

### SDMA

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/common/pto_tile.hpp>

using namespace pto;

__global__ AICORE void BatchPutSdma(
    __gm__ int32_t* remoteDst, __gm__ int32_t* localSrc,
    __gm__ uint8_t* sdmaWorkspace, __gm__ uint32_t* status)
{
    constexpr uint32_t kElementsPerWrite = 16U;
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT = GlobalTensor<int32_t, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile =
        Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    status[0] = 0U;
    ShapeDyn shape(1, 1, 1, 1, kElementsPerWrite);
    StrideDyn stride(
        kElementsPerWrite, kElementsPerWrite, kElementsPerWrite,
        kElementsPerWrite, 1);
    GT dst0(remoteDst, shape, stride);
    GT src0(localSrc, shape, stride);
    GT dst1(remoteDst + kElementsPerWrite, shape, stride);
    GT src1(localSrc + kElementsPerWrite, shape, stride);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::SDMA>(
            scratchTile, sdmaWorkspace, session)) {
        return;
    }

    constexpr uint32_t kIgnoredPeer = 0U;
    comm::TPUT_ASYNC_DEFER<comm::DmaEngine::SDMA>(
        dst0, src0, session, kIgnoredPeer);
    comm::TPUT_ASYNC_DEFER<comm::DmaEngine::SDMA>(
        dst1, src1, session, kIgnoredPeer);

    comm::AsyncEvent event =
        comm::TPUT_ASYNC_SUBMIT<comm::DmaEngine::SDMA>(
            session, kIgnoredPeer);
    if (!event.valid()) {
        return;
    }
    if (!event.Wait(session)) {
        return;
    }
    status[0] = 1U;
}
```

SDMA accepts `peer` and `jettyIndex` for interface consistency and ignores both parameters.

### URMA

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/common/pto_tile.hpp>

using namespace pto;

__global__ AICORE void BatchPutUrma(
    __gm__ int32_t* remoteDst, __gm__ int32_t* localSrc,
    __gm__ uint8_t* urmaWorkspace, __gm__ uint32_t* status,
    uint32_t peer)
{
    constexpr uint32_t kElementsPerWrite = 16U;
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT = GlobalTensor<int32_t, ShapeDyn, StrideDyn, Layout::ND>;

    status[0] = 0U;
    ShapeDyn shape(1, 1, 1, 1, kElementsPerWrite);
    StrideDyn stride(
        kElementsPerWrite, kElementsPerWrite, kElementsPerWrite,
        kElementsPerWrite, 1);
    GT dst0(remoteDst, shape, stride);
    GT src0(localSrc, shape, stride);
    GT dst1(remoteDst + kElementsPerWrite, shape, stride);
    GT src1(localSrc + kElementsPerWrite, shape, stride);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::URMA>(
            urmaWorkspace, session)) {
        return;
    }

    constexpr uint32_t kJettyIndex = 0U;
    comm::TPUT_ASYNC_DEFER<comm::DmaEngine::URMA>(
        dst0, src0, session, peer, kJettyIndex);
    comm::TPUT_ASYNC_DEFER<comm::DmaEngine::URMA>(
        dst1, src1, session, peer, kJettyIndex);

    comm::AsyncEvent event =
        comm::TPUT_ASYNC_SUBMIT<comm::DmaEngine::URMA>(
            session, peer, kJettyIndex);
    if (!event.valid()) {
        return;
    }
    if (!event.Wait(session)) {
        return;
    }
    status[0] = 1U;
}
```

`localSrc` and `remoteDst` must lie within the URMA-registered communication buffers. This example selects relative index 0. When multiple Jetties are available, any index in the available range may be selected, but every Defer and Submit in one batch must use the same index.

The example sets `status[0]` to 1 only after `Wait` returns `true`. If it returns `false`, the kernel exits with status 0, but the transfer has not thereby been canceled. The host must stop using the affected communication resources and tear down the communication context according to the runtime failure procedure.
