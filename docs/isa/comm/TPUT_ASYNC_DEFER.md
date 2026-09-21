# TPUT_ASYNC_DEFER

## Introduction

`TPUT_ASYNC_DEFER` prepares one asynchronous write from local GM to remote GM. It writes one or more transfer descriptions into backend send queues, but does not advance the hardware-visible Producer or ring the Send Doorbell, so the transfer task has not yet been published to hardware. One or more successful non-empty `TPUT_ASYNC_DEFER` calls form the current batch. A subsequent [`TPUT_ASYNC_SUBMIT`](TPUT_ASYNC_SUBMIT.md) call advances the corresponding Producer positions, rings the Send Doorbell, publishes the batch for asynchronous hardware execution, and returns one final event. `TPUT_ASYNC_DEFER` itself returns no event.

Data direction:

`srcGlobalData (local GM)` → DMA engine → `dstGlobalData (remote GM)`

## Template Parameter

`engine` selects the DMA backend at compile time:

| Engine | Platform restriction | Description |
|---|---|---|
| `DmaEngine::SDMA` (default) | A2/A3 | Uses an SDMA Channel Group. |
| `DmaEngine::URMA` | Ascend 950PR/Ascend 950DT, NPU_ARCH 3510 only | Uses one Jetty selected by `peer` and `jettyIndex`. |

Currently, only standard `TPUT_ASYNC` data transfers are supported, and A5 `DmaEngine::SDMA` is not supported.

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <
    DmaEngine engine = DmaEngine::SDMA,
    typename GlobalDstData,
    typename GlobalSrcData>
PTO_INST void TPUT_ASYNC_DEFER(
    GlobalDstData& dstGlobalData,
    GlobalSrcData& srcGlobalData,
    const AsyncSession& session,
    uint32_t peer = UINT32_MAX,
    uint32_t jettyIndex = 0U);
```

## Parameters

| Parameter | Description |
|---|---|
| `dstGlobalData` | Destination GlobalTensor for the payload in remote GM. |
| `srcGlobalData` | Source GlobalTensor in local GM; its shape and element type determine the transfer size. |
| `session` | A session successfully built by `BuildAsyncSession<engine>()`; it also holds the current batch state. |
| `peer` | Required for URMA and must identify the target rank. SDMA ignores this parameter; if omitted, it defaults to `UINT32_MAX`. |
| `jettyIndex` | Zero-based index within the Jetties available to the current AIV under the URMA workspace configuration. The default is 0. Unused by SDMA. |

The intrinsic returns `void`, creates no per-write `AsyncEvent`, and accepts no prerequisite `WaitEvents`.

## Remote-Write Semantics

One non-empty Defer call adds one remote write:

1. Validate the session, tensors, non-null pointers, and batch capacity against the session configuration.
2. Write the TPUT transfer description into the send queue.
3. Do not advance the hardware-visible Producer or ring the Send Doorbell.

A zero-length source tensor is a no-op and does not start a batch.

Defer is not guaranteed to return immediately. It may wait until resources needed to prepare the transfer task
become available, but it still does not ring the Send Doorbell or publish the current batch to hardware.

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

- The first successful non-empty Defer starts a batch; later Defer calls append writes to it.
- Every Defer and the final Submit in one batch must use the same session and the same `engine`.
- One URMA batch must always use the same `peer` and `jettyIndex`.
- A non-empty Defer sequence must be followed by `TPUT_ASYNC_SUBMIT`; there is no explicit batch-cancel interface.
- If Defer or Submit detects an argument, resource, or capacity error before publication, it discards the current
  batch's staged descriptors and batch state, and none of the previously staged content is published. Defer returns
  `void` and does not report that failure directly; if execution continues and no new successful Defer follows, the
  Submit failure path returns an invalid event.
- Between Defer and Submit, do not issue ordinary `TPUT_ASYNC`, `TGET_ASYNC`, `TPUT_ASYNC_NOTIFY`, another batch,
  or `TPREFETCH_ASYNC` through the same Channel Group or Jetty.
- From the first successful non-empty Defer through Submit, do not copy, rebuild, destroy, or transfer ownership of
  the session.
- Remote destination ranges of different writes must not overlap. Defer order must not be used to express
  dependencies between writes.
- Every source range and its contents must remain valid and unchanged until the final event returned by Submit
  completes.

### Tensor and Resource Requirements

- Source and destination tensors must have the same `RawDType`.
- Source and destination tensors must have the same layout.
- Both tensors must be flat, contiguous logical 1D tensors.
- Destination element capacity must be at least the source element count.
- Non-empty transfers require valid source and destination pointers; total element and byte counts must not overflow
  `uint64_t`.
- The interface does not validate communication-memory membership, address bounds, overlap, or target-peer ownership;
  the caller must guarantee them.
- The SDMA workspace and Channel Group must come from a valid `SdmaWorkspaceManager` environment.
- Each SDMA Defer consumes `ceil(transferBytes / blockBytes)` data SQEs. Distribution is round-robin across
  `queueNum` using the cumulative SQE index of the complete batch; it does not restart at queue 0 for each Defer.
  The batch data-SQE count assigned to each queue must be less than that queue's SQ depth.
- SDMA adds `commBlockOffset` to both tensor base addresses. The backing allocations must cover
  `[base + commBlockOffset, base + commBlockOffset + transferBytes)`, and both range-bound calculations must be
  representable without overflow.
- The URMA workspace, peer memory registration, and Jetty must come from a valid `UrmaWorkspaceManager` environment.
- URMA requires `peer` to be less than the workspace rank count and `jettyIndex < session.qpCount`.
- When one Jetty is available to the current AIV, only `jettyIndex == 0` is valid. When multiple Jetties are
  available, any index in that range may be selected.
- For URMA, the complete source range must lie within the local rank's communication buffer passed to
  `UrmaWorkspaceManager::Init()`, and the complete destination range must lie within the selected peer's communication
  buffer. These buffers must use device memory that the active CANN/HCCL runtime supports registering.
- URMA splits one remote write into WQEs of at most 256 MiB. The total WQE count of the batch must fit both the WQ and
  CQ depths of the selected Jetty.
- A batch must remain within the limits of the session configuration. Split a larger workload into multiple batches;
  the intrinsic does not automatically submit a partial batch. Earlier incomplete work may make Defer wait for
  resources, but the current batch itself must still fit the total capacity limits above.

## Completion Semantics

Defer does not return an event and does not represent transfer completion. The final `AsyncEvent` is created only by
`TPUT_ASYNC_SUBMIT`; its successful completion covers all non-empty writes in the batch. Keep every source
range, the session, workspace, SDMA scratch tile, and communication resources valid until that event completes.

The public contract does not guarantee that remote writes complete in Defer call order. Do not use that order to
express data dependencies; the receiver may also observe different writes in the batch incrementally.

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
- After Submit, the same owner may serially construct the next batch. Rebuild or destroy the session and its
  communication resources only after every outstanding event produced by that session has completed.

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

    comm::TPUT_ASYNC_DEFER<comm::DmaEngine::SDMA>(
        dst0, src0, session);
    comm::TPUT_ASYNC_DEFER<comm::DmaEngine::SDMA>(
        dst1, src1, session);

    comm::AsyncEvent event =
        comm::TPUT_ASYNC_SUBMIT<comm::DmaEngine::SDMA>(session);
    if (!event.valid()) {
        return;
    }
    if (!event.Wait(session)) {
        return;
    }
    status[0] = 1U;
}
```

SDMA ignores `peer` and `jettyIndex`; `peer` may be omitted. URMA requires an explicit valid `peer`.

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

`localSrc` and `remoteDst` must lie within the URMA-registered communication buffers. This example selects relative index 0. When multiple Jetties are available, any index in the available range may be selected, but every Defer and the final Submit in one batch must use the same index.

The example sets `status[0]` to 1 only after `Wait` returns `true`. If it returns `false`, the kernel exits with status 0, but the transfer has not thereby been canceled. The host must stop using the affected communication resources and tear down the communication context according to the runtime failure procedure.
