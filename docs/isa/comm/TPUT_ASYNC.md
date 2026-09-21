# TPUT_ASYNC

## Introduction

`TPUT_ASYNC` is an asynchronous remote write primitive. In the default `IMMEDIATE` mode it publishes a transfer from
local GM to remote GM and returns its `AsyncEvent`. In `DEFER` mode it only stages the write; no transfer starts until
`SubmitAsyncPutBatch`, and staging may wait for queue resources before returning an invalid placeholder event.

Data flow:

`srcGlobalData (local GM) -> DMA engine -> dstGlobalData (remote GM)`


## Template Parameter

- `engine`:
    - `DmaEngine::SDMA` (default)
    - `DmaEngine::URMA` (Ascend950, NPU_ARCH 3510 only)
    - `DmaEngine::RDMA` (Ascend950, NPU_ARCH 3510 only; currently supports only the HNS1825 NIC platform)

> **Important (SDMA path)**
> `TPUT_ASYNC` with `DmaEngine::SDMA` currently supports **only flat contiguous logical 1D tensors**.
> Non-1D or non-contiguous layouts are not supported by the current SDMA async implementation.


## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`.

```cpp
// A2/A3
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session,
                               AsyncPutMode mode = AsyncPutMode::IMMEDIATE,
                               WaitEvents &... events);

// A5, peer obtained from session.destRankId
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session,
                               AsyncPutMode mode = AsyncPutMode::IMMEDIATE,
                               uint32_t jettyIndex = 0U,
                               WaitEvents &... events);

// A5, explicit peer
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session, uint32_t peer,
                               AsyncPutMode mode = AsyncPutMode::IMMEDIATE,
                               uint32_t jettyIndex = 0U,
                               WaitEvents &... events);
```

The default `IMMEDIATE` mode preserves standard submission and returns its completion event. `DEFER` stages the
write and returns an invalid placeholder event (`handle == 0`). Existing calls that pass prerequisite events directly
remain source-compatible through forwarding overloads. When an A5 call passes both an explicit mode and prerequisite
events, specify `jettyIndex` before those events.

## Submission Modes

| Mode | Behavior | Return value |
|---|---|---|
| `AsyncPutMode::IMMEDIATE` (default) | Publish this remote write immediately. | A valid completion `AsyncEvent` for the submitted operation. |
| `AsyncPutMode::DEFER` | Stage this remote write in the current session without advancing the hardware-visible Producer or ringing the Send Doorbell. | An invalid placeholder event with `handle == 0`; it must not be used to determine completion. |

One or more successful non-empty Defer calls form the current aggregate batch. Publish that batch with
`SubmitAsyncPutBatch`. Only the event returned by `SubmitAsyncPutBatch` represents completion of the batch.

## SubmitAsyncPutBatch Helper

`SubmitAsyncPutBatch` is a `PTO_INTERNAL` helper function, not a separate PTO instruction:

```cpp
// A2/A3 SDMA and A5 bound-peer form
template <DmaEngine engine = DmaEngine::SDMA>
PTO_INTERNAL AsyncEvent SubmitAsyncPutBatch(const AsyncSession& session);

// A5 explicit-peer form
template <DmaEngine engine = DmaEngine::SDMA>
PTO_INTERNAL AsyncEvent SubmitAsyncPutBatch(
    const AsyncSession& session, uint32_t peer,
    uint32_t jettyIndex = 0U);
```

For A2/A3 SDMA, use the session-only form. For A5 URMA, the session-only form publishes the `peer` and
`jettyIndex` bound by the first successful Defer; the explicit-peer form requires values identical to every Defer in
the batch. A valid returned event covers every non-empty write in the batch. Check `event.valid()` before `Wait` or
`Test`.

`AsyncSession` is an engine-agnostic session object. Build once with
`BuildAsyncSession<engine>()`, then pass to all async calls and event waits.
The template `engine` parameter selects the DMA backend at compile time, making the
code forward-compatible with future engines (CCU, etc.).

## AsyncSession Construction

Use `BuildAsyncSession` from `include/pto/comm/async_common/async_event_impl.hpp`.
There are two overloads — one for SDMA and one for URMA — with different parameter lists.

### SDMA Construction (default)

```cpp
template <DmaEngine engine = DmaEngine::SDMA, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(ScratchTile &scratchTile,
                                    __gm__ uint8_t *workspace,
                                    AsyncSession &session,
                                    uint32_t syncId = 0,
                                    const sdma::SdmaBaseConfig &baseConfig = {sdma::kDefaultSdmaBlockBytes, 0, 1},
                                    uint32_t channelGroupIdx = sdma::kAutoChannelGroupIdx);
```

| Parameter | Default | Description |
|---|---|---|
| `scratchTile` | — | UB scratch tile for SDMA control metadata (see [scratchTile Role](#scratchtile-role)). |
| `workspace` | — | GM pointer allocated by host-side `SdmaWorkspaceManager`. |
| `session` | — | Output `AsyncSession` object. |
| `syncId` | `0` | MTE3/MTE2 pipe sync event id (0-7). Override if kernel uses other pipe barriers on the same id. |
| `baseConfig` | `{kDefaultSdmaBlockBytes, 0, 1}` | `{block_bytes, comm_block_offset, queue_num}`. Suitable for most single-queue transfers. |
| `channelGroupIdx` | `kAutoChannelGroupIdx` | SDMA channel group index. Default uses `get_block_idx()` internally, mapping to current AI core. Override for multi-block, concurrent, or custom channel mapping scenarios. |

### URMA Construction (NPU_ARCH 3510 only)

> URMA (User-level RDMA Memory Access) is a hardware-accelerated RDMA transport available on Ascend950 (NPU_ARCH 3510).
> URMA requires CANN Toolkit **>= 9.1.0**.

```cpp
#ifdef PTO_URMA_SUPPORTED
template <DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t *workspace,
                                    uint32_t destRankId,
                                    AsyncSession &session);
#endif
```

| Parameter | Description |
|---|---|
| `workspace` | GM pointer allocated by host-side `UrmaWorkspaceManager`. |
| `destRankId` | Remote PE rank id that this session communicates with. For `TPUT_ASYNC` this is the destination rank. |
| `session` | Output `AsyncSession` object. |

URMA does not require `scratchTile` — polling uses `ld_dev`/`st_dev` hardware intrinsics directly.

### RDMA Construction (NPU_ARCH 3510 only)

RDMA is intended for standard cross-node networking and currently supports only the HNS1825 NIC platform.

```cpp
#ifdef PTO_RDMA_SUPPORTED
template <DmaEngine engine, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(ScratchTile &scratchTile,
                                    __gm__ uint8_t *workspace,
                                    uint32_t myPe,
                                    AsyncSession &session,
                                    uint32_t syncId = 0);
#endif
```

| Parameter | Description |
|---|---|
| `scratchTile` | UB/Vec tile used for WQE/CQE control data; at least 64 bytes. |
| `workspace` | GM pointer returned by the host-side RDMA initialization flow. |
| `myPe` | Local rank id used to select the registered local memory region. |
| `session` | Output `AsyncSession` object. |
| `syncId` | MTE/scalar synchronization event id in the range 0-7. |

Build one peer-independent session, then select the destination rank with the explicit `peer` argument:

```cpp
comm::AsyncSession session;
if (comm::BuildAsyncSession<comm::DmaEngine::RDMA>(scratchTile, rdmaWorkspace, myPe, session, syncId)) {
    auto event = comm::TPUT_ASYNC<comm::DmaEngine::RDMA>(dstG, srcG, session, peer);
    (void)event.Wait(session);
}
```

## Constraints

- `GlobalSrcData::RawDType == GlobalDstData::RawDType`
- `GlobalSrcData::layout == GlobalDstData::layout`
- Both SDMA and URMA paths require source tensor to be **flat contiguous logical 1D only**
- SDMA workspace must be a valid GM pointer allocated by host-side `SdmaWorkspaceManager`
- URMA workspace must be a valid GM pointer allocated by host-side `UrmaWorkspaceManager`
- Keep the session and its workspace alive until all associated events have completed
- URMA is only available on NPU_ARCH 3510 (Ascend950)
- URMA requires CANN Toolkit **>= 9.1.0**
- The symmetric data buffer passed to `UrmaWorkspaceManager::Init()` must be backed by huge-page memory (allocate with `ACL_MEM_MALLOC_HUGE_ONLY`). The underlying MR registration requires huge-page backing; `ACL_MEM_MALLOC_HUGE_FIRST` may silently fall back to 4KB pages for small allocations, causing registration to fail
- RDMA is available only on Ascend950 / NPU_ARCH 3510 and currently supports only the HNS1825 NIC platform
- RDMA source and destination tensors must be flat contiguous logical 1D, and the complete local and remote ranges must lie within memory regions registered during host initialization
- One RDMA transfer must not exceed `0x7fffffff` bytes

If the 1D contiguous requirement is not met, current implementation returns an invalid async event (`handle == 0`).

### Deferred Batch Constraints

- Aggregate mode supports `DmaEngine::SDMA` on A2/A3 and `DmaEngine::URMA` on A5. A5 SDMA aggregate mode is not
  supported.
- A2/A3 does not expose `peer` or `jettyIndex`. A5 retains both the bound-peer and explicit-peer forms.
- The first successful non-empty Defer starts a batch. A zero-length Defer is a no-op.
- Every Defer and the final Submit must use the same session and engine. One URMA batch must also keep the same
  `peer` and `jettyIndex`.
- Do not interleave ordinary `TPUT_ASYNC`, `TGET_ASYNC`, `TPUT_ASYNC_NOTIFY`, another aggregate batch, or
  `TPREFETCH_ASYNC` on the same Channel Group or Jetty between the first Defer and Submit.
- Do not copy, rebuild, destroy, or transfer ownership of the session before Submit.
- Keep every source range, session, and workspace alive until the Submit event has completed.
- If Defer or Submit detects an argument, resource, or capacity error before publication, the current staged batch is
  discarded and is not published. Because Defer always returns an invalid placeholder, that return value does not
  distinguish success from failure; callers must satisfy all preconditions before the first Defer.
- Submit requires at least one successful non-empty Defer. An empty Submit returns an invalid event and reports a
  contract violation in assert-enabled builds.
- After Submit, the session's hidden batch state is cleared and a new batch may be staged. This does not imply that
  the previous batch has completed.
- Remote destination ranges in one batch must not overlap. Defer order must not be used to express dependencies
  between writes.

### Deferred Batch Resource Limits

- Source and destination tensors must have the same raw data type and layout, must both be flat contiguous logical 1D
  tensors, and must provide non-null pointers for every non-empty transfer.
- Tensor element counts, byte counts, and address-range calculations must fit in `uint64_t`. The destination element
  capacity must be at least the source element count.
- The interface does not validate communication-memory membership, address bounds, overlap, or target-peer ownership;
  the caller must guarantee them.
- Each SDMA Defer consumes `ceil(transferBytes / blockBytes)` data SQEs. Distribution is round-robin across
  `queueNum` using the cumulative SQE index of the complete batch. The number assigned to each queue must be less
  than that queue's SQ depth.
- SDMA adds `commBlockOffset` to both tensor base addresses. The backing allocations must cover
  `[base + commBlockOffset, base + commBlockOffset + transferBytes)` without overflow.
- URMA requires a valid registered workspace, peer memory registration, and Jetty. `peer` must be less than the
  workspace rank count and `jettyIndex < session.qpCount`.
- The complete URMA source range must be within the local communication buffer registered by
  `UrmaWorkspaceManager::Init()`, and the complete destination range must be within the selected peer's registered
  communication buffer.
- URMA splits one remote write into WQEs of at most 256 MiB. The complete batch must fit both the WQ and CQ depths of
  the selected Jetty. Split larger workloads into multiple batches; the interface does not automatically submit a
  partial batch.

## scratchTile Role

`scratchTile` is **not** the payload staging buffer for user data.
It is converted to `TmpBuffer` and used as temporary UB workspace for:

- writing/reading SDMA control words (flag, sq_tail, channel_info)
- polling event completion flags
- committing queue tail during completion

Data payload moves between GM buffers directly; `scratchTile` only supports control and synchronization metadata.

## scratchTile Type and Size Constraints

- must be a `pto::Tile` type
- must be UB/Vec tile (`ScratchTile::Loc == TileType::Vec`)
- available bytes must be at least `sizeof(uint64_t)` (8 bytes)

Recommended: `Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>` (256Byte).

## Completion Semantics (Quiet Semantics)

For `IMMEDIATE` mode, the completion mechanism differs by engine, but user-facing quiet semantics are identical:

- **SDMA**: Each `TPUT_ASYNC` submits data-transfer SQEs and flag SQEs that mark completion of that operation. `Wait` or `Test` on its returned event polls the corresponding flags to determine whether that `TPUT_ASYNC` has completed; completion also guarantees that all earlier SDMA operations in the same session have completed.
- **URMA**: `TPUT_ASYNC` submits an RDMA WRITE WQE and rings the doorbell immediately. `Wait` polls the Completion Queue (CQ) until all expected CQEs have been consumed.
- **RDMA**: `TPUT_ASYNC` submits an RDMA WRITE WQE to the queue selected by `peer`. Completion is tracked independently for each peer/queue.

- `event.Wait(session)` — blocks until **all async operations issued since the last Wait** are complete

This means after multiple `TPUT_ASYNC` calls, a single `Wait` on the last returned `AsyncEvent` drains all pending operations (similar to shmem's quiet semantics).

For RDMA operations targeting different peers, wait for the last event of each peer separately.

Up to 64 operations may be outstanding in one session before submission can apply backpressure.

After wait succeeds, all issued writes to `dstGlobalData` are complete.

For `DEFER` mode, the placeholder event returned by each `TPUT_ASYNC` is deliberately invalid. Calling `Wait` or
`Test` on that placeholder does not prove that the transfer was published or completed. Wait only on the valid event
returned by `SubmitAsyncPutBatch`. Submit only publishes the staged work; it does not mean that the batch has
completed. A successful Wait/Test on the Submit event provides sender-side completion and permits source reuse.

An aggregate batch is not a transaction. The receiver may observe writes incrementally, completed writes are not
rolled back after an error, and the public contract does not define completion order between individual writes. The
Submit event does not notify the receiver. Receiver consumption still requires an application protocol such as
`TNOTIFY`/`TWAIT` and platform-appropriate cache-visibility handling.

Only a successful Wait/Test result confirms completion. A failed result does not cancel the submitted transfer; do
not reuse source data or release/rebuild the session, workspace, scratch tile, Channel Group, or Jetty in that state.

## Concurrency and Session Ownership

- Do not use one session concurrently from multiple execution flows.
- Operations that share a channel group must also share the same session.
- Concurrent kernels, or multiple independent sessions within one kernel, must use isolated channel groups.
- Complete all outstanding events before rebuilding a session or reusing its channel group.
- Build/Rebuild, Defer, Submit, Wait, and Test on one session must execute serially.
- Distinct sessions do not imply distinct physical resources. From Build until the last event completes, do not map a
  second session to the same SDMA Channel, URMA WQ, or URMA CQ.
- In URMA `PER_PEER` mode, calls to the same peer share one physical WQ/CQ and must be serialized; different peers use
  different queues.
- In URMA `SHARED_POOL` mode, the physical Jetty is `session.qpIdxBase + jettyIndex`. Sessions on the same AIV alias
  when this physical index is equal, even when their peers differ.
- Wait/Test on a physical Channel Group, WQ, or CQ must not run concurrently with Defer, Submit, or another Wait/Test
  on the same resource.
- Rebuild or destroy a session and its communication resources only after every outstanding event on all physical
  queues used by that session has completed.
- Before releasing URMA communication resources, also synchronize every host stream that uses the communication
  context so that no later stream work can reference the context.

## Example

### Single Transfer

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/common/pto_tile.hpp>

using namespace pto;

template <typename T>
__global__ AICORE void SimplePut(__gm__ T *remoteDst, __gm__ T *localSrc,
                                 __gm__ uint8_t *sdmaWorkspace)
{
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT dstG(remoteDst, shape, stride);
    GT srcG(localSrc, shape, stride);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::SDMA>(scratchTile, sdmaWorkspace, session)) {
        return;
    }

    auto event = comm::TPUT_ASYNC<comm::DmaEngine::SDMA>(dstG, srcG, session);
    (void)event.Wait(session);
}
```

### Multiple Immediate Transfers (Quiet Semantics)

```cpp
template <typename T>
__global__ AICORE void BatchPut(__gm__ T *remoteDstBase, __gm__ T *localSrc,
                                __gm__ uint8_t *sdmaWorkspace, int nranks)
{
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT srcG(localSrc, shape, stride);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session)) {
        return;
    }

    comm::AsyncEvent lastEvent;
    for (int rank = 0; rank < nranks; ++rank) {
        GT dstG(remoteDstBase + rank * 1024, shape, stride);
        lastEvent = comm::TPUT_ASYNC(dstG, srcG, session);
    }
    (void)lastEvent.Wait(session);  // single Wait drains all pending ops
}
```

### Aggregate Batch Transfer

```cpp
template <typename GT>
AICORE void AggregatePutSdma(
    GT& dst0, GT& src0, GT& dst1, GT& src1,
    const comm::AsyncSession& session)
{
    (void)comm::TPUT_ASYNC<comm::DmaEngine::SDMA>(
        dst0, src0, session, comm::AsyncPutMode::DEFER);
    (void)comm::TPUT_ASYNC<comm::DmaEngine::SDMA>(
        dst1, src1, session, comm::AsyncPutMode::DEFER);

    auto batchEvent =
        comm::SubmitAsyncPutBatch<comm::DmaEngine::SDMA>(session);
    if (!batchEvent.valid() || !batchEvent.Wait(session)) {
        return;
    }
}
```

The invalid Defer return is a placeholder, not a success status. Validate all arguments and batch capacity before
staging; only the valid Submit event can be used for completion.

For A5 URMA with an explicit peer, keep the original peer position and use the same `peer` and `jettyIndex` for all
Defer calls and Submit:

```cpp
(void)comm::TPUT_ASYNC<comm::DmaEngine::URMA>(
    dst0, src0, session, peer, comm::AsyncPutMode::DEFER, jettyIndex);
(void)comm::TPUT_ASYNC<comm::DmaEngine::URMA>(
    dst1, src1, session, peer, comm::AsyncPutMode::DEFER, jettyIndex);
auto batchEvent = comm::SubmitAsyncPutBatch<comm::DmaEngine::URMA>(
    session, peer, jettyIndex);
```

### URMA Example (NPU_ARCH 3510)

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/common/pto_tile.hpp>

using namespace pto;

template <typename T>
__global__ AICORE void SimplePutUrma(__gm__ T *remoteDst, __gm__ T *localSrc,
                                     __gm__ uint8_t *urmaWorkspace, uint32_t destRankId)
{
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT dstG(remoteDst, shape, stride);
    GT srcG(localSrc, shape, stride);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::URMA>(urmaWorkspace, destRankId, session)) {
        return;
    }

    auto event = comm::TPUT_ASYNC<comm::DmaEngine::URMA>(dstG, srcG, session);
    (void)event.Wait(session);
}
```
