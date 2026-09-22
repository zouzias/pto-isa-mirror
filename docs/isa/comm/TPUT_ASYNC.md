# TPUT_ASYNC

## Introduction

`TPUT_ASYNC` is an asynchronous remote write primitive. By default it publishes a transfer from local GM to remote
GM and returns its `AsyncEvent`. On A2/A3 SDMA and A5 URMA, `AsyncSession::submitMode` can instead stage several
writes and publish them as a batch without changing the `TPUT_ASYNC` function signature.

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
                               WaitEvents &... events);

// A5, peer obtained from session.destRankId
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session,
                               WaitEvents &... events);

// A5, explicit peer
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session, uint32_t peer,
                               WaitEvents &... events);
```

The call signatures and the position of `peer` are unchanged. Configure batching on the Session after
`BuildAsyncSession`. Prerequisite `WaitEvents...` retain their original call form.

## Submission Modes

```cpp
session.submitMode = AsyncSubmitMode::DEFER;
session.batchSize = 16U;
```

- `AsyncSubmitMode::IMMEDIATE` (default): first publishes any pending batch for the same Session and engine, then
  publishes the current write. The returned Event covers both.
- `AsyncSubmitMode::DEFER`: stages the current write. The returned non-zero Event is a future completion-target
  snapshot. It may be waited or tested only after the physical batch containing it has been submitted.
- `AsyncSubmitMode::DEFER_AND_SUBMIT`: stages the current write and submits the remaining physical batch. The
  returned Event is immediately eligible for `Wait/Test`.

`batchSize` is the maximum number of non-empty deferred calls in one physical batch. Reaching the threshold submits
that physical batch automatically. `batchSize == UINT32_MAX` effectively disables threshold-based submission;
switch the last real write to `DEFER_AND_SUBMIT`, or execute a following same-engine immediate operation.
`batchSize == 0` disables batching, so `DEFER` and `DEFER_AND_SUBMIT` are contract violations.

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
- A2/A3 does not expose `peer`. A5 retains both the bound-peer and explicit-peer forms.
- The first successful non-empty Defer starts a batch. A zero-length Defer is a no-op.
- Every deferred write in one logical batch must use the same Session and engine. One URMA logical batch must also
  keep the same `peer`.
- A same-engine immediate `TPUT_ASYNC`, `TGET_ASYNC`, or `TPUT_ASYNC_NOTIFY` using the same Session first submits
  pending Put descriptors. `TPREFETCH_ASYNC` does so only when it reuses that external SDMA Session.
- This implicit submission establishes local publication order only. An A5 URMA notify posted on one Jetty is not a
  remote completion fence for deferred writes posted on other Jetties; wait for the Batch Event or use an explicit
  ordering protocol before treating the signal as proof that all Batch payloads are remotely visible.
- A5 `TPUT_ASYNC<SDMA>` uses the synchronous MTE fallback and never submits or otherwise changes a pending URMA
  batch.
- Do not copy, rebuild, destroy, or transfer ownership of the Session while it has pending work.
- Keep every source range, Session, and workspace alive until the corresponding Event has completed.
- If a deferred call detects an argument, resource, or capacity error before publication, the current unsubmitted
  physical batch is discarded and is not published.
- An automatically submitted physical batch clears its staged counters but remains part of the logical completion
  prefix. A subsequent deferred call begins another physical batch.
- Remote destination ranges in one batch must not overlap. Defer order must not be used to express dependencies
  between writes.
- Before submission, do not call `Wait/Test` on a deferred Event and do not pass it as a prerequisite Event. This
  release relies on the caller to enforce that rule and performs no runtime submitted-state check.

### Deferred Batch Resource Limits

- Source and destination tensors must have the same raw data type and layout, must both be flat contiguous logical 1D
  tensors, and must provide non-null pointers for every non-empty transfer.
- Tensor element counts, byte counts, and address-range calculations must fit in `uint64_t`. The destination element
  capacity must be at least the source element count.
- The interface does not validate communication-memory membership, address bounds, overlap, or target-peer ownership;
  the caller must guarantee them.
- Each SDMA Defer consumes `ceil(transferBytes / blockBytes)` data SQEs. Distribution is round-robin across
  `queueNum` using the cumulative SQE index of the current physical batch. The number assigned to each queue must be
  less than that queue's SQ depth.
- SDMA adds `commBlockOffset` to both tensor base addresses. The backing allocations must cover
  `[base + commBlockOffset, base + commBlockOffset + transferBytes)` without overflow.
- URMA requires a valid registered workspace and peer memory registration. `peer` must be less than the workspace
  rank count.
- The complete URMA source range must be within the local communication buffer registered by
  `UrmaWorkspaceManager::Init()`, and the complete destination range must be within the selected peer's registered
  communication buffer.
- URMA splits one remote write into WQEs of at most 256 MiB and distributes them deterministically over
  `session.qpCount` Jetties. Each physical batch must fit every participating WQ and CQ.

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

- For an Event returned by an `IMMEDIATE` call, `event.Wait(session)` blocks until that operation and the earlier
  same-session operations covered by its completion target are complete.

This means that after multiple `IMMEDIATE` `TPUT_ASYNC` calls, one `Wait` on the last returned `AsyncEvent` drains
the covered pending operations (similar to shmem's quiet semantics). A `DEFER` Event instead uses the prefix
snapshot semantics described below.

For RDMA operations targeting different peers, wait for the last event of each peer separately.

Up to 64 operations may be outstanding in one session before submission can apply backpressure.

After wait succeeds, all issued writes to `dstGlobalData` are complete.

For `DEFER` mode, every non-empty call returns a future completion-target snapshot. Before the physical batch
containing that snapshot is submitted, calling `Wait/Test` is unsupported. After submission, an intermediate Event
can check completion of its deferred prefix, and the Event returned by the last call covers the complete logical
prefix. A successful Wait/Test provides sender-side completion and permits reuse of the corresponding source range.

An aggregate batch is not a transaction. The receiver may observe writes incrementally, completed writes are not
rolled back after an error, and the public contract does not define completion order between individual writes. The
final Event does not notify the receiver. Receiver consumption still requires an application protocol such as
`TNOTIFY`/`TWAIT` and platform-appropriate cache-visibility handling.

Only a successful Wait/Test result confirms completion. A failed result does not cancel the submitted transfer; do
not reuse source data or release/rebuild the session, workspace, scratch tile, Channel Group, or Jetty in that state.

## SDMA Concurrency and Session Ownership

- Do not use one session concurrently from multiple execution flows.
- Operations that share a channel group must also share the same session.
- Concurrent kernels, or multiple independent sessions within one kernel, must use isolated channel groups.
- Complete all outstanding events before rebuilding a session or reusing its channel group.

### Deferred Batch Concurrency and Session Ownership

- Build/Rebuild, instruction calls, Wait, and Test on one Session must execute serially.
- Distinct sessions do not imply distinct physical resources. From Build until the last event completes, do not map a
  second session to the same SDMA Channel, URMA WQ, or URMA CQ.
- In URMA `PER_PEER` mode, calls to the same peer share one physical WQ/CQ and must be serialized; different peers use
  different queues.
- In URMA `SHARED_POOL` mode, a batching Session owns the Jetty range
  `[session.qpIdxBase, session.qpIdxBase + session.qpCount)`. Sessions alias when these ranges overlap, even when
  their peers differ.
- Wait/Test on a physical Channel Group, WQ, or CQ must not run concurrently with Defer, submission, or another Wait/Test
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

### Batch Transfer (Quiet Semantics)

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
    comm::AsyncSession& session)
{
    session.batchSize = UINT32_MAX;
    session.submitMode = comm::AsyncSubmitMode::DEFER;
    auto firstEvent =
        comm::TPUT_ASYNC<comm::DmaEngine::SDMA>(dst0, src0, session);
    session.submitMode = comm::AsyncSubmitMode::DEFER_AND_SUBMIT;
    auto batchEvent =
        comm::TPUT_ASYNC<comm::DmaEngine::SDMA>(dst1, src1, session);
    if (!batchEvent.valid() || !batchEvent.Wait(session)) {
        return;
    }
}
```

`firstEvent` is a valid future snapshot, but it must not be waited before the second call submits its physical batch.
After submission, either Event may be waited; `batchEvent` covers both writes.

For A5 URMA with an explicit peer, keep the original peer position and use the same `peer` for the complete logical
batch. WQEs are distributed over the Jetty range owned by the Session:

```cpp
session.batchSize = UINT32_MAX;
session.submitMode = comm::AsyncSubmitMode::DEFER;
auto firstEvent = comm::TPUT_ASYNC<comm::DmaEngine::URMA>(
    dst0, src0, session, peer);
session.submitMode = comm::AsyncSubmitMode::DEFER_AND_SUBMIT;
auto batchEvent = comm::TPUT_ASYNC<comm::DmaEngine::URMA>(
    dst1, src1, session, peer);
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
