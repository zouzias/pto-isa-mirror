# TGET_ASYNC

## Introduction

`TGET_ASYNC` is an asynchronous remote read primitive. It starts a transfer from remote GM to local GM and returns an `AsyncEvent` immediately.

Data flow:

`srcGlobalData (remote GM) -> DMA engine -> dstGlobalData (local GM)`

## Template Parameter

- `engine`:
    - `DmaEngine::SDMA` (default)
    - `DmaEngine::URMA` (Ascend950, NPU_ARCH 3510 only)
    - `DmaEngine::RDMA` (Ascend950, NPU_ARCH 3510 only; currently HNS1825)

> **Important (SDMA path)**
> `TGET_ASYNC` with `DmaEngine::SDMA` currently supports **only flat contiguous logical 1D tensors**.
> Non-1D or non-contiguous layouts are not supported by the current SDMA async implementation.

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`.

```cpp
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TGET_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session, WaitEvents &... events);

// A5: explicit peer overload (used by URMA/RDMA; ignored by SDMA).
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TGET_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session, uint32_t peer,
                               WaitEvents &... events);
```

`AsyncSession` is an engine-agnostic session object. Build once with
`BuildAsyncSession<engine>()`, then pass to all async calls and event waits.
The template `engine` parameter selects the DMA engine at compile time, making the
code forward-compatible with future engines (CCU, etc.).

## AsyncSession Construction

Use `BuildAsyncSession` from `include/pto/comm/async_common/async_event_impl.hpp`.
There are separate overloads for SDMA, URMA, and RDMA, with different parameter lists.

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
| `destRankId` | Remote PE rank id that this session communicates with. For `TGET_ASYNC` this is the source rank. |
| `session` | Output `AsyncSession` object. |

URMA does not require `scratchTile` — polling uses `ld_dev`/`st_dev` hardware intrinsics directly.

### RDMA Construction (HNS1825, NPU_ARCH 3510 only)

`DmaEngine::RDMA` selects the RDMA path. The Device workspace records the NIC backend; the current build supports only
HNS1825.

```cpp
#ifdef PTO_RDMA_SUPPORTED
template <DmaEngine engine, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(ScratchTile &scratchTile,
                                    __gm__ uint8_t *workspace,
                                    uint32_t myPe,
                                    AsyncSession &session,
                                    uint32_t syncId = 0);

template <DmaEngine engine, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(ScratchTile &scratchTile,
                                    __gm__ uint8_t *workspace,
                                    uint32_t destRankId,
                                    uint32_t myPe,
                                    AsyncSession &session,
                                    uint32_t syncId = 0);
#endif
```

| Parameter | Default | Description |
|---|---|---|
| `scratchTile` | — | UB/Vec scratch used to stage one HNS1825 WQE and inspect CQEs; at least 64 bytes. |
| `workspace` | — | Device workspace returned by host-side `rdma::RdmaWorkspaceManager::GetWorkspaceAddr()`. |
| `destRankId` | — | Source rank used only by the peer-bound overload. |
| `myPe` | — | Local rank id, used to select the local registered MR. |
| `session` | — | Output `AsyncSession` object. |
| `syncId` | `0` | MTE3/S pipe event id in `[0, 7]`; it must not conflict with other kernel synchronization. |

The first overload builds a peer-independent session for
`TGET_ASYNC(..., session, peer)` and can be reused across remote ranks. The second overload binds `destRankId` for the
original `TGET_ASYNC(..., session)` form.

The host must initialize the RDMA control plane and exchange peer addresses before kernel launch. See
[RDMA Backend and Host Control Plane](README.md#rdma-backend-and-host-control-plane).

## Constraints

- `GlobalSrcData::RawDType == GlobalDstData::RawDType`
- `GlobalSrcData::layout == GlobalDstData::layout`
- Both SDMA and URMA paths require source tensor to be **flat contiguous logical 1D only**
- The RDMA path requires both source and destination tensors to be **flat contiguous logical 1D**
- SDMA workspace must be a valid GM pointer allocated by host-side `SdmaWorkspaceManager`
- URMA workspace must be a valid GM pointer allocated by host-side `UrmaWorkspaceManager`
- Keep the session and its workspace alive until all associated events have completed
- URMA is only available on NPU_ARCH 3510 (Ascend950)
- URMA requires CANN Toolkit **>= 9.1.0**
- The symmetric data buffer passed to `UrmaWorkspaceManager::Init()` must be backed by huge-page memory (allocate with `ACL_MEM_MALLOC_HUGE_ONLY`). The underlying MR registration requires huge-page backing; `ACL_MEM_MALLOC_HUGE_FIRST` may silently fall back to 4KB pages for small allocations, causing registration to fail
- RDMA must be enabled at configure time and its workspace must come from `rdma::RdmaWorkspaceManager`
- The current HNS1825 RDMA backend is available only on NPU_ARCH 3510 (Ascend950)
- Both the remote source range and local destination range must be fully contained in the MRs registered by the RDMA
  manager
- A single HNS1825 transfer is limited to `0x7fffffff` bytes

If the 1D contiguous requirement is not met, current implementation returns an invalid async event (`handle == 0`).

## scratchTile Role

`scratchTile` is **not** used to hold transferred payload data.
It is converted to the common `AsyncTmpBuffer` descriptor and used as temporary UB workspace for:

- writing/reading SDMA control words (flag, sq_tail, channel_info)
- polling event completion flags
- committing queue tail during completion

The real payload path remains remote GM -> DMA engine -> local GM; `scratchTile` is only for control/synchronization metadata.

## scratchTile Type and Size Constraints

- must be a `pto::Tile` type
- must be UB/Vec tile (`ScratchTile::Loc == TileType::Vec`)
- available bytes must be at least `sizeof(uint64_t)` (8 bytes)

Recommended: `Tile<TileType::Vec, uint8_t, 1, comm::kDefaultAsyncScratchBytes>` (256Byte).

For `DmaEngine::RDMA`, the same Tile requirements apply but the available size must be at least 64 bytes. The scratch
is used for HNS1825 WQE/CQE control data, not payload data.

## Completion Semantics (Quiet Semantics)

The completion scope of `event.Wait(session)` depends on the selected engine and returned event:

- **SDMA**: Each `TGET_ASYNC` submits data-transfer SQEs and flag SQEs that mark completion of that operation. `Wait` or `Test` on its returned event polls the corresponding flags; completion also guarantees that all earlier SDMA operations submitted by the same session on the queues covered by that event have completed.
- **URMA**: `TGET_ASYNC` submits an RDMA READ WQE and rings the doorbell immediately. `Wait` consumes that peer/QP's Completion Queue (CQ) through the producer index encoded in the event.
- **RDMA/HNS1825**: `TGET_ASYNC` posts an RDMA READ WQE and rings the SQ doorbell immediately. `Wait` consumes CQEs
  through the target producer index; `Test` performs a non-consuming readiness check.

For consecutive SDMA submissions in one session, waiting on the last returned `AsyncEvent` is sufficient. For
URMA/RDMA, this is sufficient only for operations on the same peer/QP; different peers/QPs must be completed
separately.

The SDMA implementation allows up to 64 outstanding operations in one session before submission can apply
backpressure. RDMA queue capacity comes from the selected backend's HCOMM queue contexts.

After wait succeeds, all issued reads into `dstGlobalData` are complete.

## SDMA Concurrency and Session Ownership

- Do not use one session concurrently from multiple execution flows.
- Operations that share a channel group must also share the same session.
- Concurrent kernels, or multiple independent sessions within one kernel, must use isolated channel groups.
- Complete all outstanding events before rebuilding a session or reusing its channel group.

## Example

### Single Transfer

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/common/pto_tile.hpp>

using namespace pto;

template <typename T>
__global__ AICORE void SimpleGet(__gm__ T *localDst, __gm__ T *remoteSrc,
                                 __gm__ uint8_t *sdmaWorkspace)
{
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, comm::kDefaultAsyncScratchBytes>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT dstG(localDst, shape, stride);
    GT srcG(remoteSrc, shape, stride);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::SDMA>(scratchTile, sdmaWorkspace, session)) {
        return;
    }

    auto event = comm::TGET_ASYNC<comm::DmaEngine::SDMA>(dstG, srcG, session);
    (void)event.Wait(session);
}
```

### Batch Transfer (Quiet Semantics)

```cpp
template <typename T>
__global__ AICORE void BatchGet(__gm__ T *localDstBase, __gm__ T *remoteSrcBase,
                                __gm__ uint8_t *sdmaWorkspace, int nranks)
{
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, comm::kDefaultAsyncScratchBytes>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session)) {
        return;
    }

    comm::AsyncEvent lastEvent;
    for (int rank = 0; rank < nranks; ++rank) {
        GT dstG(localDstBase + rank * 1024, shape, stride);
        GT srcG(remoteSrcBase + rank * 1024, shape, stride);
        lastEvent = comm::TGET_ASYNC(dstG, srcG, session);
    }
    (void)lastEvent.Wait(session);  // single Wait drains all pending ops
}
```

### URMA Example (NPU_ARCH 3510)

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/common/pto_tile.hpp>

using namespace pto;

template <typename T>
__global__ AICORE void SimpleGetUrma(__gm__ T *localDst, __gm__ T *remoteSrc,
                                     __gm__ uint8_t *urmaWorkspace, uint32_t srcRankId)
{
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT dstG(localDst, shape, stride);
    GT srcG(remoteSrc, shape, stride);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::URMA>(urmaWorkspace, srcRankId, session)) {
        return;
    }

    auto event = comm::TGET_ASYNC<comm::DmaEngine::URMA>(dstG, srcG, session);
    (void)event.Wait(session);
}
```

### RDMA/HNS1825 Example (NPU_ARCH 3510)

The Host passes the workspace returned by `RdmaWorkspaceManager`. Remote addresses are derived from the registered peer
MR base plus an application-defined offset.

```cpp
template <typename T>
__global__ AICORE void SimpleGetRdma(__gm__ T *localDst, __gm__ uint8_t *rdmaWorkspace,
                                     uint32_t myPe, uint32_t srcRankId, uint64_t remoteOffset)
{
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, comm::kDefaultAsyncScratchBytes>;

    const uint64_t peerBase = comm::rdma::PeerMrBaseAddr(rdmaWorkspace, srcRankId);
    if (peerBase == 0) {
        return;
    }

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT dstG(localDst, shape, stride);
    GT srcG(reinterpret_cast<__gm__ T *>(peerBase + remoteOffset), shape, stride);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::RDMA>(
            scratchTile, rdmaWorkspace, myPe, session)) {
        return;
    }

    auto event = comm::TGET_ASYNC<comm::DmaEngine::RDMA>(dstG, srcG, session, srcRankId);
    if (!event.valid() || !event.Wait(session)) {
        return;
    }
}
```
