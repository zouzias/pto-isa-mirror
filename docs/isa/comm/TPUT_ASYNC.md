# TPUT_ASYNC

## Introduction

`TPUT_ASYNC` is an asynchronous remote write primitive. It starts a transfer from local GM to remote GM and returns an `AsyncEvent` immediately.

Data flow:

`srcGlobalData (local GM) -> DMA engine -> dstGlobalData (remote GM)`


## Template Parameter

- `engine`:
    - `DmaEngine::SDMA` (default)
    - `DmaEngine::URMA` (Ascend950, NPU_ARCH 3510 only)
    - `DmaEngine::RDMA` (Ascend950, NPU_ARCH 3510 only; currently HNS1825)

> **Important (SDMA path)**
> `TPUT_ASYNC` with `DmaEngine::SDMA` currently supports **only flat contiguous logical 1D tensors**.
> Non-1D or non-contiguous layouts are not supported by the current SDMA async implementation.


## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`.

```cpp
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session, WaitEvents &... events);
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
| `channelGroupIdx` | `kAutoChannelGroupIdx` | SDMA channel group index. Default uses `get_block_idx()` internally, mapping to current AI core. Override for multi-block or custom channel mapping scenarios. |

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

### RDMA Construction (HNS1825, NPU_ARCH 3510 only)

`DmaEngine::RDMA` selects the RDMA path. The Device workspace records the NIC backend; the current build supports only
HNS1825.

```cpp
#ifdef PTO_RDMA_SUPPORTED
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
| `destRankId` | — | Destination rank for this PUT session. |
| `myPe` | — | Local rank id, used to select the local registered MR. |
| `session` | — | Output `AsyncSession` object. |
| `syncId` | `0` | MTE3/S pipe event id in `[0, 7]`; it must not conflict with other kernel synchronization. |

The host must initialize the RDMA control plane and exchange peer addresses before kernel launch. See
[RDMA Backend and Host Control Plane](README.md#rdma-backend-and-host-control-plane).

## Constraints

- `GlobalSrcData::RawDType == GlobalDstData::RawDType`
- `GlobalSrcData::layout == GlobalDstData::layout`
- Both SDMA and URMA paths require source tensor to be **flat contiguous logical 1D only**
- The RDMA path requires both source and destination tensors to be **flat contiguous logical 1D**
- SDMA workspace must be a valid GM pointer allocated by host-side `SdmaWorkspaceManager`
- URMA workspace must be a valid GM pointer allocated by host-side `UrmaWorkspaceManager`
- URMA is only available on NPU_ARCH 3510 (Ascend950)
- URMA requires CANN Toolkit **>= 9.1.0**
- The symmetric data buffer passed to `UrmaWorkspaceManager::Init()` must be backed by huge-page memory (allocate with `ACL_MEM_MALLOC_HUGE_ONLY`). The underlying MR registration requires huge-page backing; `ACL_MEM_MALLOC_HUGE_FIRST` may silently fall back to 4KB pages for small allocations, causing registration to fail
- RDMA must be enabled at configure time and its workspace must come from `rdma::RdmaWorkspaceManager`
- The current HNS1825 RDMA backend is available only on NPU_ARCH 3510 (Ascend950)
- Both the local source range and remote destination range must be fully contained in the MRs registered by the RDMA
  manager
- A single HNS1825 transfer is limited to `0x7fffffff` bytes

If the 1D contiguous requirement is not met, current implementation returns an invalid async event (`handle == 0`).

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

For `DmaEngine::RDMA`, the same Tile requirements apply but the available size must be at least 64 bytes. The scratch
is used for HNS1825 WQE/CQE control data, not payload data.

## Completion Semantics (Quiet Semantics)

The completion mechanism differs by engine, but user-facing quiet semantics are identical:

- **SDMA**: `TPUT_ASYNC` only submits data transfer SQEs. The flag SQE is deferred to `Wait`, which polls the flag for completion.
- **URMA**: `TPUT_ASYNC` submits an RDMA WRITE WQE and rings the doorbell immediately. `Wait` polls the Completion Queue (CQ) until all expected CQEs have been consumed.
- **RDMA/HNS1825**: `TPUT_ASYNC` posts an RDMA WRITE WQE and rings the SQ doorbell immediately. `Wait` consumes CQEs
  through the target producer index; `Test` performs a non-consuming readiness check.

- `event.Wait(session)` — blocks until **all async operations issued since the last Wait** are complete

This means after multiple `TPUT_ASYNC` calls, a single `Wait` on the last returned `AsyncEvent` drains all pending operations (similar to shmem's quiet semantics).

After wait succeeds, all issued writes to `dstGlobalData` are complete.

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

### RDMA/HNS1825 Example (NPU_ARCH 3510)

The Host passes the workspace returned by `RdmaWorkspaceManager`. Remote addresses are derived from the registered peer
MR base plus an application-defined offset.

```cpp
template <typename T>
__global__ AICORE void SimplePutRdma(__gm__ T *localSrc, __gm__ uint8_t *rdmaWorkspace,
                                     uint32_t myPe, uint32_t destRankId, uint64_t remoteOffset)
{
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    const uint64_t peerBase = comm::rdma::PeerMrBaseAddr(rdmaWorkspace, destRankId);
    if (peerBase == 0) {
        return;
    }

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT dstG(reinterpret_cast<__gm__ T *>(peerBase + remoteOffset), shape, stride);
    GT srcG(localSrc, shape, stride);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::RDMA>(
            scratchTile, rdmaWorkspace, destRankId, myPe, session)) {
        return;
    }

    auto event = comm::TPUT_ASYNC<comm::DmaEngine::RDMA>(dstG, srcG, session);
    if (!event.valid() || !event.Wait(session)) {
        return;
    }
}
```
