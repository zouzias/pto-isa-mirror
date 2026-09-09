# TGET_ASYNC

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-26T03:14:20.711Z pushedAt=2026-08-29T09:05:18.407Z -->

## Introduction

`TGET_ASYNC` is an asynchronous remote read primitive. It initiates a transfer from remote GM to local GM and returns an `AsyncEvent` immediately.

Data flow:

`srcGlobalData (remote GM)` → DMA engine → `dstGlobalData (local GM)`

## Template Parameters

- `engine`:
    - `DmaEngine::SDMA` (default)
    - `DmaEngine::URMA` (Ascend 950PR/Ascend 950DT, NPU_ARCH 3510 only)

> **Note (SDMA path)**
> `TGET_ASYNC` with `DmaEngine::SDMA` currently **only supports flat, contiguous, logical 1D tensors**.
> The current SDMA asynchronous implementation does not support non-1D or non-contiguous layouts.

## C++ Built-in APIs

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TGET_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session, WaitEvents &... events);
```

`AsyncSession` is an engine-agnostic session object. Build it once with `BuildAsyncSession<engine>()` and pass it to all asynchronous calls and event waits. The template parameter `engine` selects the DMA backend at compile time, keeping the code forward-compatible with future engines (such as CCU).

## AsyncSession Construction

Use `BuildAsyncSession` in `include/pto/comm/async_common/async_event_impl.hpp`. This function has two overloads, one for SDMA and one for URMA, with different parameter lists.

### SDMA Construction (Default)

```cpp
template <DmaEngine engine = DmaEngine::SDMA, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(ScratchTile &scratchTile,
                                    __gm__ uint8_t *workspace,
                                    AsyncSession &session,
                                    uint32_t syncId = 0,
                                    const sdma::SdmaBaseConfig &baseConfig = {sdma::kDefaultSdmaBlockBytes, 0, 1},
                                    uint32_t channelGroupIdx = sdma::kAutoChannelGroupIdx);
```

| Parameter | Default Value | Description |
|---|---|---|
| `scratchTile` | — | UB scratch tile used for SDMA control metadata (see [Role of scratchTile](#role-of-scratchtile)). |
| `workspace` | — | GM pointer allocated by the host-side `SdmaWorkspaceManager`. |
| `session` | — | Output `AsyncSession` object. |
| `syncId` | `0` | MTE3/MTE2 pipe synchronization event ID (0-7). Override this value if the kernel uses other pipe barriers on the same ID. |
| `baseConfig` | `{kDefaultSdmaBlockBytes, 0, 1}` | `{block_bytes, comm_block_offset, queue_num}`. Suitable for most single-queue transfer scenarios. |
| `channelGroupIdx` | `kAutoChannelGroupIdx` | SDMA channel group index. By default, internally uses `get_block_idx()` to map to the current AI Core. Override this value in multi-block or custom channel mapping scenarios. |

### URMA Construction (NPU_ARCH 3510 Only)

> User-level RDMA Memory Access (URMA) is the hardware-accelerated RDMA transfer engine on Ascend 950PR/Ascend 950DT (NPU_ARCH 3510).
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
| `workspace` | GM pointer allocated by the host-side `UrmaWorkspaceManager`. |
| `destRankId` | Remote PE rank ID that this session communicates with. For `TGET_ASYNC`, this is the source rank from which data originates. |
| `session` | Output `AsyncSession` object. |

URMA does not require `scratchTile` — polling is performed directly through the `ld_dev`/`st_dev` hardware primitives.

## Constraints

- `GlobalSrcData::RawDType == GlobalDstData::RawDType`
- `GlobalSrcData::layout == GlobalDstData::layout`
- Both the SDMA and URMA paths require the source tensor to be **flat, contiguous, and logical 1D**.
- The SDMA workspace must be a valid GM pointer allocated by the host-side `SdmaWorkspaceManager`.
- The URMA workspace must be a valid GM pointer allocated by the host-side `UrmaWorkspaceManager`.
- URMA is available only on NPU_ARCH 3510 (Ascend 950PR/Ascend 950DT).
- URMA requires CANN Toolkit **>= 9.1.0**.
- The symmetric data buffer passed to `UrmaWorkspaceManager::Init()` must be backed by huge page memory (allocated with `ACL_MEM_MALLOC_HUGE_ONLY`). The underlying MR registration requires huge page backing. `ACL_MEM_MALLOC_HUGE_FIRST` may silently fall back to 4KB small pages for small allocations, causing registration to fail.

If the 1D contiguous requirement is not met, the current implementation returns an invalid async event (`handle == 0`).

## Role of scratchTile

`scratchTile` is **not** a staging buffer for transferring the data payload. It is converted to `TmpBuffer` and used as a temporary UB workspace for:

- Writing/reading SDMA control words (flag, sq_tail, channel_info)
- Polling the event completion flag
- Committing the queue tail upon completion

The actual data path is remote GM → DMA engine → local GM; `scratchTile` is used only for controlling and synchronizing metadata.

## scratchTile Type and Size Constraints

- Must be of the `pto::Tile` type.
- Must be a UB/Vec tile (`ScratchTile::Loc == TileType::Vec`).
- Must have at least `sizeof(uint64_t)` (8 bytes) of available bytes.

Recommended: `Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>` (256 bytes).

## Completion Semantics (Quiet Semantics)

The underlying completion mechanisms differ across engines, but the user-side quiet semantics behavior is consistent:

- **SDMA**: `TGET_ASYNC` only submits the data transfer SQE. The flag SQE is deferred until `Wait` is called, and completion is determined by polling the flag.
- **URMA**: `TGET_ASYNC` immediately submits the RDMA READ WQE and rings the doorbell. `Wait` polls the Completion Queue (CQ) until all expected CQEs are consumed.

- `event.Wait(session)` — blocks until **all asynchronous operations issued since the last Wait** have completed.

This means that after multiple `TGET_ASYNC` calls, you only need to call `Wait` once on the last returned `AsyncEvent` to wait for all pending operations to complete (similar to quiet semantics of shmem).

After `Wait` succeeds, all data read into the issued `dstGlobalData` is fully ready.

## Examples

### Single Transfer

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/common/pto_tile.hpp>

using namespace pto;

template <typename T>
__global__ AICORE void SimpleGet(__gm__ T *localDst, __gm__ T *remoteSrc,
                                 __gm__ uint8_t *sdmaWorkspace)
{
    using ShapeDyn  = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT        = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT dstG(localDst,  shape, stride);
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
    using ShapeDyn  = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT        = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

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
    (void)lastEvent.Wait(session);  // Wait once for all pending operations.
}
```

### URMA Examples (NPU_ARCH 3510)

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
