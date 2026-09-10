# TPUT_ASYNC

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-26T03:14:23.849Z pushedAt=2026-08-29T09:05:18.408Z -->

## Introduction

`TPUT_ASYNC` is an asynchronous remote write primitive. It initiates a transfer from local GM to remote GM and immediately returns an `AsyncEvent`.

Data flow:

`srcGlobalData (local GM)` → DMA engine → `dstGlobalData (remote GM)`

## Template Parameters

- `engine`:
    - `DmaEngine::SDMA` (default)
    - `DmaEngine::URMA` (Ascend 950PR/Ascend 950DT, NPU_ARCH 3510 only)

> **Note (SDMA path)**
> `TPUT_ASYNC` with `DmaEngine::SDMA` currently **supports only flat contiguous logical 1D tensors**.
> The current SDMA asynchronous implementation does not support non-1D or non-contiguous layouts.

## C++ Built-in APIs

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session, WaitEvents &... events);
```

`AsyncSession` is an engine-agnostic session object. Build it once using `BuildAsyncSession<engine>()` and pass it to all asynchronous calls and event waits. The template parameter `engine` selects the DMA backend at compile time, keeping the code forward-compatible with future engines (such as CCU).

## AsyncSession Construction

Use `BuildAsyncSession` in `include/pto/comm/async_common/async_event_impl.hpp`.
This function has two overloads – one for SDMA and one for URMA, with different parameter lists.

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
| `scratchTile` | - | UB scratch tile used for SDMA control metadata (see [Role of scratchTile](#role-of-scratchtile)). |
| `workspace` | - | GM pointer allocated by the host-side `SdmaWorkspaceManager`. |
| `session` | - | Output `AsyncSession` object. |
| `syncId` | `0` | MTE3/MTE2 pipe synchronization event ID (0-7). If the kernel uses other pipe barriers on the same ID, this value must be overridden. |
| `baseConfig` | `{kDefaultSdmaBlockBytes, 0, 1}` | `{block_bytes, comm_block_offset, queue_num}`. It applies to most single-queue transfer scenarios. |
| `channelGroupIdx` | `kAutoChannelGroupIdx` | SDMA channel group index. By default, `get_block_idx()` is used internally to map to the current AI Core. This value must be overridden in multi-block or custom channel mapping scenarios. |

### URMA Construction (NPU_ARCH 3510 Only)

> User-level RDMA Memory Access (URMA) is a hardware-accelerated RDMA transfer engine on Ascend 950PR/Ascend 950DT (NPU_ARCH 3510).
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
| `destRankId` | Remote PE rank ID with which this session communicates. For `TPUT_ASYNC`, this is the destination rank to which data is written. |
| `session` | Output `AsyncSession` object. |

URMA does not require `scratchTile` – polling is performed directly through the `ld_dev`/`st_dev` hardware primitives.

## Constraints

- `GlobalSrcData::RawDType == GlobalDstData::RawDType`
- `GlobalSrcData::layout == GlobalDstData::layout`
- Both the SDMA and URMA paths require the source tensor to be **flat, contiguous, logical 1D**.
- The SDMA workspace must be a valid GM pointer allocated by the host-side `SdmaWorkspaceManager`.
- The URMA workspace must be a valid GM pointer allocated by the host-side `UrmaWorkspaceManager`.
- URMA is available only on NPU_ARCH 3510 (Ascend 950PR/Ascend 950DT).
- URMA requires CANN Toolkit **>= 9.1.0**.
- The symmetric data buffer passed to `UrmaWorkspaceManager::Init()` must be backed by huge page memory (allocated with `ACL_MEM_MALLOC_HUGE_ONLY`). The underlying MR registration requires a huge page backing. `ACL_MEM_MALLOC_HUGE_FIRST` may silently fall back to 4 KB small pages for small allocations, causing registration to fail.

If the one-dimensional contiguous requirement is not met, the current implementation returns an invalid async event (`handle == 0`).

## Role of scratchTile

`scratchTile` is **not** a staging buffer for storing user data payloads.
It is converted to `TmpBuffer` and used as a temporary UB workspace for:

- Writing/reading SDMA control words (flag, sq_tail, channel_info)
- Polling the event completion flag
- Committing the queue tail upon completion

The actual data payload is transferred directly between GM buffers. `scratchTile` is used only for controlling and synchronizing metadata.

## scratchTile Type and Size Constraints

- Must be of the `pto::Tile` type.
- Must be a UB/Vec tile (`ScratchTile::Loc == TileType::Vec`).
- Must have at least `sizeof(uint64_t)` (8 bytes) of available bytes.

Recommended: `Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>` (256 bytes).

## Completion Semantics (Quiet Semantics)

The underlying completion mechanisms differ across engines, but the quiet semantics behavior on the user side is consistent:

- **SDMA**: `TPUT_ASYNC` only submits the data transfer SQE, and the flag SQE is deferred until `Wait` is called, which determines completion by polling the flag.
- **URMA**: `TPUT_ASYNC` immediately submits the RDMA WRITE WQE and rings the doorbell. `Wait` waits for all expected CQEs to be consumed by polling the Completion Queue (CQ).

- `event.Wait(session)` – blocks until **all asynchronous operations issued since the last Wait** are complete.

This means that after multiple `TPUT_ASYNC` calls, calling `Wait` once on the last returned `AsyncEvent` is sufficient to wait for all pending operations to complete (similar to the quiet semantics of shmem).

After `Wait` succeeds, all issued `dstGlobalData` writes are fully complete.

## Examples

### Single Transfer

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/common/pto_tile.hpp>

using namespace pto;

template <typename T>
__global__ AICORE void SimplePut(__gm__ T *remoteDst, __gm__ T *localSrc,
                                 __gm__ uint8_t *sdmaWorkspace)
{
    using ShapeDyn  = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT        = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT dstG(remoteDst, shape, stride);
    GT srcG(localSrc,  shape, stride);

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
    using ShapeDyn  = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT        = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
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
    (void)lastEvent.Wait(session);  // One Wait waits for all pending operations.
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
