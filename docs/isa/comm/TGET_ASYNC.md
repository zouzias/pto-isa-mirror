# TGET_ASYNC

## Introduction

`TGET_ASYNC` is an asynchronous remote read primitive. It starts a transfer from remote GM to local GM and returns an `AsyncEvent` immediately.

Data flow:

`srcGlobalData (remote GM) -> DMA engine -> dstGlobalData (local GM)`

## Template Parameter

- `engine`:
  - `DmaEngine::SDMA` (default)
  - `DmaEngine::URMA` (todo)

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
```

`AsyncSession` is an engine-agnostic session object. Build once with
`BuildAsyncSession<engine>()`, then pass to all async calls and event waits.
The template `engine` parameter selects the DMA backend at compile time, making the
code forward-compatible with future engines (URMA, CCU, etc.).

## AsyncSession Construction

Use `BuildAsyncSession` from `include/pto/comm/async/async_event_impl.hpp`:

```cpp
template <DmaEngine engine = DmaEngine::SDMA, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(ScratchTile &scratchTile,
                                    __gm__ uint8_t *workspace,
                                    AsyncSession &session,
                                    uint32_t syncId = 0,
                                    const sdma::SdmaBaseConfig &baseConfig = {1024 * 1024, 0, 1},
                                    uint32_t channelGroupIdx = sdma::kAutoChannelGroupIdx);
```

The engine template parameter selects the backend (currently only SDMA).

### Parameter Reference


| Parameter                      | Type                                             | Required | Default                                                   | Valid Range           | Description                                                                                                                                                                                                                        |
| ------------------------------ | ------------------------------------------------ | -------- | --------------------------------------------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scratchTile`                  | `Tile<TileType::Vec, uint8_t, 1, UB_ALIGN_SIZE>` | **Yes**  | —                                                         | bytes >= 8            | UB Vec tile for SDMA control-plane reads/writes (doorbell, event record, tail state). Not used for payload data. Do not reuse while async ops are in-flight.                                                                   |
| `workspace`                    | `__gm__ uint8_t` *                               | **Yes**  | —                                                         | non-`nullptr`         | Device-side SDMA context GM region (channel info, SQ ring addresses, workspace flags). Initialized by host-side `SdmaWorkspaceManager` + AICPU operator. Layout: `BatchWriteFlagInfo(64B)`                                         |
| `syncId`                       | `uint32_t`                                       | No       | `0`                                                       | 0–7 (MTE event id)    | MTE3/MTE2 pipe sync event id for `set_flag`/`wait_flag` pairs in `SetValue`/`GetValue`. Override only if kernel uses other pipe barriers on the same id.                                                                           |
| `baseConfig.block_bytes`       | `uint64_t`                                       | No       | `1048576` (1 MiB)                                         | > 0                   | Max bytes per data SQE. Total transfer splits into `ceil(transfer_size / block_bytes)` SQEs. Larger = less control overhead but coarser; per-queue SQE count (`ceil(iter_num / queue_num) + 1`) must not exceed `kSqDepth` (2048). |
| `baseConfig.comm_block_offset` | `uint64_t`                                       | No       | `0`                                                       | >= 0                  | Byte offset on both src and dst: `addr = data() + offset + idx * block_bytes`. For multi-core partitioning of a shared buffer. Caller must ensure no out-of-bounds (no runtime check).                                             |
| `baseConfig.queue_num`         | `uint32_t`                                       | No       | `1`                                                       | 1 – 48                | Parallel SDMA channels (SQ rings) for this block. SQEs distributed via `idx % queue_num`. Max concurrent blocks = `48 / queue_num`. Start with 1.                                                                                    |
| `channelGroupIdx`              | `uint32_t`                                       | No       | `kAutoChannelGroupIdx` (`UINT32_MAX`) → `get_block_idx()` | `[0, 48 / queue_num)` | Selects consecutive channel group starting at `idx * queue_num`. Out-of-range silently returns `AsyncEvent(handle=0)`.                                                                                                             |


## Constraints

- src and dst tensors must have the **same element type and memory layout**, and both must be **flat contiguous logical 1D**
- The number of elements in dst must be **>=** the number of elements in src
- src and dst GM pointers must be non-`nullptr`, i.e. the HCCL window must be initialized and the remote address must be valid
- `blockDim * queue_num` must be **<= 48** (i.e. `channelGroupIdx < 48 / queue_num`)
- The total number of SQEs on a single queue must not exceed **2048**
- workspace must be a valid GM pointer allocated by host-side `SdmaWorkspaceManager` (non-`nullptr`)
- When multiple blocks run concurrently, each block's **`channelGroupIdx` must not overlap** — otherwise SQ rings and workspace are corrupted. Using the default `kAutoChannelGroupIdx` satisfies this automatically; if set manually, the caller must ensure uniqueness
- While async operations are in-flight, **`event.Wait(session)`** must be called to wait for all work to complete before destroying the session / scratchTile or starting a new batch. Otherwise the SQ ring may overflow and overwrite unconsumed SQEs
- **`comm_block_offset + transfer_size`** must not exceed the actual size of the src/dst buffers; otherwise out-of-bounds memory access occurs

## Completion Semantics (Quiet Semantics)

`TGET_ASYNC` only submits data transfer SQEs without submitting a flag SQE. The flag SQE submission is deferred to the `Wait` call.

- `event.Wait(session)` — submits a flag SQE and blocks until **all async operations issued since the last Wait** are complete

This means after multiple `TGET_ASYNC` calls, a single `Wait` on the last returned `AsyncEvent` drains all pending operations (similar to shmem's quiet semantics).

After wait succeeds, all issued reads into `dstGlobalData` are complete.

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
    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

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
    (void)lastEvent.Wait(session);  // single Wait drains all pending ops
}
```

