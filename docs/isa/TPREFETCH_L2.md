# TPREFETCH_L2

## Introduction

Prefetch data from Global Memory (GM/HBM) into the NPU's L2 Cache via SDMA CMO (Cache Maintenance Operation, opcode=6). Unlike `TPREFETCH` which moves data from GM into UB, `TPREFETCH_L2` keeps data in L2 cache only, consuming no UB space. This allows subsequent `TLOAD` operations to hit L2 cache instead of going to GM, significantly reducing data transfer latency.

## Data Flow

```
GM / HBM  ──(SDMA CMO prefetch)──>  L2 Cache
                                       │
                                       └── subsequent TLOAD hits L2 (fast)
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`, namespace `pto::comm`:

### GlobalTensor overload

```cpp
template <typename GlobalData, typename... WaitEvents>
PTO_INST AsyncEvent TPREFETCH_L2(GlobalData &src, const AsyncSession &session,
                                  WaitEvents &... events);
```

### Raw pointer overload

```cpp
template <typename... WaitEvents>
PTO_INST AsyncEvent TPREFETCH_L2(__gm__ void *src, uint64_t bytes, const AsyncSession &session,
                                  WaitEvents &... events);
```

### Direct SdmaExecContext overloads

```cpp
template <typename GlobalData, typename... WaitEvents>
PTO_INST AsyncEvent TPREFETCH_L2(GlobalData &src, const sdma::SdmaExecContext &execCtx,
                                  WaitEvents &... events);

template <typename... WaitEvents>
PTO_INST AsyncEvent TPREFETCH_L2(__gm__ void *src, uint64_t bytes,
                                  const sdma::SdmaExecContext &execCtx,
                                  WaitEvents &... events);
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `GlobalData&` or `__gm__ void*` | Source GM region to prefetch into L2 |
| `bytes` | `uint64_t` | Byte count (raw pointer overload only) |
| `session` | `const AsyncSession&` | Async session containing SDMA context |
| `execCtx` | `const SdmaExecContext&` | SDMA execution context (direct overload) |
| `events...` | `WaitEvents&...` | Optional wait events for synchronization |

### Return Value

`AsyncEvent` — handle for asynchronous completion tracking. Use `event.Wait(session)` to block until prefetch completes, or `event.Test(session)` to poll non-blocking.

## Constraints

- Source data must be in Global Memory (GM/HBM address space).
- For the GlobalTensor overload, the tensor must be flat contiguous (packed 1D layout).
- Requires a valid `SdmaSession` or `SdmaExecContext`, initialized via `BuildSdmaSession()`.
- SDMA CMO operates at cache-line granularity; non-aligned prefetch ranges are rounded by hardware.
- On CPU simulation backend, this instruction is a no-op.

## Comparison with TPREFETCH

| Dimension | TPREFETCH | TPREFETCH_L2 |
|-----------|-----------|--------------|
| Data flow | GM → UB | GM → L2 Cache |
| Hardware path | MTE (`copy_gm_to_ubuf`) | SDMA CMO (opcode=6) |
| UB consumption | Yes (requires dst Tile) | No |
| Synchronization | Synchronous (pipeline barrier) | Asynchronous (AsyncEvent) |
| Use case | Small data preload to UB | Large data L2 warm-up |

## Examples

### Basic L2 prefetch with GlobalTensor

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;
using namespace pto::comm;

__global__ AICORE void my_kernel(__gm__ half *src, __gm__ half *dst,
                                  __gm__ uint8_t *workspace)
{
    using ScratchTile = Tile<TileType::Vec, uint32_t, 1, 16, BLayout::RowMajor>;
    ScratchTile scratch;
    TASSIGN(scratch, 0u);

    sdma::SdmaSession session;
    sdma::BuildSdmaSession(scratch, workspace, session);

    using GShape = Shape<1, 1, 1, 1, 16384>;
    using GStride = Stride<1, 1, 1, 1, 1>;
    GlobalTensor<half, GShape, GStride> srcGlobal(src);

    // Prefetch to L2
    AsyncSession asyncSession;
    asyncSession.sdmaSession = session;
    asyncSession.valid = true;
    auto evt = TPREFETCH_L2(srcGlobal, asyncSession);
    evt.Wait(asyncSession);

    // Subsequent TLOAD will hit L2
    using TileData = Tile<TileType::Vec, half, 128, 128, BLayout::RowMajor>;
    TileData tile(128, 128);
    TLOAD(tile, srcGlobal);
}
```

### Raw pointer overload

```cpp
auto evt = TPREFETCH_L2((__gm__ void *)srcPtr, totalBytes, asyncSession);
evt.Wait(asyncSession);
```
