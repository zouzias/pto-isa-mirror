# TPREFETCH_L2

## Introduction

Prefetch data from Global Memory (GM/HBM) into the NPU's L2 Cache via SDMA CMO (Cache Maintenance Operation, opcode=6). Unlike `TPREFETCH` which moves data from GM into UB, `TPREFETCH_L2` keeps data in L2 cache only, consuming no UB space. This allows subsequent `TLOAD` operations to hit L2 cache instead of going to GM, significantly reducing data transfer latency.

`TPREFETCH_L2` is logically a *memory access* instruction (it stages data from GM into the on-chip L2 cache); it just happens to use the SDMA CMO path internally. The public API therefore lives at `pto::TPREFETCH_L2`, not `pto::comm::TPREFETCH_L2` — same place as `pto::TPREFETCH`.

## Data Flow

```
GM / HBM  ──(SDMA CMO prefetch)──>  L2 Cache
                                       │
                                       └── subsequent TLOAD hits L2 (fast)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`, namespace `pto`. Two call shapes are supported, picked by which 3rd argument the caller passes (a workspace pointer or a pre-built `AsyncSession`).

### Workspace-based overloads (recommended)

```cpp
namespace pto {

template <typename GlobalData, typename... WaitEvents>
PTO_INST comm::AsyncEvent TPREFETCH_L2(GlobalData &src, __gm__ uint8_t *workspace,
                                       WaitEvents &... events);

template <typename... WaitEvents>
PTO_INST comm::AsyncEvent TPREFETCH_L2(__gm__ void *src, uint64_t bytes,
                                       __gm__ uint8_t *workspace,
                                       WaitEvents &... events);

} // namespace pto
```

A transient `AsyncSession` is built on the AICORE stack inside the call (channelGroupIdx = `get_block_idx()`, syncId = 0, queue_num = 1). `workspace` is the GM region prepared host-side via `SdmaWorkspaceManager::Init`. The returned `AsyncEvent` carries the workspace base address in its `handle` field, so:

```cpp
auto evt = pto::TPREFETCH_L2(input, bytes, workspace);
evt.Wait();              // 0-arg: rebuilds transient session from handle
// or equivalently:
evt.Wait(workspace);     // explicit workspace
```

Each call pays the session-build cost (a few hundred AICORE cycles). Use the workspace-based shape when you do **not** need to amortize that cost across many prefetch issues.

### Session-based overloads (advanced)

```cpp
namespace pto {

template <typename GlobalData, typename... WaitEvents>
PTO_INST comm::AsyncEvent TPREFETCH_L2(GlobalData &src, const comm::AsyncSession &session,
                                       WaitEvents &... events);

template <typename... WaitEvents>
PTO_INST comm::AsyncEvent TPREFETCH_L2(__gm__ void *src, uint64_t bytes,
                                       const comm::AsyncSession &session,
                                       WaitEvents &... events);

} // namespace pto
```

Build the `AsyncSession` once with `comm::BuildAsyncSession(scratchTile, workspace, session)` and pass it to every `TPREFETCH_L2` (and the matching `evt.Wait(session)`) within the same kernel. Saves the per-call session-build cost when the same workspace is reused across many prefetch issues, e.g. tight loops issuing one prefetch per chunk.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `GlobalData&` or `__gm__ void*` | Source GM region to prefetch into L2 |
| `bytes` | `uint64_t` | Byte count (raw pointer overload only) |
| `workspace` | `__gm__ uint8_t*` | SDMA workspace base from `SdmaWorkspaceManager::Init` |
| `session` | `const comm::AsyncSession&` | Pre-built async session (advanced overload) |
| `events...` | `WaitEvents&...` | Optional wait events for synchronization |

### Return Value

`comm::AsyncEvent` — handle for asynchronous completion tracking. Wait/test variants:

| Form | Use after |
|------|-----------|
| `evt.Wait()` / `evt.Test()`         | workspace-based call |
| `evt.Wait(workspace)` / `evt.Test(workspace)` | workspace-based call |
| `evt.Wait(session)` / `evt.Test(session)`     | session-based call |

Do not mix forms across call shapes: `Wait()`/`Wait(workspace)` assume the default channelGroupIdx/syncId/queue_num that the workspace-based `TPREFETCH_L2` issues. If your `TPREFETCH_L2` was issued via the session-based overload with a non-default `SdmaBaseConfig`, use `evt.Wait(session)` instead.

## Constraints

- Source data must be in Global Memory (GM/HBM address space).
- For the GlobalTensor overload, the tensor must be flat contiguous (packed 1D layout).
- The 256-byte UB scratch tile required by SDMA SQE construction is allocated on the AICORE stack at kernel-call address `0x0`. If you place anything else at that UB offset across the call site, expect undefined behaviour.
- SDMA CMO operates at cache-line granularity; non-aligned prefetch ranges are rounded by hardware.
- On CPU simulation backend, this instruction is a no-op (returns an empty `AsyncEvent`).

## Comparison with TPREFETCH

| Dimension | `pto::TPREFETCH` | `pto::TPREFETCH_L2` |
|-----------|-----------|--------------|
| Data flow | GM → UB | GM → L2 Cache |
| Hardware path | MTE (`copy_gm_to_ubuf`) | SDMA CMO (opcode=6) |
| UB consumption | Yes (requires dst Tile) | No (only 256B scratch for SQE construction) |
| Synchronization | Synchronous (pipeline barrier) | Asynchronous (`AsyncEvent`) |
| Use case | Small data preload to UB | Large data L2 warm-up |

## Examples

### Workspace-based (recommended)

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

__global__ AICORE void my_kernel(__gm__ half *src, __gm__ half *dst,
                                 __gm__ uint8_t *workspace)
{
    using GShape = Shape<1, 1, 1, 1, 16384>;
    using GStride = Stride<1, 1, 1, 1, 1>;
    GlobalTensor<half, GShape, GStride> srcGlobal(src);

    auto evt = TPREFETCH_L2(srcGlobal, workspace);
    evt.Wait();

    using TileData = Tile<TileType::Vec, half, 128, 128, BLayout::RowMajor>;
    TileData tile;
    TASSIGN(tile, 0x100);
    TLOAD(tile, srcGlobal);
}
```

### Session-based (amortize session build across many calls)

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

__global__ AICORE void streaming_kernel(__gm__ half *blocks, __gm__ uint8_t *workspace)
{
    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, 256>;
    ScratchTile scratch;
    TASSIGN(scratch, 0x0);

    comm::AsyncSession session;
    comm::BuildAsyncSession(scratch, workspace, session);

    for (int i = 0; i < N; ++i) {
        auto evt = TPREFETCH_L2(reinterpret_cast<__gm__ void *>(blocks + i * BLOCK), BLOCK_BYTES, session);
        // ... compute on block i-1 here ...
        evt.Wait(session);
        // ... TLOAD block i ...
    }
}
```

### Raw pointer overload

```cpp
auto evt = TPREFETCH_L2((__gm__ void *)srcPtr, totalBytes, workspace);
evt.Wait();
```
