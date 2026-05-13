# pto.tput_async

## Summary

`pto.tput_async` issues an asynchronous remote write and returns an `AsyncEvent` handle immediately.

## Semantics

The verified public API issues the asynchronous operation through an `AsyncSession` object and returns an `AsyncEvent`.

Completion is not implied by submission; users must later call `event.Wait(session)` or `event.Test(session)`.

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`.

```cpp
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session, WaitEvents &... events);
```

## Session Construction

The verified helper API exposes `BuildAsyncSession` overloads:

```cpp
template <DmaEngine engine = DmaEngine::SDMA, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(ScratchTile &scratchTile, __gm__ uint8_t *workspace,
                                    AsyncSession &session, uint32_t syncId = 0,
                                    const sdma::SdmaBaseConfig &baseConfig = {sdma::kDefaultSdmaBlockBytes, 0, 1},
                                    uint32_t channelGroupIdx = sdma::kAutoChannelGroupIdx);
```

```cpp
#ifdef PTO_URMA_SUPPORTED
template <DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t *workspace, uint32_t destRankId,
                                    AsyncSession &session);
#endif
```

## Constraints

!!! warning "Constraints"
    - Submission requires a previously built `AsyncSession`.
    - Event tokens passed to the wrapper are waited before async submission begins.
    - Completion must be checked through the returned `AsyncEvent` and the same logical session family.
    - Engine-specific legality constraints are implementation-dependent; the public wrapper itself does not restate all backend transport restrictions.

## `AsyncEvent` Completion API

Verified public completion methods include:

```cpp
bool AsyncEvent::Wait(const AsyncSession &session) const;
bool AsyncEvent::Test(const AsyncSession &session) const;
```

If `handle == 0`, these methods return success immediately in the verified implementation.

## Examples

```cpp
#include <pto/comm/pto_comm_inst.hpp>
using namespace pto;

void example_put_async(auto &dstG, auto &srcG, const comm::AsyncSession &session) {
    auto event = comm::TPUT_ASYNC(dstG, srcG, session);
    (void)event.Wait(session);
}
```
