#pragma once

#include <cstdint>

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#include "pto/comm/pto_comm_inst.hpp"

#ifndef V4_PTO_SYNC_INLINE
#define V4_PTO_SYNC_INLINE inline __attribute__((always_inline)) __aicore__
#endif

using PtoSignalPtr = __gm__ int32_t*;
using PtoVolatileSignalPtr = volatile __gm__ int32_t*;
#else
#ifndef V4_PTO_SYNC_INLINE
#define V4_PTO_SYNC_INLINE inline
#endif

using PtoSignalPtr = int32_t*;
using PtoVolatileSignalPtr = volatile int32_t*;
#endif

struct PtoWindowSignal {
    PtoSignalPtr base = nullptr;
    uint32_t index = 0;
};

struct PtoWindowReady {
    uint32_t windowId = 0;
    uint32_t groupId = 0;
    uint32_t rowBegin = 0;
    uint32_t rowEnd = 0;
    uint32_t epoch = 0;
};

namespace v2_pto_sync {

#if defined(__CCE_AICORE__)
V4_PTO_SYNC_INLINE PtoWindowSignal WindowSignal(__gm__ int32_t* base, uint32_t index)
{
    PtoWindowSignal signal{};
    signal.base = base;
    signal.index = index;
    return signal;
}

V4_PTO_SYNC_INLINE pto::comm::Signal CommSignal(PtoWindowSignal signal)
{
    return pto::comm::Signal(signal.base + signal.index);
}

V4_PTO_SYNC_INLINE void SynchronizePipe()
{
    pipe_barrier(PIPE_ALL);
}

V4_PTO_SYNC_INLINE void PublishGmWrites()
{
    SynchronizePipe();
    dsb(DSB_DDR);
}

static constexpr uint32_t kPtoSyncCacheLineBytes = 128;

V4_PTO_SYNC_INLINE void MakeCacheLineVisible(__gm__ void* ptr)
{
    dcci(ptr, SINGLE_CACHE_LINE);
}

V4_PTO_SYNC_INLINE void MakeCacheRangeVisible(__gm__ void* ptr, uint32_t bytes)
{
    if (ptr == nullptr || bytes == 0) {
        return;
    }
    __gm__ uint8_t* base = reinterpret_cast<__gm__ uint8_t*>(ptr);
    for (uint32_t offset = 0; offset < bytes; offset += kPtoSyncCacheLineBytes) {
        MakeCacheLineVisible(reinterpret_cast<__gm__ void*>(base + offset));
    }
}

V4_PTO_SYNC_INLINE void MakeCacheLineVisibleAndSync(__gm__ void* ptr)
{
    MakeCacheLineVisible(ptr);
    SynchronizePipe();
}

V4_PTO_SYNC_INLINE void MakeCacheRangeVisibleAndSync(__gm__ void* ptr, uint32_t bytes)
{
    MakeCacheRangeVisible(ptr, bytes);
    SynchronizePipe();
}

V4_PTO_SYNC_INLINE void StoreWindowReadyNoFence(PtoWindowSignal signal, int32_t value)
{
    if (signal.base == nullptr) {
        return;
    }
    signal.base[signal.index] = value;
}

V4_PTO_SYNC_INLINE void NotifyWindowReady(PtoWindowSignal signal, int32_t epoch)
{
    if (signal.base == nullptr) {
        return;
    }
    auto commSignal = CommSignal(signal);
    pto::comm::TNOTIFY(commSignal, epoch, pto::comm::NotifyOp::Set);
}

V4_PTO_SYNC_INLINE void PublishWindowReadyStore(PtoWindowSignal signal, int32_t epoch)
{
    StoreWindowReadyNoFence(signal, epoch);
    PublishGmWrites();
}

V4_PTO_SYNC_INLINE bool TestWindowReady(PtoWindowSignal signal,
                                        int32_t expected,
                                        pto::comm::WaitCmp cmp = pto::comm::WaitCmp::GE)
{
    if (signal.base == nullptr) {
        return true;
    }
    auto commSignal = CommSignal(signal);
    return pto::comm::TTEST(commSignal, expected, cmp);
}

V4_PTO_SYNC_INLINE void WaitWindowReady(PtoWindowSignal signal,
                                        int32_t expected,
                                        pto::comm::WaitCmp cmp = pto::comm::WaitCmp::GE)
{
    if (signal.base == nullptr) {
        return;
    }
    if (!TestWindowReady(signal, expected, cmp)) {
        auto commSignal = CommSignal(signal);
        pto::comm::TWAIT(commSignal, expected, cmp);
    }
}

V4_PTO_SYNC_INLINE bool TestCommSignalReady(pto::comm::Signal& signal,
                                            int32_t expected,
                                            pto::comm::WaitCmp cmp = pto::comm::WaitCmp::GE)
{
    return pto::comm::TTEST(signal, expected, cmp);
}

V4_PTO_SYNC_INLINE void WaitCommSignalReady(pto::comm::Signal& signal,
                                            int32_t expected,
                                            pto::comm::WaitCmp cmp = pto::comm::WaitCmp::GE)
{
    if (!TestCommSignalReady(signal, expected, cmp)) {
        pto::comm::TWAIT(signal, expected, cmp);
    }
}

V4_PTO_SYNC_INLINE void NotifyCommSignalReady(pto::comm::Signal& signal, int32_t epoch)
{
    pto::comm::TNOTIFY(signal, epoch, pto::comm::NotifyOp::Set);
}

V4_PTO_SYNC_INLINE void WaitVisibleGmValue(PtoVolatileSignalPtr signal, int32_t expected)
{
    while (*signal < expected) {
        MakeCacheLineVisible((__gm__ void*)signal);
        SynchronizePipe();
    }
}

V4_PTO_SYNC_INLINE void PublishEntryBeforeCount(__gm__ void* entryPtr,
                                                __gm__ int32_t* countPtr,
                                                int32_t countValue)
{
    MakeCacheLineVisible(entryPtr);
    PublishGmWrites();
    *countPtr = countValue;
    MakeCacheLineVisible((__gm__ void*)countPtr);
    PublishGmWrites();
}

V4_PTO_SYNC_INLINE void ResetWindowReady(PtoWindowSignal signal)
{
    PublishWindowReadyStore(signal, 0);
}
#endif

}  // namespace v2_pto_sync
