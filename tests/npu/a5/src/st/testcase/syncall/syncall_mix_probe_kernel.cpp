/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// MIX-mode probes launched with AIC + AIV together (ratio 1:2) via dav-c310
// auto-split single chevron:
//   RunMixProbe        : each core does ONE scalar GM store (can AIC write GM in
//                        MIX mode at all?).
//   RunMixBarrierProbe : each core marks stage 0, runs ONE real SYNCALL<Soft,Mix>
//                        barrier, marks stage 1 -- isolates whether the bare mix
//                        atomic barrier (no proxy/business) faults.
//   RunMixSlotBarrierProbe : same, but the per-core-slot scalar barrier candidate.
// Markers use a 128-byte (32 int32) stride so adjacent cores never share a cache
// line (the 32-byte kInt32PerCacheLine spacing caused dcci false-sharing).

#include "syncall_mix_common.hpp"

constexpr int32_t kMixProbeAicBlocks = 18;
constexpr int32_t kMixProbeParticipants = 54;
constexpr int32_t kMarkStride = 32;        // 128 bytes: no cache-line false sharing.
constexpr int32_t kSlotBarrierStride = 32; // per-core barrier slot, 128 bytes apart.

PTO_INTERNAL void BigMark(__gm__ int32_t* marker, int32_t slotIdx, int32_t stage)
{
#if defined(__DAV_CUBE__) || defined(__DAV_VEC__)
    if (marker != nullptr) {
        __gm__ int32_t* slot = marker + slotIdx * kMarkStride;
        slot[0] = stage;
        SoftDcci(static_cast<__gm__ void*>(slot));
        dsb(DSB_DDR);
    }
#else
    (void)marker;
    (void)slotIdx;
    (void)stage;
#endif
}

// Candidate MIX barrier: per-core slot, scalar-only (no st_atomic / no ld_dev /
// no shared counter). Each core publishes its own epoch into its own cache line
// and spins until every core's slot has reached that epoch. Works on both AIC
// and AIV (AIC scalar GM read/write verified by the probes).
PTO_INTERNAL void SlotBarrierOnce(__gm__ int32_t* ws, int32_t idx, int32_t total)
{
#if defined(__DAV_CUBE__) || defined(__DAV_VEC__)
    __gm__ int32_t* mySlot = ws + idx * kSlotBarrierStride;
    SoftDcci(static_cast<__gm__ void*>(mySlot));
    dsb(DSB_DDR);
    const int32_t epoch = mySlot[0] + 1;
    mySlot[0] = epoch;
    SoftDcci(static_cast<__gm__ void*>(mySlot));
    dsb(DSB_DDR);

    int32_t pollCount = 0;
    while (true) {
        int32_t ready = 0;
        for (int32_t i = 0; i < total; ++i) {
            __gm__ int32_t* s = ws + i * kSlotBarrierStride;
            SoftDcci(static_cast<__gm__ void*>(s));
            if (s[0] >= epoch) {
                ++ready;
            }
        }
        dsb(DSB_DDR);
        if (ready >= total) {
            break;
        }
        if (++pollCount >= 1000000) {
            break;
        }
    }
#else
    (void)ws;
    (void)idx;
    (void)total;
#endif
}

extern "C" __global__ AICORE void RunMixProbe_1301(__gm__ int32_t __out__* marker)
{
#if defined(__DAV_CUBE__)
    BigMark(marker, GetMixLogicalIdx(), 42);
#elif defined(__DAV_VEC__)
    BigMark(marker, GetMixLogicalIdx(), 100);
#else
    (void)marker;
#endif
}

extern "C" __global__ AICORE void RunMixBarrierProbe_1302(
    __gm__ int32_t __out__* syncWorkspace, __gm__ int32_t __out__* marker)
{
#if defined(__DAV_CUBE__) || defined(__DAV_VEC__)
    const int32_t idx = GetMixLogicalIdx();
    BigMark(marker, idx, 0);
    GlobalTensor<int32_t, pto::Shape<>, pto::Stride<>> gmWs(syncWorkspace);
    SYNCALL<SyncAllMode::Soft, SyncCoreType::Mix>(gmWs, kMixProbeParticipants);
    BigMark(marker, idx, 1);
#else
    (void)syncWorkspace;
    (void)marker;
#endif
}

extern "C" __global__ AICORE void RunMixSlotBarrierProbe_1303(
    __gm__ int32_t __out__* syncWorkspace, __gm__ int32_t __out__* marker)
{
#if defined(__DAV_CUBE__) || defined(__DAV_VEC__)
    const int32_t idx = GetMixLogicalIdx();
    BigMark(marker, idx, 0);
    SlotBarrierOnce(syncWorkspace, idx, kMixProbeParticipants);
    BigMark(marker, idx, 1);
#else
    (void)syncWorkspace;
    (void)marker;
#endif
}

void LaunchMixProbe(int32_t* marker, void* stream)
{
    RunMixProbe_1301<<<kMixProbeAicBlocks, nullptr, stream>>>(marker);
}

void LaunchMixBarrierProbe(int32_t* syncWorkspace, int32_t* marker, void* stream)
{
    RunMixBarrierProbe_1302<<<kMixProbeAicBlocks, nullptr, stream>>>(syncWorkspace, marker);
}

void LaunchMixSlotBarrierProbe(int32_t* syncWorkspace, int32_t* marker, void* stream)
{
    RunMixSlotBarrierProbe_1303<<<kMixProbeAicBlocks, nullptr, stream>>>(syncWorkspace, marker);
}
