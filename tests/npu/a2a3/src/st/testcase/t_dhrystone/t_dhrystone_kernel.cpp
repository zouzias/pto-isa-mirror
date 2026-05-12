/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#include "acl/acl.h"

using namespace pto;

namespace {
constexpr int kRecordCount = 16;
constexpr int kArrayCount = 64;
constexpr int kLoopStride = 7;
constexpr int kMixConstA = 13;
constexpr int kMixConstB = 29;
constexpr int kWarmupBlocks = 24;

struct BenchRecord {
    int next;
    int value;
    int weight;
    int flag;
};

AICORE inline int RotateIndex(int index, int delta, int mod)
{
    int rotated = index + delta;
    while (rotated >= mod) {
        rotated -= mod;
    }
    return rotated;
}

AICORE inline int MixValue(int lhs, int rhs, int step)
{
    int mixed = lhs * kMixConstA + rhs * kMixConstB + step;
    mixed ^= (lhs << 3);
    mixed ^= (rhs >> 1);
    return mixed;
}

AICORE inline void InitRecords(BenchRecord *records)
{
    for (int i = 0; i < kRecordCount; ++i) {
        records[i].next = RotateIndex(i, 3, kRecordCount);
        records[i].value = i * 17 + 5;
        records[i].weight = i * 11 + 3;
        records[i].flag = i & 1;
    }
}

AICORE inline void InitScratch(int *scratch)
{
    for (int i = 0; i < kArrayCount; ++i) {
        scratch[i] = i * i + 9;
    }
}

AICORE inline void RunIteration(BenchRecord *records, int *scratch, int step)
{
    int recordIndex = step & (kRecordCount - 1);
    for (int inner = 0; inner < kRecordCount; ++inner) {
        BenchRecord &current = records[recordIndex];
        BenchRecord &next = records[current.next];
        int scratchIndex = (recordIndex * kLoopStride + inner + step) & (kArrayCount - 1);
        int neighborIndex = (scratchIndex + current.flag + 1) & (kArrayCount - 1);

        int mixed = MixValue(current.value, next.weight, step + inner);
        scratch[scratchIndex] = mixed + scratch[neighborIndex];
        current.value = scratch[scratchIndex] - current.weight;
        next.weight = MixValue(next.weight, scratch[scratchIndex], current.flag);
        current.flag ^= (scratch[scratchIndex] & 1);
        current.next = RotateIndex(current.next, current.flag + 1, kRecordCount);
        recordIndex = current.next;
    }
}
} // namespace

template <int iteration>
__global__ AICORE void runTDhrystone()
{
    BenchRecord records[kRecordCount];
    int scratch[kArrayCount];

    InitRecords(records);
    InitScratch(scratch);

#ifdef _DEBUG
    cce::printf("Execution starts, %d runs through integer stress benchmark\n", iteration);
#endif

    uint64_t tStart = get_sys_cnt();

    for (int runIndex = 0; runIndex < iteration; ++runIndex) {
        RunIteration(records, scratch, runIndex);
    }

    pipe_barrier(PIPE_ALL);
    uint64_t tEnd = get_sys_cnt();

#ifdef _DEBUG
    int digest = scratch[(iteration + records[0].next) & (kArrayCount - 1)] ^ records[0].value ^ records[1].weight;
    cce::printf("Start @%d End @%d (%d us), digest=%d\n", int(tStart), int(tEnd), int(tEnd - tStart) * 20 / 1000,
                digest);
#endif
}

__global__ AICORE __attribute__((aic)) void warmup_kernel()
{}

template <int iteration>
void LaunchTDhrystone(void *stream)
{
    warmup_kernel<<<kWarmupBlocks, nullptr, stream>>>();
    runTDhrystone<iteration><<<1, nullptr, stream>>>();
}

template void LaunchTDhrystone<1000>(void *stream);
template void LaunchTDhrystone<2000>(void *stream);
template void LaunchTDhrystone<3000>(void *stream);
template void LaunchTDhrystone<4000>(void *stream);
template void LaunchTDhrystone<5000>(void *stream);
template void LaunchTDhrystone<6000>(void *stream);
template void LaunchTDhrystone<7000>(void *stream);
template void LaunchTDhrystone<8000>(void *stream);
