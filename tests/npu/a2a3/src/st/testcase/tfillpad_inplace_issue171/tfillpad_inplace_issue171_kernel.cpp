/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/**
 * Issue #171 Reproducer Kernel
 * 
 * Tests TFILLPAD_INPLACE with N=16 and various valid_len values.
 * According to the bug analysis:
 * - valid_len in [1,8]  -> padCols = 8 -> Path B runs -> BUG on HW (N=16) and SIM (all N)
 * - valid_len in [9,15] -> padCols = 0 -> Path B is NO-OP -> OK
 */

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <limits>
#include <algorithm>

using namespace std;
using namespace pto;

#define LOGSIZE 128

inline AICORE uint64_t get_syscnt()
{
    uint64_t syscnt;
    asm volatile("MOV %0, SYS_CNT\n" : "+l"(syscnt));
    return syscnt;
}

/**
 * Kernel template for TFILLPAD_INPLACE test
 * @param M number of rows (static)
 * @param N number of columns / stride (static)  
 * @param validLen number of valid columns (dynamic, but known at template instantiation)
 */
template <int M, int N, int validLen>
AICORE void runTFILLPAD_ISSUE171(__gm__ float *out, __gm__ float *src, __gm__ uint64_t *gLog)
{
    // Initialize stack to avoid dcache miss
    {
        uint64_t stack[1024];
        volatile uint64_t *pStack = stack;
        for (int i = 0; i < 1024; i += 8) {
            *(pStack++) = 0;
        }
        dsb(DSB_ALL);
    }

    // Preload icache
    uint64_t pc;
    asm volatile("MOV %0, PC\n" : "+l"(pc));
    preload((void *)pc, 2);
    while (get_icache_prl_st()) {
        asm("nop");
    }

    gLog += block_idx * LOGSIZE;

    __ubuf__ float *ubuf = 0x0;

    // Global tensor for input (M x N, but only first validLen columns are valid)
    using SrcShape = Shape<1, 1, 1, M, N>;
    using SrcStride = pto::Stride<M*N, M*N, M*N, N, 1>;
    using SrcGlobal = GlobalTensor<float, SrcShape, SrcStride>;
    SrcGlobal srcGlobal(src);

    // Global tensor for output (M x N)
    using DstShape = Shape<1, 1, 1, M, N>;
    using DstStride = pto::Stride<M*N, M*N, M*N, N, 1>;
    using DstGlobal = GlobalTensor<float, DstShape, DstStride>;
    DstGlobal dstGlobal(out);

    // Full tile for loading (M x N)
    using TileFull = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, -1, N, SLayout::NoneBox, 512, PadValue::Min>;
    TileFull fullTile(M);
    TASSIGN(fullTile, (uint64_t)ubuf);

    // Load full input (all N columns)
    TLOAD(fullTile, srcGlobal);

    // Now create the dynamic tile view for TFILLPAD_INPLACE
    // This tells TFILLPAD_INPLACE that only validLen columns are "valid"
    using TileDyn = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null>;
    TileDyn dynTile(M, validLen);
    TASSIGN(dynTile, (uint64_t)ubuf);  // Same address - inplace operation

    // Padded output tile (static shape M x N)
    using TilePadded = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, -1, N, SLayout::NoneBox, 512, PadValue::Min>;
    TilePadded padTile(M);
    TASSIGN(padTile, (uint64_t)ubuf);  // Same address
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    uint64_t t0 = get_syscnt();
    
    // This is the operation under test - TFILLPAD_INPLACE
    // It should fill columns [validLen, N) with -inf
    TFILLPAD_INPLACE(padTile, dynTile);
    
    uint64_t t1 = get_syscnt();
    
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    
    // Store output
    TSTORE(dstGlobal, padTile);
    
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    
    // Log timing
    *(gLog++) = t0;
    *(gLog++) = t1 - t0;
}

// Kernel instantiations for each test case
extern "C" __global__ AICORE void launchTFILLPAD_ISSUE171_1(__gm__ uint8_t *out, __gm__ uint8_t *src, 
                                                           int M, int N, int validLen, __gm__ uint64_t *gLog)
{
    // Case 1: M=16, N=16, validLen=1 (pads 15 columns) - SHOULD FAIL
    runTFILLPAD_ISSUE171<16, 16, 1>((__gm__ float *)out, (__gm__ float *)src, gLog);
}

extern "C" __global__ AICORE void launchTFILLPAD_ISSUE171_2(__gm__ uint8_t *out, __gm__ uint8_t *src,
                                                           int M, int N, int validLen, __gm__ uint64_t *gLog)
{
    // Case 2: M=16, N=16, validLen=8 (pads 8 columns) - SHOULD FAIL
    runTFILLPAD_ISSUE171<16, 16, 8>((__gm__ float *)out, (__gm__ float *)src, gLog);
}

extern "C" __global__ AICORE void launchTFILLPAD_ISSUE171_3(__gm__ uint8_t *out, __gm__ uint8_t *src,
                                                           int M, int N, int validLen, __gm__ uint64_t *gLog)
{
    // Case 3: M=16, N=16, validLen=9 (pads 7 columns) - SHOULD PASS (Path B NO-OP)
    runTFILLPAD_ISSUE171<16, 16, 9>((__gm__ float *)out, (__gm__ float *)src, gLog);
}

extern "C" __global__ AICORE void launchTFILLPAD_ISSUE171_4(__gm__ uint8_t *out, __gm__ uint8_t *src,
                                                           int M, int N, int validLen, __gm__ uint64_t *gLog)
{
    // Case 4: M=16, N=16, validLen=15 (pads 1 column) - SHOULD PASS (Path B NO-OP)
    runTFILLPAD_ISSUE171<16, 16, 15>((__gm__ float *)out, (__gm__ float *)src, gLog);
}

// Issue #171 exact cases with N=16/32/64/128, validLen=1
extern "C" __global__ AICORE void launchTFILLPAD_ISSUE171_5(__gm__ uint8_t *out, __gm__ uint8_t *src,
                                                           int M, int N, int validLen, __gm__ uint64_t *gLog)
{
    // Case 5: M=16, N=32, validLen=1 (pads 31 columns) - Issue #171 case
    runTFILLPAD_ISSUE171<16, 32, 1>((__gm__ float *)out, (__gm__ float *)src, gLog);
}

extern "C" __global__ AICORE void launchTFILLPAD_ISSUE171_6(__gm__ uint8_t *out, __gm__ uint8_t *src,
                                                           int M, int N, int validLen, __gm__ uint64_t *gLog)
{
    // Case 6: M=16, N=64, validLen=1 (pads 63 columns) - Issue #171 case
    runTFILLPAD_ISSUE171<16, 64, 1>((__gm__ float *)out, (__gm__ float *)src, gLog);
}

extern "C" __global__ AICORE void launchTFILLPAD_ISSUE171_7(__gm__ uint8_t *out, __gm__ uint8_t *src,
                                                           int M, int N, int validLen, __gm__ uint64_t *gLog)
{
    // Case 7: M=16, N=128, validLen=1 (pads 127 columns) - Issue #171 case
    runTFILLPAD_ISSUE171<16, 128, 1>((__gm__ float *)out, (__gm__ float *)src, gLog);
}

// Issue #171 cases: N=32/64/128 with validLen=N-1 (pads 1 column)
extern "C" __global__ AICORE void launchTFILLPAD_ISSUE171_8(__gm__ uint8_t *out, __gm__ uint8_t *src,
                                                           int M, int N, int validLen, __gm__ uint64_t *gLog)
{
    // Case 8: M=16, N=32, validLen=31 (pads 1 column)
    runTFILLPAD_ISSUE171<16, 32, 31>((__gm__ float *)out, (__gm__ float *)src, gLog);
}

extern "C" __global__ AICORE void launchTFILLPAD_ISSUE171_9(__gm__ uint8_t *out, __gm__ uint8_t *src,
                                                           int M, int N, int validLen, __gm__ uint64_t *gLog)
{
    // Case 9: M=16, N=64, validLen=63 (pads 1 column)
    runTFILLPAD_ISSUE171<16, 64, 63>((__gm__ float *)out, (__gm__ float *)src, gLog);
}

extern "C" __global__ AICORE void launchTFILLPAD_ISSUE171_10(__gm__ uint8_t *out, __gm__ uint8_t *src,
                                                            int M, int N, int validLen, __gm__ uint64_t *gLog)
{
    // Case 10: M=16, N=128, validLen=127 (pads 1 column)
    runTFILLPAD_ISSUE171<16, 128, 127>((__gm__ float *)out, (__gm__ float *)src, gLog);
}

// Host-side launch wrapper
template <int32_t testKey>
void launchTFILLPAD_ISSUE171(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream)
{
    if constexpr (testKey == 1) {
        launchTFILLPAD_ISSUE171_1<<<1, nullptr, stream>>>(out, src, 16, 16, 1, gLog);
    } else if constexpr (testKey == 2) {
        launchTFILLPAD_ISSUE171_2<<<1, nullptr, stream>>>(out, src, 16, 16, 8, gLog);
    } else if constexpr (testKey == 3) {
        launchTFILLPAD_ISSUE171_3<<<1, nullptr, stream>>>(out, src, 16, 16, 9, gLog);
    } else if constexpr (testKey == 4) {
        launchTFILLPAD_ISSUE171_4<<<1, nullptr, stream>>>(out, src, 16, 16, 15, gLog);
    } else if constexpr (testKey == 5) {
        launchTFILLPAD_ISSUE171_5<<<1, nullptr, stream>>>(out, src, 16, 32, 1, gLog);
    } else if constexpr (testKey == 6) {
        launchTFILLPAD_ISSUE171_6<<<1, nullptr, stream>>>(out, src, 16, 64, 1, gLog);
    } else if constexpr (testKey == 7) {
        launchTFILLPAD_ISSUE171_7<<<1, nullptr, stream>>>(out, src, 16, 128, 1, gLog);
    } else if constexpr (testKey == 8) {
        launchTFILLPAD_ISSUE171_8<<<1, nullptr, stream>>>(out, src, 16, 32, 31, gLog);
    } else if constexpr (testKey == 9) {
        launchTFILLPAD_ISSUE171_9<<<1, nullptr, stream>>>(out, src, 16, 64, 63, gLog);
    } else if constexpr (testKey == 10) {
        launchTFILLPAD_ISSUE171_10<<<1, nullptr, stream>>>(out, src, 16, 128, 127, gLog);
    }
}

// Template instantiations
template void launchTFILLPAD_ISSUE171<1>(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream);
template void launchTFILLPAD_ISSUE171<2>(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream);
template void launchTFILLPAD_ISSUE171<3>(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream);
template void launchTFILLPAD_ISSUE171<4>(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream);
template void launchTFILLPAD_ISSUE171<5>(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream);
template void launchTFILLPAD_ISSUE171<6>(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream);
template void launchTFILLPAD_ISSUE171<7>(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream);
template void launchTFILLPAD_ISSUE171<8>(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream);
template void launchTFILLPAD_ISSUE171<9>(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream);
template void launchTFILLPAD_ISSUE171<10>(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream);
