/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MOE_TOKEN_PERMUTE_REGISTER_BODY_HPP
#define MOE_TOKEN_PERMUTE_REGISTER_BODY_HPP

#include "moe_token_permute_common.hpp"
#include <pto/pto-inst.hpp>

using namespace pto;

namespace moe_token_permute {

constexpr int32_t kUbHist = 0;
constexpr int32_t kUbIdx = 32;
constexpr int32_t kUbWs = 1024;
constexpr int32_t kUbRunning = 320;
constexpr int32_t kUbAcc = 352;
constexpr int32_t kUbCpre = 384;
constexpr int32_t kUbOffsets = 416;
constexpr int32_t kUbCounters = 448;
constexpr int32_t kUbWp = 480;
constexpr int32_t kUbSio = 512;
constexpr int32_t kUbRow = 544;

PTO_INTERNAL void SetFlagPipeline(int32_t flagId)
{
    switch (flagId) {
        case 0:
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            break;
        case 1:
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
            break;
        case 2:
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
            break;
        case 3:
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
            break;
        case 10:
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
            break;
        default:
            break;
    }
}

PTO_INTERNAL void WaitFlagPipeline(int32_t flagId)
{
    switch (flagId) {
        case 0:
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            break;
        case 1:
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
            break;
        case 2:
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
            break;
        case 3:
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
            break;
        case 10:
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
            break;
        default:
            break;
    }
}

PTO_INTERNAL void InitTokenPipelineFlags()
{
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
}

PTO_INTERNAL void ClearTokenPipelineFlags()
{
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
}

PTO_INTERNAL void RunMixSyncBarrier()
{
#if defined(__DAV_CUBE__) || defined(SYNCALL_MIX_BUILD_AIC)
    SYNCALL<SyncCoreType::Mix>();
#endif
#if defined(__DAV_VEC__) || defined(SYNCALL_MIX_BUILD_AIV)
    SYNCALL<SyncCoreType::Mix>();
#endif
}

PTO_INTERNAL void InvalidateWorkspaceGm(__gm__ int32_t *workspaceGm)
{
#if defined(__DAV_VEC__) || defined(SYNCALL_MIX_BUILD_AIV)
    for (int32_t line = 0; line < kMoeWsTotal; line += 8) {
        dcci(static_cast<__gm__ void *>(workspaceGm + line), SINGLE_CACHE_LINE);
    }
    dsb(DSB_DDR);
#else
    (void)workspaceGm;
#endif
}

PTO_INTERNAL void RunMoeTokenPermuteAivPhase1(__gm__ int32_t *indicesGm, __gm__ int32_t *workspaceGm, int32_t cid)
{
#if defined(__DAV_VEC__) || defined(SYNCALL_MIX_BUILD_AIV)
    const int32_t myStart = cid * kMoeChunkSize;

    __ubuf__ int32_t *histUb = reinterpret_cast<__ubuf__ int32_t *>(kUbHist);
    __ubuf__ int32_t *idxUb = reinterpret_cast<__ubuf__ int32_t *>(kUbIdx);

    for (int32_t e = 0; e < kMoeWsSlotStride; ++e) {
        histUb[e] = 0;
    }
    pipe_barrier(PIPE_ALL);

    copy_gm_to_ubuf(static_cast<__ubuf__ void *>(idxUb), static_cast<__gm__ void *>(indicesGm + myStart), 0, 1,
                    kMoeBurstIdx, 0, 0);
    pipe_barrier(PIPE_ALL);

    for (int32_t i = 0; i < kMoeChunkSize; ++i) {
        if (myStart + i < kMoeE) {
            const int32_t expert = idxUb[i];
            histUb[expert] = histUb[expert] + 1;
        }
    }
    pipe_barrier(PIPE_ALL);

    copy_ubuf_to_gm(static_cast<__gm__ void *>(workspaceGm + cid * kMoeWsSlotStride),
                    static_cast<__ubuf__ void *>(histUb), 0, 1, 1, 0, 0);
    SetFlagPipeline(2);
    WaitFlagPipeline(2);

    pipe_barrier(PIPE_ALL);
    dcci(static_cast<__gm__ void *>(workspaceGm + cid * kMoeWsSlotStride), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
#else
    (void)indicesGm;
    (void)workspaceGm;
    (void)cid;
#endif
}

PTO_INTERNAL void RunMoeTokenPermuteAivPhase2(__gm__ half *tokensGm, __gm__ int32_t *indicesGm, __gm__ half *permOutGm,
                                              __gm__ int32_t *sioOutGm, __gm__ int32_t *workspaceGm, int32_t cid,
                                              int32_t vid)
{
#if defined(__DAV_VEC__) || defined(SYNCALL_MIX_BUILD_AIV)
    const int32_t myStart = cid * kMoeChunkSize;
    const int32_t hOff = vid * kMoeHalfH;

    __ubuf__ int32_t *idxUb = reinterpret_cast<__ubuf__ int32_t *>(kUbIdx);
    __ubuf__ int32_t *wsUb = reinterpret_cast<__ubuf__ int32_t *>(kUbWs);
    __ubuf__ int32_t *runningUb = reinterpret_cast<__ubuf__ int32_t *>(kUbRunning);
    __ubuf__ int32_t *accUb = reinterpret_cast<__ubuf__ int32_t *>(kUbAcc);
    __ubuf__ int32_t *cpreUb = reinterpret_cast<__ubuf__ int32_t *>(kUbCpre);
    __ubuf__ int32_t *offsetsUb = reinterpret_cast<__ubuf__ int32_t *>(kUbOffsets);
    __ubuf__ int32_t *countersUb = reinterpret_cast<__ubuf__ int32_t *>(kUbCounters);
    __ubuf__ int32_t *wpUb = reinterpret_cast<__ubuf__ int32_t *>(kUbWp);
    __ubuf__ int32_t *sioUb = reinterpret_cast<__ubuf__ int32_t *>(kUbSio);
    __ubuf__ half *rowUb = reinterpret_cast<__ubuf__ half *>(kUbRow);

    (void)indicesGm;
    InvalidateWorkspaceGm(workspaceGm);

    copy_gm_to_ubuf(static_cast<__ubuf__ void *>(wsUb), static_cast<__gm__ void *>(workspaceGm), 0, 1, kMoeBurstWs, 0,
                    0);
    SetFlagPipeline(3);
    WaitFlagPipeline(3);
    pipe_barrier(PIPE_ALL);

    runningUb[0] = 0;
    for (int32_t e = 0; e < kMoeNumExperts; ++e) {
        accUb[0] = 0;
        cpreUb[0] = 0;
        for (int32_t c = 0; c < kMoeNumCores; ++c) {
            accUb[0] = accUb[0] + wsUb[c * kMoeWsSlotStride + e];
            if (c < cid) {
                cpreUb[0] = cpreUb[0] + wsUb[c * kMoeWsSlotStride + e];
            }
        }
        offsetsUb[e] = runningUb[0] + cpreUb[0];
        countersUb[e] = 0;
        runningUb[0] = runningUb[0] + accUb[0];
    }

    for (int32_t i = 0; i < kMoeChunkSize; ++i) {
        if (myStart + i < kMoeE) {
            const int32_t expert = idxUb[i];
            wpUb[0] = offsetsUb[expert] + countersUb[expert];
            countersUb[expert] = countersUb[expert] + 1;
            sioUb[i] = wpUb[0];
        }
    }
    pipe_barrier(PIPE_ALL);
    SetFlagPipeline(3);
    WaitFlagPipeline(3);

    InitTokenPipelineFlags();
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    if (cid < kMoeNumTokens) {
        copy_gm_to_ubuf(static_cast<__ubuf__ void *>(rowUb),
                        static_cast<__gm__ void *>(tokensGm + cid * kMoeHiddenSize + hOff), 0, 1, kMoeBurstHalfRow, 0,
                        0);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);

    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
    if (cid < kMoeNumTokens) {
        for (int32_t k = 0; k < kMoeTopK; ++k) {
            wpUb[0] = sioUb[k];
            if (wpUb[0] < kMoeOutLen) {
                copy_ubuf_to_gm_align_b16(static_cast<__gm__ void *>(permOutGm + wpUb[0] * kMoeHiddenSize + hOff),
                                          static_cast<__ubuf__ void *>(rowUb), 0, 1, kMoeHalfH * sizeof(half), 0, 0, 0,
                                          0);
            }
        }
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    ClearTokenPipelineFlags();

    SetFlagPipeline(2);
    WaitFlagPipeline(2);
    copy_ubuf_to_gm_align_b32(static_cast<__gm__ void *>(sioOutGm + myStart), static_cast<__ubuf__ void *>(sioUb), 0, 1,
                              kMoeChunkSize * sizeof(int32_t), 0, 0, 0, 0);
    SetFlagPipeline(2);
    WaitFlagPipeline(2);
#else
    (void)tokensGm;
    (void)indicesGm;
    (void)permOutGm;
    (void)sioOutGm;
    (void)workspaceGm;
    (void)cid;
    (void)vid;
#endif
}

PTO_INTERNAL void RunMoeTokenPermuteAivBody(__gm__ half *tokensGm, __gm__ int32_t *indicesGm, __gm__ half *permOutGm,
                                            __gm__ int32_t *sioOutGm, __gm__ int32_t *workspaceGm, uint64_t fftsAddr)
{
#if defined(__DAV_VEC__) || defined(SYNCALL_MIX_BUILD_AIV)
    set_ffts_base_addr(fftsAddr);
    const int32_t cid = get_block_idx();
    const int32_t vid = get_subblockid();
    RunMixSyncBarrier();
    RunMoeTokenPermuteAivPhase1(indicesGm, workspaceGm, cid);
    RunMixSyncBarrier();
    RunMoeTokenPermuteAivPhase2(tokensGm, indicesGm, permOutGm, sioOutGm, workspaceGm, cid, vid);
#else
    (void)tokensGm;
    (void)indicesGm;
    (void)permOutGm;
    (void)sioOutGm;
    (void)workspaceGm;
    (void)fftsAddr;
#endif
}

PTO_INTERNAL void RunMoeTokenPermuteBody(__gm__ half *tokensGm, __gm__ int32_t *indicesGm, __gm__ half *permOutGm,
                                         __gm__ int32_t *sioOutGm, __gm__ int32_t *workspaceGm, uint64_t fftsAddr)
{
    set_ffts_base_addr(fftsAddr);
    RunMixSyncBarrier();
#if defined(__DAV_VEC__)
    const int32_t cid = get_block_idx();
    const int32_t vid = get_subblockid();
    RunMoeTokenPermuteAivPhase1(indicesGm, workspaceGm, cid);
#endif
    RunMixSyncBarrier();
#if defined(__DAV_VEC__)
    RunMoeTokenPermuteAivPhase2(tokensGm, indicesGm, permOutGm, sioOutGm, workspaceGm, cid, vid);
#else
    (void)tokensGm;
    (void)indicesGm;
    (void)permOutGm;
    (void)sioOutGm;
    (void)workspaceGm;
#endif
}

PTO_INTERNAL void RunMoeTokenPermuteAicOnly(__gm__ uint64_t *fftsAddrGm)
{
#if defined(SYNCALL_MIX_BUILD_AIC)
    set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddrGm));
    RunMixSyncBarrier();
    RunMixSyncBarrier();
#else
    (void)fftsAddrGm;
#endif
}

} // namespace moe_token_permute

#endif
