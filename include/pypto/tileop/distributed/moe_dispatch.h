/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_dispatch.h
 * \brief
*/

#ifndef __DISTRIBUTED_DISPATCH__
#define __DISTRIBUTED_DISPATCH__

#include "common.h"
#include "hccl_context.h"

#include <type_traits>

namespace TileOp::Distributed {
template <typename T, int32_t axisH, int32_t tRowOffset, int32_t tColOffset, int32_t tRowShape, int32_t tColShape, int32_t groupIndex>
TILEOP void SendToRoutingExpert(__gm__ int32_t *syncTensor, __ubuf__ T *tokenBuffer, __ubuf__ int32_t *expertTableUb,
    __ubuf__ int32_t *expertBuffer, __gm__ T *token, __gm__ T *shmemDataBaseAddr, __gm__ int32_t *expertTable,
    uint32_t tableOffset0, uint32_t tableOffset1, uint32_t tableRawShape0, uint32_t tableRawShape1,
    uint32_t shmemDataOffset0, uint32_t shmemDataOffset1, uint32_t shmemDataOffset2, uint32_t shmemDataOffset3,
    uint32_t shmemDataRawShape0, uint32_t shmemDataRawShape1, uint32_t shmemDataRawShape2, uint32_t shmemDataRawShape3,
    __gm__ int64_t *hcclContext)
{
    int32_t topK = tableRawShape1;
    int32_t expertTblSize = tableRawShape0 * tableRawShape1;
    int32_t lenBurst = AlignUp<int32_t>(expertTblSize * sizeof(int32_t), 32) / 32;
    copy_gm_to_ubuf(expertTableUb, expertTable, 0, 1, lenBurst, 0, 0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    __gm__ HcclCombinOpParam *winContext = (__gm__ HcclCombinOpParam *)(hcclContext[groupIndex]);
    uint64_t localUsrRankId = static_cast<uint64_t>(winContext->rankId);
    const int32_t hOutSize = axisH * sizeof(T); // 如有量化，需要量化后通信
    int32_t shmemDataLength = AlignUp<int32_t>(axisH, 512) + 512; // 512对齐，预留512三元组存储
    const int32_t tokenQuantAlign32 = AlignUp<int32_t>(hOutSize , 32) / sizeof(int32_t);
    __ubuf__ int32_t *tmpTokenBuffer = reinterpret_cast<__ubuf__ int32_t *>(tokenBuffer);

    for (int32_t row = tRowOffset; row < tRowOffset + tRowShape; ++row) {
        for (int32_t col = tColOffset; col < tColOffset + tColShape; ++col) {
            copy_gm_to_ubuf(tokenBuffer, token + row * axisH, 0, 1, hOutSize / 32, 0, 0);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            tmpTokenBuffer[tokenQuantAlign32] = static_cast<int32_t>(localUsrRankId);
            tmpTokenBuffer[tokenQuantAlign32 + 1] = row;
            tmpTokenBuffer[tokenQuantAlign32 + 2] = col;
            int32_t remoteExpertId = *(expertTableUb + row * topK + col);
            int32_t tableIndex = row * topK + col;
            int32_t remoteRankId = remoteExpertId / static_cast<int32_t>(shmemDataRawShape2);
            int32_t remoteExpertOffset = remoteExpertId % static_cast<int32_t>(shmemDataRawShape2);
            CalcOccurrences(expertTableUb, remoteExpertId, tableIndex, expertBuffer);
            int32_t tokenOffset = *(expertBuffer);
            __gm__ T* remoteShmemBaseAddr = MapVirtualAddr<T>(hcclContext, shmemDataBaseAddr, static_cast<uint32_t>(remoteRankId));
            __gm__ T* remoteShmemDataAddr = remoteShmemBaseAddr + static_cast<uint64_t>(localUsrRankId *
                    static_cast<uint64_t>(shmemDataRawShape2) * static_cast<uint64_t>(shmemDataRawShape3) +
                    static_cast<uint64_t>(remoteExpertOffset) * static_cast<uint64_t>(shmemDataRawShape3) +
                    static_cast<uint64_t>(tokenOffset) * shmemDataLength);
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            copy_ubuf_to_gm(remoteShmemDataAddr, tokenBuffer, 0 , 1, shmemDataLength * sizeof(T) / 32, 0, 0);
            set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }
}

template <typename T, int32_t bs, int32_t axisH, int32_t tileRowShape, int32_t groupIndex>
TILEOP void SendToSharedExpert(__gm__ int32_t *syncTensor, __ubuf__ T *tokenBuffer, __gm__ T *token, __gm__ T *shmemDataBaseAddr,
    uint32_t tokenOffset0, uint32_t tokenOffset1, uint32_t tokenShape0, uint32_t tokenShape1,
    uint32_t shmemDataOffset0, uint32_t shmemDataOffset1, uint32_t shmemDataOffset2, uint32_t shmemDataOffset3,
    uint32_t shmemDataRawShape0, uint32_t shmemDataRawShape1, uint32_t shmemDataRawShape2, uint32_t shmemDataRawShape3,
    __gm__ int64_t *hcclContext)
{
    __gm__ HcclCombinOpParam *winContext = (__gm__ HcclCombinOpParam *)(hcclContext[groupIndex]);
    int32_t localUsrRankId = static_cast<int32_t>(winContext->rankId);
    int32_t rankSize = static_cast<int32_t>(winContext->rankNum);
    
    constexpr int32_t hOutSize = axisH * sizeof(T);
    constexpr int32_t tokenQuantAlign32 = AlignUp<int32_t>(hOutSize, 32) / sizeof(int32_t);
    int32_t shareRankSize = 1; // 在前端赋值 shareRankCnt
    int32_t moeRankSize = rankSize - shareRankSize;
    int32_t shareOpProcessRankSize = moeRankSize / shareRankSize;

    __ubuf__ int32_t *tmpTokenBuffer = reinterpret_cast<__ubuf__ int32_t *>(tokenBuffer); // 类型转换使用
    for (int32_t row = tokenOffset0; row < tokenOffset0 + tileRowShape; ++row) {
        copy_gm_to_ubuf(tokenBuffer, token + row * axisH, 0, 1, hOutSize / 32, 0, 0);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        tmpTokenBuffer[tokenQuantAlign32] = localUsrRankId; // 暂时先不考虑 A3 数组，前端大小可以暂时用 token[1],暂不修改
        tmpTokenBuffer[tokenQuantAlign32 + 1] = row;
        tmpTokenBuffer[tokenQuantAlign32 + 2] = 1;
        // 对齐后续 sched 使用方式，连续的 moe 卡发给同一个共享专家
        uint32_t remoteRankId = (localUsrRankId - shareRankSize) / shareOpProcessRankSize;
        GM_ADDR remoteShmemBaseAddr = (GM_ADDR)MapVirtualAddr<T>(hcclContext, shmemDataBaseAddr, remoteRankId);
        GM_ADDR remoteShmemDataAddr = remoteShmemBaseAddr + localUsrRankId * shmemDataRawShape2 * shmemDataRawShape3 * sizeof(T) + row * shmemDataRawShape3 * sizeof(T);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        copy_ubuf_to_gm(remoteShmemDataAddr, tokenBuffer, 0, 1, shmemDataRawShape3 * sizeof(T) / 32, 0, 0);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
}

template <typename T, int32_t bs, int32_t axisH, int32_t tileRowShape>
TILEOP void CopyToLocalExpert(__gm__ T *expandX, __gm__ int32_t *syncTensor, __ubuf__ T *tokenBuffer, __gm__ T *token,
    uint32_t tokenOffset0, uint32_t tokenOffset1, uint32_t tokenShape0, uint32_t tokenShape1, __gm__ int64_t *hcclContext)
{
    constexpr int32_t hOutSize = axisH * sizeof(T);
    constexpr int32_t hCommuSize = hOutSize;

    for (int32_t row = tokenOffset0; row < tokenOffset0 + tileRowShape; ++row) {
        copy_gm_to_ubuf(tokenBuffer, token + row * axisH, 0, 1, hOutSize / 32, 0, 0);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        copy_ubuf_to_gm(expandX + row * axisH, tokenBuffer, 0, 1, hOutSize / 32, 0, 0);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    }
}

template <typename T, int32_t bs, int32_t topK, int32_t groupIndex, int32_t expertShape, int32_t rankShape>
TILEOP void DispatchSetFlag(__gm__ int32_t *syncDummy, __ubuf__ int32_t *statusTensor, __ubuf__ int32_t *expertTableUb,
    __ubuf__ int32_t *expertBuffer, __gm__ T *expertTable, __gm__ int32_t *shmemFlagBaseAddr, __gm__ int32_t *syncTensor,
    uint32_t shmemFlagOffset0, uint32_t shmemFlagOffset1, uint32_t shmemFlagOffset2, uint32_t shmemFlagOffset3,
    uint32_t shmemFlagRawShape0, uint32_t shmemFlagRawShape1, uint32_t shmemFlagRawShape2, uint32_t shmemFlagRawShape3,
    __gm__ int64_t *hcclContext)
{
    (void) shmemFlagOffset2;
    (void) shmemFlagOffset3;
    (void) shmemFlagRawShape0;
    __gm__ HcclCombinOpParam *winContext = (__gm__ HcclCombinOpParam *)(hcclContext[groupIndex]);
    int32_t localUsrRankId = static_cast<int32_t>(winContext->rankId);
    constexpr int32_t expertTblSize = bs * topK;
    constexpr int32_t lenBurst = AlignUp<int32_t>(expertTblSize * sizeof(int32_t), 32) / 32;
    copy_gm_to_ubuf(expertTableUb, expertTable, 0, 1, lenBurst, 0, 0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    int32_t offset = 0;
    for (int32_t rankId = shmemFlagOffset0; rankId < shmemFlagOffset0 + rankShape; ++rankId) {
        __gm__ int32_t* remoteFlagBaseAddr = MapVirtualAddr<T>(hcclContext, shmemFlagBaseAddr, rankId);
        for (int32_t dstExpertId = shmemFlagOffset1; dstExpertId < shmemFlagOffset1 + expertShape; ++dstExpertId) {
            int32_t remoteExpertId = dstExpertId + rankId * shmemFlagRawShape1;
            __gm__ int32_t* shmemFlagWriteAddr = remoteFlagBaseAddr + dstExpertId * shmemFlagRawShape2 * shmemFlagRawShape3
                 + localUsrRankId * shmemFlagRawShape3;
            statusTensor[dstExpertId * 8] = 1;
            CalcOccurrences(expertTableUb, remoteExpertId, expertTblSize, (expertBuffer + offset));
            statusTensor[dstExpertId * 8 + 1] = *(expertBuffer + offset);
            offset++;
            copy_ubuf_to_gm(shmemFlagWriteAddr, statusTensor + dstExpertId * 8, 0, 1, 1, 0, 0);
            set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        }
    }
}

TILEOP void CopyOutRecvTokenCnt(GM_ADDR outRecvTokenCntAddr, UB_ADDR recvTokenCntAddr, uint32_t tileIndex,
    uint32_t totalTileNum)
{
    DataCopyParams dataCopyParams;
    dataCopyParams.sid = 0;
    dataCopyParams.nBurst = 1; // 搬运次数
    dataCopyParams.lenBurst = 1; // 每次搬运的数据量大小，32B 为单位，1 表示每次搬运 32B
    dataCopyParams.srcStride = 0; // 前一个尾巴和下一个的开头，gap，不重要
    dataCopyParams.dstStride = 15; // dst每次跳跃15 * 32
 
    uint32_t offset = totalTileNum * 512; // 每个 op 写 48 个 512B 大小的地址
    GM_ADDR outRecvTokenCntStartAddr = outRecvTokenCntAddr + tileIndex * offset; // 本 op 偏移地址
    // 搬运需要使用同一个 src，所以需要手动循环
    for (int i = 0; i < totalTileNum; i++) {
        copy_ubuf_to_gm(outRecvTokenCntStartAddr, recvTokenCntAddr, dataCopyParams.sid, dataCopyParams.nBurst,
            dataCopyParams.lenBurst, dataCopyParams.srcStride, dataCopyParams.dstStride);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        outRecvTokenCntStartAddr += 512; // 需要将同一个 src 连续写 48 次，所以 src 不变化，dst 每次手动偏移 512B
    }
}

template <typename T>
TILEOP void ConstructOutRecvTokenCnt(__gm__ T *out, __ubuf__ uint32_t *src0, __ubuf__ uint32_t *src1,
    __ubuf__ uint32_t *dst, uint32_t cnt, __gm__ int64_t *hcclContext, DispatchInfo &dispatchInfo)
{
    GatherMaskAndSum(out, src0, src1, dst, MASK_SELECT_SEND_COUNT, cnt, hcclContext);
 
    GM_ADDR outRecvTokenCntAddr = reinterpret_cast<GM_ADDR>(out);
    UB_ADDR recvTokenCntAddr = reinterpret_cast<UB_ADDR>(src1); // src1 复用为了 sum 最终的输出
 
    CopyOutRecvTokenCnt(outRecvTokenCntAddr, recvTokenCntAddr, dispatchInfo.tileIndex, dispatchInfo.totalTileNum);
}

template <typename T>
TILEOP void MoeRankWaitFlag(__gm__ T *out, __ubuf__ uint32_t *src0, __ubuf__ uint32_t *src1, __ubuf__ uint32_t *dst,
    __gm__ int64_t *hcclContext, uint32_t processRankSize, __gm__ int32_t *shmemFlagBaseAddr, DispatchInfo &dispatchInfo)
{
    uint32_t cnt = processRankSize; // 计算次数，根据处理的expert数, 每个op需要去等
    uint32_t flagSum = 0;
    uint32_t offset = dispatchInfo.expertIndex * dispatchInfo.rankNum * 512 + dispatchInfo.rankOffset * 512;
    while (flagSum != cnt) {
        ReadFlagV2<T>(src0, offset, cnt, hcclContext, shmemFlagBaseAddr, dispatchInfo);
        WaitFlagV2<T>(out, src0, src1, dst, cnt, hcclContext);
        // src1 复用为 sum 的输出
        flagSum = src1[0];
    }
    // 理论上这里不需要再读，前面已经确保写上去了
    ConstructOutRecvTokenCnt<T>(out, src0, src1, dst, cnt, hcclContext, dispatchInfo);
    ClearFlagV2(reinterpret_cast<__ubuf__ int32_t *>(src0), offset, cnt, hcclContext, dispatchInfo, shmemFlagBaseAddr); // 暂时放在读完之后就清 flag
}
 
template <typename T>
TILEOP void ShareRankWaitFlag(__gm__ T *out, __ubuf__ uint32_t *src0, __ubuf__ uint32_t *src1, __ubuf__ uint32_t *dst,
    __gm__ int64_t *hcclContext, uint32_t processRankSize, __gm__ int32_t *shmemFlagBaseAddr, DispatchInfo &dispatchInfo)
{
    // 当前 share rank 接收的起始的 moe rank id
    uint32_t startMoeRankId = dispatchInfo.tileIndex * processRankSize + dispatchInfo.shareRankCnt;
    if (dispatchInfo.tileIndex != 0) {
        return;
    }
    uint32_t cnt = processRankSize; // 计算次数，共享专家固定处理 8 个卡
    uint32_t flagSum = 0;
    uint32_t offset = startMoeRankId * 512; // 每张卡占据 512B，flag 读取偏移为 rankID * 512B
    while (flagSum != cnt) {
        ReadFlagV2<T>(src0, offset, cnt, hcclContext, shmemFlagBaseAddr, dispatchInfo);
        WaitFlagV2<T>(out, src0, src1, dst, cnt, hcclContext);
        // src1 复用为 sum 的输出
        flagSum = src1[0];
    }
    ClearFlagV2(reinterpret_cast<__ubuf__ int32_t *>(src0), offset, cnt, hcclContext, dispatchInfo, shmemFlagBaseAddr); // 暂时放在读完之后就清 flag
}
 
// tileIndex 覆写为了 tileOpIndex 计数
template<typename T, uint32_t tileIndex, uint32_t groupIndex, uint32_t shareRankCnt, uint32_t totalTileNum, int32_t rankShape, int32_t expertNum>
TILEOP void FFNSched(__gm__ T *out, __ubuf__ int32_t *buffer, __gm__ int32_t *dummy, __gm__ int32_t *shmemFlagBaseAddr,
    uint32_t flagShmemOffset0, uint32_t flagShmemOffset1, uint32_t flagShmemOffset2, uint32_t flagShmemOffset3,
    uint32_t flagShmemShape0, uint32_t flagShmemShape1, uint32_t flagShmemShape2, uint32_t flagShmemShape3, __gm__ int64_t *hcclContext)
{
    __gm__ HcclCombinOpParam *winContext = (__gm__ HcclCombinOpParam *)(hcclContext[groupIndex]);
    DispatchInfo dispatchInfo = {tileIndex, groupIndex, 0, 0, rankShape, 
        static_cast<int32_t>(flagShmemOffset2), 0, 0, 0, 0, totalTileNum, shareRankCnt, expertNum,
        static_cast<int32_t>(winContext->rankNum), static_cast<int32_t>(flagShmemOffset1)};

    __ubuf__ uint8_t *tmpUb = reinterpret_cast<__ubuf__ uint8_t *>(buffer);
    uint32_t offset = 0;
    __ubuf__ uint32_t *src0 = reinterpret_cast<__ubuf__ uint32_t *>(tmpUb + offset);
    uint32_t moeOpProcessRankSize = dispatchInfo.rankShape;
    uint32_t src0Size = moeOpProcessRankSize * 32; // 每个 op 最多等待的 flag 卡数，最多是 8 个，8 * 32 = 256B
    offset += src0Size;
    __ubuf__ uint32_t *src1 = reinterpret_cast<__ubuf__ uint32_t *>(tmpUb + offset);
    uint32_t src1Size = 32; // 第一次是 mask，32B，第二次复用为 sum 的结果，一个 float 4B；所以最大为 32B
    offset += src1Size;
    __ubuf__ uint32_t *dst = reinterpret_cast<__ubuf__ uint32_t *>(tmpUb + offset);
    uint32_t dstSize = (moeOpProcessRankSize * 4 + 31) / 32 * 32; // src0 中挑出来的 int 个数，最大是 8 个，正好是 32B
    offset += dstSize;

    MoeRankWaitFlag<T>(out, src0, src1, dst, hcclContext, moeOpProcessRankSize, shmemFlagBaseAddr, dispatchInfo);
}

TILEOP void ReadRecvTokenCnt(__ubuf__ uint32_t *recvTokenCnt, __gm__ uint32_t *src,
    DispatchInfo &dispatchInfo, __gm__ int64_t *hcclContext, uint32_t tileCnt)
{
    DataCopyParams gmToUbParams;
    gmToUbParams.sid = 0;
    gmToUbParams.nBurst = dispatchInfo.tileIndex; // 搬运次数，只搬运能用到的数据即可

    gmToUbParams.lenBurst = 1; // cnt 有效数据只有 4B，搬运 32B 即可
    // 前一个尾巴和下一个的开头，gap，最大值 65535，每次偏移 48*512B，48 * 512 / 32 - 1
    gmToUbParams.srcStride = dispatchInfo.totalTileNum * 512 / 32 - 1;
    gmToUbParams.dstStride = 0; // 前一个尾巴和下一个开头，gap，搬运成连续
 
    // 每个 tile 都读自己 index 的 gm 地址，避免地址交织
    GM_ADDR thisTileStartSrcAddr = reinterpret_cast<GM_ADDR>(src) + dispatchInfo.tileIndex * 512;

    copy_gm_to_ubuf(recvTokenCnt, thisTileStartSrcAddr, gmToUbParams.sid, gmToUbParams.nBurst, gmToUbParams.lenBurst,
        gmToUbParams.srcStride, gmToUbParams.dstStride);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
}

template<typename T, int32_t expertShape>
TILEOP void FFNValidCnt(__gm__ int32_t *validCnt, __ubuf__ int32_t *buffer, __gm__ int32_t *gmRecvTokenCnt, __gm__ int32_t *shmemFlagBaseAddr,
    uint32_t shmemFlagOffset0, uint32_t shmemFlagOffset1, uint32_t shmemFlagOffset2, uint32_t shmemFlagOffset3,
    uint32_t shmemFlagRawShape0, uint32_t shmemFlagRawShape1, uint32_t shmemFlagRawShape2, uint32_t shmemFlagRawShape3, __gm__ int64_t *hcclContext)
{
    int32_t localUsrRankId = static_cast<int32_t>(shmemFlagOffset0);
    __gm__ int32_t *winFlagBaseAddr = MapVirtualAddr<int32_t>(hcclContext, shmemFlagBaseAddr, localUsrRankId);
    __ubuf__ uint32_t *flag = reinterpret_cast<__ubuf__ uint32_t*>(buffer);
    uint32_t flagSize = 32;
    __ubuf__ int32_t *receiveCnt = reinterpret_cast<__ubuf__ int32_t *>(buffer + flagSize);
    int32_t offsetResult = 0;
    for (int32_t expertId = shmemFlagOffset1; expertId < shmemFlagOffset1 + expertShape; ++expertId) {
        uint32_t receiveToken = 0;
        for (int32_t rankId = 0; rankId < shmemFlagRawShape0; ++rankId) {
            uint32_t thisRankFlagOffset = expertId * shmemFlagRawShape0 * 512 + rankId * 512;
            GM_ADDR winFlagReadStartAddr = (GM_ADDR) winFlagBaseAddr + thisRankFlagOffset;
            DataCopyParams dataCopyParams;
            dataCopyParams.sid = 0;
            dataCopyParams.nBurst = 1;
            dataCopyParams.lenBurst = 1;
            dataCopyParams.srcStride = 15;
            dataCopyParams.dstStride = 0;
            set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
            copy_gm_to_ubuf(flag, winFlagReadStartAddr, dataCopyParams.sid, dataCopyParams.nBurst, dataCopyParams.lenBurst,
                dataCopyParams.srcStride, dataCopyParams.dstStride);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            receiveToken += flag[1];
        }
        pipe_barrier(PIPE_ALL);
        receiveCnt[offsetResult++] = receiveToken;
    }
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    TileOp::UBCopyOut<int32_t, 1, expertShape, expertShape, expertShape>(validCnt + shmemFlagOffset1, receiveCnt);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
}

template<typename T>
TILEOP void CombineInfoCopyOut(__gm__ int32_t *combineInfo, __ubuf__ uint8_t *combineInfoBuffer, DispatchInfo &dispatchInfo,
    __gm__ int64_t *hcclContext, __gm__ T *shmemDataBaseAddr, __gm__ int32_t *shmemFlagBaseAddr,
    uint32_t rankSize, uint32_t bs, uint32_t shmemLength)
{
    __gm__ HcclCombinOpParam *winContext = (__gm__ HcclCombinOpParam *)(hcclContext[dispatchInfo.groupIndex]);
    uint32_t localUsrRankId = winContext->rankId;
    GM_ADDR localShmemBaseAddr = (GM_ADDR)MapVirtualAddr<T>(hcclContext, shmemDataBaseAddr, localUsrRankId);
    __ubuf__ int32_t *combineBuffer = reinterpret_cast<__ubuf__ int32_t *>(combineInfoBuffer);
    __ubuf__ uint32_t *flag = reinterpret_cast<__ubuf__ uint32_t *>(combineBuffer);
    __ubuf__ int32_t *buffer = reinterpret_cast<__ubuf__ int32_t *>(combineBuffer + 32); // flag两位有效，预留32足够
    uint32_t tokenCnt = 0; // 本 op 处理的 cnt 总数

    for (int32_t rankId = dispatchInfo.rankOffset; rankId < dispatchInfo.rankOffset + dispatchInfo.rankShape; rankId++) {
        GM_ADDR thisRankExpertAddrBase = localShmemBaseAddr + rankId * dispatchInfo.expertNumPerRank * bs * shmemLength * sizeof(T);
        GM_ADDR thisRankExpertTokenAddr = thisRankExpertAddrBase + dispatchInfo.expertIndex * bs * shmemLength * sizeof(T);
        __gm__ int32_t *thisExpertCombineAddr = combineInfo + tokenCnt * MOE_COMBINE_INFO_NUM;
        GM_ADDR thisRankExpertCombineAddr = thisRankExpertTokenAddr + AlignUp<uint64_t>(dispatchInfo.colShape, 512) * sizeof(T);

        uint32_t thisRankFlagOffset = dispatchInfo.expertIndex * rankSize * 512 + rankId * 512; // 每个flag大小位512B
        ReadFlagV2(flag, thisRankFlagOffset, 1, hcclContext, shmemFlagBaseAddr, dispatchInfo); // 每次读取一张卡的 flag, 512B, 存到 UB 是 32B
        pipe_barrier(PIPE_ALL);
        uint32_t thisRankSendTokenCnt = flag[1]; // count 在 flag 第二个数

        uint32_t shmemColBurst = shmemLength * sizeof(T) / sizeof(int32_t);
        __gm__ int32_t *combineExpertAddr = reinterpret_cast<__gm__ int32_t*>(thisRankExpertCombineAddr);
        for (int i = 0; i < thisRankSendTokenCnt; i++) {
            set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
            TileOp::UBCopyIn<int32_t, 1, MOE_COMBINE_INFO_NUM, 8, MOE_COMBINE_INFO_NUM>(buffer, combineExpertAddr + i * shmemColBurst);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TileOp::UBCopyOut<int32_t, 1, MOE_COMBINE_INFO_NUM, MOE_COMBINE_INFO_NUM, 8>(thisExpertCombineAddr + i * MOE_COMBINE_INFO_NUM, buffer);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        tokenCnt += thisRankSendTokenCnt;
    }
}

template<typename T>
TILEOP void MoeRankWinCopyOut(__gm__ T *expandX, __gm__ uint32_t *validCnt, __ubuf__ uint8_t *buffer,
    DispatchInfo &dispatchInfo, __gm__ int64_t *hcclContext, __gm__ T *shmemDataBaseAddr, __gm__ int32_t *shmemFlagBaseAddr,
    uint32_t rankSize, uint32_t bs, uint32_t shmemLength)
{
    uint32_t offset = 0;
    __ubuf__ T *token = reinterpret_cast<__ubuf__ T *>(buffer + offset);
    uint32_t tokenSize = dispatchInfo.colShape * sizeof(T); // 这里需要为真实的 colShape
    offset = offset + tokenSize;
    __ubuf__ uint32_t *flag = reinterpret_cast<__ubuf__ uint32_t *>(buffer + offset);
    uint32_t flagSize = 32; // 读取的一张卡的 flag，一共两个 int 有效数字，32B 即可存放
    offset = offset + flagSize;

    __gm__ HcclCombinOpParam *winContext = (__gm__ HcclCombinOpParam *)(hcclContext[dispatchInfo.groupIndex]);
    uint32_t localUsrRankId = winContext->rankId;
    GM_ADDR localShmemBaseAddr = (GM_ADDR)MapVirtualAddr<T>(hcclContext, shmemDataBaseAddr, localUsrRankId);

    uint32_t tokenCnt = 0; // 本 op 处理的 cnt 总数
    uint16_t lenBurst = static_cast<uint16_t>(dispatchInfo.colShape * sizeof(T) / 32);
    DataCopyParams gmToUbParams = {0, 1, lenBurst, 0, 0};
    DataCopyParams ubToGmParams = {0, 1, lenBurst, 0, 0};
    for (int32_t rankId = dispatchInfo.rankOffset; rankId < dispatchInfo.rankOffset + dispatchInfo.rankShape; rankId++) {
        GM_ADDR thisRankExpertAddrBase = localShmemBaseAddr + rankId * dispatchInfo.expertNumPerRank * bs * shmemLength * sizeof(T);
        GM_ADDR thisRankOutAddr = reinterpret_cast<GM_ADDR>(expandX) + tokenCnt * dispatchInfo.colShape * sizeof(T);
        GM_ADDR thisRankExpertTokenAddr = thisRankExpertAddrBase + dispatchInfo.expertIndex * bs * shmemLength * sizeof(T);

        uint32_t thisRankFlagOffset = dispatchInfo.expertIndex * rankSize * 512 + rankId * 512; // 当前 rank 在 flag 的偏移
        ReadFlagV2(flag, thisRankFlagOffset, 1, hcclContext, shmemFlagBaseAddr, dispatchInfo); // 每次读取一张卡的 flag, 512B, 存到 UB 是 32B
        uint32_t thisRankSendTokenCnt = flag[1]; // count 在 flag 第二个数
        tokenCnt += thisRankSendTokenCnt; 
        for (int i = 0; i < thisRankSendTokenCnt; i++) {
            CopyGmToGm(thisRankOutAddr + i * dispatchInfo.colShape * sizeof(T), 
            thisRankExpertTokenAddr + i * shmemLength * sizeof(T), token, gmToUbParams, ubToGmParams);
        }
    }
}

// 对于 MOE 专家卡，搬出需要确定当前 TileOp 收到了多少个 token，获取偏移
template<typename T1, typename T2>
TILEOP void MoeRankCopyOut(__gm__ T1 *out, __gm__ uint32_t *validCnt, __ubuf__ uint8_t *buffer,
    __gm__ uint32_t *gmRecvTokenCnt, DispatchInfo &dispatchInfo, __gm__ int64_t *hcclContext, __gm__ T2 *shmemDataBaseAddr,
    __gm__ int32_t *shmemFlagBaseAddr, uint32_t rankSize, uint32_t bs, uint32_t shmemLength, bool copyOutData)
{
    int tileCnt = dispatchInfo.totalTileNum;
    uint32_t offset = 0;
    __ubuf__ uint32_t *recvTokenCnt = reinterpret_cast<__ubuf__ uint32_t *>(buffer + offset);
    uint32_t recvTokenCntSize = (tileCnt * 32 + 255) / 256 * 256;
    offset = offset + recvTokenCntSize;
    // 第一次作为 GatherMask 的 mask，只有两个数；第二次作为 cumSum 的输出，只有一个数；但是会使用指令清空，最小 256B；所以最终 256B
    __ubuf__ uint32_t *cumSumDst = reinterpret_cast<__ubuf__ uint32_t *>(buffer + offset);
    uint32_t cumSumDstSize = 256;
    offset = offset + cumSumDstSize;
    __ubuf__ uint32_t *gatherMaskDst = reinterpret_cast<__ubuf__ uint32_t *>(buffer + offset);
    // 作为 gathermask 的 dst，最多会存放 totalTileNum 个 int
    uint32_t gatherMaskDstSize = (tileCnt * 4 + 31) / 32 * 32;
    offset = offset + gatherMaskDstSize;
    ReadRecvTokenCnt(recvTokenCnt, gmRecvTokenCnt, dispatchInfo, hcclContext, tileCnt);

    // 计算的最终结果在 cumSumDst 中，一个 float 类型数据
    // tileIndex 就是计算 recvTokenOffset 时 sum 的计算 cnt
    CumSum(cumSumDst, recvTokenCnt, gatherMaskDst, MASK_SELECT_RECV_TOKEN_CNT, dispatchInfo.tileIndex);
    uint32_t recvTokenOffset = cumSumDst[0];

    if (copyOutData) {
        uint32_t expandXOffset = recvTokenOffset * dispatchInfo.colShape;
        __gm__ T2* expandX = reinterpret_cast<__gm__ T2 *>(out);
        MoeRankWinCopyOut<T2>(expandX + expandXOffset, validCnt, buffer, dispatchInfo, hcclContext, shmemDataBaseAddr,
        shmemFlagBaseAddr, rankSize, bs, shmemLength);
    } else {
        int32_t infoOffset = static_cast<int32_t>(recvTokenOffset * MOE_COMBINE_INFO_NUM);
        __gm__ int32_t *combineInfo = reinterpret_cast<__gm__ int32_t *>(out);
        CombineInfoCopyOut<T2>(combineInfo + infoOffset, buffer, dispatchInfo, hcclContext, shmemDataBaseAddr,
        shmemFlagBaseAddr, rankSize, bs, shmemLength);
    }
}

template<typename T>
TILEOP void ShareRankWinCopyOut(__gm__ T *expandX, __ubuf__ uint8_t *buffer, uint32_t processTokenCnt,
    uint32_t recvTokenOffset, DispatchInfo &dispatchInfo, __gm__ int64_t *hcclContext, uint32_t processMoeRankCnt, __gm__ T *shmemDataAddr,
    uint32_t shmemDataRawShape2, uint32_t shmemDataRawShape3)
{
    if (processTokenCnt == 0) { // bs 泛化场景下可能存在不够 48 个 op 切分的场景
        return;
    }
    __gm__ HcclCombinOpParam *winContext = (__gm__ HcclCombinOpParam *)(hcclContext[dispatchInfo.groupIndex]);
    uint32_t localUsrRankId = winContext->rankId;
    GM_ADDR winTokenAddr = (GM_ADDR)MapVirtualAddr<T>(hcclContext, shmemDataAddr, localUsrRankId);
    uint32_t shareRankCnt = dispatchInfo.shareRankCnt;
    uint32_t processMoeRankStartIdx = localUsrRankId * processMoeRankCnt + shareRankCnt; // 本共享专家接收的 moe 专家起始 rank id
    // 每张卡在 win 区都占据完整大小，row，col，共享专家处理从 startIdx 开始的连续 64 个 token（处理 scale + 三元组）
    GM_ADDR thisRankWinTokenAddr = winTokenAddr + processMoeRankStartIdx * shmemDataRawShape2 * shmemDataRawShape3 * sizeof(T);
    GM_ADDR thisTileWinTokenAddr = thisRankWinTokenAddr + recvTokenOffset * shmemDataRawShape3 * sizeof(T);
 
    uint32_t offset = 0;
    __ubuf__ T *token = reinterpret_cast<__ubuf__ T *>(buffer + offset);
    uint32_t tokenSize = dispatchInfo.colShape * sizeof(T);
    offset = offset + tokenSize;
 
    DataCopyParams gmToUbParams;
    gmToUbParams.sid = 0;
    gmToUbParams.nBurst = 1; // 搬运次数，防止 UB 越界，也为了避免 UB 大小不固定
    gmToUbParams.lenBurst = dispatchInfo.colShape * sizeof(T) / 32; // x 实际 col 大小，不包括后面追加的东西
    gmToUbParams.srcStride = 0; // 前一个尾巴和下一个的开头，gap，这里根据 token 结构需要跳跃 7168+512
    gmToUbParams.dstStride = 0; // 前一个尾巴和下一个开头，gap
 
    DataCopyParams ubToGmParams;
    ubToGmParams.sid = 0;
    ubToGmParams.nBurst = 1; // 搬运次数，
    ubToGmParams.lenBurst = dispatchInfo.colShape * sizeof(T) / 32; // col 就是 token 实际列宽
    ubToGmParams.srcStride = 0; // 前一个尾巴和下一个的开头，gap
    ubToGmParams.dstStride = 0; // 前一个尾巴和下一个开头，gap
 
    GM_ADDR thisTileOutAddr = reinterpret_cast<GM_ADDR>(expandX);
    for (int i = 0; i < processTokenCnt; i++) {
        CopyGmToGm(thisTileOutAddr, thisTileWinTokenAddr, token, gmToUbParams, ubToGmParams);
        thisTileWinTokenAddr += shmemDataRawShape3 * sizeof(T); // colShape 需要修改为冗余长度的 shape
        thisTileOutAddr += dispatchInfo.colShape * sizeof(T);
    }
}

// 对于共享专家卡，其收到的 token 个数是固定的，直接切分搬出即可，不需要 cumsum 计算
template<typename T>
TILEOP void ShareRankCopyOut(__gm__ T *expandX, __ubuf__ uint8_t *buffer, DispatchInfo &dispatchInfo,
    __gm__ int64_t *hcclContext, __gm__ T*shmemDataBaseAddr, uint32_t shmemDataRawShape2, uint32_t shmemDataRawShape3)
{
    __gm__ HcclCombinOpParam *winContext = (__gm__ HcclCombinOpParam *)(hcclContext[dispatchInfo.groupIndex]);
 
    // 理论上不需要切分 TileOp, 但是为了和 moe 专家切分保持一致，按照 token cnt 切分
    uint32_t vectorCnt = dispatchInfo.totalTileNum; // 理论上的核数，device 实际核数不是 48 也不影响
    // 每个共享专家处理的 moe 卡数
    uint32_t processMoeRankCnt = (winContext->rankNum - dispatchInfo.shareRankCnt) / dispatchInfo.shareRankCnt;
    // bs * processMoeRankCnt, bs 典型值 8，从某张卡开始连续的 64 个 token
    uint32_t tokenCntRecvFromMoeRank = processMoeRankCnt * dispatchInfo.rowShape;
    uint32_t tailProcessTokenCnt = tokenCntRecvFromMoeRank / vectorCnt; // 尾块处理 token 个数
    uint32_t tileCnt = tokenCntRecvFromMoeRank % vectorCnt; // 整块个数
    uint32_t tileProcessTokenCnt = tailProcessTokenCnt + 1; // 整块处理 token 个数
 
    uint32_t recvTokenOffset = 0; // 本 TileOp 处理的 token 的偏移
    uint32_t processTokenCnt = 0;
    if (dispatchInfo.tileIndex < tileCnt) {
        recvTokenOffset = dispatchInfo.tileIndex * tileProcessTokenCnt;
        processTokenCnt = tileProcessTokenCnt;
    } else {
        recvTokenOffset = tileCnt * tileProcessTokenCnt + (dispatchInfo.tileIndex - tileCnt) * tailProcessTokenCnt;
        processTokenCnt = tailProcessTokenCnt;
    }
    uint32_t outOffset = dispatchInfo.rowShape * dispatchInfo.colShape; // 共享专家本身的 8 个 token 放在最开头
    outOffset += recvTokenOffset * dispatchInfo.colShape; // 元素个数
    ShareRankWinCopyOut<T>(expandX + outOffset, buffer, processTokenCnt, recvTokenOffset, dispatchInfo, hcclContext,
        processMoeRankCnt, shmemDataBaseAddr, shmemDataRawShape2, shmemDataRawShape3);
}

template<typename T, uint32_t tileIndex, uint32_t groupIndex, uint32_t shareRankCnt, uint32_t totalTileNum, uint32_t rankShape, uint32_t axisH, uint32_t bs, uint32_t expandXRow>
TILEOP void FFNBatching(__gm__ T *expandX, __gm__ int32_t *validCnt, __ubuf__ int32_t *buffer,
    __gm__ T *shmemDataBaseAddr, __gm__ int32_t *shmemFlagBaseAddr, __gm__ int32_t *gmRecvTokenCnt,
    uint32_t shmemDataOffset0, uint32_t shmemDataOffset1, uint32_t shmemDataOffset2, uint32_t shmemDataOffset3,
    uint32_t shmemDataShape0, uint32_t shmemDataShape1, uint32_t shmemDataShape2, uint32_t shmemDataShape3, __gm__ int64_t *hcclContext)
{
    DispatchInfo dispatchInfo = {tileIndex, groupIndex, 0, 0, rankShape,
        static_cast<int32_t>(shmemDataOffset1), bs, 0, axisH, 0, totalTileNum, shareRankCnt,
        static_cast<int32_t>(shmemDataShape2), static_cast<int32_t>(shmemDataShape0),
        static_cast<int32_t>(shmemDataOffset2)};
    int32_t shmemLength = AlignUp<int32_t>(axisH, 512) + 512;

    MoeRankCopyOut<T, T>(expandX, reinterpret_cast<__gm__ uint32_t *>(validCnt),
        reinterpret_cast<__ubuf__ uint8_t *>(buffer), reinterpret_cast<__gm__ uint32_t *>(gmRecvTokenCnt),
        dispatchInfo, hcclContext, shmemDataBaseAddr, shmemFlagBaseAddr, shmemDataShape0, bs, shmemLength, true);
}

template<typename T, uint32_t tileIndex, uint32_t groupIndex, uint32_t shareRankCnt, uint32_t totalTileNum, uint32_t rankShape, uint32_t axisH, uint32_t bs, uint32_t expandXRow>
TILEOP void FFNCombineInfo(__gm__ int32_t *combineInfo, __ubuf__ int32_t *buffer, __gm__ T *shmemDataBaseAddr, 
    __gm__ int32_t *shmemFlagBaseAddr, __gm__ int32_t *gmRecvTokenCnt, uint32_t shmemDataOffset0,
    uint32_t shmemDataOffset1, uint32_t shmemDataOffset2, uint32_t shmemDataOffset3, uint32_t shmemDataShape0,
    uint32_t shmemDataShape1, uint32_t shmemDataShape2, uint32_t shmemDataShape3, __gm__ int64_t *hcclContext)
{
    DispatchInfo dispatchInfo = {tileIndex, groupIndex, 0, 0, rankShape,
        static_cast<int32_t>(shmemDataOffset1), bs, 0, axisH, 0, totalTileNum, shareRankCnt,
        static_cast<int32_t>(shmemDataShape2), static_cast<int32_t>(shmemDataShape0),
        static_cast<int32_t>(shmemDataOffset2)};
    int32_t shmemLength = AlignUp<int32_t>(axisH, 512) + 512;

    MoeRankCopyOut<int32_t, T>(combineInfo, nullptr, reinterpret_cast<__ubuf__ uint8_t *>(buffer), reinterpret_cast<__gm__ uint32_t *>(gmRecvTokenCnt),
        dispatchInfo, hcclContext, shmemDataBaseAddr, shmemFlagBaseAddr, shmemDataShape0, bs, shmemLength, false);
}
} // namespace TileOp::Distributed

#endif