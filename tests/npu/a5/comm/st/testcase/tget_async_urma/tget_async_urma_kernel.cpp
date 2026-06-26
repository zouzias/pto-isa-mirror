/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <pto/pto-inst.hpp>

#include "../common.hpp"
#include "pto/common/pto_tile.hpp"
#ifdef PTO_URMA_SUPPORTED
#include "pto/comm/async/urma/urma_async_intrin.hpp"
#endif

// ============================================================================
// TGET_ASYNC via URMA — device kernel.
// ============================================================================

#ifdef COMM_DEBUG
enum UrmaDebugSlot : uint32_t
{
    DBG_MAGIC = 0,
    DBG_TARGET_PEER,
    DBG_LOCAL_TOKEN_ID,
    DBG_REMOTE_TID,
    DBG_REMOTE_TPN,
    DBG_REMOTE_ADDR,
    DBG_SQE_CTRL,
    DBG_SQE_RMT_OBJ_ID,
    DBG_SQE_RMT_TOKEN_VALUE,
    DBG_SQE_RMT_ADDR,
    DBG_SQE_RMT_EID_L,
    DBG_SQE_RMT_EID_H,
    DBG_SGE_LEN,
    DBG_SGE_TOKEN_ID,
    DBG_SGE_VA,
    DBG_CQE_CTRL,
    DBG_CQE_BYTE_CNT,
    DBG_CQE_TPN,
    DBG_CQE_DATA,
    DBG_SQ_HEAD,
    DBG_CQ_TAIL,
    DBG_POST_REMOTE_ADDR,
    DBG_POST_LOCAL_ADDR,
    DBG_WQE_BUF_ADDR,
    DBG_WQE_HEAD_ADDR,
    DBG_EID_PTR,
    DBG_EID_DATA_L,
    DBG_EID_DATA_H,
    DBG_WQN,
    DBG_DB_ADDR,
    DBG_DB_VALUE,
    DBG_SLOT_COUNT
};
#endif // COMM_DEBUG

#if defined(PTO_URMA_SUPPORTED) && defined(COMM_DEBUG)
AICORE inline void SnapshotUrmaDebug(__gm__ uint8_t *urmaWorkspace, __gm__ uint8_t *debugBase, uint32_t targetPeer,
                                     uint64_t postRemoteAddr, uint64_t postLocalAddr)
{
    using namespace pto::comm::urma;
    __gm__ uint64_t *dbg = reinterpret_cast<__gm__ uint64_t *>(debugBase);
    __gm__ UrmaInfo *info = reinterpret_cast<__gm__ UrmaInfo *>(urmaWorkspace);
    const uint32_t qpNum = info->qpNum;

    __gm__ UrmaMemInfo *remoteMem = reinterpret_cast<__gm__ UrmaMemInfo *>(info->memPtr) + targetPeer;
    __gm__ UrmaWQCtx *wq = reinterpret_cast<__gm__ UrmaWQCtx *>(info->sqPtr + targetPeer * qpNum * sizeof(UrmaWQCtx));
    __gm__ UrmaCqCtx *cq = reinterpret_cast<__gm__ UrmaCqCtx *>(info->scqPtr + targetPeer * qpNum * sizeof(UrmaCqCtx));

    DcciCachelines(reinterpret_cast<__gm__ uint8_t *>(wq->headAddr), sizeof(uint32_t));
    DcciCachelines(reinterpret_cast<__gm__ uint8_t *>(cq->tailAddr), sizeof(uint32_t));

    uint32_t sqHead = static_cast<uint32_t>(ld_dev(reinterpret_cast<__gm__ uint32_t *>(wq->headAddr), 0));
    uint32_t cqTail = static_cast<uint32_t>(ld_dev(reinterpret_cast<__gm__ uint32_t *>(cq->tailAddr), 0));
    uint32_t wqeIdx = (sqHead == 0) ? 0 : ((sqHead - 1U) & (wq->depth - 1U));
    uint32_t cqeIdx = (cqTail == 0) ? 0 : ((cqTail - 1U) & (cq->depth - 1U));

    __gm__ uint8_t *wqeAddr =
        reinterpret_cast<__gm__ uint8_t *>(wq->bufAddr + (1U << wq->wqeShiftSize) * wqeIdx);
    __gm__ UrmaJfcCqeCtx *cqe =
        reinterpret_cast<__gm__ UrmaJfcCqeCtx *>(cq->bufAddr + (1U << cq->cqeShiftSize) * cqeIdx);

    pipe_barrier(PIPE_ALL);
    DcciCachelines(wqeAddr, kUrmaSqeSizeBytes + kUrmaSgeSizeBytes);
    DcciCachelines(reinterpret_cast<__gm__ uint8_t *>(cqe), sizeof(UrmaJfcCqeCtx));

    dbg[DBG_MAGIC] = 0x554d524144424731ULL;
    dbg[DBG_TARGET_PEER] = targetPeer;
    dbg[DBG_LOCAL_TOKEN_ID] = info->localTokenId;
    dbg[DBG_REMOTE_TID] = remoteMem->tid;
    dbg[DBG_REMOTE_TPN] = remoteMem->tpn;
    dbg[DBG_REMOTE_ADDR] = remoteMem->addr;
    dbg[DBG_POST_REMOTE_ADDR] = postRemoteAddr;
    dbg[DBG_POST_LOCAL_ADDR] = postLocalAddr;
    dbg[DBG_WQE_BUF_ADDR] = wq->bufAddr;
    dbg[DBG_WQE_HEAD_ADDR] = wq->headAddr;
    dbg[DBG_SQE_CTRL] = (static_cast<uint64_t>(*reinterpret_cast<__gm__ uint32_t *>(wqeAddr + 4)) << 32) |
                        *reinterpret_cast<__gm__ uint32_t *>(wqeAddr);
    dbg[DBG_SQE_RMT_OBJ_ID] = *reinterpret_cast<__gm__ uint32_t *>(wqeAddr + 12);
    dbg[DBG_SQE_RMT_TOKEN_VALUE] = *reinterpret_cast<__gm__ uint32_t *>(wqeAddr + 32);
    dbg[DBG_SQE_RMT_ADDR] =
        (static_cast<uint64_t>(*reinterpret_cast<__gm__ uint32_t *>(wqeAddr + kUrmaSqeRmtAddrHOffset)) << 32) |
        *reinterpret_cast<__gm__ uint32_t *>(wqeAddr + kUrmaSqeRmtAddrLOffset);
    dbg[DBG_SQE_RMT_EID_L] = *reinterpret_cast<__gm__ uint64_t *>(wqeAddr + kUrmaSqeRmtEidLOffset);
    dbg[DBG_SQE_RMT_EID_H] = *reinterpret_cast<__gm__ uint64_t *>(wqeAddr + kUrmaSqeRmtEidHOffset);
    dbg[DBG_SGE_LEN] = *reinterpret_cast<__gm__ uint32_t *>(wqeAddr + kUrmaSqeSizeBytes);
    dbg[DBG_SGE_TOKEN_ID] = *reinterpret_cast<__gm__ uint32_t *>(wqeAddr + kUrmaSqeSizeBytes + 4);
    dbg[DBG_SGE_VA] = *reinterpret_cast<__gm__ uint64_t *>(wqeAddr + kUrmaSqeSizeBytes + 8);
    dbg[DBG_CQE_CTRL] = (static_cast<uint64_t>(cqe->status) << 56) | (static_cast<uint64_t>(cqe->substatus) << 48) |
                        (static_cast<uint64_t>(cqe->owner) << 40) | (static_cast<uint64_t>(cqe->opcode) << 32) |
                        cqe->entryIdx;
    dbg[DBG_CQE_BYTE_CNT] = cqe->byteCnt;
    dbg[DBG_CQE_TPN] = cqe->tpn;
    dbg[DBG_CQE_DATA] = (static_cast<uint64_t>(cqe->dataH) << 32) | cqe->dataL;
    dbg[DBG_SQ_HEAD] = sqHead;
    dbg[DBG_CQ_TAIL] = cqTail;
    dbg[DBG_EID_PTR] = remoteMem->eidAddr;
    __gm__ uint64_t *eidRaw = reinterpret_cast<__gm__ uint64_t *>(remoteMem->eidAddr);
    DcciCachelines(reinterpret_cast<__gm__ uint8_t *>(eidRaw), 16);
    dbg[DBG_EID_DATA_L] = eidRaw[0];
    dbg[DBG_EID_DATA_H] = eidRaw[1];
    dbg[DBG_WQN] = wq->wqn;
    dbg[DBG_DB_ADDR] = wq->dbAddr;
    uint64_t dbVal = (static_cast<uint64_t>(wq->wqn) & 0xFFFFFFUL) |
                     ((static_cast<uint64_t>(sqHead % 65536U)) << 32UL) |
                     ((static_cast<uint64_t>(wq->sl) & 0x7UL) << 48UL);
    dbg[DBG_DB_VALUE] = dbVal;

    pipe_barrier(PIPE_ALL);
    DcciCachelines(reinterpret_cast<__gm__ uint8_t *>(dbg), DBG_SLOT_COUNT * sizeof(uint64_t));
}
#endif

template <typename T, size_t count>
__global__ AICORE void TGetAsyncUrmaKernelImpl(__gm__ T *localBuf, int nranks, int my_rank, int first_rank_id,
                                               int root_rank, int elem_offset, int elem_count,
                                               __gm__ uint8_t *urmaWorkspace)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

    if (elem_count <= 0 || elem_offset < 0 || elem_offset + elem_count > static_cast<int>(count)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    ShapeDyn shape(1, 1, 1, 1, elem_count);
    StrideDyn stride(elem_count, elem_count, elem_count, elem_count, 1);

    constexpr size_t kDataOffset = 64 * sizeof(int32_t);

    __gm__ T *sendBuf = reinterpret_cast<__gm__ T *>(reinterpret_cast<__gm__ uint8_t *>(localBuf) + kDataOffset);
    __gm__ T *recvBuf = sendBuf + count;

    pipe_barrier(PIPE_ALL);

    if (my_rank == root_rank) {
#ifdef PTO_URMA_SUPPORTED
        const int my_peer = my_rank - first_rank_id;
        for (int target_peer = 0; target_peer < nranks; ++target_peer) {
            if (target_peer == my_peer) {
                continue;
            }
            uint64_t peerBase = pto::comm::urma::UrmaPeerMrBaseAddr(urmaWorkspace, static_cast<uint32_t>(target_peer));
            __gm__ T *remoteSendBuf = reinterpret_cast<__gm__ T *>(peerBase + kDataOffset) + elem_offset;
            __gm__ T *localRecvBuf = recvBuf + target_peer * count + elem_offset;
            Global remoteSendG(remoteSendBuf, shape, stride);
            Global localRecvG(localRecvBuf, shape, stride);

            pto::comm::AsyncSession session;
            pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(urmaWorkspace, static_cast<uint32_t>(target_peer),
                                                                     session);
            auto event = pto::comm::TGET_ASYNC<pto::comm::DmaEngine::URMA>(localRecvG, remoteSendG, session);
            if (!event.valid()) {
                trap();
            }
#ifdef COMM_DEBUG
            SnapshotUrmaDebug(urmaWorkspace, reinterpret_cast<__gm__ uint8_t *>(localBuf),
                              static_cast<uint32_t>(target_peer),
                              reinterpret_cast<uint64_t>(remoteSendBuf),
                              reinterpret_cast<uint64_t>(localRecvBuf));
#endif
            if (!event.Wait(session)) {
                trap();
            }
        }
#endif
    }

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Verify root-get results: each remote rank's data should match pattern i + rank * 10000.
// ============================================================================
template <typename T, size_t count>
bool VerifyRootGetResults(const uint8_t *output_host, int n_ranks, int first_rank_id, int root_rank, int rank_id,
                          int deviceId, int syncRet)
{
    const int root_peer = root_rank - first_rank_id;
    for (int src_peer = 0; src_peer < n_ranks; ++src_peer) {
        if (src_peer == root_peer)
            continue;
        const int src_logical = first_rank_id + src_peer;
        const size_t base = static_cast<size_t>(src_peer) * count;
        for (size_t i = 0; i < count; ++i) {
            T value = reinterpret_cast<const T *>(output_host)[base + i];
            T expected = static_cast<T>(i + src_logical * 10000);
            if (value != expected) {
                std::cerr << "Rank " << rank_id << " Device " << deviceId << " SyncRet " << syncRet
                          << " Expected: " << (float)expected << " Actual: " << (float)value << std::endl;
                return false;
            }
        }
    }
    return true;
}

#ifdef COMM_DEBUG
void DumpUrmaDebugSlots(const uint64_t *dbg)
{
    std::cerr << "[URMA][DBG] deviceSnapshot magic=0x" << std::hex << dbg[DBG_MAGIC] << " targetPeer=0x"
              << dbg[DBG_TARGET_PEER] << " localTokenId=0x" << dbg[DBG_LOCAL_TOKEN_ID] << " remoteTid=0x"
              << dbg[DBG_REMOTE_TID] << " remoteTpn=0x" << dbg[DBG_REMOTE_TPN] << " remoteMrAddr=0x"
              << dbg[DBG_REMOTE_ADDR] << " postRemoteAddr=0x" << dbg[DBG_POST_REMOTE_ADDR] << " postLocalAddr=0x"
              << dbg[DBG_POST_LOCAL_ADDR] << " wqeBufAddr=0x" << dbg[DBG_WQE_BUF_ADDR] << " wqeHeadAddr=0x"
              << dbg[DBG_WQE_HEAD_ADDR] << std::endl;
    std::cerr << "[URMA][DBG] deviceSnapshot sqeCtrl=0x" << dbg[DBG_SQE_CTRL] << " rmtObjId=0x"
              << dbg[DBG_SQE_RMT_OBJ_ID] << " rmtTokenValue=0x" << dbg[DBG_SQE_RMT_TOKEN_VALUE] << " rmtAddr=0x"
              << dbg[DBG_SQE_RMT_ADDR] << " rmtEidL=0x" << dbg[DBG_SQE_RMT_EID_L] << " rmtEidH=0x"
              << dbg[DBG_SQE_RMT_EID_H] << std::endl;
    std::cerr << "[URMA][DBG] deviceSnapshot sgeLen=0x" << dbg[DBG_SGE_LEN] << " sgeTokenId=0x" << dbg[DBG_SGE_TOKEN_ID]
              << " sgeVa=0x" << dbg[DBG_SGE_VA] << " cqeCtrl=0x" << dbg[DBG_CQE_CTRL] << " cqeByteCnt=0x"
              << dbg[DBG_CQE_BYTE_CNT] << " cqeTpn=0x" << dbg[DBG_CQE_TPN] << " cqeData=0x" << dbg[DBG_CQE_DATA]
              << " sqHead=0x" << dbg[DBG_SQ_HEAD] << " cqTail=0x" << dbg[DBG_CQ_TAIL] << std::dec << std::endl;
    std::cerr << "[URMA][DBG] deviceSnapshot eidPtr=0x" << std::hex << dbg[DBG_EID_PTR] << " eidDataL=0x"
              << dbg[DBG_EID_DATA_L] << " eidDataH=0x" << dbg[DBG_EID_DATA_H] << std::dec << std::endl;
    std::cerr << "[URMA][DBG] deviceSnapshot wqn=0x" << std::hex << dbg[DBG_WQN] << " dbAddr=0x" << dbg[DBG_DB_ADDR]
              << " dbValue=0x" << dbg[DBG_DB_VALUE] << std::dec << std::endl;
}
#endif

// ============================================================================
// Host-side runner.
// ============================================================================
template <typename T, size_t count>
bool RunGetAsyncUrmaRootGetKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, int first_rank_id,
                                  int root_rank)
{
    const size_t recv_elems = static_cast<size_t>(n_ranks) * count;
    size_t commBytesNeeded = 64 * sizeof(int32_t) + (static_cast<size_t>(n_ranks) + 1) * count * sizeof(T);

    UrmaTestContext ctx;
    if (!ctx.Setup(rank_id, n_ranks, n_devices, first_device_id, root_rank, commBytesNeeded)) {
        return false;
    }

    uint8_t *input_host = nullptr, *output_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void **>(&input_host), count * sizeof(T));
    aclrtMallocHost(reinterpret_cast<void **>(&output_host), recv_elems * sizeof(T));
    if (!input_host || !output_host) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        ctx.Cleanup();
        return false;
    }
    for (size_t i = 0; i < count; ++i)
        reinterpret_cast<T *>(input_host)[i] = static_cast<T>(i + rank_id * 10000);
    for (size_t i = 0; i < recv_elems; ++i)
        reinterpret_cast<T *>(output_host)[i] = static_cast<T>(-1);

    constexpr size_t kDataOffset = 64 * sizeof(int32_t);
    T *sendBuf = reinterpret_cast<T *>(reinterpret_cast<uint8_t *>(ctx.devBuf) + kDataOffset);
    T *recvBuf = sendBuf + count;
    aclrtMemcpy(sendBuf, count * sizeof(T), input_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(recvBuf, recv_elems * sizeof(T), output_host, recv_elems * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    CommMpiBarrier();

    TGetAsyncUrmaKernelImpl<T, count><<<1, nullptr, ctx.stream>>>(
        reinterpret_cast<T *>(ctx.devBuf), n_ranks, rank_id, first_rank_id, root_rank, 0, static_cast<int>(count),
        reinterpret_cast<uint8_t *>(ctx.urmaMgr.GetWorkspaceAddr()));
    int syncRet = aclrtSynchronizeStream(ctx.stream);
#ifdef COMM_DEBUG
    if (rank_id == root_rank) {
        uint64_t debugSlots[64]{};
        aclrtMemcpy(debugSlots, sizeof(debugSlots), ctx.devBuf, sizeof(debugSlots), ACL_MEMCPY_DEVICE_TO_HOST);
        DumpUrmaDebugSlots(debugSlots);
    }
#endif

    CommMpiBarrier();

    bool is_ok = true;
    if (rank_id == root_rank) {
        aclrtMemcpy(output_host, recv_elems * sizeof(T), recvBuf, recv_elems * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
        is_ok = VerifyRootGetResults<T, count>(output_host, n_ranks, first_rank_id, root_rank, rank_id, ctx.deviceId,
                                               syncRet);
    }

    aclrtFreeHost(input_host);
    aclrtFreeHost(output_host);
    ctx.Cleanup();

    return is_ok;
}

// ============================================================================
// MPI-based multi-rank launch.
// ============================================================================
template <typename T, size_t count>
bool RunGetAsyncUrmaRootGet(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunUrmaTestMpiLaunch(n_ranks, n_devices, first_rank_id, first_device_id,
                                RunGetAsyncUrmaRootGetKernel<T, count>);
}

// Explicit instantiations
template bool RunGetAsyncUrmaRootGet<float, 256>(int, int, int, int);
template bool RunGetAsyncUrmaRootGet<int32_t, 4096>(int, int, int, int);
template bool RunGetAsyncUrmaRootGet<uint8_t, 512>(int, int, int, int);
template bool RunGetAsyncUrmaRootGet<float, 524288>(int, int, int,
                                                    int);                    // MR = 8MB (>2MB)
template bool RunGetAsyncUrmaRootGet<int32_t, 67108864>(int, int, int, int); // MR ≈ 770MB (>512MB)
