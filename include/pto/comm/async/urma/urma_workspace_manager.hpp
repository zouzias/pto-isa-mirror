/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_WORKSPACE_MANAGER_HPP
#define PTO_COMM_ASYNC_URMA_WORKSPACE_MANAGER_HPP

#if defined(__CCE_KT_TEST__)
#error "urma_workspace_manager.hpp is a host-only header and cannot be included in device code."
#endif

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "securec.h"

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_rank_graph.h"

#include "pto/comm/async/urma/urma_types.hpp"
#include "pto/comm/async/urma/urma_hccl_defs.hpp"

namespace pto {
namespace comm {
namespace urma {

// ============================================================================
// UrmaWorkspaceManager: HCCL-based URMA workspace initialization.
//
// Uses HCCL public APIs for connection establishment (HcclCommMemReg,
// HcclRankGraphGetLinks, HcclChannelAcquire), then reads ChannelEntity from
// device/host and fills UrmaInfo for AIV-side urma_async_intrin.hpp.
// Flow:
//   1. HcclCommMemReg (register symmetric memory)
//   2. HcclRankGraphGetLinks (find UBC_TP/CTP links per peer)
//   3. HcclChannelAcquire(COMM_ENGINE_AIV) — returns device ChannelEntity* handles
//      (requires CANN with ConvertAivChannelHandlesToDevicePtrs in HcclChannelAcquire)
//   4. Read back ChannelEntity + sub-structures from device
//   5. Convert to UrmaInfo format and copy to device
// ============================================================================
class UrmaWorkspaceManager {
public:
    UrmaWorkspaceManager() = default;
    ~UrmaWorkspaceManager()
    {
        Finalize();
    }

    UrmaWorkspaceManager(const UrmaWorkspaceManager &) = delete;
    UrmaWorkspaceManager &operator=(const UrmaWorkspaceManager &) = delete;

    bool Init(HcclComm comm, uint32_t rankId, uint32_t rankCount, void *symmetricAddr, uint64_t symmetricSize)
    {
        comm_ = comm;
        rankId_ = rankId;
        rankCount_ = rankCount;
        symmetricAddr_ = symmetricAddr;
        symmetricSize_ = symmetricSize;

        if (!RegisterMemory()) {
            Finalize();
            return false;
        }
        if (!BuildChannels()) {
            Finalize();
            return false;
        }
        if (!ExtractAndFillUrmaInfo()) {
            Finalize();
            return false;
        }

        initialized_ = true;
        return true;
    }

    void Finalize()
    {
        FreeDeviceAddr(urmaInfoDevice_);
        FreeDeviceAddr(eidDevice_);
        channelHandles_.clear();
        initialized_ = false;
    }

    void *GetWorkspaceAddr() const
    {
        return urmaInfoDevice_;
    }

private:
    bool RegisterMemory()
    {
        CommMem mem{};
        mem.type = COMM_MEM_TYPE_DEVICE;
        mem.addr = symmetricAddr_;
        mem.size = symmetricSize_;

        HcclResult ret = HcclCommMemReg(comm_, kUrmaSymMemTag, &mem, &memHandle_);
        if (ret != HCCL_SUCCESS) {
            std::cerr << "[URMA] HcclCommMemReg failed: " << static_cast<int>(ret) << std::endl;
            return false;
        }
        return true;
    }

    bool BuildChannels()
    {
        std::vector<HcclChannelDesc> descs;
        descs.reserve(rankCount_ - 1);

        for (uint32_t peer = 0; peer < rankCount_; ++peer) {
            if (peer == rankId_) {
                continue;
            }

            uint32_t netLayer = 0;
            uint32_t linkNum = 0;
            CommLink *linkList = nullptr;
            HcclResult rc = HcclRankGraphGetLinks(comm_, netLayer, rankId_, peer, &linkList, &linkNum);
            if (rc != HCCL_SUCCESS) {
                std::cerr << "[URMA] HcclRankGraphGetLinks peer=" << peer << " ret=" << static_cast<int>(rc)
                          << std::endl;
                return false;
            }

            bool found = false;
            for (uint32_t i = 0; i < linkNum; ++i) {
                CommProtocol proto = linkList[i].linkAttr.linkProtocol;
                if (proto != kCommProtocolUbcCtp && proto != kCommProtocolUbcTp) {
                    continue;
                }

                HcclChannelDesc desc;
                HcclChannelDescInit(&desc, 1);
                desc.remoteRank = peer;
                desc.notifyNum = 0;
                desc.channelProtocol = proto;
                desc.localEndpoint = linkList[i].srcEndpointDesc;
                desc.remoteEndpoint = linkList[i].dstEndpointDesc;
                desc.memHandles = &memHandle_;
                desc.memHandleNum = 1;
                descs.push_back(desc);
                found = true;
                break;
            }
            if (!found) {
                std::cerr << "[URMA] rank=" << rankId_ << " no UBC_TP/CTP link to peer=" << peer << std::endl;
                return false;
            }
        }

        channelHandles_.resize(descs.size());
        HcclResult rc = HcclChannelAcquire(comm_, COMM_ENGINE_AIV, descs.data(), static_cast<uint32_t>(descs.size()),
                                           channelHandles_.data());
        if (rc != HCCL_SUCCESS) {
            std::cerr << "[URMA] HcclChannelAcquire failed: " << static_cast<int>(rc) << std::endl;
            return false;
        }
        return true;
    }

    bool ExtractAndFillUrmaInfo()
    {
        std::vector<UrmaWQCtx> wqList(rankCount_);
        std::vector<UrmaCqCtx> cqList(rankCount_);
        std::vector<UrmaMemInfo> memList(rankCount_);
        std::vector<uint8_t> eidTable(rankCount_ * 16, 0);
        uint32_t localTokenId = 0;

        if (!ExtractPerPeerInfo(wqList, cqList, memList, eidTable, localTokenId)) {
            return false;
        }
        if (!AllocAndCopyEidTable(eidTable, memList)) {
            return false;
        }
        if (!BuildAndCopyUrmaInfoTable(wqList, cqList, memList, localTokenId)) {
            return false;
        }

        std::cerr << "[URMA] UrmaInfo OK rank=" << rankId_ << " localTokenId=0x" << std::hex << localTokenId
                  << std::dec << std::endl;
        return true;
    }

    bool ExtractPerPeerInfo(std::vector<UrmaWQCtx> &wqList, std::vector<UrmaCqCtx> &cqList,
                            std::vector<UrmaMemInfo> &memList, std::vector<uint8_t> &eidTable,
                            uint32_t &localTokenId)
    {
        uint32_t channelIdx = 0;
        for (uint32_t peer = 0; peer < rankCount_; ++peer) {
            if (peer == rankId_) {
                wqList[peer] = UrmaWQCtx{};
                cqList[peer] = UrmaCqCtx{};
                memList[peer] = UrmaMemInfo{};
                memList[peer].addr = reinterpret_cast<uint64_t>(symmetricAddr_);
                memList[peer].len = static_cast<uint32_t>(symmetricSize_);
                continue;
            }
            if (!ExtractSinglePeer(peer, channelIdx, wqList, cqList, memList, eidTable, localTokenId)) {
                return false;
            }
            ++channelIdx;
        }
        return true;
    }

    bool ExtractSinglePeer(uint32_t peer, uint32_t channelIdx, std::vector<UrmaWQCtx> &wqList,
                           std::vector<UrmaCqCtx> &cqList, std::vector<UrmaMemInfo> &memList,
                           std::vector<uint8_t> &eidTable, uint32_t &localTokenId)
    {
        ChannelHandle handle = channelHandles_[channelIdx];
        if (handle != 0 && static_cast<uint64_t>(handle) < kDeviceVaThreshold) {
            std::cerr << "[URMA] ChannelHandle looks like host pointer (0x" << std::hex << handle << std::dec
                      << ") for peer=" << peer
                      << ". Upgrade CANN: HcclChannelAcquire must return device ChannelEntity pointers for AIV URMA."
                      << std::endl;
            return false;
        }

        ChannelEntity hostEntity{};
        SqContext sq{};
        CqContext cq{};
        RegedBufferEntity remoteBuf{};
        RegedBufferEntity localBuf{};

        if (handle == 0 || !TryReadChannelEntity(handle, peer, hostEntity, sq, cq, remoteBuf, localBuf)) {
            std::cerr << "[URMA] Cannot read ChannelEntity for peer=" << peer << " handle=0x" << std::hex
                      << static_cast<uint64_t>(handle) << std::dec << std::endl;
            return false;
        }

        RegedBufferEntity symRemoteBuf{};
        uint64_t symRmaAddr = 0;
        uint32_t symRmaSize = 0;
        if (!SelectSymmetricRemoteBuffer(handle, peer, hostEntity, symRemoteBuf, symRmaAddr, symRmaSize)) {
            return false;
        }
        RegedBufferEntity symLocalBuf{};
        if (SelectSymmetricLocalBuffer(hostEntity, peer, symLocalBuf) && symLocalBuf.type == REGED_BUFFER_RMA) {
            localTokenId = symLocalBuf.bufferInfo.rma.protectionInfo.memInfo.ub.tokenId;
        }

        FillWqCtx(wqList[peer], sq);
        FillCqCtx(cqList[peer], cq);
        FillMemInfo(memList[peer], sq, symRemoteBuf, symRmaAddr, symRmaSize);

        (void)memcpy_s(&eidTable[peer * 16], 16, sq.contextInfo.ubJfs.remoteEID, 16);

        std::cerr << "[URMA] peer=" << peer << " tpId=" << memList[peer].tpn << " rmtAddr=0x" << std::hex
                  << memList[peer].addr << " sqVa=0x" << wqList[peer].bufAddr << " dbAddr=0x"
                  << wqList[peer].dbAddr << std::dec << std::endl;
        return true;
    }

    static void FillWqCtx(UrmaWQCtx &wq, const SqContext &sq)
    {
        wq.wqn = sq.contextInfo.ubJfs.jfsID;
        wq.bufAddr = sq.contextInfo.ubJfs.sqVa;
        wq.wqeShiftSize = Log2U32(sq.contextInfo.ubJfs.wqeSize);
        wq.depth = sq.contextInfo.ubJfs.sqDepth;
        wq.headAddr = sq.contextInfo.ubJfs.headAddr;
        wq.tailAddr = sq.contextInfo.ubJfs.tailAddr;
        wq.dbMode = UrmaDbMode::SW_DB;
        wq.dbAddr = sq.contextInfo.ubJfs.dbVa;
        wq.sl = 0;
    }

    static void FillCqCtx(UrmaCqCtx &cqCtx, const CqContext &cq)
    {
        cqCtx.cqn = cq.contextInfo.ubJfc.jfcID;
        cqCtx.bufAddr = cq.contextInfo.ubJfc.scqVa;
        cqCtx.cqeShiftSize = Log2U32(cq.contextInfo.ubJfc.cqeSize);
        cqCtx.depth = cq.contextInfo.ubJfc.cqDepth;
        cqCtx.headAddr = cq.contextInfo.ubJfc.headAddr;
        cqCtx.tailAddr = cq.contextInfo.ubJfc.tailAddr;
        cqCtx.dbMode = UrmaDbMode::SW_DB;
        cqCtx.dbAddr = cq.contextInfo.ubJfc.dbVa;
    }

    static void FillMemInfo(UrmaMemInfo &mem, const SqContext &sq, const RegedBufferEntity &symRemoteBuf,
                            uint64_t symRmaAddr, uint32_t symRmaSize)
    {
        mem.tokenValueValid = true;
        mem.rmtJettyType = 1;
        mem.targetHint = 0;
        mem.tpn = sq.contextInfo.ubJfs.tpID;
        mem.tid = symRemoteBuf.bufferInfo.rma.protectionInfo.memInfo.ub.tokenId >> kUrmaTokenIdTidShift;
        mem.rmtTokenValue = symRemoteBuf.bufferInfo.rma.protectionInfo.memInfo.ub.tokenValue;
        mem.len = symRmaSize;
        mem.addr = symRmaAddr;
    }

    bool AllocAndCopyEidTable(const std::vector<uint8_t> &eidTable, std::vector<UrmaMemInfo> &memList)
    {
        size_t eidDevSize = rankCount_ * 16;
        aclError err = aclrtMalloc(&eidDevice_, eidDevSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMalloc(eidTable) failed: " << err << std::endl;
            return false;
        }
        err = aclrtMemcpy(eidDevice_, eidDevSize, eidTable.data(), eidDevSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMemcpy(eidTable) failed: " << err << std::endl;
            return false;
        }
        for (uint32_t peer = 0; peer < rankCount_; ++peer) {
            memList[peer].eidAddr = reinterpret_cast<uint64_t>(static_cast<uint8_t *>(eidDevice_) + peer * 16);
        }
        return true;
    }

    bool BuildAndCopyUrmaInfoTable(const std::vector<UrmaWQCtx> &wqList, const std::vector<UrmaCqCtx> &cqList,
                                   const std::vector<UrmaMemInfo> &memList, uint32_t localTokenId)
    {
        constexpr uint32_t qpNum = 1;
        size_t totalSize =
            sizeof(UrmaInfo) + rankCount_ * (2U * sizeof(UrmaWQCtx) * qpNum + 2U * sizeof(UrmaCqCtx) * qpNum +
                                             sizeof(UrmaMemInfo) * qpNum);

        aclError err = aclrtMalloc(&urmaInfoDevice_, totalSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMalloc(urmaInfo) failed: " << err << std::endl;
            return false;
        }

        std::vector<uint8_t> hostBuf(totalSize, 0);
        FillUrmaInfoLayout(hostBuf, wqList, cqList, memList, localTokenId);

        err = aclrtMemcpy(urmaInfoDevice_, totalSize, hostBuf.data(), totalSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMemcpy(urmaInfo) failed: " << err << std::endl;
            return false;
        }
        return true;
    }

    void FillUrmaInfoLayout(std::vector<uint8_t> &hostBuf, const std::vector<UrmaWQCtx> &wqList,
                            const std::vector<UrmaCqCtx> &cqList, const std::vector<UrmaMemInfo> &memList,
                            uint32_t localTokenId)
    {
        constexpr uint32_t qpNum = 1;
        auto *info = reinterpret_cast<UrmaInfo *>(hostBuf.data());
        info->qpNum = qpNum;
        info->localTokenId = localTokenId;
        info->rankCount = rankCount_;

        uint8_t *devAddr = static_cast<uint8_t *>(urmaInfoDevice_) + sizeof(UrmaInfo);
        info->sqPtr = reinterpret_cast<uint64_t>(devAddr);
        devAddr += sizeof(UrmaWQCtx) * rankCount_ * qpNum;
        info->rqPtr = reinterpret_cast<uint64_t>(devAddr);
        devAddr += sizeof(UrmaWQCtx) * rankCount_ * qpNum;
        info->scqPtr = reinterpret_cast<uint64_t>(devAddr);
        devAddr += sizeof(UrmaCqCtx) * rankCount_ * qpNum;
        info->rcqPtr = reinterpret_cast<uint64_t>(devAddr);
        devAddr += sizeof(UrmaCqCtx) * rankCount_ * qpNum;
        info->memPtr = reinterpret_cast<uint64_t>(devAddr);

        uint8_t *hostAddr = hostBuf.data() + sizeof(UrmaInfo);
        auto *sqArr = reinterpret_cast<UrmaWQCtx *>(hostAddr);
        hostAddr += sizeof(UrmaWQCtx) * rankCount_ * qpNum;
        auto *rqArr = reinterpret_cast<UrmaWQCtx *>(hostAddr);
        hostAddr += sizeof(UrmaWQCtx) * rankCount_ * qpNum;
        auto *scqArr = reinterpret_cast<UrmaCqCtx *>(hostAddr);
        hostAddr += sizeof(UrmaCqCtx) * rankCount_ * qpNum;
        auto *rcqArr = reinterpret_cast<UrmaCqCtx *>(hostAddr);
        hostAddr += sizeof(UrmaCqCtx) * rankCount_ * qpNum;
        auto *memArr = reinterpret_cast<UrmaMemInfo *>(hostAddr);

        for (uint32_t rank = 0; rank < rankCount_; ++rank) {
            sqArr[rank] = wqList[rank];
            rqArr[rank] = wqList[rank];
            scqArr[rank] = cqList[rank];
            rcqArr[rank] = cqList[rank];
            memArr[rank] = memList[rank];
        }
    }

    static bool IsLikelyDevicePtr(const void *ptr)
    {
        return reinterpret_cast<uintptr_t>(ptr) >= kDeviceVaThreshold;
    }

    static bool IsValidChannelEntityHeader(const ChannelEntity &entity)
    {
        const uint32_t magic = entity.abiHeader.magicWord;
        if (magic != kHcclChannelEntityMagic && magic != kHcommChannelEntityMagic) {
            return false;
        }
        return entity.engine == COMM_ENGINE_AIV;
    }

    bool CopyChannelSubStruct(const void *srcPtr, void *dst, size_t size, uint32_t peer, const char *name) const
    {
        if (srcPtr == nullptr) {
            return false;
        }
        if (IsLikelyDevicePtr(srcPtr)) {
            aclError err = aclrtMemcpy(dst, size, srcPtr, size, ACL_MEMCPY_DEVICE_TO_HOST);
            if (err != ACL_SUCCESS) {
                std::cerr << "[URMA] aclrtMemcpy(" << name << ") peer=" << peer << " err=" << err << std::endl;
                return false;
            }
            return true;
        }
        errno_t rc = memcpy_s(dst, size, srcPtr, size);
        return rc == EOK;
    }

    bool FillChannelSubStructs(uint32_t peer, const ChannelEntity &hostEntity, SqContext &sq, CqContext &cq,
                               RegedBufferEntity &remoteBuf, RegedBufferEntity &localBuf) const
    {
        if (hostEntity.sqContextAddr != nullptr && hostEntity.sqNum > 0) {
            if (!CopyChannelSubStruct(hostEntity.sqContextAddr, &sq, sizeof(SqContext), peer, "SqContext")) {
                return false;
            }
        }

        if (hostEntity.cqContextAddr != nullptr && hostEntity.cqNum > 0) {
            if (!CopyChannelSubStruct(hostEntity.cqContextAddr, &cq, sizeof(CqContext), peer, "CqContext")) {
                return false;
            }
        }

        if (hostEntity.remoteBufferAddr != nullptr && hostEntity.remoteBufferNum > 0) {
            if (!CopyChannelSubStruct(hostEntity.remoteBufferAddr, &remoteBuf, sizeof(RegedBufferEntity), peer,
                                      "RemoteBuffer")) {
                return false;
            }
        }

        if (hostEntity.localBufferAddr != nullptr && hostEntity.localBufferNum > 0) {
            if (!CopyChannelSubStruct(hostEntity.localBufferAddr, &localBuf, sizeof(RegedBufferEntity), peer,
                                      "LocalBuffer")) {
                return false;
            }
        }
        return true;
    }

    bool TryReadChannelEntity(ChannelHandle handle, uint32_t peer, ChannelEntity &hostEntity, SqContext &sq,
                              CqContext &cq, RegedBufferEntity &remoteBuf, RegedBufferEntity &localBuf)
    {
        void *devEntityPtr = reinterpret_cast<void *>(static_cast<uintptr_t>(handle));
        aclError err = aclrtMemcpy(&hostEntity, sizeof(ChannelEntity), devEntityPtr, sizeof(ChannelEntity),
                                   ACL_MEMCPY_DEVICE_TO_HOST);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMemcpy(ChannelEntity) peer=" << peer << " err=" << err << std::endl;
            return false;
        }

        if (!IsValidChannelEntityHeader(hostEntity)) {
            std::cerr << "[URMA] invalid ChannelEntity header peer=" << peer << " magic=0x" << std::hex
                      << hostEntity.abiHeader.magicWord << " engine=" << std::dec << static_cast<int>(hostEntity.engine)
                      << std::endl;
            return false;
        }

        return FillChannelSubStructs(peer, hostEntity, sq, cq, remoteBuf, localBuf);
    }

    bool GetRemoteMemByTag(ChannelHandle handle, uint32_t peer, void **outAddr, uint64_t *outSize) const
    {
        uint32_t memNum = 0;
        CommMem *remoteMems = nullptr;
        char **memTags = nullptr;
        HcclResult rc = HcclChannelGetRemoteMems(comm_, handle, &memNum, &remoteMems, &memTags);
        if (rc != HCCL_SUCCESS) {
            std::cerr << "[URMA] HcclChannelGetRemoteMems peer=" << peer << " ret=" << static_cast<int>(rc)
                      << std::endl;
            return false;
        }
        for (uint32_t i = 0; i < memNum; ++i) {
            const char *tag = memTags[i] ? memTags[i] : "";
            if (strcmp(tag, kUrmaSymMemTag) == 0) {
                *outAddr = remoteMems[i].addr;
                *outSize = remoteMems[i].size;
                return true;
            }
        }
        std::cerr << "[URMA] peer=" << peer << " tag " << kUrmaSymMemTag << " not found" << std::endl;
        return false;
    }

    bool ReadRegedBufferEntityAt(RegedBufferEntity *array, uint32_t count, uint32_t index, uint32_t peer,
                                 RegedBufferEntity &out) const
    {
        if (array == nullptr || index >= count) {
            return false;
        }
        const void *ptr = reinterpret_cast<const void *>(reinterpret_cast<uintptr_t>(array) +
                                                         static_cast<uintptr_t>(index) * sizeof(RegedBufferEntity));
        return CopyChannelSubStruct(ptr, &out, sizeof(RegedBufferEntity), peer, "RegedBufferEntity");
    }

    bool SelectSymmetricRemoteBuffer(ChannelHandle handle, uint32_t peer, const ChannelEntity &entity,
                                     RegedBufferEntity &selected, uint64_t &rmaAddr, uint32_t &rmaSize) const
    {
        void *symAddr = nullptr;
        uint64_t symSize = 0;
        if (!GetRemoteMemByTag(handle, peer, &symAddr, &symSize)) {
            return false;
        }
        rmaAddr = reinterpret_cast<uint64_t>(symAddr);
        rmaSize = static_cast<uint32_t>(symSize);

        if (entity.remoteBufferAddr == nullptr || entity.remoteBufferNum == 0) {
            return false;
        }

        bool found = false;
        for (uint32_t i = 0; i < entity.remoteBufferNum; ++i) {
            RegedBufferEntity buf{};
            if (!ReadRegedBufferEntityAt(entity.remoteBufferAddr, entity.remoteBufferNum, i, peer, buf)) {
                continue;
            }
            if (buf.type != REGED_BUFFER_RMA) {
                continue;
            }
            if (buf.bufferInfo.rma.addr != rmaAddr || buf.bufferInfo.rma.size != symmetricSize_) {
                continue;
            }
            selected = buf;
            found = true;
            break;
        }
        if (!found) {
            std::cerr << "[URMA] peer=" << peer << " no RegedBufferEntity matches " << kUrmaSymMemTag << std::endl;
            return false;
        }
        return true;
    }

    bool SelectSymmetricLocalBuffer(const ChannelEntity &entity, uint32_t peer, RegedBufferEntity &selected) const
    {
        if (entity.localBufferAddr == nullptr || entity.localBufferNum == 0) {
            return false;
        }
        for (uint32_t i = 0; i < entity.localBufferNum; ++i) {
            RegedBufferEntity buf{};
            if (!ReadRegedBufferEntityAt(entity.localBufferAddr, entity.localBufferNum, i, peer, buf)) {
                continue;
            }
            if (buf.type == REGED_BUFFER_RMA && buf.bufferInfo.rma.size == symmetricSize_) {
                selected = buf;
                return true;
            }
        }
        for (uint32_t i = 0; i < entity.localBufferNum; ++i) {
            RegedBufferEntity buf{};
            if (!ReadRegedBufferEntityAt(entity.localBufferAddr, entity.localBufferNum, i, peer, buf)) {
                continue;
            }
            if (buf.type == REGED_BUFFER_RMA) {
                selected = buf;
                return true;
            }
        }
        return false;
    }

    static uint32_t Log2U32(uint32_t n)
    {
        return (n <= 1) ? 0 : __builtin_ctz(n);
    }

    static void FreeDeviceAddr(void *&addr)
    {
        if (addr) {
            aclrtFree(addr);
            addr = nullptr;
        }
    }

    static constexpr const char *kUrmaSymMemTag = "pto_urma_sym";
    // HcclChannelDescInit uses HCCL_CHANNEL_MAGIC_WORD; AivUrmaChannel copies desc.header into entity.abiHeader.
    static constexpr uint32_t kHcclChannelEntityMagic = 0x0f0f0f0fU;
    // HcommChannelDescInit uses HCOMM_CHANNEL_MAGIC_WORD (reserved for forward compatibility).
    static constexpr uint32_t kHcommChannelEntityMagic = 0x0fcf0f0fU;
    static constexpr uint64_t kDeviceVaThreshold = 0x100000000000ULL;
    static constexpr CommProtocol kCommProtocolUbcCtp = static_cast<CommProtocol>(4);
    static constexpr CommProtocol kCommProtocolUbcTp = static_cast<CommProtocol>(5);

    HcclComm comm_{nullptr};
    uint32_t rankId_{0};
    uint32_t rankCount_{0};
    void *symmetricAddr_{nullptr};
    uint64_t symmetricSize_{0};
    HcclMemHandle memHandle_{nullptr};

    std::vector<ChannelHandle> channelHandles_;

    void *urmaInfoDevice_{nullptr};
    void *eidDevice_{nullptr};

    bool initialized_{false};
};

} // namespace urma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_URMA_WORKSPACE_MANAGER_HPP
