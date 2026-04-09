/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_COMM_ASYNC_URMA_WORKSPACE_MANAGER_HPP
#define PTO_NPU_COMM_ASYNC_URMA_WORKSPACE_MANAGER_HPP

#if defined(__CCE_KT_TEST__)
#error "urma_workspace_manager.hpp is a host-only header and cannot be included in device code."
#endif

#include <cstdint>
#include <cstddef>
#include <iostream>
#include <vector>

#include "acl/acl.h"

#include "pto/npu/comm/async/urma/urma_types.hpp"
#include "pto/npu/comm/async/urma/urma_hccp_loader.hpp"

namespace pto {
namespace comm {
namespace urma {

// ============================================================================
// UrmaBootstrapHandle: generic cross-rank information exchange abstraction.
// ============================================================================
struct UrmaBootstrapHandle {
    int (*allgather)(const void *sendbuf, void *recvbuf, int size, void *ctx);
    int (*barrier)(void *ctx);
    void *ctx;
};

// ============================================================================
// UrmaWorkspaceManager: Host-side URMA workspace initialization.
//
// 10-step initialization:
//   1. TsdProcessOpen
//   2. RaInit
//   3. RaCtxInit (+ RaGetDevEidInfo + RaCtxTokenIdAlloc)
//   4. RaCtxLmemRegister (register symmetric memory as MR)
//   5. JFCCreate (Channel + CQ with device-side polling)
//   6. JettyCreate (QP with SQ/RQ)
//   7. JettyImport (allgather QpKey, RaCtxQpImport per remote rank)
//   8. JettyBind (skip in RM mode)
//   9. RmemImport (allgather MR, RaCtxRmemImport per remote rank)
//  10. FillUrmaInfo (construct device layout + aclrtMemcpy)
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

    bool Init(uint32_t deviceId, uint32_t rankId, uint32_t rankCount, void *symmetricAddr, uint64_t symmetricSize,
              const UrmaBootstrapHandle &bootstrap)
    {
        deviceId_ = deviceId;
        rankId_ = rankId;
        rankCount_ = rankCount;
        symmetricAddr_ = symmetricAddr;
        symmetricSize_ = symmetricSize;
        bootstrap_ = bootstrap;

        wqInfoList_.resize(rankCount_);
        cqInfoList_.resize(rankCount_);
        ubMemInfoList_.resize(rankCount_);
        hccpEidList_.resize(rankCount_);
        tpnList_.resize(rankCount_, 0);
        qpKeyList_.resize(rankCount_);
        allQpImportInfoT_.resize(rankCount_);
        remoteQpHandles_.resize(rankCount_, nullptr);
        rmemHandles_.resize(rankCount_, nullptr);

        auto &loader = HccpV2Loader::Instance();
        if (!loader.Load()) {
            std::cerr << "[URMA] Failed to load HCCP V2 libraries" << std::endl;
            return false;
        }

        if (!OpenTsd())
            return false;
        if (!RaInit())
            return false;
        if (!RaCtxInit())
            return false;
        if (!RegisterMR())
            return false;
        if (!JFCCreate())
            return false;
        if (!JettyCreate())
            return false;
        if (!JettyImport())
            return false;
        if (!JettyBind())
            return false;
        if (!RmemImport())
            return false;
        if (!FillUrmaInfo())
            return false;

        initialized_ = true;
        return true;
    }

    void Finalize()
    {
        FreeDeviceAddrs();
        DestroyHccpHandles();
        initialized_ = false;
    }

    void *GetWorkspaceAddr() const
    {
        return urmaInfoDevice_;
    }

private:
    static uint32_t Log2U32(uint32_t n)
    {
        return (n <= 1) ? 0 : __builtin_ctz(n);
    }

    // Byte-swap and cross-swap the two 64-bit halves of an EID (network → host byte order).
    // Uses __builtin_memcpy to avoid strict-aliasing UB when type-punning uint8_t[] ↔ uint64_t.
    static void SwapEidByteOrder(hccp::HccpEid &eid)
    {
        static_assert(sizeof(hccp::HccpEid) == 16, "EID must be 16 bytes");
        uint64_t lo, hi;
        __builtin_memcpy(&lo, eid.raw, sizeof(uint64_t));
        __builtin_memcpy(&hi, eid.raw + sizeof(uint64_t), sizeof(uint64_t));
        lo = __builtin_bswap64(lo);
        hi = __builtin_bswap64(hi);
        __builtin_memcpy(eid.raw, &hi, sizeof(uint64_t));
        __builtin_memcpy(eid.raw + sizeof(uint64_t), &lo, sizeof(uint64_t));
    }

    // Compute UrmaInfo pointer layout from a base address.
    // Sets sqPtr/rqPtr/scqPtr/rcqPtr/memPtr in copyInfo relative to baseAddr.
    void ComputeInfoLayout(UrmaInfo *copyInfo, uint8_t *baseAddr)
    {
        constexpr uint32_t qpNum = 1;
        uint8_t *cursor = baseAddr + sizeof(UrmaInfo);
        copyInfo->sqPtr = reinterpret_cast<uint64_t>(cursor);
        cursor += sizeof(UrmaWQCtx) * rankCount_ * qpNum;
        copyInfo->rqPtr = reinterpret_cast<uint64_t>(cursor);
        cursor += sizeof(UrmaWQCtx) * rankCount_ * qpNum;
        copyInfo->scqPtr = reinterpret_cast<uint64_t>(cursor);
        cursor += sizeof(UrmaCqCtx) * rankCount_ * qpNum;
        copyInfo->rcqPtr = reinterpret_cast<uint64_t>(cursor);
        cursor += sizeof(UrmaCqCtx) * rankCount_ * qpNum;
        copyInfo->memPtr = reinterpret_cast<uint64_t>(cursor);
    }

    void FreeDeviceAddrs()
    {
        if (urmaInfoDevice_) {
            aclrtFree(urmaInfoDevice_);
            urmaInfoDevice_ = nullptr;
        }
        if (hccpEidDevice_) {
            aclrtFree(hccpEidDevice_);
            hccpEidDevice_ = nullptr;
        }
        if (cqPiAddr_) {
            aclrtFree(cqPiAddr_);
            cqPiAddr_ = nullptr;
        }
        if (cqCiAddr_) {
            aclrtFree(cqCiAddr_);
            cqCiAddr_ = nullptr;
        }
        if (sqPiAddr_) {
            aclrtFree(sqPiAddr_);
            sqPiAddr_ = nullptr;
        }
        if (sqCiAddr_) {
            aclrtFree(sqCiAddr_);
            sqCiAddr_ = nullptr;
        }
    }

    void DestroyHccpHandles()
    {
        auto &api = HccpV2Loader::Instance();

        if (transportMode_ != hccp::CONN_RM && qpHandle_ && api.raCtxQpUnbind) {
            api.raCtxQpUnbind(qpHandle_);
        }
        for (uint32_t i = 0; i < rankCount_; ++i) {
            if (i != rankId_ && remoteQpHandles_[i] && api.raCtxQpUnimport) {
                api.raCtxQpUnimport(ctxHandle_, remoteQpHandles_[i]);
                remoteQpHandles_[i] = nullptr;
            }
        }
        for (uint32_t i = 0; i < rankCount_; ++i) {
            if (i != rankId_ && rmemHandles_[i] && api.raCtxRmemUnimport) {
                api.raCtxRmemUnimport(ctxHandle_, rmemHandles_[i]);
                rmemHandles_[i] = nullptr;
            }
        }

        if (api.raCtxQpDestroy && qpHandle_) {
            api.raCtxQpDestroy(qpHandle_);
            qpHandle_ = nullptr;
        }
        if (api.raCtxCqDestroy && cqHandle_ && ctxHandle_) {
            api.raCtxCqDestroy(ctxHandle_, cqHandle_);
            cqHandle_ = nullptr;
        }
        if (api.raCtxChanDestroy && chanHandle_ && ctxHandle_) {
            api.raCtxChanDestroy(ctxHandle_, chanHandle_);
            chanHandle_ = nullptr;
        }
        if (api.raCtxLmemUnregister && lmemHandle_ && ctxHandle_) {
            api.raCtxLmemUnregister(ctxHandle_, lmemHandle_);
            lmemHandle_ = nullptr;
        }
        if (api.raCtxTokenIdFree && tokenIdHandle_ && ctxHandle_) {
            api.raCtxTokenIdFree(ctxHandle_, tokenIdHandle_);
            tokenIdHandle_ = nullptr;
        }
        if (api.raCtxDeinit && ctxHandle_) {
            api.raCtxDeinit(ctxHandle_);
            ctxHandle_ = nullptr;
        }
    }

    // Step 1: Open TSD (process-level singleton, guarded by static flag)
    bool OpenTsd()
    {
        if (tsdOpened_)
            return true;
        auto &api = HccpV2Loader::Instance();
        hccp::ProcOpenArgs args{};
        args.procType = hccp::TSD_SUB_PROC_HCCP;
        char paramStr[] = "--hdcType=18";
        hccp::ProcExtParam extParam{paramStr, sizeof("--hdcType=18")};
        args.extParamList = &extParam;
        args.extParamCnt = 1;
        int subPid = 0;
        args.subPid = &subPid;

        int ret = api.tsdProcessOpen(deviceId_, &args);
        if (ret != 0) {
            std::cerr << "[URMA] TsdProcessOpen failed: " << ret << std::endl;
            return false;
        }
        tsdOpened_ = true;
        return true;
    }

    // Step 2: Initialize RA (process-level singleton, guarded by static flag)
    bool RaInit()
    {
        if (raInitialized_)
            return true;
        auto &api = HccpV2Loader::Instance();
        hccp::RaInitConfig config{};
        config.phyId = deviceId_;
        config.nicPosition = hccp::NETWORK_OFFLINE;
        config.hdcType = hccp::HDC_SERVICE_TYPE_RDMA_V2;
        config.enableHdcAsync = true;

        int ret = api.raInit(&config);
        if (ret != 0) {
            std::cerr << "[URMA] RaInit failed: " << ret << std::endl;
            return false;
        }
        raInitialized_ = true;
        return true;
    }

    // Step 3: Create RA context (+ get EID + allocate token)
    bool RaCtxInit()
    {
        auto &api = HccpV2Loader::Instance();
        hccp::RaInfo info{hccp::NETWORK_OFFLINE, deviceId_};

        unsigned int eidNum = 0;
        int ret = api.raGetDevEidInfoNum(info, &eidNum);
        if (ret != 0 || eidNum == 0) {
            std::cerr << "[URMA] RaGetDevEidInfoNum failed: ret=" << ret << " eidNum=" << eidNum << std::endl;
            return false;
        }

        std::vector<hccp::DevEidInfo> eidInfoList(eidNum);
        unsigned int infoListNum = eidNum;
        ret = api.raGetDevEidInfoList(info, eidInfoList.data(), &infoListNum);
        if (ret != 0 || infoListNum != eidNum) {
            std::cerr << "[URMA] RaGetDevEidInfoList failed: ret=" << ret << std::endl;
            return false;
        }

        hccp::CtxInitAttr attr{};
        attr.phyId = deviceId_;
        attr.ub.eid = eidInfoList[0].eid;
        attr.ub.eidIndex = eidInfoList[0].eidIndex;

        hccp::CtxInitCfg cfg{};
        cfg.mode = hccp::NETWORK_OFFLINE;

        ret = api.raCtxInit(&cfg, &attr, &ctxHandle_);
        if (ret != 0) {
            std::cerr << "[URMA] RaCtxInit failed: " << ret << std::endl;
            return false;
        }

        localHccpEid_ = attr.ub.eid;
        SwapEidByteOrder(localHccpEid_);

        hccp::HccpTokenId tokenId{0};
        ret = api.raCtxTokenIdAlloc(ctxHandle_, &tokenId, &tokenIdHandle_);
        if (ret != 0) {
            std::cerr << "[URMA] RaCtxTokenIdAlloc failed: " << ret << std::endl;
            return false;
        }
        return true;
    }

    // Step 4: Register symmetric memory as MR
    bool RegisterMR()
    {
        auto &api = HccpV2Loader::Instance();
        hccp::MrRegInfoT mrInfo{};
        mrInfo.in.mem.addr = reinterpret_cast<uint64_t>(symmetricAddr_);
        mrInfo.in.mem.size = symmetricSize_;
        mrInfo.in.ub.tokenValue = hccp::kTokenValue;
        mrInfo.in.ub.tokenIdHandle = tokenIdHandle_;
        mrInfo.in.ub.flags.bs.access = hccp::MEM_SEG_ACCESS_DEFAULT;
        mrInfo.in.ub.flags.bs.cacheable = 0;
        mrInfo.in.ub.flags.bs.tokenIdValid = 1;
        mrInfo.in.ub.flags.bs.nonPin = 0;
        mrInfo.in.ub.flags.bs.userIova = 0;
        mrInfo.in.ub.flags.bs.tokenPolicy = hccp::TOKEN_POLICY_PLAIN_TEXT;

        int ret = api.raCtxLmemRegister(ctxHandle_, &mrInfo, &lmemHandle_);
        if (ret != 0) {
            std::cerr << "[URMA] RaCtxLmemRegister failed: " << ret << std::endl;
            return false;
        }

        localMR_.address = mrInfo.in.mem.addr;
        localMR_.size = mrInfo.in.mem.size;
        localMR_.lmemHandle = lmemHandle_;
        localMR_.key = mrInfo.out.key;
        localMR_.tokenId = mrInfo.out.ub.tokenId;
        localMR_.tokenValue = hccp::kTokenValue;
        localMR_.targetSegHandle = mrInfo.out.ub.targetSegHandle;
        localMR_.tokenIdHandle = tokenIdHandle_;
        localMR_.cacheable = 0;
        localMR_.access = hccp::MEM_SEG_ACCESS_DEFAULT;

        localMemInfo_.tokenValueValid = true;
        localMemInfo_.rmtJettyType = 1;
        localMemInfo_.targetHint = 0;
        localMemInfo_.tpn = 0;
        localMemInfo_.tid = mrInfo.out.ub.tokenId >> 8;
        localMemInfo_.rmtTokenValue = hccp::kTokenValue;
        localMemInfo_.len = static_cast<uint32_t>(symmetricSize_);
        localMemInfo_.addr = reinterpret_cast<uint64_t>(symmetricAddr_);

        return true;
    }

    // Step 5: Create JFC (Channel + CQ)
    bool JFCCreate()
    {
        auto &api = HccpV2Loader::Instance();

        hccp::ChanInfoT chanInfo{};
        chanInfo.in.dataPlaneFlag.bs.poolCqCstm = 1;
        int ret = api.raCtxChanCreate(ctxHandle_, &chanInfo, &chanHandle_);
        if (ret != 0) {
            std::cerr << "[URMA] RaCtxChanCreate failed: " << ret << std::endl;
            return false;
        }

        cqInfo_.in.chanHandle = chanHandle_;
        cqInfo_.in.depth = hccp::kCqDepthDefault;
        cqInfo_.in.ub.userCtx = 0;
        cqInfo_.in.ub.mode = hccp::JFC_MODE_USER_CTL_NORMAL;
        cqInfo_.in.ub.ceqn = 0;
        cqInfo_.in.ub.flag.bs.lockFree = 0;
        cqInfo_.in.ub.flag.bs.jfcInline = 0;

        ret = api.raCtxCqCreate(ctxHandle_, &cqInfo_, &cqHandle_);
        if (ret != 0) {
            std::cerr << "[URMA] RaCtxCqCreate failed: " << ret << std::endl;
            return false;
        }

        UrmaCqCtx localCq{};
        localCq.cqn = 0;
        localCq.bufAddr = cqInfo_.out.bufAddr;
        localCq.cqeShiftSize = Log2U32(cqInfo_.out.cqeSize);
        localCq.depth = cqInfo_.in.depth;

        if (aclrtMalloc(&cqPiAddr_, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS || !cqPiAddr_) {
            std::cerr << "[URMA] aclrtMalloc for cqPiAddr failed" << std::endl;
            return false;
        }
        aclrtMemset(cqPiAddr_, sizeof(uint32_t), 0, sizeof(uint32_t));
        localCq.headAddr = reinterpret_cast<uintptr_t>(cqPiAddr_);
        if (aclrtMalloc(&cqCiAddr_, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS || !cqCiAddr_) {
            std::cerr << "[URMA] aclrtMalloc for cqCiAddr failed" << std::endl;
            return false;
        }
        aclrtMemset(cqCiAddr_, sizeof(uint32_t), 0, sizeof(uint32_t));
        localCq.tailAddr = reinterpret_cast<uintptr_t>(cqCiAddr_);
        localCq.dbMode = UrmaDbMode::SW_DB;
        localCq.dbAddr = cqInfo_.out.swdbAddr;

        bootstrap_.allgather(&localCq, cqInfoList_.data(), sizeof(UrmaCqCtx), bootstrap_.ctx);
        return true;
    }

    // Step 6: Create Jetty QP
    bool JettyCreate()
    {
        auto &api = HccpV2Loader::Instance();

        hccp::QpCreateAttr qpAttr{};
        qpAttr.scqHandle = cqHandle_;
        qpAttr.rcqHandle = cqHandle_;
        qpAttr.srqHandle = cqHandle_;
        qpAttr.sqDepth = hccp::kSqDepthDefault;
        qpAttr.rqDepth = hccp::kRqDepthDefault;
        qpAttr.transportMode = transportMode_;
        qpAttr.ub.mode = hccp::JETTY_MODE_USER_CTL_NORMAL;
        qpAttr.ub.jettyId = 0;
        qpAttr.ub.flag.value = 1;
        qpAttr.ub.jfsFlag.value = 2;
        qpAttr.ub.tokenValue = hccp::kTokenValue;
        qpAttr.ub.priority = 0;
        qpAttr.ub.rnrRetry = hccp::kRnrRetryCountDefault;
        qpAttr.ub.errTimeout = 0;
        qpAttr.ub.extMode.piType = false;
        qpAttr.ub.extMode.cstmFlag.bs.sqCstm = 0;
        qpAttr.ub.extMode.sqebbNum = hccp::kSqDepthDefault;
        qpAttr.ub.tokenIdHandle = tokenIdHandle_;

        int ret = api.raCtxQpCreate(ctxHandle_, &qpAttr, &qpCreateInfo_, &qpHandle_);
        if (ret != 0) {
            std::cerr << "[URMA] RaCtxQpCreate failed: " << ret << std::endl;
            return false;
        }

        UrmaWQCtx localWq{};
        localWq.wqn = 0;
        localWq.bufAddr = qpCreateInfo_.ub.sqBuffVa;
        localWq.wqeShiftSize = Log2U32(static_cast<uint32_t>(qpCreateInfo_.ub.wqebbSize));
        localWq.depth = qpAttr.sqDepth;

        if (aclrtMalloc(&sqPiAddr_, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS || !sqPiAddr_) {
            std::cerr << "[URMA] aclrtMalloc for sqPiAddr failed" << std::endl;
            return false;
        }
        aclrtMemset(sqPiAddr_, sizeof(uint32_t), 0, sizeof(uint32_t));
        localWq.headAddr = reinterpret_cast<uintptr_t>(sqPiAddr_);
        if (aclrtMalloc(&sqCiAddr_, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS || !sqCiAddr_) {
            std::cerr << "[URMA] aclrtMalloc for sqCiAddr failed" << std::endl;
            return false;
        }
        aclrtMemset(sqCiAddr_, sizeof(uint32_t), 0, sizeof(uint32_t));
        localWq.tailAddr = reinterpret_cast<uintptr_t>(sqCiAddr_);
        localWq.dbMode = UrmaDbMode::SW_DB;
        localWq.dbAddr = qpCreateInfo_.ub.dbAddr;
        localWq.sl = 0;

        bootstrap_.allgather(&localWq, wqInfoList_.data(), sizeof(UrmaWQCtx), bootstrap_.ctx);
        return true;
    }

    // Step 7: Exchange QP info and import remote Jetty QPs
    bool JettyImport()
    {
        auto &api = HccpV2Loader::Instance();

        hccp::QpImportInfoT localQpImport{};
        localQpImport.in.ub.mode = hccp::JETTY_IMPORT_MODE_NORMAL;
        localQpImport.in.ub.tokenValue = hccp::kTokenValue;
        localQpImport.in.ub.policy = hccp::JETTY_GRP_POLICY_RR;
        localQpImport.in.ub.type = hccp::TARGET_TYPE_JETTY;
        localQpImport.in.ub.flag.bs.tokenPolicy = hccp::TOKEN_POLICY_PLAIN_TEXT;
        localQpImport.in.ub.tpType = 1;

        bootstrap_.allgather(&localQpImport, allQpImportInfoT_.data(), sizeof(hccp::QpImportInfoT), bootstrap_.ctx);
        bootstrap_.allgather(&qpCreateInfo_.key, qpKeyList_.data(), sizeof(hccp::QpKeyT), bootstrap_.ctx);

        for (uint32_t i = 0; i < rankCount_; ++i) {
            if (i == rankId_)
                continue;
            allQpImportInfoT_[i].in.key = qpKeyList_[i];
            int ret = api.raCtxQpImport(ctxHandle_, &allQpImportInfoT_[i], &remoteQpHandles_[i]);
            if (ret != 0) {
                std::cerr << "[URMA] RaCtxQpImport for rank " << i << " failed: " << ret << std::endl;
                return false;
            }
            tpnList_[i] = allQpImportInfoT_[i].out.ub.tpn;
        }
        return true;
    }

    // Step 8: Bind (skip in RM mode)
    bool JettyBind()
    {
        if (transportMode_ == hccp::CONN_RM)
            return true;
        auto &api = HccpV2Loader::Instance();
        for (uint32_t i = 0; i < rankCount_; ++i) {
            if (i == rankId_)
                continue;
            int ret = api.raCtxQpBind(qpHandle_, remoteQpHandles_[i]);
            if (ret != 0) {
                std::cerr << "[URMA] RaCtxQpBind for rank " << i << " failed: " << ret << std::endl;
                return false;
            }
        }
        return true;
    }

    // Step 9: Exchange MR info and import remote memory
    bool RmemImport()
    {
        auto &api = HccpV2Loader::Instance();
        std::vector<hccp::RegMemResultInfo> mrList(rankCount_);
        bootstrap_.allgather(&localMR_, mrList.data(), sizeof(hccp::RegMemResultInfo), bootstrap_.ctx);

        for (uint32_t i = 0; i < rankCount_; ++i) {
            if (i == rankId_) {
                rmemHandles_[i] = lmemHandle_;
                continue;
            }
            hccp::MrImportInfoT mrImport{};
            mrImport.in.key = mrList[i].key;
            mrImport.in.ub.tokenValue = mrList[i].tokenValue;
            mrImport.in.ub.flags.bs.cacheable = mrList[i].cacheable;
            mrImport.in.ub.flags.bs.access = mrList[i].access;

            int ret = api.raCtxRmemImport(ctxHandle_, &mrImport, &rmemHandles_[i]);
            if (ret != 0) {
                std::cerr << "[URMA] RaCtxRmemImport for rank " << i << " failed: " << ret << std::endl;
                return false;
            }
        }
        return true;
    }

    // Step 10: Construct UrmaInfo on host, copy to device
    bool FillUrmaInfo()
    {
        bootstrap_.allgather(&localMemInfo_, ubMemInfoList_.data(), sizeof(UrmaMemInfo), bootstrap_.ctx);
        bootstrap_.allgather(&localHccpEid_, hccpEidList_.data(), sizeof(hccp::HccpEid), bootstrap_.ctx);
        bootstrap_.barrier(bootstrap_.ctx);

        if (!AllocAndCopyEidTable())
            return false;

        constexpr uint32_t qpNum = 1;
        size_t totalSize =
            sizeof(UrmaInfo) + rankCount_ * (2U * sizeof(UrmaWQCtx) * qpNum + 2U * sizeof(UrmaCqCtx) * qpNum +
                                             sizeof(UrmaMemInfo) * qpNum);

        aclError err = aclrtMalloc(&urmaInfoDevice_, totalSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMalloc for urmaInfo failed: " << err << std::endl;
            return false;
        }

        std::vector<uint8_t> hostBuf(totalSize, 0);
        auto *copyInfo = reinterpret_cast<UrmaInfo *>(hostBuf.data());
        copyInfo->qpNum = qpNum;
        copyInfo->localTokenId = localMR_.tokenId;

        // Phase 1: build layout with host-temporary pointers to populate per-rank arrays
        ComputeInfoLayout(copyInfo, hostBuf.data());
        FillPerRankData(copyInfo);

        // Phase 2: rewrite pointers to device virtual addresses
        ComputeInfoLayout(copyInfo, reinterpret_cast<uint8_t *>(urmaInfoDevice_));

        err = aclrtMemcpy(urmaInfoDevice_, totalSize, hostBuf.data(), totalSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMemcpy for urmaInfo failed: " << err << std::endl;
            aclrtFree(urmaInfoDevice_);
            urmaInfoDevice_ = nullptr;
            return false;
        }
        return true;
    }

    bool AllocAndCopyEidTable()
    {
        size_t eidTableSize = rankCount_ * sizeof(hccp::HccpEid);
        aclError err = aclrtMalloc(&hccpEidDevice_, eidTableSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMalloc for hccpEid failed: " << err << std::endl;
            return false;
        }
        err = aclrtMemcpy(hccpEidDevice_, eidTableSize, hccpEidList_.data(), eidTableSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMemcpy for hccpEid failed: " << err << std::endl;
            return false;
        }
        return true;
    }

    void FillPerRankData(UrmaInfo *copyInfo)
    {
        auto &localWq = wqInfoList_[rankId_];
        auto &localCq = cqInfoList_[rankId_];

        for (uint32_t rank = 0; rank < rankCount_; ++rank) {
            ubMemInfoList_[rank].tpn = tpnList_[rank];

            reinterpret_cast<UrmaWQCtx *>(copyInfo->sqPtr)[rank] = localWq;
            reinterpret_cast<UrmaWQCtx *>(copyInfo->rqPtr)[rank] = localWq;
            reinterpret_cast<UrmaCqCtx *>(copyInfo->scqPtr)[rank] = localCq;
            reinterpret_cast<UrmaCqCtx *>(copyInfo->rcqPtr)[rank] = localCq;
            reinterpret_cast<UrmaMemInfo *>(copyInfo->memPtr)[rank] = ubMemInfoList_[rank];
            reinterpret_cast<UrmaMemInfo *>(copyInfo->memPtr)[rank].eidAddr =
                reinterpret_cast<uint64_t>(static_cast<hccp::HccpEid *>(hccpEidDevice_) + rank);
        }
    }

    // Configuration
    uint32_t deviceId_{0};
    uint32_t rankId_{0};
    uint32_t rankCount_{0};
    void *symmetricAddr_{nullptr};
    uint64_t symmetricSize_{0};
    UrmaBootstrapHandle bootstrap_{};
    hccp::TransportModeT transportMode_{hccp::CONN_RM};

    // HCCP V2 handles
    void *ctxHandle_{nullptr};
    void *chanHandle_{nullptr};
    void *tokenIdHandle_{nullptr};
    void *lmemHandle_{nullptr};
    void *cqHandle_{nullptr};
    void *qpHandle_{nullptr};
    std::vector<void *> remoteQpHandles_;
    std::vector<void *> rmemHandles_;

    // HCCP V2 state
    hccp::HccpEid localHccpEid_{};
    hccp::RegMemResultInfo localMR_{};
    hccp::CqInfoT cqInfo_{};
    hccp::QpCreateInfo qpCreateInfo_{};
    UrmaMemInfo localMemInfo_{};

    // Device-side allocations
    void *urmaInfoDevice_{nullptr};
    void *hccpEidDevice_{nullptr};
    void *cqPiAddr_{nullptr};
    void *cqCiAddr_{nullptr};
    void *sqPiAddr_{nullptr};
    void *sqCiAddr_{nullptr};

    // Allgather results
    std::vector<UrmaWQCtx> wqInfoList_;
    std::vector<UrmaCqCtx> cqInfoList_;
    std::vector<UrmaMemInfo> ubMemInfoList_;
    std::vector<hccp::HccpEid> hccpEidList_;
    std::vector<uint32_t> tpnList_;
    std::vector<hccp::QpKeyT> qpKeyList_;
    std::vector<hccp::QpImportInfoT> allQpImportInfoT_;

    bool initialized_{false};

    // Process-level singletons
    static inline bool tsdOpened_{false};
    static inline bool raInitialized_{false};
};

} // namespace urma
} // namespace comm
} // namespace pto

#endif // PTO_NPU_COMM_ASYNC_URMA_WORKSPACE_MANAGER_HPP
