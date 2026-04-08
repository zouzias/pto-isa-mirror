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
#include <cstring>
#include <iostream>
#include <vector>
#include <mutex>

#include <dlfcn.h>
#include "acl/acl.h"

#include "pto/npu/comm/async/urma/urma_types.hpp"

namespace pto {
namespace comm {
namespace urma {

// ============================================================================
// UrmaBootstrapHandle: generic cross-rank information exchange abstraction.
// Users bind the actual implementation (MPI, TCP socket, etc.) at init time.
// ============================================================================
struct UrmaBootstrapHandle {
    int (*allgather)(const void *sendbuf, void *recvbuf, int size, void *ctx);
    int (*barrier)(void *ctx);
    void *ctx;
};

// ============================================================================
// HCCP V2 type definitions — self-contained, no CANN header dependency.
// Binary-compatible with CANN HCCP V2 library.
// Only the minimal subset required by UrmaWorkspaceManager is defined.
// ============================================================================
namespace hccp {

constexpr int32_t kDevEidInfoMaxName = 64;
constexpr int32_t kDevQpKeySize = 64;
constexpr int32_t kMemKeySize = 128;
constexpr uint32_t kTokenValue = 0;
constexpr uint32_t kCqDepthDefault = 16384;
constexpr uint32_t kSqDepthDefault = 4096;
constexpr uint32_t kRqDepthDefault = 256;
constexpr uint8_t kRnrRetryCountDefault = 7;

// --- Enums ---

enum HccpNetworkMode {
    NETWORK_PEER_ONLINE = 0,
    NETWORK_OFFLINE,
    NETWORK_ONLINE,
};

enum DrvHdcServiceType : int {
    HDC_SERVICE_TYPE_RDMA = 6,
    HDC_SERVICE_TYPE_RDMA_V2 = 18,
};

enum SubProcType {
    TSD_SUB_PROC_HCCP = 0,
    TSD_SUB_PROC_COMPUTE = 1,
    TSD_SUB_PROC_MAX = 0xFF,
};

enum JfcMode {
    JFC_MODE_NORMAL = 0,
    JFC_MODE_STARS_POLL = 1,
    JFC_MODE_CCU_POLL = 2,
    JFC_MODE_USER_CTL_NORMAL = 3,
    JFC_MODE_MAX,
};

enum JettyMode {
    JETTY_MODE_URMA_NORMAL = 0,
    JETTY_MODE_CACHE_LOCK_DWQE = 1,
    JETTY_MODE_CCU = 2,
    JETTY_MODE_USER_CTL_NORMAL = 3,
    JETTY_MODE_CCU_TA_CACHE = 4,
    JETTY_MODE_MAX,
};

enum TransportModeT {
    CONN_RM = 1,
    CONN_RC = 2,
};

enum TokenPolicy : uint32_t {
    TOKEN_POLICY_NONE = 0,
    TOKEN_POLICY_PLAIN_TEXT = 1,
    TOKEN_POLICY_SIGNED = 2,
    TOKEN_POLICY_ALL_ENCRYPTED = 3,
};

enum JettyImportMode {
    JETTY_IMPORT_MODE_NORMAL = 0,
    JETTY_IMPORT_MODE_EXP = 1,
};

enum JettyGrpPolicy : uint32_t {
    JETTY_GRP_POLICY_RR = 0,
    JETTY_GRP_POLICY_HASH_HINT = 1,
};

enum TargetType {
    TARGET_TYPE_JFR = 0,
    TARGET_TYPE_JETTY = 1,
    TARGET_TYPE_JETTY_GROUP = 2,
};

enum MemSegAccessFlags {
    MEM_SEG_ACCESS_LOCAL_ONLY = 1,
    MEM_SEG_ACCESS_READ = (1 << 1),
    MEM_SEG_ACCESS_WRITE = (1 << 2),
    MEM_SEG_ACCESS_ATOMIC = (1 << 3),
    MEM_SEG_ACCESS_DEFAULT = MEM_SEG_ACCESS_READ | MEM_SEG_ACCESS_WRITE | MEM_SEG_ACCESS_ATOMIC,
};

// --- Basic structs ---

struct ProcExtParam {
    const char *paramInfo;
    uint64_t paramLen;
};

struct ProcEnvParam {
    const char *envName;
    uint64_t nameLen;
    const char *envValue;
    uint64_t valueLen;
};

struct ProcOpenArgs {
    SubProcType procType;
    ProcEnvParam *envParaList;
    uint64_t envCnt;
    const char *filePath;
    uint64_t pathLen;
    ProcExtParam *extParamList;
    uint64_t extParamCnt;
    int *subPid;
};

struct RaInitConfig {
    unsigned int phyId;
    HccpNetworkMode nicPosition;
    DrvHdcServiceType hdcType;
    bool enableHdcAsync;
};

struct RaInfo {
    HccpNetworkMode mode;
    unsigned int phyId;
};

union HccpEid {
    uint8_t raw[16];
    struct {
        uint64_t reserved;
        uint32_t prefix;
        uint32_t addr;
    } in4;
    struct {
        uint64_t subnetPrefix;
        uint64_t interfaceId;
    } in6;
};

struct DevEidInfo {
    char name[kDevEidInfoMaxName];
    uint32_t type;
    uint32_t eidIndex;
    HccpEid eid;
    uint32_t dieId;
    uint32_t chipId;
    uint32_t funcId;
    uint32_t resv;
};

struct CtxInitCfg {
    HccpNetworkMode mode;
    union {
        struct {
            bool disabledLiteThread;
        } rdma;
    };
};

struct CtxInitAttr {
    unsigned int phyId;
    union {
        uint8_t _rdmaPad[24]; // rdma branch is 24 bytes; keeps union size correct without netinet headers
        struct {
            uint32_t eidIndex;
            HccpEid eid;
        } ub;
    };
    uint32_t resv[16];
};

struct HccpTokenId {
    uint32_t tokenId;
};

// --- Channel / CQ ---

union DataPlaneCstmFlag {
    struct {
        uint32_t poolCqCstm : 1;
        uint32_t reserved : 31;
    } bs;
    uint32_t value;
};

struct ChanInfoT {
    struct {
        DataPlaneCstmFlag dataPlaneFlag;
    } in;
    struct {
        int fd;
    } out;
};

union JfcFlag {
    struct {
        uint32_t lockFree : 1;
        uint32_t jfcInline : 1;
        uint32_t reserved : 30;
    } bs;
    uint32_t value;
};

struct CqCreateAttr {
    void *chanHandle;
    uint32_t depth;
    union {
        struct {
            uint64_t cqContext;
            uint32_t mode;
            uint32_t compVector;
        } rdma;
        struct {
            uint64_t userCtx;
            JfcMode mode;
            uint32_t ceqn;
            JfcFlag flag;
            struct {
                bool valid;
                uint32_t cqeFlag;
            } ccuExCfg;
        } ub;
    };
};

struct CqCreateInfo {
    uint64_t va;
    uint32_t id;
    uint32_t cqeSize;
    uint64_t bufAddr;
    uint64_t swdbAddr;
};

struct CqInfoT {
    CqCreateAttr in;
    CqCreateInfo out;
};

// --- QP / Jetty ---

union JettyFlag {
    struct {
        uint32_t shareJfr : 1;
        uint32_t reserved : 31;
    } bs;
    uint32_t value;
};

union JfsFlag {
    struct {
        uint32_t lockFree : 1;
        uint32_t errorSuspend : 1;
        uint32_t outorderComp : 1;
        uint32_t orderType : 8;
        uint32_t multiPath : 1;
        uint32_t reserved : 20;
    } bs;
    uint32_t value;
};

union CstmJfsFlag {
    struct {
        uint32_t sqCstm : 1;
        uint32_t dbCstm : 1;
        uint32_t dbCtlCstm : 1;
        uint32_t reserved : 29;
    } bs;
    uint32_t value;
};

struct JettyQueCfgEx {
    uint32_t buffSize;
    uint64_t buffVa;
};

struct QpCreateAttr {
    void *scqHandle;
    void *rcqHandle;
    void *srqHandle;
    uint32_t sqDepth;
    uint32_t rqDepth;
    TransportModeT transportMode;
    union {
        struct {
            uint32_t mode;
            uint32_t udpSport;
            uint8_t trafficClass;
            uint8_t sl;
            uint8_t timeout;
            uint8_t rnrRetry;
            uint8_t retryCnt;
        } rdm;
        struct {
            JettyMode mode;
            uint32_t jettyId;
            JettyFlag flag;
            JfsFlag jfsFlag;
            void *tokenIdHandle;
            uint32_t tokenValue;
            uint8_t priority;
            uint8_t rnrRetry;
            uint8_t errTimeout;
            union {
                struct {
                    JettyQueCfgEx sq;
                    bool piType;
                    CstmJfsFlag cstmFlag;
                    uint32_t sqebbNum;
                } extMode;
                struct {
                    bool lockFlag;
                    uint32_t sqeBufIdx;
                } taCacheMode;
            };
        } ub;
    };
    uint32_t resv[16];
};

struct QpKeyT {
    uint8_t value[kDevQpKeySize];
    uint8_t size;
};

struct QpCreateInfo {
    QpKeyT key;
    union {
        struct {
            uint32_t qpn;
        } rdma;
        struct {
            uint32_t uasid;
            uint32_t id;
            uint64_t sqBuffVa;
            uint64_t wqebbSize;
            uint64_t dbAddr;
            uint32_t dbTokenId;
            uint64_t ciAddr;
        } ub;
    };
    uint64_t va;
    uint32_t resv[16U];
};

// --- Jetty Import ---

union ImportJettyFlag {
    struct {
        uint32_t tokenPolicy : 3;
        uint32_t orderType : 8;
        uint32_t shareTp : 1;
        uint32_t reserved : 20;
    } bs;
    uint32_t value;
};

struct JettyImportExpCfg {
    uint64_t tpHandle;
    uint64_t peerTpHandle;
    uint64_t tag;
    uint32_t txPsn;
    uint32_t rxPsn;
    uint32_t rsv[16];
};

struct QpImportAttr {
    QpKeyT key;
    union {
        struct {
            JettyImportMode mode;
            uint32_t tokenValue;
            JettyGrpPolicy policy;
            TargetType type;
            ImportJettyFlag flag;
            JettyImportExpCfg expImportCfg;
            uint32_t tpType;
        } ub;
    };
    uint32_t resv[7];
};

struct QpImportInfo {
    union {
        struct {
            uint64_t tjettyHandle;
            uint32_t tpn;
        } ub;
    };
    uint32_t resv[8];
};

struct QpImportInfoT {
    QpImportAttr in;
    QpImportInfo out;
};

// --- Memory Region ---

struct MemKey {
    uint8_t value[kMemKeySize];
    uint8_t size;
};

struct MemInfo {
    uint64_t addr;
    uint64_t size;
};

union RegSegFlag {
    struct {
        uint32_t tokenPolicy : 3;
        uint32_t cacheable : 1;
        uint32_t dsva : 1;
        uint32_t access : 6;
        uint32_t nonPin : 1;
        uint32_t userIova : 1;
        uint32_t tokenIdValid : 1;
        uint32_t reserved : 18;
    } bs;
    uint32_t value;
};

struct MemRegAttr {
    MemInfo mem;
    union {
        struct {
            int access;
        } rdma;
        struct {
            RegSegFlag flags;
            uint32_t tokenValue;
            void *tokenIdHandle;
        } ub;
    };
    uint32_t resv[8];
};

struct MemRegInfo {
    MemKey key;
    union {
        struct {
            uint32_t lkey;
        } rdma;
        struct {
            uint32_t tokenId;
            uint64_t targetSegHandle;
        } ub;
    };
    uint32_t resv[8U];
};

struct MrRegInfoT {
    MemRegAttr in;
    MemRegInfo out;
};

union ImportSegFlag {
    struct {
        uint32_t cacheable : 1;
        uint32_t access : 6;
        uint32_t mapping : 1;
        uint32_t reserved : 24;
    } bs;
    uint32_t value;
};

struct MemImportAttr {
    MemKey key;
    union {
        struct {
            ImportSegFlag flags;
            uint64_t mappingAddr;
            uint32_t tokenValue;
        } ub;
    };
    uint32_t resv[4];
};

struct MemImportInfo {
    union {
        struct {
            uint32_t key;
        } rdma;
        struct {
            uint64_t targetSegHandle;
        } ub;
    };
    uint32_t resv[4];
};

struct MrImportInfoT {
    MemImportAttr in;
    MemImportInfo out;
};

struct RegMemResultInfo {
    uint32_t reserved;
    uint64_t address;
    uint64_t size;
    void *lmemHandle;
    MemKey key;
    uint32_t tokenId;
    uint32_t tokenValue;
    uint64_t targetSegHandle;
    void *tokenIdHandle;
    uint32_t cacheable;
    int32_t access;
};

// --- Function pointer typedefs (match HCCP V2 library ABI) ---

using RaInitFn = int (*)(RaInitConfig *);
using TsdProcessOpenFn = uint32_t (*)(uint32_t, ProcOpenArgs *);
using TsdProcessCloseFn = uint32_t (*)(uint32_t, int);
using RaGetDevEidInfoNumFn = int (*)(RaInfo, unsigned int *);
using RaGetDevEidInfoListFn = int (*)(RaInfo, DevEidInfo[], unsigned int *);
using RaCtxInitFn = int (*)(CtxInitCfg *, CtxInitAttr *, void **);
using RaCtxDeinitFn = int (*)(void *);
using RaCtxChanCreateFn = int (*)(void *, ChanInfoT *, void **);
using RaCtxChanDestroyFn = int (*)(void *, void *);
using RaCtxCqCreateFn = int (*)(void *, CqInfoT *, void **);
using RaCtxCqDestroyFn = int (*)(void *, void *);
using RaCtxQpCreateFn = int (*)(void *, QpCreateAttr *, QpCreateInfo *, void **);
using RaCtxQpDestroyFn = int (*)(void *);
using RaCtxTokenIdAllocFn = int (*)(void *, HccpTokenId *, void **);
using RaCtxTokenIdFreeFn = int (*)(void *, void *);
using RaCtxQpImportFn = int (*)(void *, QpImportInfoT *, void **);
using RaCtxQpUnimportFn = int (*)(void *, void *);
using RaCtxQpBindFn = int (*)(void *, void *);
using RaCtxQpUnbindFn = int (*)(void *);
using RaCtxLmemRegisterFn = int (*)(void *, MrRegInfoT *, void **);
using RaCtxLmemUnregisterFn = int (*)(void *, void *);
using RaCtxRmemImportFn = int (*)(void *, MrImportInfoT *, void **);
using RaCtxRmemUnimportFn = int (*)(void *, void *);

} // namespace hccp

// ============================================================================
// HccpV2Loader: dynamic loader for HCCP V2 libraries.
// Thread-safe singleton.
// ============================================================================
class HccpV2Loader {
public:
    static HccpV2Loader &Instance()
    {
        static HccpV2Loader inst;
        return inst;
    }

    bool Load()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (loaded_) {
            return true;
        }

        hcclV1Handle_ = dlopen("libhccl.so", RTLD_NOW);
        if (!hcclV1Handle_) {
            std::cerr << "[URMA] Failed to load libhccl.so: " << dlerror() << std::endl;
            return false;
        }
        hcclV2Handle_ = dlopen("libhccl_v2.so", RTLD_NOW);
        if (!hcclV2Handle_) {
            std::cerr << "[URMA] Failed to load libhccl_v2.so: " << dlerror() << std::endl;
            CleanupUnlocked();
            return false;
        }
        raHandle_ = dlopen("libra.so", RTLD_NOW);
        if (!raHandle_) {
            std::cerr << "[URMA] Failed to load libra.so: " << dlerror() << std::endl;
            CleanupUnlocked();
            return false;
        }
        tsdHandle_ = dlopen("libtsdclient.so", RTLD_NOW);
        if (!tsdHandle_) {
            std::cerr << "[URMA] Failed to load libtsdclient.so: " << dlerror() << std::endl;
            CleanupUnlocked();
            return false;
        }

        if (!LoadSymbols()) {
            CleanupUnlocked();
            return false;
        }
        loaded_ = true;
        return true;
    }

    void Cleanup()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        CleanupUnlocked();
    }

    hccp::RaInitFn raInit{nullptr};
    hccp::TsdProcessOpenFn tsdProcessOpen{nullptr};
    hccp::TsdProcessCloseFn tsdProcessClose{nullptr};
    hccp::RaGetDevEidInfoNumFn raGetDevEidInfoNum{nullptr};
    hccp::RaGetDevEidInfoListFn raGetDevEidInfoList{nullptr};
    hccp::RaCtxInitFn raCtxInit{nullptr};
    hccp::RaCtxDeinitFn raCtxDeinit{nullptr};
    hccp::RaCtxChanCreateFn raCtxChanCreate{nullptr};
    hccp::RaCtxChanDestroyFn raCtxChanDestroy{nullptr};
    hccp::RaCtxCqCreateFn raCtxCqCreate{nullptr};
    hccp::RaCtxCqDestroyFn raCtxCqDestroy{nullptr};
    hccp::RaCtxQpCreateFn raCtxQpCreate{nullptr};
    hccp::RaCtxQpDestroyFn raCtxQpDestroy{nullptr};
    hccp::RaCtxTokenIdAllocFn raCtxTokenIdAlloc{nullptr};
    hccp::RaCtxTokenIdFreeFn raCtxTokenIdFree{nullptr};
    hccp::RaCtxQpImportFn raCtxQpImport{nullptr};
    hccp::RaCtxQpUnimportFn raCtxQpUnimport{nullptr};
    hccp::RaCtxQpBindFn raCtxQpBind{nullptr};
    hccp::RaCtxQpUnbindFn raCtxQpUnbind{nullptr};
    hccp::RaCtxLmemRegisterFn raCtxLmemRegister{nullptr};
    hccp::RaCtxLmemUnregisterFn raCtxLmemUnregister{nullptr};
    hccp::RaCtxRmemImportFn raCtxRmemImport{nullptr};
    hccp::RaCtxRmemUnimportFn raCtxRmemUnimport{nullptr};

private:
    HccpV2Loader() = default;
    ~HccpV2Loader() { CleanupUnlocked(); }
    HccpV2Loader(const HccpV2Loader &) = delete;
    HccpV2Loader &operator=(const HccpV2Loader &) = delete;

    void CleanupUnlocked()
    {
        loaded_ = false;
        auto safeClose = [](void *&h) {
            if (h) {
                dlclose(h);
                h = nullptr;
            }
        };
        safeClose(tsdHandle_);
        safeClose(raHandle_);
        safeClose(hcclV2Handle_);
        safeClose(hcclV1Handle_);
    }

    template <typename T>
    bool LoadSym(T &fn, void *lib, const char *primary, const char *fallback)
    {
        fn = reinterpret_cast<T>(dlsym(lib, primary));
        if (!fn && fallback) {
            fn = reinterpret_cast<T>(dlsym(lib, fallback));
        }
        if (!fn) {
            std::cerr << "[URMA] Failed to load symbol " << primary << ": " << dlerror() << std::endl;
            return false;
        }
        return true;
    }

    bool LoadSymbols()
    {
        bool ok = true;
        ok = ok && LoadSym(raInit, hcclV2Handle_, "RaInit", "ra_init");
        ok = ok && LoadSym(tsdProcessOpen, tsdHandle_, "TsdProcessOpen", "tsd_process_open");
        ok = ok && LoadSym(tsdProcessClose, tsdHandle_, "TsdProcessClose", "tsd_process_close");
        ok = ok && LoadSym(raGetDevEidInfoNum, hcclV2Handle_, "RaGetDevEidInfoNum", "ra_get_dev_eid_info_num");
        ok = ok && LoadSym(raGetDevEidInfoList, hcclV2Handle_, "RaGetDevEidInfoList", "ra_get_dev_eid_info_list");
        ok = ok && LoadSym(raCtxInit, hcclV2Handle_, "RaCtxInit", "ra_ctx_init");
        ok = ok && LoadSym(raCtxDeinit, hcclV2Handle_, "RaCtxDeinit", "ra_ctx_deinit");
        ok = ok && LoadSym(raCtxChanCreate, raHandle_, "RaCtxChanCreate", "ra_ctx_chan_create");
        ok = ok && LoadSym(raCtxChanDestroy, raHandle_, "RaCtxChanDestroy", "ra_ctx_chan_destroy");
        ok = ok && LoadSym(raCtxCqCreate, hcclV2Handle_, "RaCtxCqCreate", "ra_ctx_cq_create");
        ok = ok && LoadSym(raCtxCqDestroy, hcclV2Handle_, "RaCtxCqDestroy", "ra_ctx_cq_destroy");
        ok = ok && LoadSym(raCtxQpCreate, hcclV2Handle_, "RaCtxQpCreate", "ra_ctx_qp_create");
        ok = ok && LoadSym(raCtxQpDestroy, hcclV2Handle_, "RaCtxQpDestroy", "ra_ctx_qp_destroy");
        ok = ok && LoadSym(raCtxTokenIdAlloc, hcclV2Handle_, "RaCtxTokenIdAlloc", "ra_ctx_token_id_alloc");
        ok = ok && LoadSym(raCtxTokenIdFree, hcclV2Handle_, "RaCtxTokenIdFree", "ra_ctx_token_id_free");
        ok = ok && LoadSym(raCtxQpImport, hcclV2Handle_, "RaCtxQpImport", "ra_ctx_qp_import");
        ok = ok && LoadSym(raCtxQpUnimport, hcclV2Handle_, "RaCtxQpUnimport", "ra_ctx_qp_unimport");
        ok = ok && LoadSym(raCtxQpBind, hcclV2Handle_, "RaCtxQpBind", "ra_ctx_qp_bind");
        ok = ok && LoadSym(raCtxQpUnbind, hcclV2Handle_, "RaCtxQpUnbind", "ra_ctx_qp_unbind");
        ok = ok && LoadSym(raCtxLmemRegister, hcclV2Handle_, "RaCtxLmemRegister", "ra_ctx_lmem_register");
        ok = ok && LoadSym(raCtxLmemUnregister, hcclV2Handle_, "RaCtxLmemUnregister", "ra_ctx_lmem_unregister");
        ok = ok && LoadSym(raCtxRmemImport, hcclV2Handle_, "RaCtxRmemImport", "ra_ctx_rmem_import");
        ok = ok && LoadSym(raCtxRmemUnimport, hcclV2Handle_, "RaCtxRmemUnimport", "ra_ctx_rmem_unimport");
        return ok;
    }

    std::mutex mutex_;
    bool loaded_{false};
    void *hcclV1Handle_{nullptr};
    void *hcclV2Handle_{nullptr};
    void *raHandle_{nullptr};
    void *tsdHandle_{nullptr};
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
    ~UrmaWorkspaceManager() { Finalize(); }

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

        if (!OpenTsd()) return false;
        if (!RaInit()) return false;
        if (!RaCtxInit()) return false;
        if (!RegisterMR()) return false;
        if (!JFCCreate()) return false;
        if (!JettyCreate()) return false;
        if (!JettyImport()) return false;
        if (!JettyBind()) return false;
        if (!RmemImport()) return false;
        if (!FillUrmaInfo()) return false;

        initialized_ = true;
        return true;
    }

    void Finalize()
    {
        if (!initialized_) return;

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

        auto &api = HccpV2Loader::Instance();

        if (transportMode_ != hccp::CONN_RM && qpHandle_) {
            if (api.raCtxQpUnbind) api.raCtxQpUnbind(qpHandle_);
        }
        for (uint32_t i = 0; i < rankCount_; ++i) {
            if (i == rankId_ || !remoteQpHandles_[i]) continue;
            if (api.raCtxQpUnimport) api.raCtxQpUnimport(ctxHandle_, remoteQpHandles_[i]);
        }
        for (uint32_t i = 0; i < rankCount_; ++i) {
            if (i == rankId_ || !rmemHandles_[i]) continue;
            if (api.raCtxRmemUnimport) api.raCtxRmemUnimport(ctxHandle_, rmemHandles_[i]);
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

        initialized_ = false;
    }

    void *GetWorkspaceAddr() const { return urmaInfoDevice_; }

private:
    static uint32_t Log2U32(uint32_t n) { return (n <= 1) ? 0 : __builtin_ctz(n); }

    // Step 1: Open TSD (process-level singleton, guarded by static flag)
    bool OpenTsd()
    {
        if (tsdOpened_) {
            return true;
        }
        auto &api = HccpV2Loader::Instance();
        hccp::ProcOpenArgs args{};
        args.procType = hccp::TSD_SUB_PROC_HCCP;
        args.filePath = nullptr;
        args.pathLen = 0;
        char paramStr[] = "--hdcType=18";
        hccp::ProcExtParam extParam{};
        extParam.paramInfo = paramStr;
        extParam.paramLen = sizeof("--hdcType=18");
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
        if (raInitialized_) {
            return true;
        }
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

        hccp::RaInfo info{};
        info.phyId = deviceId_;
        info.mode = hccp::NETWORK_OFFLINE;

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

        // Byte-swap EID for device-side use (network byte order → host byte order)
        localHccpEid_ = attr.ub.eid;
        uint64_t eidL, eidH;
        std::memcpy(&eidL, localHccpEid_.raw, sizeof(uint64_t));
        std::memcpy(&eidH, localHccpEid_.raw + sizeof(uint64_t), sizeof(uint64_t));
        eidL = __builtin_bswap64(eidL);
        eidH = __builtin_bswap64(eidH);
        std::memcpy(localHccpEid_.raw, &eidH, sizeof(uint64_t));
        std::memcpy(localHccpEid_.raw + sizeof(uint64_t), &eidL, sizeof(uint64_t));

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

        UdmaCqCtx localCq{};
        localCq.cqn = 0;
        localCq.bufAddr = cqInfo_.out.bufAddr;
        localCq.cqeShiftSize = Log2U32(cqInfo_.out.cqeSize);
        localCq.depth = cqInfo_.in.depth;

        aclrtMalloc(&cqPiAddr_, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemset(cqPiAddr_, sizeof(uint32_t), 0, sizeof(uint32_t));
        localCq.headAddr = reinterpret_cast<uintptr_t>(cqPiAddr_);

        aclrtMalloc(&cqCiAddr_, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemset(cqCiAddr_, sizeof(uint32_t), 0, sizeof(uint32_t));
        localCq.tailAddr = reinterpret_cast<uintptr_t>(cqCiAddr_);

        localCq.dbMode = UdmaDbMode::SW_DB;
        localCq.dbAddr = cqInfo_.out.swdbAddr;

        bootstrap_.allgather(&localCq, cqInfoList_.data(), sizeof(UdmaCqCtx), bootstrap_.ctx);

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
        qpAttr.ub.flag.value = 1;    // URMA_SHARE_JFR
        qpAttr.ub.jfsFlag.value = 2; // errorSuspend=1
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

        UdmaWQCtx localWq{};
        localWq.wqn = 0;
        localWq.bufAddr = qpCreateInfo_.ub.sqBuffVa;
        localWq.wqeShiftSize = Log2U32(static_cast<uint32_t>(qpCreateInfo_.ub.wqebbSize));
        localWq.depth = qpAttr.sqDepth;

        aclrtMalloc(&sqPiAddr_, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemset(sqPiAddr_, sizeof(uint32_t), 0, sizeof(uint32_t));
        localWq.headAddr = reinterpret_cast<uintptr_t>(sqPiAddr_);

        aclrtMalloc(&sqCiAddr_, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemset(sqCiAddr_, sizeof(uint32_t), 0, sizeof(uint32_t));
        localWq.tailAddr = reinterpret_cast<uintptr_t>(sqCiAddr_);

        localWq.dbMode = UdmaDbMode::SW_DB;
        localWq.dbAddr = qpCreateInfo_.ub.dbAddr;
        localWq.sl = 0;

        bootstrap_.allgather(&localWq, wqInfoList_.data(), sizeof(UdmaWQCtx), bootstrap_.ctx);

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
        localQpImport.in.ub.tpType = 1; // ctp mode

        bootstrap_.allgather(&localQpImport, allQpImportInfoT_.data(), sizeof(hccp::QpImportInfoT), bootstrap_.ctx);
        bootstrap_.allgather(&qpCreateInfo_.key, qpKeyList_.data(), sizeof(hccp::QpKeyT), bootstrap_.ctx);

        for (uint32_t i = 0; i < rankCount_; ++i) {
            if (i == rankId_) continue;
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
        if (transportMode_ == hccp::CONN_RM) {
            return true;
        }
        auto &api = HccpV2Loader::Instance();
        for (uint32_t i = 0; i < rankCount_; ++i) {
            if (i == rankId_) continue;
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

    // Step 10: Construct UdmaInfo on host, copy to device (binary layout preserved)
    // Layout: [UdmaInfo header] [WQCtx*N SQ] [WQCtx*N RQ] [CqCtx*N SCQ] [CqCtx*N RCQ] [MemInfo*N]
    bool FillUrmaInfo()
    {
        bootstrap_.allgather(&localMemInfo_, ubMemInfoList_.data(), sizeof(UdmaMemInfo), bootstrap_.ctx);
        bootstrap_.allgather(&localHccpEid_, hccpEidList_.data(), sizeof(hccp::HccpEid), bootstrap_.ctx);
        bootstrap_.barrier(bootstrap_.ctx);

        aclError err =
            aclrtMalloc(&hccpEidDevice_, rankCount_ * sizeof(hccp::HccpEid), ACL_MEM_MALLOC_HUGE_FIRST);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMalloc for hccpEid failed: " << err << std::endl;
            return false;
        }
        err = aclrtMemcpy(hccpEidDevice_, rankCount_ * sizeof(hccp::HccpEid), hccpEidList_.data(),
                          rankCount_ * sizeof(hccp::HccpEid), ACL_MEMCPY_HOST_TO_DEVICE);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMemcpy for hccpEid failed: " << err << std::endl;
            return false;
        }

        constexpr uint32_t qpNum = 1;
        size_t wqSize = sizeof(UdmaWQCtx) * qpNum;
        size_t cqSize = sizeof(UdmaCqCtx) * qpNum;
        size_t oneQpSize = 2U * (wqSize + cqSize) + sizeof(UdmaMemInfo) * qpNum;
        size_t totalSize = sizeof(UdmaInfo) + oneQpSize * rankCount_;

        err = aclrtMalloc(&urmaInfoDevice_, totalSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMalloc for urmaInfo failed: " << err << std::endl;
            return false;
        }

        // Phase 1: build layout in host buffer using host temporary addresses
        std::vector<uint8_t> hostBuf(totalSize, 0);
        auto *copyInfo = reinterpret_cast<UdmaInfo *>(hostBuf.data());
        copyInfo->qpNum = qpNum;
        copyInfo->localTokenId = localMR_.tokenId;
        copyInfo->sqPtr = reinterpret_cast<uint64_t>(copyInfo + 1);
        copyInfo->rqPtr = reinterpret_cast<uint64_t>(
            reinterpret_cast<UdmaWQCtx *>(copyInfo->sqPtr) + rankCount_ * qpNum);
        copyInfo->scqPtr = reinterpret_cast<uint64_t>(
            reinterpret_cast<UdmaWQCtx *>(copyInfo->rqPtr) + rankCount_ * qpNum);
        copyInfo->rcqPtr = reinterpret_cast<uint64_t>(
            reinterpret_cast<UdmaCqCtx *>(copyInfo->scqPtr) + rankCount_ * qpNum);
        copyInfo->memPtr = reinterpret_cast<uint64_t>(
            reinterpret_cast<UdmaCqCtx *>(copyInfo->rcqPtr) + rankCount_ * qpNum);

        for (uint32_t rank = 0; rank < rankCount_; ++rank) {
            ubMemInfoList_[rank].tpn = tpnList_[rank];

            auto &localWq = wqInfoList_[rankId_];
            auto &localCq = cqInfoList_[rankId_];

            reinterpret_cast<UdmaWQCtx *>(copyInfo->sqPtr)[rank] = localWq;
            reinterpret_cast<UdmaWQCtx *>(copyInfo->rqPtr)[rank] = localWq;
            reinterpret_cast<UdmaCqCtx *>(copyInfo->scqPtr)[rank] = localCq;
            reinterpret_cast<UdmaCqCtx *>(copyInfo->rcqPtr)[rank] = localCq;
            reinterpret_cast<UdmaMemInfo *>(copyInfo->memPtr)[rank] = ubMemInfoList_[rank];
            reinterpret_cast<UdmaMemInfo *>(copyInfo->memPtr)[rank].eidAddr =
                reinterpret_cast<uint64_t>(static_cast<hccp::HccpEid *>(hccpEidDevice_) + rank);
        }

        // Phase 2: rewrite pointers from host addresses to device virtual addresses
        auto *devBase = reinterpret_cast<UdmaInfo *>(urmaInfoDevice_);
        copyInfo->sqPtr = reinterpret_cast<uint64_t>(devBase + 1);
        copyInfo->rqPtr = reinterpret_cast<uint64_t>(
            reinterpret_cast<UdmaWQCtx *>(copyInfo->sqPtr) + rankCount_ * qpNum);
        copyInfo->scqPtr = reinterpret_cast<uint64_t>(
            reinterpret_cast<UdmaWQCtx *>(copyInfo->rqPtr) + rankCount_ * qpNum);
        copyInfo->rcqPtr = reinterpret_cast<uint64_t>(
            reinterpret_cast<UdmaCqCtx *>(copyInfo->scqPtr) + rankCount_ * qpNum);
        copyInfo->memPtr = reinterpret_cast<uint64_t>(
            reinterpret_cast<UdmaCqCtx *>(copyInfo->rcqPtr) + rankCount_ * qpNum);

        err = aclrtMemcpy(urmaInfoDevice_, totalSize, hostBuf.data(), totalSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMemcpy for urmaInfo failed: " << err << std::endl;
            aclrtFree(urmaInfoDevice_);
            urmaInfoDevice_ = nullptr;
            return false;
        }

        return true;
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
    UdmaMemInfo localMemInfo_{};

    // Device-side allocations
    void *urmaInfoDevice_{nullptr};
    void *hccpEidDevice_{nullptr};
    void *cqPiAddr_{nullptr};
    void *cqCiAddr_{nullptr};
    void *sqPiAddr_{nullptr};
    void *sqCiAddr_{nullptr};

    // Allgather results
    std::vector<UdmaWQCtx> wqInfoList_;
    std::vector<UdmaCqCtx> cqInfoList_;
    std::vector<UdmaMemInfo> ubMemInfoList_;
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
