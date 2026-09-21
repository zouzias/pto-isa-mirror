/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_WORKSPACE_HPP
#define PTO_COMM_WORKSPACE_HPP

#if defined(__CCE_KT_TEST__)
#error "workspace.hpp is a host-only header and cannot be included in device code."
#endif

#include <cstdint>
#include <new>
#include <string>
#include <vector>

#include "pto/common/arch_macro.hpp"
#include "pto/comm/dma_engine.hpp"

#ifndef PTO_COMM_WORKSPACE_SDMA_SUPPORTED
#define PTO_COMM_WORKSPACE_SDMA_SUPPORTED 1
#endif

#ifndef PTO_COMM_WORKSPACE_URMA_SUPPORTED
#if defined(PTO_URMA_SUPPORTED) || (!defined(PTO_NPU_ARCH_A2A3) && !defined(__CCE_KT_TEST__))
#define PTO_COMM_WORKSPACE_URMA_SUPPORTED 1
#else
#define PTO_COMM_WORKSPACE_URMA_SUPPORTED 0
#endif
#endif

#ifndef PTO_COMM_WORKSPACE_RDMA_SUPPORTED
#if defined(PTO_RDMA_SUPPORTED) || defined(PTO_RDMA_BACKEND_HNS_1825_SUPPORTED)
#define PTO_COMM_WORKSPACE_RDMA_SUPPORTED 1
#else
#define PTO_COMM_WORKSPACE_RDMA_SUPPORTED 0
#endif
#endif

#if PTO_COMM_WORKSPACE_SDMA_SUPPORTED
#include "pto/comm/async/sdma/sdma_workspace_manager.hpp"
#endif

#if PTO_COMM_WORKSPACE_URMA_SUPPORTED
#include "pto/comm/async/urma/urma_workspace_manager.hpp"
#endif

#if PTO_COMM_WORKSPACE_RDMA_SUPPORTED
#include "pto/comm/async/rdma/rdma_workspace_manager.hpp"
#endif

namespace pto {
namespace comm {

enum class WorkspaceStatus : int32_t {
    Ok = 0,
    InvalidArgument,
    Unsupported,
    EngineFailure,
};

struct RdmaTransportConfig {
    uint32_t phyId{0};
    std::string localIp;
    uint16_t basePort{60032};
    std::vector<std::string> peerIps;
    std::vector<uint32_t> peerPhyIds;
    std::vector<uint64_t> peerSymAddrs;
    uint32_t traceId{0};
};

struct WorkspaceRequest {
    void* hcclComm{nullptr};
    uint32_t rankId{0};
    uint32_t rankNum{0};
    void* symmetricAddr{nullptr};
    uint64_t symmetricBytes{0};
    const RdmaTransportConfig* rdma{nullptr};
};

struct Workspace {
    void* addr{nullptr};
    uint64_t bytes{0};
    DmaEngine engine{DmaEngine::SDMA};
    void* impl{nullptr};
};

namespace detail {

inline bool HasWorkspaceResource(const Workspace& ws) { return ws.addr != nullptr || ws.impl != nullptr; }

inline void ClearWorkspace(Workspace* ws)
{
    if (ws != nullptr) {
        *ws = Workspace{};
    }
}

inline WorkspaceStatus ValidateOutputWorkspace(Workspace* out)
{
    if (out == nullptr || HasWorkspaceResource(*out)) {
        return WorkspaceStatus::InvalidArgument;
    }
    return WorkspaceStatus::Ok;
}

inline void FillWorkspace(Workspace* out, DmaEngine engine, void* addr, uint64_t bytes, void* impl)
{
    out->addr = addr;
    out->bytes = bytes;
    out->engine = engine;
    out->impl = impl;
}

#if PTO_COMM_WORKSPACE_SDMA_SUPPORTED
inline WorkspaceStatus CreateSdmaWorkspace(Workspace* out)
{
    auto* manager = new (std::nothrow) sdma::SdmaWorkspaceManager();
    if (manager == nullptr) {
        return WorkspaceStatus::EngineFailure;
    }
    if (!manager->Init() || manager->GetWorkspaceAddr() == nullptr) {
        delete manager;
        return WorkspaceStatus::EngineFailure;
    }
    FillWorkspace(out, DmaEngine::SDMA, manager->GetWorkspaceAddr(), sdma::kSdmaWorkspaceBytes, manager);
    return WorkspaceStatus::Ok;
}
#endif

#if PTO_COMM_WORKSPACE_URMA_SUPPORTED
inline bool IsValidUrmaRequest(const WorkspaceRequest& req)
{
    return req.hcclComm != nullptr && req.rankNum != 0 && req.rankId < req.rankNum && req.symmetricAddr != nullptr &&
           req.symmetricBytes != 0;
}

inline WorkspaceStatus CreateUrmaWorkspace(const WorkspaceRequest& req, Workspace* out)
{
    if (!IsValidUrmaRequest(req)) {
        return WorkspaceStatus::InvalidArgument;
    }
    auto* manager = new (std::nothrow) urma::UrmaWorkspaceManager();
    if (manager == nullptr) {
        return WorkspaceStatus::EngineFailure;
    }
    auto comm = reinterpret_cast<HcclComm>(req.hcclComm);
    if (!manager->Init(comm, req.rankId, req.rankNum, req.symmetricAddr, req.symmetricBytes) ||
        manager->GetWorkspaceAddr() == nullptr) {
        delete manager;
        return WorkspaceStatus::EngineFailure;
    }
    FillWorkspace(out, DmaEngine::URMA, manager->GetWorkspaceAddr(), manager->GetWorkspaceSize(), manager);
    return WorkspaceStatus::Ok;
}
#endif

#if PTO_COMM_WORKSPACE_RDMA_SUPPORTED
inline WorkspaceStatus MapRdmaPreflight(rdma::WorkspaceInitResult status)
{
    return status == rdma::WorkspaceInitResult::READY ? WorkspaceStatus::Ok : WorkspaceStatus::Unsupported;
}

inline bool IsValidRdmaRequest(const WorkspaceRequest& req)
{
    return req.rdma != nullptr && req.rankNum != 0 && req.rankId < req.rankNum && req.symmetricAddr != nullptr &&
           req.symmetricBytes != 0 && req.rdma->peerIps.size() == req.rankNum &&
           req.rdma->peerPhyIds.size() == req.rankNum && req.rdma->peerSymAddrs.size() == req.rankNum &&
           !req.rdma->localIp.empty();
}

inline rdma::WorkspaceConfig MakeRdmaWorkspaceConfig(const WorkspaceRequest& req)
{
    rdma::WorkspaceConfig config{};
    config.rankId = req.rankId;
    config.rankCount = req.rankNum;
    config.phyId = req.rdma->phyId;
    config.localIp = req.rdma->localIp;
    config.basePort = req.rdma->basePort;
    config.peerIps = req.rdma->peerIps;
    config.peerPhyIds = req.rdma->peerPhyIds;
    config.peerSymAddrs = req.rdma->peerSymAddrs;
    config.symmetricAddr = req.symmetricAddr;
    config.symmetricSize = req.symmetricBytes;
    return config;
}

inline WorkspaceStatus CreateRdmaWorkspace(const WorkspaceRequest& req, Workspace* out)
{
    if (!IsValidRdmaRequest(req)) {
        return WorkspaceStatus::InvalidArgument;
    }
    const WorkspaceStatus preflight = MapRdmaPreflight(rdma::RdmaWorkspaceManager::Preflight());
    if (preflight != WorkspaceStatus::Ok) {
        return preflight;
    }
    auto* manager = new (std::nothrow) rdma::RdmaWorkspaceManager();
    if (manager == nullptr) {
        return WorkspaceStatus::EngineFailure;
    }
    manager->SetTraceId(req.rdma->traceId);
    const rdma::WorkspaceInitResult result = manager->Init(MakeRdmaWorkspaceConfig(req));
    if (result != rdma::WorkspaceInitResult::READY || manager->GetWorkspaceAddr() == nullptr) {
        delete manager;
        return result == rdma::WorkspaceInitResult::DISABLED ? WorkspaceStatus::Unsupported :
                                                               WorkspaceStatus::EngineFailure;
    }
    FillWorkspace(out, DmaEngine::RDMA, manager->GetWorkspaceAddr(), manager->GetWorkspaceSize(), manager);
    return WorkspaceStatus::Ok;
}
#endif

} // namespace detail

inline WorkspaceStatus CreateWorkspace(DmaEngine engine, const WorkspaceRequest& req, Workspace* out)
{
    (void)req;
    const WorkspaceStatus outputStatus = detail::ValidateOutputWorkspace(out);
    if (outputStatus != WorkspaceStatus::Ok) {
        return outputStatus;
    }
    switch (engine) {
#if PTO_COMM_WORKSPACE_SDMA_SUPPORTED
        case DmaEngine::SDMA:
            return detail::CreateSdmaWorkspace(out);
#endif
#if PTO_COMM_WORKSPACE_URMA_SUPPORTED
        case DmaEngine::URMA:
            return detail::CreateUrmaWorkspace(req, out);
#endif
#if PTO_COMM_WORKSPACE_RDMA_SUPPORTED
        case DmaEngine::RDMA:
            return detail::CreateRdmaWorkspace(req, out);
#endif
        default:
            return WorkspaceStatus::Unsupported;
    }
}

inline void DestroyWorkspace(Workspace* ws)
{
    if (ws == nullptr || ws->impl == nullptr) {
        detail::ClearWorkspace(ws);
        return;
    }
    switch (ws->engine) {
#if PTO_COMM_WORKSPACE_SDMA_SUPPORTED
        case DmaEngine::SDMA:
            delete static_cast<sdma::SdmaWorkspaceManager*>(ws->impl);
            break;
#endif
#if PTO_COMM_WORKSPACE_URMA_SUPPORTED
        case DmaEngine::URMA:
            delete static_cast<urma::UrmaWorkspaceManager*>(ws->impl);
            break;
#endif
#if PTO_COMM_WORKSPACE_RDMA_SUPPORTED
        case DmaEngine::RDMA:
            delete static_cast<rdma::RdmaWorkspaceManager*>(ws->impl);
            break;
#endif
        default:
            break;
    }
    detail::ClearWorkspace(ws);
}

inline void AbandonWorkspace(Workspace* ws) { detail::ClearWorkspace(ws); }

} // namespace comm
} // namespace pto

#endif // PTO_COMM_WORKSPACE_HPP
