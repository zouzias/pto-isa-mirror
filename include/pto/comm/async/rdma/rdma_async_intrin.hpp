/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// RDMA device dispatch. Public async code enters here, and the workspace backend
// id selects the compiled NIC implementation.

#ifndef PTO_COMM_ASYNC_RDMA_ASYNC_INTRIN_HPP
#define PTO_COMM_ASYNC_RDMA_ASYNC_INTRIN_HPP

#include "pto/comm/async_common/async_types.hpp"
#include "pto/comm/async/rdma/rdma_device_common.hpp"
#ifdef PTO_RDMA_BACKEND_HNS_1825_SUPPORTED
#include "pto/comm/async/rdma/backends/hns_1825/hns_1825_backend.hpp"
#endif

namespace pto {
namespace comm {
namespace rdma {

AICORE inline bool BuildSession(
    __gm__ uint8_t* workspace, uint32_t destRankId, uint32_t myPe, const sdma::TmpBuffer& tmpBuf, uint32_t syncId,
    RdmaSession& session)
{
    session = {};
    __gm__ RdmaInfo* info = reinterpret_cast<__gm__ RdmaInfo*>(workspace);
    if (!IsWorkspaceHeaderValid(info)) {
        return false;
    }
    switch (info->backend) {
#ifdef PTO_RDMA_BACKEND_HNS_1825_SUPPORTED
        case RdmaBackend::HNS_1825:
            return hns_1825::BuildSession(workspace, destRankId, myPe, tmpBuf, syncId, session);
#endif
        default:
            return false;
    }
}

AICORE inline uint64_t Write(const RdmaExecContext& ctx, __gm__ uint8_t* dst, __gm__ uint8_t* src, uint64_t len)
{
    switch (ctx.backend) {
#ifdef PTO_RDMA_BACKEND_HNS_1825_SUPPORTED
        case RdmaBackend::HNS_1825:
            return hns_1825::Write(ctx, dst, src, len);
#endif
        default:
            return EncodeErrorHandle(kRdmaBackendUnavailableError);
    }
}

AICORE inline uint64_t Read(const RdmaExecContext& ctx, __gm__ uint8_t* dst, __gm__ uint8_t* src, uint64_t len)
{
    switch (ctx.backend) {
#ifdef PTO_RDMA_BACKEND_HNS_1825_SUPPORTED
        case RdmaBackend::HNS_1825:
            return hns_1825::Read(ctx, dst, src, len);
#endif
        default:
            return EncodeErrorHandle(kRdmaBackendUnavailableError);
    }
}

AICORE inline uint64_t PeerMrBaseAddr(__gm__ uint8_t* workspace, uint32_t peer)
{
    __gm__ RdmaInfo* info = reinterpret_cast<__gm__ RdmaInfo*>(workspace);
    if (!IsWorkspaceHeaderValid(info)) {
        return 0;
    }
    switch (info->backend) {
#ifdef PTO_RDMA_BACKEND_HNS_1825_SUPPORTED
        case RdmaBackend::HNS_1825:
            return hns_1825::PeerMrBaseAddr(workspace, peer);
#endif
        default:
            return 0;
    }
}

AICORE inline uint32_t WaitEventStatus(uint64_t handle, const RdmaEventContext& ctx)
{
    if (handle == 0) {
        return 0;
    }
    if (IsErrorHandle(handle)) {
        return static_cast<uint32_t>(handle);
    }
    switch (ctx.backend) {
#ifdef PTO_RDMA_BACKEND_HNS_1825_SUPPORTED
        case RdmaBackend::HNS_1825:
            return hns_1825::WaitEventStatus(handle, ctx);
#endif
        default:
            return kRdmaBackendUnavailableError;
    }
}

AICORE inline bool WaitEvent(uint64_t handle, const RdmaEventContext& ctx) { return WaitEventStatus(handle, ctx) == 0; }

AICORE inline bool TestEvent(uint64_t handle, const RdmaEventContext& ctx)
{
    if (handle == 0 || IsErrorHandle(handle)) {
        return true;
    }
    switch (ctx.backend) {
#ifdef PTO_RDMA_BACKEND_HNS_1825_SUPPORTED
        case RdmaBackend::HNS_1825:
            return hns_1825::TestEvent(handle, ctx);
#endif
        default:
            return false;
    }
}

} // namespace rdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_RDMA_ASYNC_INTRIN_HPP
