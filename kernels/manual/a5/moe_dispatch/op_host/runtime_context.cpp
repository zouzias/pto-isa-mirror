/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "op_host/runtime_context.hpp"

namespace moe_dispatch {

void StandaloneHcclContext::AttachExternalRemoteWindowContext(PtoRemoteWindowContext *ctx)
{
    remoteWindowContext = ctx;
    ownsRemoteWindowContext = false;
}

void StandaloneHcclContext::ReleaseRemoteWindowContext()
{
    if (ownsRemoteWindowContext && remoteWindowContext != nullptr) {
        aclrtFree(remoteWindowContext);
    }
    remoteWindowContext = nullptr;
    ownsRemoteWindowContext = false;
}

bool StandaloneHcclContext::LoadHostRemoteWindowContextFromDevice()
{
    if (remoteWindowContext == nullptr) {
        return false;
    }
    return aclrtMemcpy(&hostRemoteWindowContext, sizeof(hostRemoteWindowContext), remoteWindowContext,
                       sizeof(hostRemoteWindowContext), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
}

bool StandaloneHcclContext::CopyHostRemoteWindowContextToDevice()
{
    if (!ValidateHostRemoteWindowContext(hostRemoteWindowContext)) {
        return false;
    }
    void *devMem = nullptr;
    if (aclrtMalloc(&devMem, sizeof(PtoRemoteWindowContext), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        devMem == nullptr) {
        return false;
    }
    if (aclrtMemcpy(devMem, sizeof(PtoRemoteWindowContext), &hostRemoteWindowContext, sizeof(PtoRemoteWindowContext),
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        aclrtFree(devMem);
        return false;
    }
    remoteWindowContext = reinterpret_cast<PtoRemoteWindowContext *>(devMem);
    ownsRemoteWindowContext = true;
    return true;
}

bool ValidateHostRemoteWindowContext(const PtoRemoteWindowContext &ctx)
{
    if (ctx.rankSize == 0 || ctx.rankSize > kMaxMoeDispatchRanks || ctx.rank >= ctx.rankSize || ctx.windowBytes == 0) {
        return false;
    }
    for (uint32_t rank = 0; rank < ctx.rankSize; ++rank) {
        if (ctx.windowIn[rank] == 0 || ctx.windowOut[rank] == 0) {
            return false;
        }
    }
    return true;
}

void DestroyStandaloneRankRuntime(StandaloneRankRuntime &runtime)
{
    runtime.hccl.ReleaseRemoteWindowContext();
    if (runtime.computeStream != nullptr) {
        aclrtDestroyStream(runtime.computeStream);
        runtime.computeStream = nullptr;
    }
}

} // namespace moe_dispatch
