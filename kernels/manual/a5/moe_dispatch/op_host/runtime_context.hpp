/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MOE_DISPATCH_RUNTIME_CONTEXT_HPP_
#define MOE_DISPATCH_RUNTIME_CONTEXT_HPP_

#include "common.h"

#include <cstdint>

#include "acl/acl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_types.h"

namespace moe_dispatch {

using RtStream = void *;

struct StandaloneHcclContext {
    int rankId = 0;
    int worldSize = 0;
    int deviceId = 0;
    RtStream hcclStream = nullptr;
    HcclComm comm = nullptr;
    PtoRemoteWindowContext *remoteWindowContext = nullptr;
    PtoRemoteWindowContext hostRemoteWindowContext{};
    bool ownsRemoteWindowContext = false;

    void AttachExternalRemoteWindowContext(PtoRemoteWindowContext *ctx);
    void ReleaseRemoteWindowContext();
    bool LoadHostRemoteWindowContextFromDevice();
    bool CopyHostRemoteWindowContextToDevice();
};

struct StandaloneRankRuntime {
    StandaloneHcclContext hccl;
    aclrtStream computeStream = nullptr;
};

bool ValidateHostRemoteWindowContext(const PtoRemoteWindowContext &ctx);
void DestroyStandaloneRankRuntime(StandaloneRankRuntime &runtime);

} // namespace moe_dispatch

#endif // MOE_DISPATCH_RUNTIME_CONTEXT_HPP_
