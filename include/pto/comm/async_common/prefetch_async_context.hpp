/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_COMMON_PREFETCH_ASYNC_CONTEXT_HPP
#define PTO_COMM_ASYNC_COMMON_PREFETCH_ASYNC_CONTEXT_HPP

#include "pto/common/type.hpp"
#include "pto/comm/async_common/async_types.hpp"

namespace pto {
namespace detail {

struct PrefetchAsyncContextBase {
    __gm__ uint8_t* workspace{nullptr};
    comm::AsyncSession session;
    comm::AsyncSession* externalSession{nullptr};

    AICORE PrefetchAsyncContextBase() = default;
    AICORE explicit PrefetchAsyncContextBase(__gm__ uint8_t* workspace_) : workspace(workspace_) {}
    AICORE PrefetchAsyncContextBase(__gm__ uint8_t* workspace_, comm::AsyncSession* externalSession_)
        : workspace(workspace_), externalSession(externalSession_)
    {}

    AICORE comm::AsyncSession& GetSession() { return externalSession != nullptr ? *externalSession : session; }
    AICORE const comm::AsyncSession& GetSession() const
    {
        return externalSession != nullptr ? *externalSession : session;
    }
};

} // namespace detail
} // namespace pto

#endif // PTO_COMM_ASYNC_COMMON_PREFETCH_ASYNC_CONTEXT_HPP
