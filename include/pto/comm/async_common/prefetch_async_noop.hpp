/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_COMMON_PREFETCH_ASYNC_NOOP_HPP
#define PTO_COMM_ASYNC_COMMON_PREFETCH_ASYNC_NOOP_HPP

#include "pto/comm/async_common/prefetch_async_context.hpp"

namespace pto {

struct PrefetchAsyncContext : detail::PrefetchAsyncContextBase {
    using detail::PrefetchAsyncContextBase::PrefetchAsyncContextBase;
};

template <typename GlobalData>
PTO_INTERNAL comm::AsyncEvent TPREFETCH_ASYNC_IMPL(GlobalData& /*src*/, PrefetchAsyncContext& /*ctx*/)
{
    return comm::AsyncEvent(0, comm::DmaEngine::SDMA);
}

} // namespace pto

#endif // PTO_COMM_ASYNC_COMMON_PREFETCH_ASYNC_NOOP_HPP
