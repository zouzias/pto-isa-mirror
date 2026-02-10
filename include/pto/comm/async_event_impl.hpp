/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_EVENT_IMPL_HPP
#define PTO_COMM_ASYNC_EVENT_IMPL_HPP

#include "pto/comm/comm_types.hpp"
#include "pto/comm/sdma_async_intrin.hpp"

namespace pto {
namespace comm {

PTO_INTERNAL inline void AsyncEvent::Wait() const
{
    if (handle == 0) {
        return;
    }
    if (engine == DmaEngine::SDMA) {
        (void)sdma::detail::sdma_wait_event(handle);
        return;
    }
    (void)engine;
}

PTO_INTERNAL inline bool AsyncEvent::Test() const
{
    if (handle == 0) {
        return true;
    }
    if (engine == DmaEngine::SDMA) {
        return sdma::detail::sdma_test_event(handle);
    }
    return false;
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_EVENT_IMPL_HPP
