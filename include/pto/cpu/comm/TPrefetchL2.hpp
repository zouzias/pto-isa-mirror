/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_COMM_TPREFETCH_L2_HPP
#define PTO_CPU_COMM_TPREFETCH_L2_HPP

#include "pto/comm/comm_types.hpp"
#include "pto/comm/async/async_types.hpp"

namespace pto {
namespace comm {

template <typename GlobalData>
PTO_INTERNAL AsyncEvent TPREFETCH_L2_IMPL(GlobalData & /*src*/, const AsyncSession & /*session*/)
{
    return AsyncEvent(0, DmaEngine::SDMA);
}

PTO_INTERNAL AsyncEvent TPREFETCH_L2_IMPL(__gm__ void * /*src*/, uint64_t /*bytes*/,
                                           const AsyncSession & /*session*/)
{
    return AsyncEvent(0, DmaEngine::SDMA);
}

template <typename GlobalData>
PTO_INTERNAL AsyncEvent TPREFETCH_L2_IMPL(GlobalData & /*src*/, const sdma::SdmaExecContext & /*execCtx*/)
{
    return AsyncEvent(0, DmaEngine::SDMA);
}

PTO_INTERNAL AsyncEvent TPREFETCH_L2_IMPL(__gm__ void * /*src*/, uint64_t /*bytes*/,
                                           const sdma::SdmaExecContext & /*execCtx*/)
{
    return AsyncEvent(0, DmaEngine::SDMA);
}

} // namespace comm
} // namespace pto

#endif // PTO_CPU_COMM_TPREFETCH_L2_HPP
