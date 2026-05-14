/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_TPREFETCH_L2_HPP
#define PTO_CPU_TPREFETCH_L2_HPP

// CPU-sim no-op stubs for TPREFETCH_L2. The L2 cache concept does not exist
// in CPU simulation, so these overloads keep API surface compatibility but
// always return an empty AsyncEvent. Functional correctness on CPU is checked
// by simply running TLOAD afterwards (cache state is irrelevant).

#include "pto/comm/comm_types.hpp"
#include "pto/comm/async_common/async_types.hpp"

namespace pto {

template <typename GlobalData>
PTO_INTERNAL comm::AsyncEvent TPREFETCH_L2_IMPL(GlobalData & /*src*/, __gm__ uint8_t * /*workspace*/)
{
    return comm::AsyncEvent(0, comm::DmaEngine::SDMA);
}

PTO_INTERNAL comm::AsyncEvent TPREFETCH_L2_IMPL(__gm__ void * /*src*/, uint64_t /*bytes*/,
                                                __gm__ uint8_t * /*workspace*/)
{
    return comm::AsyncEvent(0, comm::DmaEngine::SDMA);
}

template <typename GlobalData>
PTO_INTERNAL comm::AsyncEvent TPREFETCH_L2_IMPL(GlobalData & /*src*/, const comm::AsyncSession & /*session*/)
{
    return comm::AsyncEvent(0, comm::DmaEngine::SDMA);
}

PTO_INTERNAL comm::AsyncEvent TPREFETCH_L2_IMPL(__gm__ void * /*src*/, uint64_t /*bytes*/,
                                                const comm::AsyncSession & /*session*/)
{
    return comm::AsyncEvent(0, comm::DmaEngine::SDMA);
}

template <typename GlobalData>
PTO_INTERNAL comm::AsyncEvent TPREFETCH_L2_IMPL(GlobalData & /*src*/, const comm::sdma::SdmaExecContext & /*execCtx*/)
{
    return comm::AsyncEvent(0, comm::DmaEngine::SDMA);
}

PTO_INTERNAL comm::AsyncEvent TPREFETCH_L2_IMPL(__gm__ void * /*src*/, uint64_t /*bytes*/,
                                                const comm::sdma::SdmaExecContext & /*execCtx*/)
{
    return comm::AsyncEvent(0, comm::DmaEngine::SDMA);
}

} // namespace pto

#endif // PTO_CPU_TPREFETCH_L2_HPP
