/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_A2A3_TPUT_ASYNC_BATCH_HPP
#define PTO_COMM_A2A3_TPUT_ASYNC_BATCH_HPP

#include "pto/comm/async_common/TPutAsyncBatchCommonDetail.hpp"
#include "pto/comm/async/sdma/sdma_async_batch.hpp"

namespace pto {
namespace comm {

template <DmaEngine engine, typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent
TPUT_ASYNC_DEFER_IMPL(GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, const AsyncSession& session)
{
    static_assert(engine == DmaEngine::SDMA, "TPUT_ASYNC(DEFER): only SDMA is supported on A2/A3.");
    if (!detail::ValidateAsyncPutBatchSession<engine>(session)) {
        return {};
    }
    uint64_t totalBytes = 0U;
    if (!detail::TPutAsyncBatchValidatePayload(dstGlobalData, srcGlobalData, session, totalBytes) || totalBytes == 0U) {
        return {};
    }
    return sdma::detail::SdmaDeferAsyncPut(
        reinterpret_cast<__gm__ uint8_t*>(dstGlobalData.data()),
        reinterpret_cast<__gm__ uint8_t*>(srcGlobalData.data()), totalBytes, session);
}

template <DmaEngine engine>
PTO_INTERNAL AsyncEvent FlushPendingAsyncPut(const AsyncSession& session)
{
    static_assert(engine == DmaEngine::SDMA, "TPUT_ASYNC batch flush: only SDMA is supported on A2/A3.");
    if (!detail::ValidateAsyncPutBatchSession<engine>(session)) {
        return {};
    }
    return sdma::detail::SdmaFlushPendingAsyncPut(session);
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_A2A3_TPUT_ASYNC_BATCH_HPP
