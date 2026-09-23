/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_A5_TPUT_ASYNC_BATCH_HPP
#define PTO_COMM_A5_TPUT_ASYNC_BATCH_HPP

#include "pto/comm/async_common/TPutAsyncBatchCommonDetail.hpp"
#ifdef PTO_URMA_SUPPORTED
#include "pto/comm/async/urma/urma_async_batch.hpp"
#endif

namespace pto {
namespace comm {

template <DmaEngine engine, typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_DEFER_IMPL(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, const AsyncSession& session, uint32_t peer)
{
    static_assert(
        engine == DmaEngine::URMA,
        "TPUT_ASYNC(DEFER): A5 SDMA uses a synchronous MTE fallback; aggregate PUT requires URMA.");
#ifdef PTO_URMA_SUPPORTED
    uint64_t totalBytes = 0U;
    if (!detail::TPutAsyncBatchValidatePayload(dstGlobalData, srcGlobalData, session, totalBytes) || totalBytes == 0U) {
        return {};
    }
    return urma::detail::UrmaDeferAsyncPut(
        reinterpret_cast<__gm__ uint8_t*>(dstGlobalData.data()),
        reinterpret_cast<__gm__ uint8_t*>(srcGlobalData.data()), totalBytes, session, peer);
#else
    (void)dstGlobalData;
    (void)srcGlobalData;
    (void)session;
    (void)peer;
    static_assert(engine != DmaEngine::URMA, "TPUT_ASYNC(DEFER): URMA requires NPU_ARCH 3510.");
    return {};
#endif
}

template <DmaEngine engine>
PTO_INTERNAL AsyncEvent FlushPendingAsyncPut(const AsyncSession& session, uint32_t peer)
{
    static_assert(
        engine == DmaEngine::URMA,
        "TPUT_ASYNC batch flush: A5 SDMA uses a synchronous MTE fallback; aggregate PUT requires URMA.");
#ifdef PTO_URMA_SUPPORTED
    return urma::detail::UrmaFlushPendingAsyncPut(session, peer);
#else
    (void)session;
    (void)peer;
    static_assert(engine != DmaEngine::URMA, "TPUT_ASYNC batch flush: URMA requires NPU_ARCH 3510.");
    return {};
#endif
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_A5_TPUT_ASYNC_BATCH_HPP
