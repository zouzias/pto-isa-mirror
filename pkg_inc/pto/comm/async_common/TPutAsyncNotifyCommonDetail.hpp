/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TPUT_ASYNC_NOTIFY_COMMON_DETAIL_HPP
#define PTO_COMM_TPUT_ASYNC_NOTIFY_COMMON_DETAIL_HPP

#include "pto/comm/async_common/TPutAsyncCommonDetail.hpp"

namespace pto {
namespace comm {
namespace detail {

template <typename GlobalDstData, typename GlobalSrcData, typename GlobalSignalData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_NOTIFY_SDMA_IMPL(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, GlobalSignalData& dstSignalData, int32_t signalValue,
    NotifyOp notifyOp, const AsyncSession& session)
{
    (void)TPutAsyncCheckTensorCompatibility<GlobalDstData, GlobalSrcData>();
    static_assert(
        std::is_same_v<typename GlobalSignalData::RawDType, int32_t>, "TPUT_ASYNC_NOTIFY: signal type must be int32_t");

    PTO_ASSERT(
        session.valid && session.engine == DmaEngine::SDMA && session.channelGroupIdx == 0U,
        "TPUT_ASYNC_NOTIFY: A2/A3 first version requires a valid SDMA session with channelGroupIdx == 0.");
    PTO_ASSERT(
        srcGlobalData.data() != nullptr && dstGlobalData.data() != nullptr && dstSignalData.data() != nullptr,
        "TPUT_ASYNC_NOTIFY: src, dst and signal pointers must not be null.");
    PTO_ASSERT(
        (reinterpret_cast<uint64_t>(dstSignalData.data()) & (alignof(int32_t) - 1U)) == 0U,
        "TPUT_ASYNC_NOTIFY: A2/A3 SDMA signal address must be 4-byte aligned.");
    PTO_ASSERT(
        TPutAsyncIsFlatContiguous1D(srcGlobalData) && TPutAsyncIsFlatContiguous1D(dstGlobalData),
        "TPUT_ASYNC_NOTIFY: src and dst tensors must be flat contiguous 1D.");

    const uint32_t srcElems = TPutAsyncGetTotalElemCount(srcGlobalData);
    const uint32_t dstElems = TPutAsyncGetTotalElemCount(dstGlobalData);
    PTO_ASSERT(dstElems >= srcElems, "TPUT_ASYNC_NOTIFY: dst buffer too small for src data.");

    using T = typename GlobalSrcData::RawDType;
    const uint64_t eventHandle = sdma::__sdma_put_async_notify(
        dstGlobalData.data(), srcGlobalData.data(), dstSignalData.data(), signalValue, notifyOp,
        static_cast<uint64_t>(srcElems) * sizeof(T), session);
    return AsyncEvent(eventHandle, DmaEngine::SDMA);
}

} // namespace detail
} // namespace comm
} // namespace pto

#endif // PTO_COMM_TPUT_ASYNC_NOTIFY_COMMON_DETAIL_HPP
