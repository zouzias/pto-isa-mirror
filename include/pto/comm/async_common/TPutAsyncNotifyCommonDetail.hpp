/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

AICORE inline bool TPutAsyncNotifyRangesOverlap(uint64_t lhs, uint64_t lhsBytes, uint64_t rhs, uint64_t rhsBytes)
{
    return lhs <= rhs ? rhs - lhs < lhsBytes : lhs - rhs < rhsBytes;
}

template <typename GlobalDstData, typename GlobalSrcData, typename GlobalSignalData>
PTO_INTERNAL uint64_t TPutAsyncNotifyValidateTensors(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, GlobalSignalData& dstSignalData)
{
    (void)TPutAsyncCheckTensorCompatibility<GlobalDstData, GlobalSrcData>();
    static_assert(
        std::is_same_v<typename GlobalSignalData::RawDType, int32_t>, "TPUT_ASYNC_NOTIFY: signal type must be int32_t");

    PTO_ASSERT(
        srcGlobalData.data() != nullptr && dstGlobalData.data() != nullptr && dstSignalData.data() != nullptr,
        "TPUT_ASYNC_NOTIFY: src, dst and signal pointers must not be null.");
    PTO_ASSERT(
        TPutAsyncIsFlatContiguous1D(srcGlobalData) && TPutAsyncIsFlatContiguous1D(dstGlobalData),
        "TPUT_ASYNC_NOTIFY: src and dst tensors must be flat contiguous 1D.");
    PTO_ASSERT(
        (reinterpret_cast<uint64_t>(dstSignalData.data()) & (alignof(int32_t) - 1U)) == 0U,
        "TPUT_ASYNC_NOTIFY: signal address must be 4-byte aligned.");
    PTO_ASSERT(
        TPutAsyncGetTotalElemCount(dstSignalData) == 1U,
        "TPUT_ASYNC_NOTIFY: signal tensor must contain exactly one int32_t element.");

    const uint32_t srcElems = TPutAsyncGetTotalElemCount(srcGlobalData);
    const uint32_t dstElems = TPutAsyncGetTotalElemCount(dstGlobalData);
    PTO_ASSERT(srcElems > 0U, "TPUT_ASYNC_NOTIFY: payload must not be empty.");
    PTO_ASSERT(dstElems >= srcElems, "TPUT_ASYNC_NOTIFY: dst buffer is smaller than src.");

    using T = typename GlobalSrcData::RawDType;
    const uint64_t transferBytes = static_cast<uint64_t>(srcElems) * sizeof(T);
    PTO_ASSERT(
        !TPutAsyncNotifyRangesOverlap(
            reinterpret_cast<uint64_t>(dstGlobalData.data()), transferBytes,
            reinterpret_cast<uint64_t>(dstSignalData.data()), sizeof(int32_t)),
        "TPUT_ASYNC_NOTIFY: payload destination and signal must not overlap.");
    return transferBytes;
}

template <typename GlobalDstData, typename GlobalSrcData, typename GlobalSignalData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_NOTIFY_SDMA_IMPL(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, GlobalSignalData& dstSignalData, int32_t signalValue,
    NotifyOp notifyOp, const AsyncSession& session)
{
    const uint64_t transferBytes = TPutAsyncNotifyValidateTensors(dstGlobalData, srcGlobalData, dstSignalData);
    PTO_ASSERT(
        session.valid && session.engine == DmaEngine::SDMA, "TPUT_ASYNC_NOTIFY: SDMA requires a valid SDMA session.");
    PTO_ASSERT(
        notifyOp == NotifyOp::Set || notifyOp == NotifyOp::AtomicAdd,
        "TPUT_ASYNC_NOTIFY: SDMA notifyOp must be Set or AtomicAdd.");

    const uint64_t eventHandle = sdma::__sdma_put_async_notify(
        dstGlobalData.data(), srcGlobalData.data(), dstSignalData.data(), signalValue, notifyOp, transferBytes,
        session);
    return AsyncEvent(eventHandle, DmaEngine::SDMA);
}

} // namespace detail
} // namespace comm
} // namespace pto

#endif // PTO_COMM_TPUT_ASYNC_NOTIFY_COMMON_DETAIL_HPP
