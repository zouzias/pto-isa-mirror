/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_A5_TPUT_ASYNC_NOTIFY_HPP
#define PTO_COMM_A5_TPUT_ASYNC_NOTIFY_HPP

#include "pto/comm/a5/TNotify.hpp"
#include "pto/comm/a5/async/TPutAsync.hpp"

namespace pto {
namespace comm {
namespace detail {

// A5 keeps DmaEngine::SDMA as an API-compatible name for the synchronous MTE
// fallback. Complete every payload chunk before issuing the Scalar notification,
// then return handle 0 to represent an already-completed operation.
//
// Intended ordering and concurrency semantics:
// - Repeated calls from one AICore are fully serialized.
// - AtomicAdd to one signal may aggregate completions from multiple AICores or
//   ranks, provided every payload uses a non-conflicting destination range.
// - Set does not define a winner when multiple AICores or ranks write one signal;
//   use one signal slot per producer or AtomicAdd for completion counting.
// Receiver-side visibility and multi-producer cases require dedicated hardware
// stress tests; the current ST covers sequential calls from one sender AICore.
template <typename GlobalDstData, typename GlobalSrcData, typename GlobalSignalData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_NOTIFY_MTE_FALLBACK(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, GlobalSignalData& dstSignalData, int32_t signalValue,
    NotifyOp notifyOp, const AsyncSession& session)
{
    static_assert(
        std::is_same_v<typename GlobalSignalData::RawDType, int32_t>, "TPUT_ASYNC_NOTIFY: signal type must be int32_t");

    PTO_ASSERT(
        session.valid && session.engine == DmaEngine::SDMA,
        "TPUT_ASYNC_NOTIFY: A5 MTE fallback requires a valid SDMA session.");
    PTO_ASSERT(dstSignalData.data() != nullptr, "TPUT_ASYNC_NOTIFY: signal pointer must not be null.");
    PTO_ASSERT(
        (reinterpret_cast<uint64_t>(dstSignalData.data()) & (alignof(int32_t) - 1U)) == 0U,
        "TPUT_ASYNC_NOTIFY: A5 Scalar signal address must be 4-byte aligned.");
    PTO_ASSERT(
        notifyOp == NotifyOp::Set || notifyOp == NotifyOp::AtomicAdd,
        "TPUT_ASYNC_NOTIFY: notifyOp must be Set or AtomicAdd.");

    // TPUT_ASYNC_MTE_FALLBACK uses MTE3 -> MTE2 events for UB reuse. Add an
    // explicit all-pipeline barrier before the Scalar notification so payload
    // completion does not rely on an implicit MTE-to-Scalar ordering assumption.
    (void)TPUT_ASYNC_MTE_FALLBACK(dstGlobalData, srcGlobalData, session);
    pipe_barrier(PIPE_ALL);
    TNOTIFY_IMPL(dstSignalData, signalValue, notifyOp);
    return AsyncEvent(0, DmaEngine::SDMA);
}

} // namespace detail

template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData, typename GlobalSignalData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_NOTIFY_IMPL(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, GlobalSignalData& dstSignalData, int32_t signalValue,
    NotifyOp notifyOp, const AsyncSession& session)
{
    if constexpr (engine == DmaEngine::SDMA) {
        return detail::TPUT_ASYNC_NOTIFY_MTE_FALLBACK(
            dstGlobalData, srcGlobalData, dstSignalData, signalValue, notifyOp, session);
    } else {
        static_assert(engine == DmaEngine::SDMA, "TPUT_ASYNC_NOTIFY: A5 URMA path is not implemented yet");
        return {};
    }
}

template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData, typename GlobalSignalData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_NOTIFY_IMPL(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, GlobalSignalData& dstSignalData, int32_t signalValue,
    NotifyOp notifyOp, const AsyncSession& session, uint32_t peer)
{
    (void)peer;
    return TPUT_ASYNC_NOTIFY_IMPL<engine>(dstGlobalData, srcGlobalData, dstSignalData, signalValue, notifyOp, session);
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_A5_TPUT_ASYNC_NOTIFY_HPP
