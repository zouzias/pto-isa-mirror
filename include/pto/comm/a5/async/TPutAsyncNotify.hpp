/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "pto/comm/async_common/TPutAsyncNotifyCommonDetail.hpp"

namespace pto {
namespace comm {
namespace detail {

template <typename GlobalDstData, typename GlobalSrcData, typename GlobalSignalData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_NOTIFY_MTE_IMPL(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, GlobalSignalData& dstSignalData, int32_t signalValue,
    NotifyOp notifyOp, const AsyncSession& session)
{
    (void)TPutAsyncNotifyValidateTensors(dstGlobalData, srcGlobalData, dstSignalData);
    PTO_ASSERT(
        session.valid && session.engine == DmaEngine::SDMA, "TPUT_ASYNC_NOTIFY: SDMA requires a valid SDMA session.");
    PTO_ASSERT(
        notifyOp == NotifyOp::Set || notifyOp == NotifyOp::AtomicAdd,
        "TPUT_ASYNC_NOTIFY: SDMA notifyOp must be Set or AtomicAdd.");

    (void)TPUT_ASYNC_MTE_FALLBACK(dstGlobalData, srcGlobalData, session);
    pipe_barrier(PIPE_ALL);
    TNOTIFY_IMPL(dstSignalData, signalValue, notifyOp);
    return AsyncEvent(0, DmaEngine::SDMA);
}

#ifdef PTO_URMA_SUPPORTED
template <typename GlobalDstData, typename GlobalSrcData, typename GlobalSignalData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_NOTIFY_URMA_IMPL(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, GlobalSignalData& dstSignalData, int32_t signalValue,
    NotifyOp notifyOp, const AsyncSession& session, uint32_t peer)
{
    const uint64_t transferBytes = TPutAsyncNotifyValidateTensors(dstGlobalData, srcGlobalData, dstSignalData);
    PTO_ASSERT(
        session.valid && session.engine == DmaEngine::URMA, "TPUT_ASYNC_NOTIFY: URMA requires a valid URMA session.");
    PTO_ASSERT(
        notifyOp == NotifyOp::Set || notifyOp == NotifyOp::AtomicAdd,
        "TPUT_ASYNC_NOTIFY: URMA notifyOp must be Set or AtomicAdd.");
    PTO_ASSERT(
        transferBytes <= urma::kUrmaMaxWqeTransferBytes, "TPUT_ASYNC_NOTIFY: URMA payload must not exceed 256MB.");

    const uint64_t eventHandle = urma::__urma_put_async_notify(
        reinterpret_cast<__gm__ uint8_t*>(dstGlobalData.data()),
        reinterpret_cast<__gm__ uint8_t*>(srcGlobalData.data()), transferBytes,
        reinterpret_cast<__gm__ int32_t*>(dstSignalData.data()), signalValue, notifyOp, session, peer);
    return AsyncEvent(eventHandle, DmaEngine::URMA);
}
#endif

#ifdef PTO_RDMA_SUPPORTED
template <typename GlobalDstData, typename GlobalSrcData, typename GlobalSignalData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_NOTIFY_RDMA_IMPL(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, GlobalSignalData& dstSignalData, int32_t signalValue,
    NotifyOp notifyOp, const AsyncSession& session, uint32_t peer)
{
    const uint64_t transferBytes = TPutAsyncNotifyValidateTensors(dstGlobalData, srcGlobalData, dstSignalData);
    PTO_ASSERT(
        session.valid && session.engine == DmaEngine::RDMA, "TPUT_ASYNC_NOTIFY: RDMA requires a valid RDMA session.");
    PTO_ASSERT(notifyOp == NotifyOp::Set, "TPUT_ASYNC_NOTIFY: RDMA currently supports NotifyOp::Set only.");
    PTO_ASSERT(transferBytes <= 0x7fffffffULL, "TPUT_ASYNC_NOTIFY: RDMA payload must not exceed 0x7fffffff bytes.");

    const uint64_t eventHandle = rdma::WriteNotify(
        session, reinterpret_cast<__gm__ uint8_t*>(dstGlobalData.data()),
        reinterpret_cast<__gm__ uint8_t*>(srcGlobalData.data()), transferBytes,
        reinterpret_cast<__gm__ int32_t*>(dstSignalData.data()), signalValue, peer);
    return AsyncEvent(eventHandle, DmaEngine::RDMA);
}
#endif

} // namespace detail

template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData, typename GlobalSignalData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_NOTIFY_IMPL(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, GlobalSignalData& dstSignalData, int32_t signalValue,
    NotifyOp notifyOp, const AsyncSession& session, uint32_t peer)
{
    if constexpr (engine == DmaEngine::SDMA) {
        (void)peer;
        return detail::TPUT_ASYNC_NOTIFY_MTE_IMPL(
            dstGlobalData, srcGlobalData, dstSignalData, signalValue, notifyOp, session);
    } else if constexpr (engine == DmaEngine::URMA) {
#ifdef PTO_URMA_SUPPORTED
        return detail::TPUT_ASYNC_NOTIFY_URMA_IMPL(
            dstGlobalData, srcGlobalData, dstSignalData, signalValue, notifyOp, session, peer);
#else
        static_assert(engine != DmaEngine::URMA, "TPUT_ASYNC_NOTIFY: URMA requires NPU_ARCH 3510.");
        return AsyncEvent(0, engine);
#endif
    } else if constexpr (engine == DmaEngine::RDMA) {
#ifdef PTO_RDMA_SUPPORTED
        return detail::TPUT_ASYNC_NOTIFY_RDMA_IMPL(
            dstGlobalData, srcGlobalData, dstSignalData, signalValue, notifyOp, session, peer);
#else
        static_assert(engine != DmaEngine::RDMA, "TPUT_ASYNC_NOTIFY: RDMA support is not enabled for this build.");
        return AsyncEvent(0, engine);
#endif
    } else {
        static_assert(
            engine == DmaEngine::SDMA || engine == DmaEngine::URMA || engine == DmaEngine::RDMA,
            "TPUT_ASYNC_NOTIFY: unsupported DMA engine.");
        return AsyncEvent(0, engine);
    }
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_A5_TPUT_ASYNC_NOTIFY_HPP
