/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TPUT_ASYNC_COMPAT_HPP
#define PTO_COMM_TPUT_ASYNC_COMPAT_HPP

namespace pto {
namespace comm {

#if defined(PTO_NPU_ARCH_A2A3)
template <
    DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData, typename FirstWaitEvent,
    typename... WaitEvents>
PTO_INST std::enable_if_t<!std::is_same_v<std::decay_t<FirstWaitEvent>, AsyncPutMode>, AsyncEvent> TPUT_ASYNC(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, const AsyncSession& session, FirstWaitEvent& firstEvent,
    WaitEvents&... events)
{
    return TPUT_ASYNC<engine>(dstGlobalData, srcGlobalData, session, AsyncPutMode::IMMEDIATE, firstEvent, events...);
}
#elif defined(PTO_NPU_ARCH_A5)
template <
    DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData, typename FirstWaitEvent,
    typename... WaitEvents>
PTO_INST std::enable_if_t<
    !std::is_same_v<std::decay_t<FirstWaitEvent>, AsyncPutMode> &&
        !std::is_same_v<std::decay_t<FirstWaitEvent>, uint32_t>,
    AsyncEvent>
TPUT_ASYNC(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, const AsyncSession& session, FirstWaitEvent& firstEvent,
    WaitEvents&... events)
{
    return TPUT_ASYNC<engine>(
        dstGlobalData, srcGlobalData, session, AsyncPutMode::IMMEDIATE, 0U, firstEvent, events...);
}

template <
    DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData, typename FirstWaitEvent,
    typename... WaitEvents>
PTO_INST std::enable_if_t<!std::is_same_v<std::decay_t<FirstWaitEvent>, AsyncPutMode>, AsyncEvent> TPUT_ASYNC(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, const AsyncSession& session, uint32_t peer,
    FirstWaitEvent& firstEvent, WaitEvents&... events)
{
    return TPUT_ASYNC<engine>(
        dstGlobalData, srcGlobalData, session, peer, AsyncPutMode::IMMEDIATE, 0U, firstEvent, events...);
}
#endif

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TPUT_ASYNC_COMPAT_HPP
