/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_INST_HPP
#define PTO_COMM_INST_HPP

#include "pto/comm/comm_types.hpp"
#include "pto/comm/pto_comm_instr_impl.hpp"
#include "pto/common/event.hpp"

namespace pto {
namespace comm {

// ============================================================================
// TPUT: Remote write operation - write local data to remote NPU's memory
// Data flow: srcGlobalData (local GM) → stagingTileData (UB) → dstGlobalData (remote GM)
// ============================================================================

template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TPUT(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData, 
                          TileData &stagingTileData, WaitEvents&... events)
{
    WaitAllEvents(events...);
    TPUT_IMPL(dstGlobalData, srcGlobalData, stagingTileData);
    return {};
}

// ============================================================================
// TGET: Remote read operation - read remote NPU's data to local memory
// Data flow: srcGlobalData (remote GM) → stagingTileData (UB) → dstGlobalData (local GM)
// ============================================================================

template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGET(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData, 
                          TileData &stagingTileData, WaitEvents&... events)
{
    WaitAllEvents(events...);
    TGET_IMPL(dstGlobalData, srcGlobalData, stagingTileData);
    return {};
}

// ============================================================================
// TPUT_ASYNC: Asynchronous remote write - direct GM to GM transfer
// Data flow: srcGlobalData (local GM) → DMA Engine → dstGlobalData (remote GM)
// 
// Template Parameters:
//   - engine: DmaEngine::SDMA (default) - System DMA, supports 2D transfer
//             DmaEngine::URMA - A5 URMA, supports 1D transfer
//
// Returns: AsyncEvent for synchronization with TSYNC
// ============================================================================

template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData, WaitEvents&... events)
{
    WaitAllEvents(events...);
    return TPUT_ASYNC_IMPL<engine>(dstGlobalData, srcGlobalData);
}

// ============================================================================
// TGET_ASYNC: Asynchronous remote read - direct GM to GM transfer
// Data flow: srcGlobalData (remote GM) → DMA Engine → dstGlobalData (local GM)
// 
// Template Parameters:
//   - engine: DmaEngine::SDMA (default) - System DMA, supports 2D transfer
//             DmaEngine::URMA - A5 URMA, supports 1D transfer
//
// Returns: AsyncEvent for synchronization with TSYNC
// ============================================================================

template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TGET_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData, WaitEvents&... events)
{
    WaitAllEvents(events...);
    return TGET_ASYNC_IMPL<engine>(dstGlobalData, srcGlobalData);
}

// ============================================================================
// TNOTIFY: Send flag notification to remote NPU
// Signal type must be int32_t
// ============================================================================

template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST void TNOTIFY(GlobalSignalData &dstSignalData, int32_t value, NotifyOp op, WaitEvents&... events)
{
    WaitAllEvents(events...);
    TNOTIFY_IMPL(dstSignalData, value, op);
}

// ============================================================================
// TWAIT: Blocking wait until signal(s) meet comparison condition
// Used in conjunction with TNOTIFY for flag-based synchronization
// Signal type must be int32_t
// 
// For signal matrix: Shape determines the 2D region to wait on. All signals must satisfy.
// ============================================================================

template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST void TWAIT(GlobalSignalData &signalData, int32_t cmpValue, WaitCmp cmp, WaitEvents&... events)
{
    WaitAllEvents(events...);
    TWAIT_IMPL(signalData, cmpValue, cmp);
}

// ============================================================================
// TTEST: Non-blocking test if signal(s) meet comparison condition
// Returns true if condition is satisfied, false otherwise
// 
// For signal matrix: Returns true only if ALL signals satisfy the condition.
// ============================================================================

template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST bool TTEST(GlobalSignalData &signalData, int32_t cmpValue, WaitCmp cmp, WaitEvents&... events)
{
    WaitAllEvents(events...);
    return TTEST_IMPL(signalData, cmpValue, cmp);
}

// ============================================================================
// TGATHER: Gather operation - root collects data from all ranks
// Only the root needs to execute. Non-root ranks ensure source buffers are ready.
// ============================================================================

template <typename ParallelGroupT, typename GlobalDstData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(ParallelGroupT &parallelGroup, GlobalDstData &dstGlobalData, 
                             TileData &stagingTileData, WaitEvents&... events)
{
    WaitAllEvents(events...);
    ::pto::comm::TGATHER_IMPL(parallelGroup, dstGlobalData, stagingTileData);
    return {};
}

// ============================================================================
// TSCATTER: Scatter operation - root distributes data to all ranks
// Only the root needs to execute. Non-root ranks ensure destination buffers are allocated.
// ============================================================================

template <typename ParallelGroupT, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(ParallelGroupT &parallelGroup, GlobalSrcData &srcGlobalData, 
                              TileData &stagingTileData, WaitEvents&... events)
{
    WaitAllEvents(events...);
    ::pto::comm::TSCATTER_IMPL(parallelGroup, srcGlobalData, stagingTileData);
    return {};
}

// ============================================================================
// TBROADCAST: Broadcast data from current NPU (root) to all ranks
// The calling NPU (parallelGroup.my_rank) is the root.
// Only the root needs to execute.
// ============================================================================

template <typename ParallelGroupT, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TBROADCAST(ParallelGroupT &parallelGroup, GlobalSrcData &srcGlobalData, 
                                TileData &stagingTileData, WaitEvents&... events)
{
    WaitAllEvents(events...);
    TBROADCAST_IMPL(parallelGroup, srcGlobalData, stagingTileData);
    return {};
}

// ============================================================================
// TREDUCE: Reduce operation - root gathers and reduces data from all ranks
// Only the root needs to execute. Non-root ranks ensure source buffers are ready.
// ============================================================================

template <typename ParallelGroupT, typename GlobalDstData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TREDUCE(ParallelGroupT &parallelGroup, GlobalDstData &dstGlobalData, 
                             TileData &accTileData, TileData &recvTileData, 
                             ReduceOp op, WaitEvents&... events)
{
    WaitAllEvents(events...);
    TREDUCE_IMPL(parallelGroup, dstGlobalData, accTileData, recvTileData, op);
    return {};
}

// ============================================================================
// TALLREDUCE: Reduce operation with ping-pong double buffering
// Only the root needs to execute. Non-root ranks ensure source buffers are ready.
// ============================================================================

template <typename ParallelGroupT, typename GlobalDstData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TREDUCE_PINGPONG(ParallelGroupT &parallelGroup, GlobalDstData &dstGlobalData, 
                                     TileData &accTileData, TileData &pingTileData, TileData &pongTileData, 
                                     ReduceOp op, WaitEvents&... events)
{
    WaitAllEvents(events...);
    TREDUCE_PINGPONG_IMPL(parallelGroup, dstGlobalData, accTileData, pingTileData, pongTileData, op);
    return {};
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_INST_HPP
