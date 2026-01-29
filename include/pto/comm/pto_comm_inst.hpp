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

namespace pto {
namespace comm {

// ============================================================================
// TPUT: Remote write operation - write local data to remote PE's memory
// Note: UB tile must be pre-allocated by compiler
// ============================================================================

template <typename GlobalDstData, typename GlobalSrcData, typename TileData>
PTO_INST void TPUT(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal, TileData &ubTile)
{
    TPUT_IMPL(dstGlobal, srcGlobal, ubTile);
}

// ============================================================================
// TPUT_SDMA: Asynchronous remote write operation using SDMA engine
// Direct GM to GM transfer without UB staging
// Returns SdmaEvent for synchronization
// ============================================================================

template <typename GlobalDstData, typename GlobalSrcData>
PTO_INST SdmaEvent TPUT_SDMA(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal)
{
    return TPUT_SDMA_IMPL(dstGlobal, srcGlobal);
}

template <typename GlobalDstData, typename GlobalSrcData>
PTO_INST SdmaEvent TPUT_SDMA(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal,
                              uint32_t numRows, uint32_t numCols)
{
    return TPUT_SDMA_IMPL(dstGlobal, srcGlobal, numRows, numCols);
}

// ============================================================================
// TGET: Remote read operation - read remote PE's data to local memory
// Note: UB tile must be pre-allocated by compiler
// ============================================================================

template <typename GlobalDstData, typename GlobalSrcData, typename TileData>
PTO_INST void TGET(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal, TileData &ubTile)
{
    TGET_IMPL(dstGlobal, srcGlobal, ubTile);
}

// ============================================================================
// TGET_SDMA: Asynchronous remote read operation using SDMA engine
// Direct GM to GM transfer without UB staging
// Returns SdmaEvent for synchronization
// Data flow: srcGlobal (remote GM) -> dstGlobal (local GM)
// ============================================================================

template <typename GlobalDstData, typename GlobalSrcData>
PTO_INST SdmaEvent TGET_SDMA(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal)
{
    return TGET_SDMA_IMPL(dstGlobal, srcGlobal);
}

template <typename GlobalDstData, typename GlobalSrcData>
PTO_INST SdmaEvent TGET_SDMA(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal,
                              uint32_t numRows, uint32_t numCols)
{
    return TGET_SDMA_IMPL(dstGlobal, srcGlobal, numRows, numCols);
}

PTO_INST void TBARRIER()
{
    TBARRIER_IMPL();
}

template <typename GlobalSignalData>
PTO_INST void TBARRIER(GlobalSignalData *barrierSignals, int nranks, int my_rank)
{
    TBARRIER_IMPL(barrierSignals, nranks, my_rank);
}

// ============================================================================
// TALLREDUCE: All-reduce operation across parallel group
// Note: All UB tiles (accTile, pingTile, pongTile) must be pre-allocated by compiler
// ============================================================================

template <typename ParallelGroup, typename GlobalDstData, typename TileData>
PTO_INST void TALLREDUCE(ParallelGroup &parallelGroup, GlobalDstData &dstGlobal, 
                         TileData &accTile, TileData &pingTile, TileData &pongTile)
{
    TALLREDUCE_IMPL(parallelGroup, dstGlobal, accTile, pingTile, pongTile);
}

// ============================================================================
// TALLGATHER: All-gather operation across parallel group
// Note: UB tile must be pre-allocated by compiler
// ============================================================================

template <typename ParallelGroup, typename GlobalDstData, typename TileData>
PTO_INST void TALLGATHER(ParallelGroup &parallelGroup, GlobalDstData &dstGlobal, TileData &ubTile)
{
    TALLGATHER_IMPL(parallelGroup, dstGlobal, ubTile);
}

// ============================================================================
// TBROADCAST: Broadcast data from root rank to all ranks
// Note: UB tile must be pre-allocated by compiler
// ============================================================================

template <typename ParallelGroup, typename GlobalSrcData, typename TileData>
PTO_INST void TBROADCAST(ParallelGroup &parallelGroup, GlobalSrcData &srcGlobal, int root, TileData &ubTile)
{
    TBROADCAST_IMPL(parallelGroup, srcGlobal, root, ubTile);
}

// ============================================================================
// TNOTIFY: Send flag notification to remote PE
// Signal type is int32_t
// ============================================================================

// Compile-time specified NotifyOp (recommended, zero overhead)
template <NotifyOp op = NotifyOp::Set, typename GlobalSignalData>
PTO_INST void TNOTIFY(GlobalSignalData &dstSignal, int32_t value = 1)
{
    TNOTIFY_IMPL<op>(dstSignal, value);
}

// Runtime specified NotifyOp
template <typename GlobalSignalData>
PTO_INST void TNOTIFY(GlobalSignalData &dstSignal, int32_t value, NotifyOp op)
{
    TNOTIFY_IMPL(dstSignal, value, op);
}

// ============================================================================
// TWAIT: Wait until signal(s) meet comparison condition
// Used in conjunction with TNOTIFY for synchronization
// Signal type is int32_t
// ============================================================================

// Compile-time specified comparison (recommended, zero overhead)
template <WaitCmp cmp = WaitCmp::EQ, typename GlobalSignalData>
PTO_INST void TWAIT(GlobalSignalData &signal, int32_t cmpValue)
{
    TWAIT_IMPL<cmp>(signal, cmpValue);
}

// Runtime specified comparison
template <typename GlobalSignalData>
PTO_INST void TWAIT(GlobalSignalData &signal, WaitCmp cmp, int32_t cmpValue)
{
    TWAIT_IMPL(signal, cmp, cmpValue);
}

// Wait for all signals in array to meet condition
template <WaitCmp cmp = WaitCmp::EQ, typename GlobalSignalData>
PTO_INST void TWAIT_ALL(GlobalSignalData *signals, int count, int32_t cmpValue)
{
    TWAIT_ALL_IMPL<cmp>(signals, count, cmpValue);
}

// Runtime specified comparison
template <typename GlobalSignalData>
PTO_INST void TWAIT_ALL(GlobalSignalData *signals, int count, WaitCmp cmp, int32_t cmpValue)
{
    TWAIT_ALL_IMPL(signals, count, cmp, cmpValue);
}

// ============================================================================
// TTEST: Non-blocking test if signal(s) meet comparison condition
// Returns true if condition is satisfied, false otherwise
// ============================================================================

// Compile-time specified comparison (recommended, zero overhead)
template <WaitCmp cmp = WaitCmp::EQ, typename GlobalSignalData>
PTO_INST bool TTEST(GlobalSignalData &signal, int32_t cmpValue)
{
    return TTEST_IMPL<cmp>(signal, cmpValue);
}

// Runtime specified comparison
template <typename GlobalSignalData>
PTO_INST bool TTEST(GlobalSignalData &signal, WaitCmp cmp, int32_t cmpValue)
{
    return TTEST_IMPL(signal, cmp, cmpValue);
}

// Test all signals in array (returns true only if ALL meet condition)
template <WaitCmp cmp = WaitCmp::EQ, typename GlobalSignalData>
PTO_INST bool TTEST_ALL(GlobalSignalData *signals, int count, int32_t cmpValue)
{
    return TTEST_ALL_IMPL<cmp>(signals, count, cmpValue);
}

// Runtime specified comparison
template <typename GlobalSignalData>
PTO_INST bool TTEST_ALL(GlobalSignalData *signals, int count, WaitCmp cmp, int32_t cmpValue)
{
    return TTEST_ALL_IMPL(signals, count, cmp, cmpValue);
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_INST_HPP
