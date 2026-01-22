#ifndef PTO_COMM_INST_HPP
#define PTO_COMM_INST_HPP

#include "pto/comm/comm_types.hpp"
#include "pto/comm/pto_comm_instr_impl.hpp"

#define MAP_INSTR_IMPL(API, ...) API##_IMPL(__VA_ARGS__)

namespace pto {
namespace comm {

template < typename GlobalDstData, typename GlobalSrcData>
PTO_INST void TPUT(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal)
{
    MAP_INSTR_IMPL(TPUT, dstGlobal, srcGlobal);
}

template < typename GlobalDstData, typename GlobalSrcData>
PTO_INST void TGET(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal)
{
    MAP_INSTR_IMPL(TGET, dstGlobal, srcGlobal);
}

PTO_INST void TQUIET()
{
    MAP_INSTR_IMPL(TQUIET);
}

PTO_INST void TBARRIER()
{
    MAP_INSTR_IMPL(TBARRIER);
}


template <typename ParallelGroup, typename GlobalDstData>
PTO_INST void TALLREDUCE(ParallelGroup &parallelGroup, GlobalDstData &dstGlobal)
{
    MAP_INSTR_IMPL(TALLREDUCE, parallelGroup, dstGlobal);
}

template <typename ParallelGroup, typename GlobalDstData>
PTO_INST void TALLGATHER(ParallelGroup &parallelGroup, GlobalDstData &dstGlobal)
{
    MAP_INSTR_IMPL(TALLGATHER, parallelGroup, dstGlobal);
}

template <typename ParallelGroup, typename GlobalSrcData>
PTO_INST void TBROADCAST(ParallelGroup &parallelGroup, GlobalSrcData &srcGlobal, int root)
{
    MAP_INSTR_IMPL(TBROADCAST, parallelGroup, srcGlobal, root);
}

// ============================================================================
// TNOTIFY: Send flag notification to remote PE
// dstSignal's data() and GetRank() already contain target location and PE info
// Signal type is int32_t, compatible with shmem signal API
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
// signal contains the local signal to wait on
// Signal type is int32_t, compatible with shmem signal API
// ============================================================================

// Compile-time specified comparison (recommended, zero overhead)
// Default: wait until signal == cmpValue
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
// Compile-time specified comparison
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
// Used for polling-based synchronization with timeout or interleaved work
// ============================================================================

// Compile-time specified comparison (recommended, zero overhead)
// Returns true if signal meets condition
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
// Compile-time specified comparison
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

