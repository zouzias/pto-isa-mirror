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

PTO_INST void TWAIT()
{
    MAP_INSTR_IMPL(TWAIT);
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

} // namespace comm
} // namespace pto

#endif // PTO_COMM_INST_HPP

