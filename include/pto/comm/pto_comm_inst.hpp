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


} // namespace comm
} // namespace pto

#endif // PTO_COMM_INST_HPP

