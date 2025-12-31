#ifndef PTO_COMM_TALLGATHER_HPP
#define PTO_COMM_TALLGATHER_HPP

#include <type_traits>

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/comm/comm_types.hpp"
#include "pto/comm/backend/backend_selector.hpp"

namespace pto {
namespace comm {

template <BackendKind backend = BackendKind::Shmem, typename ParallelGroupType, typename GlobalDstData>
AICORE void TALLGATHER_IMPL(ParallelGroupType &srcPG, GlobalDstData &dstGlobal)
{
    // Use type extraction to get GlobalData type (if not ParallelGroup<...>, this will trigger a compile error)
    using GlobalData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    
    static_assert(std::is_same_v<GlobalData, typename GlobalDstData::GlobalDataType>, "GlobalData type mismatch!");

    // Check PG size 
    PTO_ASSERT(pg.size() > 1, "ParallelGroup size must be greater than 1!");
    // Check PG tensors have the same shape (runtime check)
    for (int i = 1; i < pg.size(); i++) {
        PTO_ASSERT(pg[i].GetShape(0) == pg[0].GetShape(0) &&
                   pg[i].GetShape(1) == pg[0].GetShape(1) &&
                   pg[i].GetShape(2) == pg[0].GetShape(2) &&
                   pg[i].GetShape(3) == pg[0].GetShape(3) &&
                   pg[i].GetShape(4) == pg[0].GetShape(4),
                   "All tensors in ParallelGroup must have the same shape!");
    }

    BackendSelector<backend>::type::template AllGather(pg, dstGlobal);
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TALLGATHER_HPP
