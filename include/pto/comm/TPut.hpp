#ifndef PTO_COMM_TPUT_HPP
#define PTO_COMM_TPUT_HPP

#include "pto/comm/comm_types.hpp"
#include "pto/comm/backend/backend_selector.hpp"
#include "pto/common/type.hpp"

namespace pto {
namespace comm {
    template <BackendKind backend = BackendKind::Shmem, typename GlobalSrcData, typename GlobalDstData>
    AICORE void TPUT_IMPL(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal)
    {
        PTO_ASSERT(GlobalSrcData.GetShape(0) > 0 && GlobalSrcData.GetShape(1) > 0 && GlobalSrcData.GetShape(2) > 0 
        && GlobalSrcData.GetShape(3) > 0 && GlobalSrcData.GetShape(4) > 0 && 
        GlobalDstData.GetShape(0) > 0 && GlobalDstData.GetShape(1) > 0 && GlobalDstData.GetShape(2) > 0 
        && GlobalDstData.GetShape(3) > 0 && GlobalDstData.GetShape(4) > 0, 
        "The shape of src and dst must be greater than 0!");

        constexpr bool isSameLayout = (GlobalSrcData::layout == GlobalDstData::layout);
        PTO_ASSERT(isSameLayout, "Src and dst layout must be same!");

        BackendSelector<backend>::type::template Put(dstGlobal, srcGlobal);
    }
} // namespace comm
} // namespace pto

#endif // PTO_COMM_TPUT_HPP

