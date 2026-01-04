#ifndef PTO_COMM_TBROADCAST_HPP
#define PTO_COMM_TBROADCAST_HPP

#include "pto/comm/comm_types.hpp"
#include "pto/comm/backend/backend_selector.hpp"

namespace pto {
namespace comm {

template <BackendKind backend = BackendKind::Shmem, typename ParallelGroupType, typename GlobalSrcData>
AICORE void TBROADCAST_IMPL(ParallelGroupType &pg, GlobalSrcData &srcGlobal, int root)
{
    BackendSelector<backend>::type::template BroadCast(pg, srcGlobal, root);
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TBROADCAST_HPP

