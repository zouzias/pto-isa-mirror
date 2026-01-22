#ifndef PTO_COMM_TQUIET_HPP
#define PTO_COMM_TQUIET_HPP

#include "pto/comm/backend/backend_selector.hpp"

namespace pto {
namespace comm {

template <BackendKind backend = BackendKind::Shmem>
AICORE void TQUIET_IMPL()
{
    BackendSelector<backend>::type::Quiet();
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TQUIET_HPP

