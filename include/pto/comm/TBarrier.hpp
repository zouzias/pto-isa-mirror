#ifndef PTO_COMM_TBARRIER_HPP
#define PTO_COMM_TBARRIER_HPP

#include "pto/comm/backend/backend_selector.hpp"

namespace pto {
namespace comm {

template <BackendKind backend = BackendKind::Shmem>
AICORE void TBARRIER_IMPL()
{
    BackendSelector<backend>::type::Barrier();
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TBARRIER_HPP

