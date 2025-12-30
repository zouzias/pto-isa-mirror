#ifndef PTO_COMM_TWAIT_HPP
#define PTO_COMM_TWAIT_HPP

#include "pto/comm/backend/backend_selector.hpp"

namespace pto {
namespace comm {

template <BackendKind backend = BackendKind::Shmem>
AICORE void TWAIT_IMPL()
{
    BackendSelector<backend>::type::Wait();
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TWAIT_HPP

