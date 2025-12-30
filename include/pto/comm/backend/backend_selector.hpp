#ifndef PTO_COMM_DETAIL_BACKEND_SELECTOR_HPP
#define PTO_COMM_DETAIL_BACKEND_SELECTOR_HPP

#include "pto/comm/backend/shmem/shmem_backend.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

template <BackendKind backend>
struct BackendSelector;

template <>
struct BackendSelector<BackendKind::Shmem> {
    using type = backend::ShmemBackend;
};

} // namespace comm
} // namespace pto

#endif // PTO_COMM_DETAIL_BACKEND_SELECTOR_HPP





