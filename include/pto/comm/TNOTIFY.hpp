#ifndef PTO_COMM_TNOTIFY_HPP
#define PTO_COMM_TNOTIFY_HPP

#include "pto/comm/comm_types.hpp"
#include "pto/comm/backend/backend_selector.hpp"
#include "pto/common/type.hpp"

namespace pto {
namespace comm {

// NotifyOp is defined in comm_types.hpp

// TNOTIFY_IMPL: Send flag notification to remote PE
// dstSignal's data() pointer and GetRank() already contain target location info
// - dstSignal: Remote signal GlobalTensor (contains target address and PE info)
// - value: Value to set/accumulate
// Signal type must be int32_t (compatible with shmem signal API)
template <NotifyOp op = NotifyOp::Set, BackendKind backend = BackendKind::Shmem, typename GlobalSignalData>
AICORE void TNOTIFY_IMPL(GlobalSignalData &dstSignal, int32_t value = 1)
{
    // Type check via sizeof to ensure 32-bit type
    static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
        "TNOTIFY: signal type must be 32-bit (int32_t)");

    BackendSelector<backend>::type::template Notify<op>(dstSignal, value);
}

// TNOTIFY_IMPL: Runtime specified NotifyOp version
template <BackendKind backend = BackendKind::Shmem, typename GlobalSignalData>
AICORE void TNOTIFY_IMPL(GlobalSignalData &dstSignal, int32_t value, NotifyOp op)
{
    static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
        "TNOTIFY: signal type must be 32-bit (int32_t)");

    BackendSelector<backend>::type::Notify(dstSignal, value, op);
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TNOTIFY_HPP
