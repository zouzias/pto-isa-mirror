#ifndef PTO_COMM_TWAIT_HPP
#define PTO_COMM_TWAIT_HPP

#include "pto/comm/comm_types.hpp"
#include "pto/comm/backend/backend_selector.hpp"
#include "pto/common/type.hpp"

namespace pto {
namespace comm {

// ============================================================================
// TWAIT: Wait until signal(s) meet comparison condition
// Used in conjunction with TNOTIFY for synchronization
// Signal is stored in GlobalTensor data structure
// ============================================================================

// TWAIT_IMPL: Wait for single signal to meet condition
// - signal: GlobalTensor containing the signal to wait on (local memory)
// - cmpValue: Value to compare against
// Signal type must be int32_t (compatible with shmem signal API)

// Compile-time specified comparison (recommended, zero overhead)
template <WaitCmp cmp = WaitCmp::EQ, BackendKind backend = BackendKind::Shmem, typename GlobalSignalData>
AICORE void TWAIT_IMPL(GlobalSignalData &signal, int32_t cmpValue)
{
    static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
        "TWAIT: signal type must be 32-bit (int32_t)");

    BackendSelector<backend>::type::template SignalWait<cmp>(signal, cmpValue);
}

// Runtime specified comparison
template <BackendKind backend = BackendKind::Shmem, typename GlobalSignalData>
AICORE void TWAIT_IMPL(GlobalSignalData &signal, WaitCmp cmp, int32_t cmpValue)
{
    static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
        "TWAIT: signal type must be 32-bit (int32_t)");

    BackendSelector<backend>::type::SignalWait(signal, cmp, cmpValue);
}

// TWAIT_IMPL: Wait for all signals in array to meet condition
// - signals: Pointer to array of GlobalTensors containing signals
// - count: Number of signals in array
// - cmpValue: Value to compare against

// Compile-time specified comparison
template <WaitCmp cmp = WaitCmp::EQ, BackendKind backend = BackendKind::Shmem, typename GlobalSignalData>
AICORE void TWAIT_ALL_IMPL(GlobalSignalData *signals, int count, int32_t cmpValue)
{
    static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
        "TWAIT_ALL: signal type must be 32-bit (int32_t)");

    BackendSelector<backend>::type::template SignalWaitAll<cmp>(signals, count, cmpValue);
}

// Runtime specified comparison
template <BackendKind backend = BackendKind::Shmem, typename GlobalSignalData>
AICORE void TWAIT_ALL_IMPL(GlobalSignalData *signals, int count, WaitCmp cmp, int32_t cmpValue)
{
    static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
        "TWAIT_ALL: signal type must be 32-bit (int32_t)");

    BackendSelector<backend>::type::SignalWaitAll(signals, count, cmp, cmpValue);
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TWAIT_HPP
