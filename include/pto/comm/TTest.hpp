#ifndef PTO_COMM_TTEST_HPP
#define PTO_COMM_TTEST_HPP

#include "pto/comm/comm_types.hpp"
#include "pto/comm/backend/backend_selector.hpp"
#include "pto/common/type.hpp"

namespace pto {
namespace comm {

// ============================================================================
// TTEST: Non-blocking test if signal(s) meet comparison condition
// Returns true if condition is satisfied, false otherwise
// Used for polling-based synchronization with timeout or interleaved work
// ============================================================================

// TTEST_IMPL: Test single signal condition (non-blocking)
// Returns true if signal meets condition, false otherwise
// - signal: GlobalTensor containing the signal to test (local memory)
// - cmpValue: Value to compare against
// Signal type must be int32_t (compatible with shmem signal API)

// Compile-time specified comparison (recommended, zero overhead)
template <WaitCmp cmp = WaitCmp::EQ, BackendKind backend = BackendKind::Shmem, typename GlobalSignalData>
AICORE bool TTEST_IMPL(GlobalSignalData &signal, int32_t cmpValue)
{
    static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
        "TTEST: signal type must be 32-bit (int32_t)");

    return BackendSelector<backend>::type::template SignalTest<cmp>(signal, cmpValue);
}

// Runtime specified comparison
template <BackendKind backend = BackendKind::Shmem, typename GlobalSignalData>
AICORE bool TTEST_IMPL(GlobalSignalData &signal, WaitCmp cmp, int32_t cmpValue)
{
    static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
        "TTEST: signal type must be 32-bit (int32_t)");

    return BackendSelector<backend>::type::SignalTest(signal, cmp, cmpValue);
}

// TTEST_ALL_IMPL: Test if all signals in array meet condition (non-blocking)
// Returns true only if ALL signals satisfy the condition
// - signals: Pointer to array of GlobalTensors containing signals
// - count: Number of signals in array
// - cmpValue: Value to compare against

// Compile-time specified comparison
template <WaitCmp cmp = WaitCmp::EQ, BackendKind backend = BackendKind::Shmem, typename GlobalSignalData>
AICORE bool TTEST_ALL_IMPL(GlobalSignalData *signals, int count, int32_t cmpValue)
{
    static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
        "TTEST_ALL: signal type must be 32-bit (int32_t)");

    return BackendSelector<backend>::type::template SignalTestAll<cmp>(signals, count, cmpValue);
}

// Runtime specified comparison
template <BackendKind backend = BackendKind::Shmem, typename GlobalSignalData>
AICORE bool TTEST_ALL_IMPL(GlobalSignalData *signals, int count, WaitCmp cmp, int32_t cmpValue)
{
    static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
        "TTEST_ALL: signal type must be 32-bit (int32_t)");

    return BackendSelector<backend>::type::SignalTestAll(signals, count, cmp, cmpValue);
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TTEST_HPP
