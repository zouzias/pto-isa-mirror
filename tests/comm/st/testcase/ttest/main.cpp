// Test TTEST (non-blocking signal test) operations via PTO (Shmem backend)
// TTEST returns true if signal meets comparison condition, false otherwise

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include "pto/comm/comm_types.hpp"

// Function declarations implemented in ttest_kernel.cpp

// TTEST True: Test returns true when condition is met
bool RunTTestTrue(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// TTEST False: Test returns false when condition is not met
bool RunTTestFalse(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// TTEST Compare: Test with different comparison operators
template <pto::comm::WaitCmp cmp>
bool RunTTestCompare(int n_ranks, int n_devices, int first_rank_id, int first_device_id,
                     int32_t signalValue, int32_t cmpValue, bool expectedResult);

// TTEST Polling with Timeout: Polling loop pattern
bool RunTTestPollingTimeout(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// TTEST NE: Test not-equal comparison
bool RunTTestNE(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// ============================================================================
// TTEST Basic Tests
// ============================================================================

// Test TTEST returns true when signal == expected value
TEST(TTest, True_EQ) { ASSERT_TRUE(RunTTestTrue(2, 2, 0, 0)); }

// Test TTEST returns false when signal != expected value
TEST(TTest, False_EQ) { ASSERT_TRUE(RunTTestFalse(2, 2, 0, 0)); }

// Test TTEST with NE (not equal) comparison
TEST(TTest, NE_True) { ASSERT_TRUE(RunTTestNE(2, 2, 0, 0)); }

// ============================================================================
// TTEST Comparison Operator Tests
// ============================================================================

// GE (>=): signal=100, test >= 50 -> should be true
TEST(TTest, GE_True) {
    ASSERT_TRUE((RunTTestCompare<pto::comm::WaitCmp::GE>(2, 2, 0, 0, 100, 50, true)));
}

// GE (>=): signal=100, test >= 100 -> should be true (equal case)
TEST(TTest, GE_Equal) {
    ASSERT_TRUE((RunTTestCompare<pto::comm::WaitCmp::GE>(2, 2, 0, 0, 100, 100, true)));
}

// GT (>): signal=100, test > 50 -> should be true
TEST(TTest, GT_True) {
    ASSERT_TRUE((RunTTestCompare<pto::comm::WaitCmp::GT>(2, 2, 0, 0, 100, 50, true)));
}

// GT (>): signal=100, test > 100 -> should be false (equal, not greater)
TEST(TTest, GT_False) {
    ASSERT_TRUE((RunTTestCompare<pto::comm::WaitCmp::GT>(2, 2, 0, 0, 100, 100, false)));
}

// LE (<=): signal=50, test <= 100 -> should be true
TEST(TTest, LE_True) {
    ASSERT_TRUE((RunTTestCompare<pto::comm::WaitCmp::LE>(2, 2, 0, 0, 50, 100, true)));
}

// LT (<): signal=50, test < 100 -> should be true
TEST(TTest, LT_True) {
    ASSERT_TRUE((RunTTestCompare<pto::comm::WaitCmp::LT>(2, 2, 0, 0, 50, 100, true)));
}

// LT (<): signal=100, test < 100 -> should be false (equal, not less)
TEST(TTest, LT_False) {
    ASSERT_TRUE((RunTTestCompare<pto::comm::WaitCmp::LT>(2, 2, 0, 0, 100, 100, false)));
}

// ============================================================================
// TTEST Polling Pattern Tests
// ============================================================================

// Test polling loop with TTEST until signal arrives
TEST(TTest, PollingTimeout) { ASSERT_TRUE(RunTTestPollingTimeout(2, 2, 0, 0)); }

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
