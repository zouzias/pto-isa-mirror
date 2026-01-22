// Test TWAIT and TTEST operations via PTO (Shmem backend)
// TWAIT: Blocking wait for signal condition
// TTEST: Non-blocking test of signal condition

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

// Function declarations implemented in twait_kernel.cpp

// TWAIT Basic: Rank 0 sends signal to rank 1, rank 1 waits (blocking)
bool RunTWaitBasic(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// TWAIT Compare: Test different comparison operators (GE, LE, etc.)
bool RunTWaitCompare(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int32_t notifyValue);

// TTEST Basic: Non-blocking signal test
bool RunTTestBasic(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// TTEST Polling: Polling loop with TTEST until condition met
bool RunTTestPolling(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// TWAIT Atomic: Wait for atomic counter to reach threshold
bool RunTWaitAtomic(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// ============================================================================
// TWAIT Tests
// ============================================================================

// Basic TWAIT test: wait for signal == expected value
TEST(TWait, Basic_2Ranks) { ASSERT_TRUE(RunTWaitBasic(2, 2, 0, 0)); }

// TWAIT with GE comparison: wait for signal >= 100
TEST(TWait, Compare_GE_2Ranks) { ASSERT_TRUE(RunTWaitCompare(2, 2, 0, 0, 150)); }

// TWAIT with atomic add: multiple ranks contribute, one waits for threshold
TEST(TWait, Atomic_2Ranks) { ASSERT_TRUE(RunTWaitAtomic(2, 2, 0, 0)); }
TEST(TWait, Atomic_4Ranks) { ASSERT_TRUE(RunTWaitAtomic(4, 4, 0, 0)); }

// ============================================================================
// TTEST Tests
// ============================================================================

// Basic TTEST: non-blocking check for signal condition
TEST(TTest, Basic_2Ranks) { ASSERT_TRUE(RunTTestBasic(2, 2, 0, 0)); }

// TTEST polling loop: poll until signal reaches target value
TEST(TTest, Polling_2Ranks) { ASSERT_TRUE(RunTTestPolling(2, 2, 0, 0)); }

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
