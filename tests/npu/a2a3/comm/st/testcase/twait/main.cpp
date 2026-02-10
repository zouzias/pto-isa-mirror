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

// TWAIT Atomic: Wait for atomic counter to reach threshold
bool RunTWaitAtomic(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// TWAIT Matrix: Wait on 2D signal matrix
template <int Rows, int Cols>
bool RunTWaitMatrix(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// TWAIT SubRegion: Wait on a sub-region of 2D signal matrix
template <int FullCols, int SubRows, int SubCols>
bool RunTWaitSubRegion(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// TWAIT Multi-Phase: rank 0 updates signal in phases, rank 1 waits in phases
bool RunTWaitMultiPhase(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

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

// TWAIT 2D signal matrix
TEST(TWait, Matrix2D_2Ranks) { ASSERT_TRUE((RunTWaitMatrix<4, 8>(2, 2, 0, 0))); }
TEST(TWait, Matrix2D_2Ranks_Large) { ASSERT_TRUE((RunTWaitMatrix<7, 13>(2, 2, 0, 0))); }

// TWAIT multi-phase update
TEST(TWait, MultiPhase_2Ranks) { ASSERT_TRUE(RunTWaitMultiPhase(2, 2, 0, 0)); }

// TWAIT sub-region of signal matrix
TEST(TWait, SubRegion_4x8_of_16) { ASSERT_TRUE((RunTWaitSubRegion<16, 4, 8>(2, 2, 0, 0))); }