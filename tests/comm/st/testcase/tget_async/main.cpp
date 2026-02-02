// Test TGET_ASYNC (remote read) operation via PTO with SHMEM backend
// Ring communication pattern: each rank reads data from next rank

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

// Declaration of 1D test functions implemented in tget_async_kernel.cpp
template <typename T, size_t count>
bool RunGetAsyncRing(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Declaration of 2D test functions implemented in tget_async_kernel.cpp
template <typename T, size_t rows, size_t cols>
bool RunGetAsyncRing2D(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// ============================================================================
// 1D Vector Tile Tests
// ============================================================================
TEST(TGetAsync, Vec_FloatSmall) { ASSERT_TRUE((RunGetAsyncRing<float, 256>(2, 2, 0, 0))); }
TEST(TGetAsync, Vec_Int32Large) { ASSERT_TRUE((RunGetAsyncRing<int32_t, 4096>(2, 2, 0, 0))); }
TEST(TGetAsync, Vec_Uint8Small) { ASSERT_TRUE((RunGetAsyncRing<uint8_t, 512>(2, 2, 0, 0))); }

// ============================================================================
// 2D Shape Tests (using Vec Tile with 2D shape)
// ============================================================================
TEST(TGetAsync, Shape2D_Float16x16) { ASSERT_TRUE((RunGetAsyncRing2D<float, 16, 16>(2, 2, 0, 0))); }
TEST(TGetAsync, Shape2D_Float8x32) { ASSERT_TRUE((RunGetAsyncRing2D<float, 8, 32>(2, 2, 0, 0))); }
TEST(TGetAsync, Shape2D_Int32_4x64) { ASSERT_TRUE((RunGetAsyncRing2D<int32_t, 4, 64>(2, 2, 0, 0))); }

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
