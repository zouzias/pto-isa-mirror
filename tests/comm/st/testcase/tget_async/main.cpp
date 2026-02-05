// Test TGET_ASYNC (remote read) operation via PTO with SHMEM backend
// Root rank reads data from all other ranks

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

// Declaration of 1D test functions implemented in tget_async_kernel.cpp
template <typename T, size_t count>
bool RunGetAsyncRootGet(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// ============================================================================
// 1D Vector Tile Tests
// ============================================================================
TEST(TGetAsync, Vec_FloatSmall) { ASSERT_TRUE((RunGetAsyncRootGet<float, 256>(2, 2, 0, 0))); }
TEST(TGetAsync, Vec_Int32Large) { ASSERT_TRUE((RunGetAsyncRootGet<int32_t, 4096>(2, 2, 0, 0))); }
TEST(TGetAsync, Vec_Uint8Small) { ASSERT_TRUE((RunGetAsyncRootGet<uint8_t, 512>(2, 2, 0, 0))); }

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
