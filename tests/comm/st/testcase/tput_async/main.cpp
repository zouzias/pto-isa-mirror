// 验证通过 PTO TPut Async（SDMA 后端）进行环形互传：从前一 rank 拉取数据
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

// 声明在 tput_async_kernel.cpp 中实现的设备侧逻辑（返回执行是否成功）
// 1D Vector Tile tests
template <typename T, size_t count>
bool RunPutAsyncRing(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// 2D Matrix Tile tests
template <typename T, size_t rows, size_t cols>
bool RunPutAsyncRing2D(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// ============================================================================
// 1D Vector Tile Tests
// ============================================================================
TEST(TPutAsync, Vec_FloatSmall) { ASSERT_TRUE((RunPutAsyncRing<float, 256>(4, 4, 0, 0))); }
TEST(TPutAsync, Vec_Int32Large) { ASSERT_TRUE((RunPutAsyncRing<int32_t, 4096>(2, 2, 0, 0))); }
TEST(TPutAsync, Vec_Uint8Small) { ASSERT_TRUE((RunPutAsyncRing<uint8_t, 512>(8, 8, 0, 0))); }

// ============================================================================
// 2D Shape Tests (GlobalTensor with 2D shape, using Vec Tile for transfer)
// ============================================================================
TEST(TPutAsync, Shape2D_Float16x16) { ASSERT_TRUE((RunPutAsyncRing2D<float, 16, 16>(2, 2, 0, 0))); }
TEST(TPutAsync, Shape2D_Float8x32) { ASSERT_TRUE((RunPutAsyncRing2D<float, 8, 32>(2, 2, 0, 0))); }
TEST(TPutAsync, Shape2D_Int32_4x64) { ASSERT_TRUE((RunPutAsyncRing2D<int32_t, 4, 64>(2, 2, 0, 0))); }

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
