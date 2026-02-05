// 验证通过 PTO TPut Async（SDMA 后端）进行 root-put：root 向其他 rank 写入数据
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

// 声明在 tput_async_kernel.cpp 中实现的设备侧逻辑（返回执行是否成功）
// 1D Vector Tile tests (root puts to all other ranks)
template <typename T, size_t count>
bool RunPutAsyncRootPut(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// ============================================================================
// 1D Vector Tile Tests
// ============================================================================
TEST(TPutAsync, Vec_FloatSmall) { ASSERT_TRUE((RunPutAsyncRootPut<float, 256>(4, 4, 0, 0))); }
TEST(TPutAsync, Vec_Int32Large) { ASSERT_TRUE((RunPutAsyncRootPut<int32_t, 4096>(2, 2, 0, 0))); }
TEST(TPutAsync, Vec_Uint8Small) { ASSERT_TRUE((RunPutAsyncRootPut<uint8_t, 512>(8, 8, 0, 0))); }

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
