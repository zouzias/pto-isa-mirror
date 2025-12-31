// 验证通过 PTO TGet（Shmem 后端）进行环形互传：从前一 rank 拉取数据
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

// 声明在 tget_kernel.cpp 中实现的设备侧逻辑（返回执行是否成功）
template <typename T, size_t count>
bool RunAllGather(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

TEST(TAllGatger, FloatSmall) { ASSERT_TRUE((RunAllGather<float, 256>(4, 4, 0, 0))); }
TEST(TAllGatger, Int32Large) { ASSERT_TRUE((RunAllGather<int32_t, 4096>(2, 2, 0, 0))); }
TEST(TAllGatger, Uint8Small) { ASSERT_TRUE((RunAllGather<uint8_t, 512>(8, 8, 0, 0))); }

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

