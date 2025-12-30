// 验证通过 PTO TGet（Shmem 后端）进行环形互传：从前一 rank 拉取数据
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

// 声明在 tget_kernel.cpp 中实现的设备侧逻辑（返回执行是否成功）
template <typename T, size_t count>
bool RunGetRing(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

TEST(TGet, FloatSmall) { ASSERT_TRUE((RunGetRing<float, 256>(4, 4, 0, 0))); }
TEST(TGet, Int32Large) { ASSERT_TRUE((RunGetRing<int32_t, 4096>(2, 2, 0, 0))); }
TEST(TGet, Uint8Small) { ASSERT_TRUE((RunGetRing<uint8_t, 512>(8, 8, 0, 0))); }

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

