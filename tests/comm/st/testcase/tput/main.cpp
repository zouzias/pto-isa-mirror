// 验证通过 PTO TPut（Shmem 后端）进行环形互传：从前一 rank 拉取数据
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

// 声明在 tput_kernel.cpp 中实现的设备侧逻辑（返回执行是否成功）
template <typename T, size_t count>
bool RunPutRing(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

TEST(TPut, FloatSmall) { ASSERT_TRUE((RunPutRing<float, 256>(4, 4, 0, 0))); }
TEST(TPut, Int32Large) { ASSERT_TRUE((RunPutRing<int32_t, 4096>(2, 2, 0, 0))); }
TEST(TPut, Uint8Small) { ASSERT_TRUE((RunPutRing<uint8_t, 512>(8, 8, 0, 0))); }

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

