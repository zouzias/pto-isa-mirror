// 验证对称内存分配/释放
#include "common.hpp"
#include <gtest/gtest.h>

TEST(ShmemMalloc, AllocFree)
{
    ShmemEnv env;
    ASSERT_TRUE(ShmemInitFromEnv(env));

    constexpr size_t count = 1024;
    int *buf = static_cast<int *>(shmem_malloc(count * sizeof(int)));
    ASSERT_NE(buf, nullptr);

    // 简单写入并 barrier，确保对称堆正常工作
    for (size_t i = 0; i < count; ++i) {
        buf[i] = static_cast<int>(env.rank * 1000 + i);
    }
    shmem_barrier_all();

    shmem_free(buf);
    shmem_barrier_all();
    ShmemFinalize();
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

