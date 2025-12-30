// 验证 shmem 初始化/结束，rank 检查
#include "common.hpp"
#include <gtest/gtest.h>

TEST(ShmemInit, Basic)
{
    ShmemEnv env;
    ASSERT_TRUE(ShmemInitFromEnv(env));
    EXPECT_EQ(shmem_my_pe(), env.rank);
    EXPECT_EQ(shmem_n_pes(), env.size);
    shmem_barrier_all();
    ShmemFinalize();
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

