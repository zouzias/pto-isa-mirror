// 验证通过 PTO TPut Async（SDMA 后端）进行 root-put：root 向其他 rank 写入数据
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

#include "tput_async_kernel.h"

// ============================================================================
// 1D Vector Tile Tests
// ============================================================================
TEST(TPutAsync, Vec_FloatSmall)
{
    ASSERT_TRUE((RunPutAsyncRootPut<float, 256>(4, 4, 0, 0)));
}
TEST(TPutAsync, Vec_Int32Large)
{
    ASSERT_TRUE((RunPutAsyncRootPut<int32_t, 4096>(2, 2, 0, 0)));
}
TEST(TPutAsync, Vec_Uint8Small)
{
    ASSERT_TRUE((RunPutAsyncRootPut<uint8_t, 512>(8, 8, 0, 0)));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
