#include <gtest/gtest.h>

template <typename T, size_t count>
bool RunBroadCast(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int root);

TEST(TBroadCast, FloatSmallRoot0) {
    EXPECT_TRUE((RunBroadCast<float, 256>(4, 4, 0, 0, 0)));
}

TEST(TBroadCast, Int32LargeRoot1) {
    EXPECT_TRUE((RunBroadCast<int32_t, 4096>(2, 2, 0, 0, 1)));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


