#include <gtest/gtest.h>

template <typename T, size_t count>
bool RunAllReduce(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

TEST(TAllReduce, FloatSmall2Rank) {
    EXPECT_TRUE((RunAllReduce<float, 256>(2, 2, 0, 0)));
}

TEST(TAllReduce, Int32Large2Rank) {
    EXPECT_TRUE((RunAllReduce<int32_t, 4096>(2, 2, 0, 0)));
}

TEST(TAllReduce, Int32Small2Rank) {
    EXPECT_TRUE((RunAllReduce<int32_t, 512>(2, 2, 0, 0)));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

