#include <gtest/gtest.h>

template <typename T, size_t count>
bool RunAllReduce(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

TEST(TAllReduce, FloatSmall) {
    EXPECT_TRUE((RunAllReduce<float, 256>(4, 4, 0, 0)));
}

TEST(TAllReduce, Int32Large) {
    EXPECT_TRUE((RunAllReduce<int32_t, 4096>(2, 2, 0, 0)));
}

TEST(TAllReduce, Int32Small) {
    EXPECT_TRUE((RunAllReduce<int32_t, 512>(8, 8, 0, 0)));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

