// 验证通过 PTO TNOTIFY（Shmem 后端）进行多节点 flag 通告
// 测试 AtomicAdd 和 Set 两种模式

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

// 声明在 tnotify_kernel.cpp 中实现的测试函数

// 测试 AtomicAdd 模式：多个 rank 对同一计数器进行原子加
bool RunNotifyAtomicAdd(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// 测试 Set 模式：设置远端信号
bool RunNotifySet(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// 测试 Scoreboard 模式：使用指针偏移通知不同槽位
template <size_t numSlots>
bool RunNotifyScoreboard(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// AtomicAdd 测试
TEST(TNotify, AtomicAdd_2Ranks) { ASSERT_TRUE(RunNotifyAtomicAdd(2, 2, 0, 0)); }
TEST(TNotify, AtomicAdd_4Ranks) { ASSERT_TRUE(RunNotifyAtomicAdd(4, 4, 0, 0)); }

// Set 测试
TEST(TNotify, Set_2Ranks) { ASSERT_TRUE(RunNotifySet(2, 2, 0, 0)); }
TEST(TNotify, Set_4Ranks) { ASSERT_TRUE(RunNotifySet(4, 4, 0, 0)); }

// Scoreboard 测试
TEST(TNotify, Scoreboard_4Slots) { ASSERT_TRUE((RunNotifyScoreboard<4>(4, 4, 0, 0))); }
TEST(TNotify, Scoreboard_8Slots) { ASSERT_TRUE((RunNotifyScoreboard<8>(4, 4, 0, 0))); }

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
