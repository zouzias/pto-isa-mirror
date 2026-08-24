/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

    constexpr int kFifoDepth = 2;
    using CubeTile = Tile<TileType::Mat, T, rows, cols>;
    using VecTile = Tile<TileType::Vec, T, rows, cols>;
    using Pipe = TPipe<kPipeFlagId + 5, Direction::DIR_C2V, sizeof(T) * VecTile::Numel, kFifoDepth, 2, true>;

    NPU_MEMORY_CLEAR();
    Pipe::reset_for_cpu_sim();
    Pipe pipe((__gm__ void *)nullptr, kVecConsumerBase, 0x0);
    std::vector<std::vector<T>> lane0Actual(kFifoDepth + 1);
    std::vector<std::vector<T>> lane1Actual(kFifoDepth);

    {
        cpu_sim::ScopedExecutionContext producerCtx(0, 0, 1);
        for (int iter = 0; iter < kFifoDepth; ++iter) {
            CubeTile cubeTile;
            TASSIGN(cubeTile, 0x0);
            fillCubeTile<T, rows, cols>(cubeTile, iter);
            TPUSH<Pipe, CubeTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, cubeTile);
        }
    }

    {
        cpu_sim::ScopedExecutionContext lane0Ctx(0, 0, 2);
        for (int iter = 0; iter < kFifoDepth; ++iter) {
            VecTile vecTile;
            TASSIGN(vecTile, 0x4000);
            TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
            lane0Actual[iter].assign(vecTile.data(), vecTile.data() + vecTile.Numel);
        }
        for (int iter = 0; iter < kFifoDepth; ++iter) {
            TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
        }
    }

    std::atomic<bool> lane0ThirdPopDone{false};
    std::thread lane0ThirdPop([&]() {
        cpu_sim::ScopedExecutionContext lane0Ctx(0, 0, 2);
        VecTile vecTile;
        TASSIGN(vecTile, 0x4000);
        TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
        lane0Actual[kFifoDepth].assign(vecTile.data(), vecTile.data() + vecTile.Numel);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
        lane0ThirdPopDone.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(lane0ThirdPopDone.load(std::memory_order_acquire));

    {
        cpu_sim::ScopedExecutionContext lane1Ctx(0, 1, 2);
        for (int iter = 0; iter < kFifoDepth; ++iter) {
            VecTile vecTile;
            TASSIGN(vecTile, 0x8000);
            TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
            lane1Actual[iter].assign(vecTile.data(), vecTile.data() + vecTile.Numel);
            TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
        }
    }

    {
        cpu_sim::ScopedExecutionContext producerCtx(0, 0, 1);
        CubeTile cubeTile;
        TASSIGN(cubeTile, 0x0);
        fillCubeTile<T, rows, cols>(cubeTile, kFifoDepth);
        TPUSH<Pipe, CubeTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, cubeTile);
    }

    lane0ThirdPop.join();
    EXPECT_TRUE(lane0ThirdPopDone.load(std::memory_order_acquire));

    for (int iter = 0; iter < kFifoDepth + 1; ++iter) {
        const auto expected = makeExpected<T, rows, cols>(iter);
        EXPECT_TRUE(ResultCmp(expected, lane0Actual[iter], 0));
    }
    for (int iter = 0; iter < kFifoDepth; ++iter) {
        EXPECT_TRUE(std::all_of(lane1Actual[iter].begin(), lane1Actual[iter].end(),
                                [](T value) { return value == static_cast<T>(0); }));
    }

    auto &sharedState = Pipe::GetSharedState();
    std::lock_guard<std::mutex> lock(sharedState.mutex);
    EXPECT_EQ(sharedState.occupied, 1);
    EXPECT_EQ(sharedState.popped_not_freed_by_lane[0], 0);
    EXPECT_EQ(sharedState.popped_not_freed_by_lane[1], 0);
