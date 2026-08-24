/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

    constexpr int kRows = 16;
    constexpr int kCols = 128;
    constexpr int kLaneRows = kRows / 2;
    constexpr int kSlotBytes = 8192;
    constexpr int kFifoDepth = 4;
    using Pipe = TPipe<30, Direction::DIR_BOTH, kSlotBytes, kFifoDepth, 4, false>;
    using CubeProdTile = Tile<TileType::Acc, float, kRows, kCols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;
    using VecTile = Tile<TileType::Vec, float, kLaneRows, kCols, BLayout::RowMajor, kLaneRows, kCols>;
    using CubeConsTile = Tile<TileType::Mat, float, kRows, kCols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;

    NPU_MEMORY_CLEAR();
    Pipe::reset_for_cpu_sim();
    std::vector<uint8_t> gmStorage(static_cast<std::size_t>(Pipe::RingFiFo::SLOT_SIZE) * Pipe::RingFiFo::SLOT_NUM);
    Pipe pipe(reinterpret_cast<__gm__ void *>(gmStorage.data()), 0x0, 0x0);

    auto pushC2V = [&](int base) {
        cpu_sim::ScopedExecutionContext cubeCtx(0, 0, 1);
        CubeProdTile cubeTile(kRows, kCols);
        TASSIGN(cubeTile, 0x0);
        fillFullTile(cubeTile, base);
        TPUSH<Pipe, CubeProdTile, TileSplitAxis::TILE_UP_DOWN>(pipe, cubeTile);
    };

    auto popC2V = [&](int base) {
        for (int lane = 0; lane < 2; ++lane) {
            cpu_sim::ScopedExecutionContext laneCtx(0, lane, 2);
            VecTile vecTile;
            TASSIGN(vecTile, 0x4000 + lane * 0x4000);
            TPOP<Pipe, VecTile, TileSplitAxis::TILE_UP_DOWN>(pipe, vecTile);
            const auto actual = readFullTile(vecTile);
            const auto expected = makeLinearExpected<float, kRows, kCols>(base, lane * kLaneRows, kLaneRows);
            EXPECT_TRUE(ResultCmp(expected, actual, 0));
            TFREE<Pipe, TileSplitAxis::TILE_UP_DOWN>(pipe);
        }
    };

    auto pushV2C = [&]() {
        for (int lane = 0; lane < 2; ++lane) {
            cpu_sim::ScopedExecutionContext laneCtx(0, lane, 2);
            VecTile vecTile;
            TASSIGN(vecTile, 0x4000 + lane * 0x4000);
            fillFullTile(vecTile, 7000 + lane * 1000);
            TPUSH<Pipe, VecTile, TileSplitAxis::TILE_UP_DOWN>(pipe, vecTile);
        }
    };

    auto popV2C = [&]() {
        cpu_sim::ScopedExecutionContext cubeCtx(0, 0, 1);
        CubeConsTile cubeTile(kRows, kCols);
        TASSIGN(cubeTile, 0x0);
        TPOP<Pipe, CubeConsTile, TileSplitAxis::TILE_UP_DOWN>(pipe, cubeTile);
        const auto actual = readFullTile(cubeTile);
        std::vector<float> expected;
        const auto upper = makeLinearExpected<float, kLaneRows, kCols>(7000, 0, kLaneRows);
        const auto lower = makeLinearExpected<float, kLaneRows, kCols>(8000, 0, kLaneRows);
        expected.insert(expected.end(), upper.begin(), upper.end());
        expected.insert(expected.end(), lower.begin(), lower.end());
        EXPECT_TRUE(ResultCmp(expected, actual, 0));
        TFREE<Pipe, TileSplitAxis::TILE_UP_DOWN>(pipe);
    };

    pushC2V(1000);
    pushC2V(2000);
    popC2V(1000);
    pushV2C();
    popV2C();
    pushC2V(3000);
    popC2V(2000);
    popC2V(3000);

    auto &sharedState = Pipe::GetSharedState();
    std::lock_guard<std::mutex> lock(sharedState.mutex);
    EXPECT_EQ(sharedState.occupied, 0);
