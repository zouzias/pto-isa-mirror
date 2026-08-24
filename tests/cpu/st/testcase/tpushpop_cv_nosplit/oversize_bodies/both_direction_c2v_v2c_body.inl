/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

    constexpr int kFifoDepth = 4;
    using CubeProdTile = Tile<TileType::Acc, T, rows, cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;
    using VecConsTile = Tile<TileType::Vec, T, rows, cols>;
    using VecProdTile = Tile<TileType::Vec, T, rows, cols>;
    using CubeConsTile = Tile<TileType::Mat, T, rows, cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    using Pipe = TPipe<kPipeFlagId + 8, Direction::DIR_BOTH, sizeof(T) * VecConsTile::Numel, kFifoDepth, 2, true>;

    NPU_MEMORY_CLEAR();
    Pipe::reset_for_cpu_sim();
    std::vector<uint8_t> gmSlotStorage(Pipe::RingFiFo::SLOT_SIZE * Pipe::RingFiFo::SLOT_NUM);
    auto *gmSlotBuffer = reinterpret_cast<__gm__ void *>(gmSlotStorage.data());
    Pipe pipe(gmSlotBuffer, kVecConsumerBase, kVecConsumerBase + 0x4000);
    std::vector<T> c2vActual(VecConsTile::Numel);
    std::vector<T> secondC2vActual(VecConsTile::Numel);
    std::vector<T> v2cActual(CubeConsTile::Numel);

    auto pushCube = [&](int iter) {
        cpu_sim::ScopedExecutionContext cubeCtx(0, 0, 1);
        CubeProdTile cubeTile(rows, cols);
        TASSIGN(cubeTile, 0x0);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                cubeTile.data()[GetTileElementOffset<CubeProdTile>(r, c)] =
                    static_cast<T>(iter * 1000 + r * cols + c + 1);
            }
        }
        TPUSH<Pipe, CubeProdTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, cubeTile);
    };

    pushCube(0);
    {
        cpu_sim::ScopedExecutionContext lane0Ctx(0, 0, 2);
        VecConsTile vecTile;
        TASSIGN(vecTile, 0x4000);
        TPOP<Pipe, VecConsTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
        c2vActual.assign(vecTile.data(), vecTile.data() + vecTile.Numel);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
    }

    {
        cpu_sim::ScopedExecutionContext lane1Ctx(0, 1, 2);
        VecConsTile vecTile;
        TASSIGN(vecTile, 0x8000);
        TPOP<Pipe, VecConsTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
    }

    {
        cpu_sim::ScopedExecutionContext vecCtx(0, 0, 2);
        VecProdTile vecTile;
        TASSIGN(vecTile, 0x4000);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                vecTile.data()[GetTileElementOffset<VecProdTile>(r, c)] = static_cast<T>(1000 + r * cols + c + 1);
            }
        }
        TPUSH<Pipe, VecProdTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
    }

    pushCube(1);
    {
        cpu_sim::ScopedExecutionContext lane0Ctx(0, 0, 2);
        VecConsTile vecTile;
        TASSIGN(vecTile, 0x4000);
        TPOP<Pipe, VecConsTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
        secondC2vActual.assign(vecTile.data(), vecTile.data() + vecTile.Numel);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
    }

    {
        cpu_sim::ScopedExecutionContext lane1Ctx(0, 1, 2);
        VecConsTile vecTile;
        TASSIGN(vecTile, 0x8000);
        TPOP<Pipe, VecConsTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
    }

    {
        cpu_sim::ScopedExecutionContext cubeCtx(0, 0, 1);
        CubeConsTile cubeTile(rows, cols);
        TASSIGN(cubeTile, 0x0);
        TPOP<Pipe, CubeConsTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, cubeTile);
        v2cActual.assign(cubeTile.data(), cubeTile.data() + cubeTile.Numel);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
    }

    std::vector<T> c2vExpected(VecConsTile::Numel);
    std::vector<T> secondC2vExpected(VecConsTile::Numel);
    std::vector<T> v2cExpected(CubeConsTile::Numel);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            c2vExpected[r * cols + c] = static_cast<T>(r * cols + c + 1);
            secondC2vExpected[r * cols + c] = static_cast<T>(1000 + r * cols + c + 1);
            v2cExpected[GetTileElementOffset<CubeConsTile>(r, c)] = static_cast<T>(1000 + r * cols + c + 1);
        }
    }
    EXPECT_TRUE(ResultCmp(c2vExpected, c2vActual, 0));
    EXPECT_TRUE(ResultCmp(secondC2vExpected, secondC2vActual, 0));
    EXPECT_TRUE(ResultCmp(v2cExpected, v2cActual, 0));

    auto &sharedState = Pipe::GetSharedState();
    std::lock_guard<std::mutex> lock(sharedState.mutex);
    EXPECT_EQ(sharedState.occupied, 0);
