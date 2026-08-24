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
    using CubeProdTile = Tile<TileType::Acc, T, rows, cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;
    using VecConsTile = Tile<TileType::Vec, T, rows, cols>;
    using VecProdTile = Tile<TileType::Vec, T, rows, cols>;
    using CubeConsTile = Tile<TileType::Mat, T, rows, cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    using Pipe = TPipe<kPipeFlagId + 9, Direction::DIR_BOTH, sizeof(T) * VecConsTile::Numel, kFifoDepth, 2, true>;

    NPU_MEMORY_CLEAR();
    Pipe::reset_for_cpu_sim();
    std::vector<uint8_t> gmSlotStorage(Pipe::RingFiFo::SLOT_SIZE * Pipe::RingFiFo::SLOT_NUM);
    auto *gmSlotBuffer = reinterpret_cast<__gm__ void *>(gmSlotStorage.data());
    Pipe pipe(gmSlotBuffer, kVecConsumerBase, kVecConsumerBase + 0x4000);

    // base: X=1, Y=1001, Z=2001 so each C2V tile is distinguishable per slot.
    auto pushCube = [&](int base) {
        cpu_sim::ScopedExecutionContext cubeCtx(0, 0, 1);
        CubeProdTile cubeTile(rows, cols);
        TASSIGN(cubeTile, 0x0);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                cubeTile.data()[GetTileElementOffset<CubeProdTile>(r, c)] = static_cast<T>(base + r * cols + c);
            }
        }
        TPUSH<Pipe, CubeProdTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, cubeTile);
    };
    auto popLaneFree = [&](uint32_t subblock, uint32_t gmOffset, std::vector<T> *capture) {
        cpu_sim::ScopedExecutionContext laneCtx(0, subblock, 2);
        VecConsTile vecTile;
        TASSIGN(vecTile, gmOffset);
        TPOP<Pipe, VecConsTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
        if (capture != nullptr) {
            capture->assign(vecTile.data(), vecTile.data() + vecTile.Numel);
        }
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
    };

    // Leading V2C push (vector lane 0) consumed by the cube: offsets the producer
    // cursor to slot 1 while the consumer lane cursors stay at slot 0.
    {
        cpu_sim::ScopedExecutionContext vecCtx(0, 0, 2);
        VecProdTile vecTile;
        TASSIGN(vecTile, 0x4000);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                vecTile.data()[GetTileElementOffset<VecProdTile>(r, c)] = static_cast<T>(5000 + r * cols + c);
            }
        }
        TPUSH<Pipe, VecProdTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
    }
    {
        cpu_sim::ScopedExecutionContext cubeCtx(0, 0, 1);
        CubeConsTile cubeTile(rows, cols);
        TASSIGN(cubeTile, 0x0);
        TPOP<Pipe, CubeConsTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, cubeTile);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
    }

    pushCube(1);    // X -> slot1
    pushCube(1001); // Y -> slot0 ; ring now full

    std::atomic<bool> zDone{false};
    std::thread zPush([&]() {
        pushCube(2001); // Z must wait until slot1 is free
        zDone.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    EXPECT_FALSE(zDone.load(std::memory_order_acquire));

    // Both lanes drain slot0 (Y). This frees slot0 and drops occupied below the
    // cap, but the cursor still points at slot1 (X), which no lane has consumed.
    std::vector<T> yActual;
    popLaneFree(0, 0x4000, &yActual);
    popLaneFree(1, 0x8000, nullptr);

    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    // slot1 (X) is still unconsumed, so Z must remain blocked. If Z has already
    // completed it reused slot1 and overwrote X; record the failure and stop
    // before the corrupted ring deadlocks the remaining pops (fail fast, no hang).
    const bool zStoleSlot = zDone.load(std::memory_order_acquire);
    EXPECT_FALSE(zStoleSlot) << "producer reused an out-of-order-freed slot (slot1/X)";
    if (zStoleSlot) {
        zPush.join();
        return;
    }

    // Both lanes drain slot1 (X). Only now is slot1 free and Z may proceed.
    std::vector<T> xActual;
    popLaneFree(0, 0x4000, &xActual);
    popLaneFree(1, 0x8000, nullptr);

    zPush.join();
    EXPECT_TRUE(zDone.load(std::memory_order_acquire));

    // Drain Z from slot1.
    std::vector<T> zActual;
    popLaneFree(0, 0x4000, &zActual);
    popLaneFree(1, 0x8000, nullptr);

    std::vector<T> xExpected(VecConsTile::Numel);
    std::vector<T> yExpected(VecConsTile::Numel);
    std::vector<T> zExpected(VecConsTile::Numel);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            xExpected[r * cols + c] = static_cast<T>(1 + r * cols + c);
            yExpected[r * cols + c] = static_cast<T>(1001 + r * cols + c);
            zExpected[r * cols + c] = static_cast<T>(2001 + r * cols + c);
        }
    }
    EXPECT_TRUE(ResultCmp(yExpected, yActual, 0)); // slot0, consumed first
    EXPECT_TRUE(ResultCmp(xExpected, xActual, 0)); // slot1, never corrupted by Z
    EXPECT_TRUE(ResultCmp(zExpected, zActual, 0));

    auto &sharedState = Pipe::GetSharedState();
    std::lock_guard<std::mutex> lock(sharedState.mutex);
    EXPECT_EQ(sharedState.occupied, 0);
