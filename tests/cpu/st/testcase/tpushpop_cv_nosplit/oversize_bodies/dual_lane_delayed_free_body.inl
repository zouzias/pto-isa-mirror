    constexpr int kFifoDepth = 2;
    using CubeTile = Tile<TileType::Mat, T, rows, cols>;
    using VecTile = Tile<TileType::Vec, T, rows, cols>;
    using Pipe = TPipe<kPipeFlagId + 4, Direction::DIR_C2V, sizeof(T) * VecTile::Numel, kFifoDepth, 2, true>;

    NPU_MEMORY_CLEAR();
    Pipe::reset_for_cpu_sim();
    Pipe pipe((__gm__ void *)nullptr, kVecConsumerBase, 0x0);
    std::vector<std::vector<T>> lane0Actual(kFifoDepth);
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

    {
        auto &sharedState = Pipe::GetSharedState();
        std::lock_guard<std::mutex> lock(sharedState.mutex);
        EXPECT_EQ(sharedState.occupied, kFifoDepth);
    }

    {
        cpu_sim::ScopedExecutionContext lane1Ctx(0, 1, 2);
        for (int iter = 0; iter < kFifoDepth; ++iter) {
            VecTile vecTile;
            TASSIGN(vecTile, 0x8000);
            TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
            lane1Actual[iter].assign(vecTile.data(), vecTile.data() + vecTile.Numel);
        }
        for (int iter = 0; iter < kFifoDepth; ++iter) {
            TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
        }
    }

    for (int iter = 0; iter < kFifoDepth; ++iter) {
        const auto expected = makeExpected<T, rows, cols>(iter);
        EXPECT_TRUE(ResultCmp(expected, lane0Actual[iter], 0));
        EXPECT_TRUE(std::all_of(lane1Actual[iter].begin(), lane1Actual[iter].end(),
                                [](T value) { return value == static_cast<T>(0); }));
    }

    auto &sharedState = Pipe::GetSharedState();
    std::lock_guard<std::mutex> lock(sharedState.mutex);
    EXPECT_EQ(sharedState.occupied, 0);
    EXPECT_EQ(sharedState.popped_not_freed_by_lane[0], 0);
    EXPECT_EQ(sharedState.popped_not_freed_by_lane[1], 0);
