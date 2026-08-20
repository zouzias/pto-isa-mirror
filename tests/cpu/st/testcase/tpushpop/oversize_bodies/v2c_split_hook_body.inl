    using VecTile = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, 8, 16>;
    using MatTile = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16>;

    HookedV2CPipe::SharedStateStorage storage{};
    g_pipe_hook_call_count.store(0, std::memory_order_relaxed);
    g_pipe_hook_storage = &storage;
    g_pipe_hook_size = 0;
    g_pipe_hook_last_key = 0;

    ScopedCpuStubHooks hooks(nullptr, reinterpret_cast<void*>(MockPipeSharedStateHook));
    HookedV2CPipe::reset_for_cpu_sim();

    HookedV2CPipe producer0((__gm__ void*)nullptr, 0x0, 0x10000);
    HookedV2CPipe producer1((__gm__ void*)nullptr, 0x0, 0x10000);
    HookedV2CPipe consumer((__gm__ void*)nullptr, 0x0, 0x10000);
    VecTile topHalf;
    VecTile bottomHalf;
    MatTile dst;
    TASSIGN(topHalf, 0);
    TASSIGN(bottomHalf, VecTile::Numel * sizeof(VecTile::DType));
    TASSIGN(dst, 2 * VecTile::Numel * sizeof(VecTile::DType));
    fillTile<float, 8, 16, TileType::Vec>(topHalf, 0);
    fillTile<float, 8, 16, TileType::Vec>(bottomHalf, 1);
    std::fill(dst.data(), dst.data() + dst.Numel, 0.0f);

    {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 2);
        TPUSH<HookedV2CPipe, VecTile, TileSplitAxis::TILE_UP_DOWN>(producer0, topHalf);
    }

    auto& state = HookedV2CPipe::GetSharedState();
    EXPECT_EQ(state.occupied, 0);
    EXPECT_EQ(state.next_producer_slot, 0);
    EXPECT_EQ(state.producers_done[0], 0x1u);
    EXPECT_EQ(state.producers_allocated[0], 0x1u);

    {
        cpu_sim::ScopedExecutionContext ctx(0, 1, 2);
        TPUSH<HookedV2CPipe, VecTile, TileSplitAxis::TILE_UP_DOWN>(producer1, bottomHalf);
    }

    EXPECT_EQ(state.occupied, 1);
    EXPECT_EQ(state.next_producer_slot, 1);
    EXPECT_EQ(state.producers_done[0], 0u);
    EXPECT_EQ(state.producers_allocated[0], 0u);

    {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 1);
        TPOP<HookedV2CPipe, MatTile, TileSplitAxis::TILE_UP_DOWN>(consumer, dst);
        TFREE<HookedV2CPipe, TileSplitAxis::TILE_UP_DOWN>(consumer);
    }

    for (int r = 0; r < topHalf.GetValidRow(); ++r) {
        for (int c = 0; c < topHalf.GetValidCol(); ++c) {
            EXPECT_EQ(
                dst.data()[GetTileElementOffset<MatTile>(r, c)], topHalf.data()[GetTileElementOffset<VecTile>(r, c)]);
            EXPECT_EQ(
                dst.data()[GetTileElementOffset<MatTile>(r + topHalf.GetValidRow(), c)],
                bottomHalf.data()[GetTileElementOffset<VecTile>(r, c)]);
        }
    }

    EXPECT_GT(g_pipe_hook_call_count.load(std::memory_order_relaxed), 0u);
    EXPECT_EQ(g_pipe_hook_size, sizeof(HookedV2CPipe::SharedStateStorage));
    EXPECT_NE(g_pipe_hook_last_key, 0u);
