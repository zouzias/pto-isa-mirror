/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

    using VecTile = DirBothVecTile<SplitAxis>;
    using MatTile = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16>;
    using AccTile = TileAcc<float, 16, 16>;
    using Pipe = TPipe<FlagId, Direction::DIR_BOTH, sizeof(float) * MatTile::Numel, 2>;

    Pipe::reset_for_cpu_sim();
    Pipe vecProducer0((__gm__ void*)nullptr, 0x0, 0x10000);
    Pipe vecProducer1((__gm__ void*)nullptr, 0x0, 0x10000);
    Pipe cubePipe((__gm__ void*)nullptr, 0x0, 0x10000);
    Pipe vecConsumer0((__gm__ void*)nullptr, 0x0, 0x10000);
    Pipe vecConsumer1((__gm__ void*)nullptr, 0x0, 0x10000);

    VecTile src0;
    VecTile src1;
    MatTile poppedMat;
    AccTile accSrc;
    VecTile dst0;
    VecTile dst1;

    TASSIGN(src0, 0x0);
    TASSIGN(src1, VecTile::Numel * sizeof(float));
    TASSIGN(poppedMat, 2 * VecTile::Numel * sizeof(float));
    TASSIGN(accSrc, 2 * VecTile::Numel * sizeof(float) + MatTile::Numel * sizeof(float));
    TASSIGN(dst0, 2 * VecTile::Numel * sizeof(float) + MatTile::Numel * sizeof(float) + AccTile::Numel * sizeof(float));
    TASSIGN(
        dst1, 2 * VecTile::Numel * sizeof(float) + MatTile::Numel * sizeof(float) + AccTile::Numel * sizeof(float) +
                  VecTile::Numel * sizeof(float));

    fillTileSequence(src0, 1.0f);
    fillTileSequence(src1, 1001.0f);
    fillTileSequence(accSrc, 2001.0f);
    std::fill(poppedMat.data(), poppedMat.data() + poppedMat.Numel, 0.0f);
    std::fill(dst0.data(), dst0.data() + dst0.Numel, 0.0f);
    std::fill(dst1.data(), dst1.data() + dst1.Numel, 0.0f);

    {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 2);
        TPUSH<Pipe, VecTile, SplitAxis>(vecProducer0, src0);
    }
    {
        cpu_sim::ScopedExecutionContext ctx(0, 1, 2);
        TPUSH<Pipe, VecTile, SplitAxis>(vecProducer1, src1);
    }

    std::atomic<bool> vec0Done{false};
    std::atomic<bool> vec1Done{false};
    std::thread consumerThread0([&]() {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 2);
        TPOP<Pipe, VecTile, SplitAxis>(vecConsumer0, dst0);
        vec0Done.store(true, std::memory_order_release);
    });
    std::thread consumerThread1([&]() {
        cpu_sim::ScopedExecutionContext ctx(0, 1, 2);
        TPOP<Pipe, VecTile, SplitAxis>(vecConsumer1, dst1);
        vec1Done.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    EXPECT_FALSE(vec0Done.load(std::memory_order_acquire));
    EXPECT_FALSE(vec1Done.load(std::memory_order_acquire));

    {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 1);
        TPOP<Pipe, MatTile, SplitAxis>(cubePipe, poppedMat);
        TFREE<Pipe, SplitAxis>(cubePipe);
    }

    {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 1);
        TPUSH<Pipe, AccTile, SplitAxis>(cubePipe, accSrc);
    }

    consumerThread0.join();
    consumerThread1.join();

    expectVecMatchesAccSplit<SplitAxis>(dst0, accSrc, 0);
    expectVecMatchesAccSplit<SplitAxis>(dst1, accSrc, 1);

    {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 2);
        TFREE<Pipe, SplitAxis>(vecConsumer0);
    }
    {
        cpu_sim::ScopedExecutionContext ctx(0, 1, 2);
        TFREE<Pipe, SplitAxis>(vecConsumer1);
    }
