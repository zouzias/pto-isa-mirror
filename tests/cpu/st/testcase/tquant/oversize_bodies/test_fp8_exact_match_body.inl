/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

    using SrcTile = Tile<TileType::Vec, SrcType, 16, 32>;
    using ScaleTile = Tile<TileType::Vec, float, 16, 32>;
    using DstTile = Tile<TileType::Vec, int8_t, 16, 32>;
    using ExpTile = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, 16, 1>;
    using MaxTile = Tile<TileType::Vec, float, 16, 32, BLayout::RowMajor, 16, 1>;
    SrcTile src;
    ScaleTile scaling;
    DstTile dst;
    ExpTile expTile;
    MaxTile max;
    size_t addr = 0;
    TASSIGN(src, addr);
    addr += SrcTile::Numel * sizeof(typename SrcTile::DType);
    TASSIGN(scaling, addr);
    addr += ScaleTile::Numel * sizeof(typename ScaleTile::DType);
    TASSIGN(dst, addr);
    addr += DstTile::Numel * sizeof(typename DstTile::DType);
    TASSIGN(expTile, addr);
    addr += ExpTile::Numel * sizeof(typename ExpTile::DType);
    TASSIGN(max, addr);

    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            double fraction = pow(static_cast<double>(r * SrcTile::Cols + c) / SrcTile::Numel, 10);
            double base =
                fraction * ((r + c) % 2 ? std::numeric_limits<SrcType>::max() : std::numeric_limits<SrcType>::lowest());
            src.data()[GetTileElementOffset<SrcTile>(r, c)] = static_cast<SrcType>(base);
        }
    }
    src.data()[GetTileElementOffset<SrcTile>(1, 0)] = BitsToFloat(0x04600001u);
    src.data()[GetTileElementOffset<SrcTile>(2, 0)] = std::nextafter(448.0f, std::numeric_limits<float>::infinity());
    src.data()[GetTileElementOffset<SrcTile>(3, 0)] = -896.0f;

    TQUANT<QuantType::MXFP8>(dst, src, &expTile, &max, &scaling);

    for (int row = 0; row < 4; ++row) {
        float maxAbs = 0.0f;
        for (int col = 0; col < 32; ++col) {
            maxAbs =
                std::max(maxAbs, std::fabs(static_cast<float>(src.data()[GetTileElementOffset<SrcTile>(row, col)])));
        }
        const uint8_t expectedExp = cpu_quant::ComputeMxSharedExponent<QuantType::MXFP8, QuantScaleAlg::OCP>(maxAbs);
        const float expectedScaling =
            cpu_quant::ComputeMxGroupScaling<QuantType::MXFP8, QuantScaleAlg::OCP>(maxAbs, expectedExp);
        EXPECT_EQ(expTile.data()[GetTileElementOffset<ExpTile>(row, 0)], expectedExp);
        EXPECT_FLOAT_EQ(max.data()[row], maxAbs);
        for (int col = 0; col < 32; ++col) {
            EXPECT_FLOAT_EQ(scaling.data()[row], expectedScaling);
            const uint8_t expectedByte =
                EncodeE4M3Fn(src.data()[GetTileElementOffset<SrcTile>(row, col)] * expectedScaling);
            EXPECT_EQ(static_cast<uint8_t>(dst.data()[GetTileElementOffset<DstTile>(row, col)]), expectedByte);
        }
    }
