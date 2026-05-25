#pragma once

#include "kernel_operator.h"

#if defined(__CCE_AICORE__)
#include <pto/pto-inst.hpp>
#endif

#ifndef V4_FORCE_INLINE_AICORE
#define V4_FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__
#endif

namespace v2_compute_vec {

constexpr uint32_t kComputeTileRows = 8;
constexpr uint32_t kGmm1OutputElems = 4;
constexpr uint32_t kDequantInt8TileCols = 32;
constexpr uint32_t kDequantFloatTileCols = 8;
constexpr uint32_t kSwiGluFloatTileCols = 16;
constexpr uint64_t kSwiGluGateTileOffset = 0x0;
constexpr uint64_t kSwiGluUpTileOffset = 0x400;
constexpr uint64_t kSwiGluNegGateTileOffset = 0x800;
constexpr uint64_t kSwiGluExpTileOffset = 0xc00;
constexpr uint64_t kSwiGluDenomTileOffset = 0x1000;
constexpr uint64_t kSwiGluSigmoidTileOffset = 0x1400;
constexpr uint64_t kSwiGluProductTileOffset = 0x1800;
constexpr uint64_t kSwiGluScaledTileOffset = 0x1c00;
constexpr uint64_t kSwiGluAbsTileOffset = 0x2000;
constexpr uint64_t kSwiGluRowMaxTileOffset = 0x2400;
constexpr uint64_t kSwiGluScale2TileOffset = 0x2800;
constexpr uint64_t kSwiGluInvScaleTileOffset = 0x2c00;
constexpr uint64_t kSwiGluQuantTileOffset = 0x3000;
constexpr uint64_t kSwiGluRowTileOffset = 0x3800;

#if defined(__CCE_AICORE__)

using DequantShape = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
using DequantStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;

template <typename T, pto::Layout Layout = pto::Layout::ND>
using DequantGlobal = pto::GlobalTensor<T, DequantShape, DequantStride, Layout>;

using DequantScaleLoadTile = pto::Tile<pto::TileType::Vec,
                                       float,
                                       kComputeTileRows,
                                       kDequantFloatTileCols,
                                       pto::BLayout::RowMajor,
                                       pto::DYNAMIC,
                                       1,
                                       pto::SLayout::NoneBox,
                                       512,
                                       pto::PadValue::Zero>;
using DequantScaleExpandedTile = pto::Tile<pto::TileType::Vec,
                                           float,
                                           kComputeTileRows,
                                           kDequantFloatTileCols,
                                           pto::BLayout::RowMajor,
                                           pto::DYNAMIC,
                                           kDequantFloatTileCols>;
using DequantCompactRowTile = pto::Tile<pto::TileType::Vec,
                                        float,
                                        1,
                                        kDequantFloatTileCols,
                                        pto::BLayout::RowMajor,
                                        1,
                                        pto::DYNAMIC>;

using SwiGluShape = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
using SwiGluStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
using SwiGluGlobal = pto::GlobalTensor<float, SwiGluShape, SwiGluStride>;
using SwiGluQuantGlobal = pto::GlobalTensor<int8_t, SwiGluShape, SwiGluStride>;
using SwiGluFloatTile = pto::Tile<pto::TileType::Vec,
                                  float,
                                  kComputeTileRows,
                                  kSwiGluFloatTileCols,
                                  pto::BLayout::RowMajor,
                                  pto::DYNAMIC,
                                  pto::DYNAMIC>;
using SwiGluRowStatTile = pto::Tile<pto::TileType::Vec,
                                    float,
                                    kComputeTileRows,
                                    kSwiGluFloatTileCols,
                                    pto::BLayout::RowMajor,
                                    pto::DYNAMIC,
                                    kSwiGluFloatTileCols>;
using SwiGluQuantSrcTile = pto::Tile<pto::TileType::Vec,
                                     float,
                                     kComputeTileRows,
                                     kDequantInt8TileCols,
                                     pto::BLayout::RowMajor,
                                     pto::DYNAMIC,
                                     pto::DYNAMIC>;
using SwiGluQuantTile = pto::Tile<pto::TileType::Vec,
                                  int8_t,
                                  kComputeTileRows,
                                  kDequantInt8TileCols,
                                  pto::BLayout::RowMajor,
                                  pto::DYNAMIC,
                                  pto::DYNAMIC,
                                  pto::SLayout::NoneBox,
                                  512,
                                  pto::PadValue::Zero>;
using SwiGluQuantRowTile = pto::Tile<pto::TileType::Vec,
                                     int8_t,
                                     1,
                                     kDequantInt8TileCols,
                                     pto::BLayout::RowMajor,
                                     1,
                                     pto::DYNAMIC>;

V4_FORCE_INLINE_AICORE DequantGlobal<float> MakeScaleView(__gm__ float* scale1,
                                                           uint32_t rowBegin,
                                                           uint32_t rows,
                                                           uint32_t scaleStrideBytes)
{
    auto* rowScale = reinterpret_cast<__gm__ float*>(
        reinterpret_cast<__gm__ uint8_t*>(scale1) + static_cast<uint64_t>(rowBegin) * scaleStrideBytes);
    const uint32_t rowStrideElems = scaleStrideBytes / sizeof(float);
    DequantShape shape(1, 1, 1, rows, 1);
    DequantStride stride(rowStrideElems * rows, rowStrideElems * rows, rowStrideElems * rows, rowStrideElems, 1);
    return DequantGlobal<float>(rowScale, shape, stride);
}

V4_FORCE_INLINE_AICORE SwiGluQuantGlobal MakeSwiGluQuantRowView(__gm__ int8_t* rowSwiGluQ,
                                                                uint32_t rowStrideElems,
                                                                uint32_t cols)
{
    SwiGluShape shape(1, 1, 1, 1, cols);
    SwiGluStride stride(rowStrideElems, rowStrideElems, rowStrideElems, rowStrideElems, 1);
    return SwiGluQuantGlobal(rowSwiGluQ, shape, stride);
}

V4_FORCE_INLINE_AICORE SwiGluGlobal MakeScale2View(__gm__ float* tileScale2, uint32_t rows)
{
    SwiGluShape shape(1, 1, 1, rows, 1);
    SwiGluStride stride(rows, rows, rows, 1, 1);
    return SwiGluGlobal(tileScale2, shape, stride);
}


#endif

}  // namespace v2_compute_vec
