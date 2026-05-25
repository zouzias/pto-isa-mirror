#pragma once

#include "compute_vec_views.hpp"

#if defined(__CCE_AICORE__)
namespace v2_compute_vec {

__tf__ __aicore__ inline void ExpandSwiGluQuantSrc(typename SwiGluQuantSrcTile::TileDType __out__ dst,
                                                   typename SwiGluFloatTile::TileDType __in__ src,
                                                   uint32_t rows,
                                                   uint32_t cols)
{
    __ubuf__ float* dstPtr = reinterpret_cast<__ubuf__ float*>(__cce_get_tile_ptr(dst));
    __ubuf__ float* srcPtr = reinterpret_cast<__ubuf__ float*>(__cce_get_tile_ptr(src));
    for (uint32_t row = 0; row < kComputeTileRows; ++row) {
        for (uint32_t col = 0; col < kDequantInt8TileCols; ++col) {
            dstPtr[static_cast<uint64_t>(row) * kDequantInt8TileCols + col] =
                row < rows && col < cols ? srcPtr[static_cast<uint64_t>(row) * kSwiGluFloatTileCols + col] : 0.0f;
        }
    }
}

__tf__ __aicore__ inline void CompactSwiGluQuantRow(typename SwiGluQuantRowTile::TileDType __out__ row,
                                                    typename SwiGluQuantTile::TileDType __in__ src,
                                                    uint32_t localRow,
                                                    uint32_t cols,
                                                    bool valid)
{
    __ubuf__ int8_t* rowPtr = reinterpret_cast<__ubuf__ int8_t*>(__cce_get_tile_ptr(row));
    __ubuf__ int8_t* srcPtr = reinterpret_cast<__ubuf__ int8_t*>(__cce_get_tile_ptr(src));
    for (uint32_t col = 0; col < kDequantInt8TileCols; ++col) {
        rowPtr[col] = valid && col < cols ? srcPtr[static_cast<uint64_t>(localRow) * kDequantInt8TileCols + col] : 0;
    }
}

V4_FORCE_INLINE_AICORE void StoreSwiGluQuantRows(__gm__ int8_t* tileSwiGluQ,
                                                 SwiGluQuantTile& quantTile,
                                                 uint32_t rows,
                                                 uint32_t k2Total,
                                                 uint32_t colBegin,
                                                 uint32_t cols)
{
    SwiGluQuantRowTile rowTile(cols);
    pto::TASSIGN(rowTile, kSwiGluRowTileOffset);
    for (uint32_t localRow = 0; localRow < kComputeTileRows; ++localRow) {
        pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();
        CompactSwiGluQuantRow(rowTile.data(), quantTile.data(), localRow, cols, localRow < rows);
        pto::PtoSetWaitFlag<PIPE_S, PIPE_MTE3>();
        auto* rowOut = tileSwiGluQ + static_cast<uint64_t>(localRow) * k2Total + colBegin;
        auto rowGlobal = MakeSwiGluQuantRowView(rowOut, k2Total, cols);
        pto::TSTORE(rowGlobal, rowTile);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    }
}

__tf__ __aicore__ inline void CompactScale2Row(typename DequantCompactRowTile::TileDType __out__ row,
                                               typename SwiGluRowStatTile::TileDType __in__ src,
                                               uint32_t localRow,
                                               bool valid)
{
    __ubuf__ float* rowPtr = reinterpret_cast<__ubuf__ float*>(__cce_get_tile_ptr(row));
    __ubuf__ float* srcPtr = reinterpret_cast<__ubuf__ float*>(__cce_get_tile_ptr(src));
    rowPtr[0] = valid ? srcPtr[static_cast<uint64_t>(localRow) * kSwiGluFloatTileCols] : 0.0f;
}

V4_FORCE_INLINE_AICORE void StoreScale2Rows(__gm__ float* tileScale2,
                                            SwiGluRowStatTile& scale2Tile,
                                            uint32_t rows)
{
    DequantCompactRowTile rowTile(1);
    pto::TASSIGN(rowTile, kSwiGluRowTileOffset);
    for (uint32_t localRow = 0; localRow < kComputeTileRows; ++localRow) {
        pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();
        CompactScale2Row(rowTile.data(), scale2Tile.data(), localRow, localRow < rows);
        pto::PtoSetWaitFlag<PIPE_S, PIPE_MTE3>();
        auto rowGlobal = MakeScale2View(tileScale2 + localRow, 1);
        pto::TSTORE(rowGlobal, rowTile);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    }
}

constexpr uint32_t kSwigluFp16TileCols = 16;

using SwigluHalfLoadTile = pto::Tile<pto::TileType::Vec,
                                     half,
                                     kComputeTileRows,
                                     kSwigluFp16TileCols,
                                     pto::BLayout::RowMajor,
                                     pto::DYNAMIC,
                                     pto::DYNAMIC>;
using SwigluCvtFloatTile = pto::Tile<pto::TileType::Vec,
                                     float,
                                     kComputeTileRows,
                                     kSwigluFp16TileCols,
                                     pto::BLayout::RowMajor,
                                     pto::DYNAMIC,
                                     pto::DYNAMIC>;

constexpr uint64_t kSwFp16GateHalfOff   = 0x0;
constexpr uint64_t kSwFp16UpHalfOff     = 0x200;
constexpr uint64_t kSwFp16GateFloatOff  = 0x400;
constexpr uint64_t kSwFp16UpFloatOff    = 0x800;
constexpr uint64_t kSwFp16NegGateOff    = 0xc00;
constexpr uint64_t kSwFp16ExpOff        = 0x1000;
constexpr uint64_t kSwFp16DenomOff      = 0x1400;
constexpr uint64_t kSwFp16SigmoidOff    = 0x1800;
constexpr uint64_t kSwFp16ProductOff    = 0x1c00;
constexpr uint64_t kSwFp16ScaledOff     = 0x2000;
constexpr uint64_t kSwFp16AbsOff        = 0x2400;
constexpr uint64_t kSwFp16RowMaxOff     = 0x2800;
constexpr uint64_t kSwFp16ChunkMaxOff   = 0x2880;
constexpr uint64_t kSwFp16Scale2Off     = 0x2c00;
constexpr uint64_t kSwFp16InvScaleOff   = 0x3000;
constexpr uint64_t kSwFp16QuantOff      = 0x3400;
constexpr uint64_t kSwFp16QuantSrcOff   = 0x3800;
constexpr uint64_t kSwFp16RowTileOff    = 0x3c00;
constexpr uint64_t kSwFp16ScaleLoadOff  = 0x4000;
constexpr uint64_t kSwFp16ScaleExpOff   = 0x4400;

using SwigluFp16Global = pto::GlobalTensor<half,
    pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>,
    pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>,
    pto::Layout::ND>;

V4_FORCE_INLINE_AICORE SwigluFp16Global MakeSwigluFp16InputView(__gm__ half* tileGmm1,
                                                                 uint32_t offset,
                                                                 uint32_t rows,
                                                                 uint32_t cols,
                                                                 uint32_t gmm1StrideN)
{
    auto* input = tileGmm1 + offset;
    pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC> shape(rows, cols);
    pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1> stride(
        gmm1StrideN * rows, gmm1StrideN * rows, gmm1StrideN * rows, gmm1StrideN, 1);
    return SwigluFp16Global(input, shape, stride);
}

V4_FORCE_INLINE_AICORE void RunSwiGluFp16TilePtoBody(__gm__ half* tileGmm1,
                                                      __gm__ float* perTokenScale1,
                                                      __gm__ int8_t* tileSwiGluQ,
                                                      __gm__ float* tileScale2,
                                                      uint32_t rows,
                                                      uint32_t gmm1N)
{
    const uint32_t k2Total = gmm1N / 2;
    if (rows == 0 || k2Total == 0 || tileGmm1 == nullptr || tileSwiGluQ == nullptr || tileScale2 == nullptr) {
        return;
    }

    SwiGluFloatTile gateTile(rows, kSwiGluFloatTileCols);
    SwiGluFloatTile upTile(rows, kSwiGluFloatTileCols);
    SwiGluFloatTile negGateTile(rows, kSwiGluFloatTileCols);
    SwiGluFloatTile expTile(rows, kSwiGluFloatTileCols);
    SwiGluFloatTile denomTile(rows, kSwiGluFloatTileCols);
    SwiGluFloatTile sigmoidTile(rows, kSwiGluFloatTileCols);
    SwiGluFloatTile productTile(rows, kSwiGluFloatTileCols);
    SwiGluFloatTile scaledTile(rows, kSwiGluFloatTileCols);
    SwiGluFloatTile absTile(rows, kSwiGluFloatTileCols);
    SwiGluRowStatTile rowMaxTile(rows);
    SwiGluRowStatTile chunkMaxTile(rows);
    SwiGluRowStatTile scale2Tile(rows);
    SwiGluRowStatTile invScaleTile(rows);
    SwiGluQuantSrcTile quantSrcTile(rows, kDequantInt8TileCols);
    SwiGluQuantTile quantTile(rows, kDequantInt8TileCols);

    SwigluHalfLoadTile gateHalfTile(rows, kSwigluFp16TileCols);
    SwigluHalfLoadTile upHalfTile(rows, kSwigluFp16TileCols);

    DequantScaleLoadTile scaleLoadTile(rows);
    DequantScaleExpandedTile scaleExpandedTile(rows);

    pto::TASSIGN(gateTile, kSwiGluGateTileOffset);
    pto::TASSIGN(upTile, kSwiGluUpTileOffset);
    pto::TASSIGN(negGateTile, kSwiGluNegGateTileOffset);
    pto::TASSIGN(expTile, kSwiGluExpTileOffset);
    pto::TASSIGN(denomTile, kSwiGluDenomTileOffset);
    pto::TASSIGN(sigmoidTile, kSwiGluSigmoidTileOffset);
    pto::TASSIGN(productTile, kSwiGluProductTileOffset);
    pto::TASSIGN(scaledTile, kSwiGluScaledTileOffset);
    pto::TASSIGN(absTile, kSwiGluAbsTileOffset);
    pto::TASSIGN(rowMaxTile, kSwiGluRowMaxTileOffset);
    pto::TASSIGN(chunkMaxTile, kSwiGluRowMaxTileOffset + 0x80);
    pto::TASSIGN(scale2Tile, kSwiGluScale2TileOffset);
    pto::TASSIGN(invScaleTile, kSwiGluInvScaleTileOffset);
    pto::TASSIGN(quantSrcTile, kSwiGluQuantTileOffset + 0x400);
    pto::TASSIGN(quantTile, kSwiGluQuantTileOffset);
    pto::TASSIGN(gateHalfTile, kSwFp16GateHalfOff);
    pto::TASSIGN(upHalfTile, kSwFp16UpHalfOff);
    pto::TASSIGN(scaleLoadTile, kSwFp16ScaleLoadOff);
    pto::TASSIGN(scaleExpandedTile, kSwFp16ScaleExpOff);

    auto scaleGlobal = MakeScaleView(perTokenScale1, 0, rows, sizeof(float));
    pto::TLOAD(scaleLoadTile, scaleGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pto::TROWEXPAND(scaleExpandedTile, scaleLoadTile);
    pipe_barrier(PIPE_V);

    pto::TEXPANDS(rowMaxTile, 0.0f);

    for (uint32_t colBegin = 0; colBegin < k2Total; colBegin += kSwiGluFloatTileCols) {
        const uint32_t cols = k2Total - colBegin < kSwiGluFloatTileCols ? k2Total - colBegin : kSwiGluFloatTileCols;
        gateHalfTile = SwigluHalfLoadTile(rows, cols);
        upHalfTile = SwigluHalfLoadTile(rows, cols);
        gateTile = SwiGluFloatTile(rows, cols);
        upTile = SwiGluFloatTile(rows, cols);
        negGateTile = SwiGluFloatTile(rows, cols);
        expTile = SwiGluFloatTile(rows, cols);
        denomTile = SwiGluFloatTile(rows, cols);
        sigmoidTile = SwiGluFloatTile(rows, cols);
        productTile = SwiGluFloatTile(rows, cols);
        scaledTile = SwiGluFloatTile(rows, cols);
        absTile = SwiGluFloatTile(rows, cols);
        pto::TASSIGN(gateHalfTile, kSwFp16GateHalfOff);
        pto::TASSIGN(upHalfTile, kSwFp16UpHalfOff);
        pto::TASSIGN(gateTile, kSwiGluGateTileOffset);
        pto::TASSIGN(upTile, kSwiGluUpTileOffset);
        pto::TASSIGN(negGateTile, kSwiGluNegGateTileOffset);
        pto::TASSIGN(expTile, kSwiGluExpTileOffset);
        pto::TASSIGN(denomTile, kSwiGluDenomTileOffset);
        pto::TASSIGN(sigmoidTile, kSwiGluSigmoidTileOffset);
        pto::TASSIGN(productTile, kSwiGluProductTileOffset);
        pto::TASSIGN(scaledTile, kSwiGluScaledTileOffset);
        pto::TASSIGN(absTile, kSwiGluAbsTileOffset);

        auto gateGlobal = MakeSwigluFp16InputView(tileGmm1, colBegin, rows, cols, gmm1N);
        auto upGlobal = MakeSwigluFp16InputView(tileGmm1, k2Total + colBegin, rows, cols, gmm1N);
        pto::TLOAD(gateHalfTile, gateGlobal);
        pto::TLOAD(upHalfTile, upGlobal);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        pto::TCVT(gateTile, gateHalfTile, pto::RoundMode::CAST_RINT);
        pipe_barrier(PIPE_V);
        pto::TCVT(upTile, upHalfTile, pto::RoundMode::CAST_RINT);
        pipe_barrier(PIPE_V);

        pto::TROWEXPANDMUL(gateTile, gateTile, scaleExpandedTile);
        pipe_barrier(PIPE_V);
        pto::TROWEXPANDMUL(upTile, upTile, scaleExpandedTile);
        pipe_barrier(PIPE_V);

        pto::TMULS(negGateTile, gateTile, -1.0f);
        pipe_barrier(PIPE_V);
        pto::TEXP(expTile, negGateTile);
        pipe_barrier(PIPE_V);
        pto::TADDS(denomTile, expTile, 1.0f);
        pipe_barrier(PIPE_V);
        pto::TDIVS(sigmoidTile, 1.0f, denomTile);
        pipe_barrier(PIPE_V);
        pto::TMUL(productTile, gateTile, sigmoidTile);
        pipe_barrier(PIPE_V);
        pto::TMUL(scaledTile, productTile, upTile);
        pipe_barrier(PIPE_V);

        pto::TABS(absTile, scaledTile);
        pipe_barrier(PIPE_V);
        pto::TROWMAX(chunkMaxTile, absTile, gateTile);
        pipe_barrier(PIPE_V);
        pto::TMAX(rowMaxTile, rowMaxTile, chunkMaxTile);
        pipe_barrier(PIPE_V);
    }

    pto::TMULS(scale2Tile, rowMaxTile, 1.0f / 127.0f);
    pipe_barrier(PIPE_V);
    pto::TMAXS(scale2Tile, scale2Tile, 1.0e-8f);
    pipe_barrier(PIPE_V);
    pto::TDIVS(invScaleTile, 1.0f, scale2Tile);
    pipe_barrier(PIPE_V);
    StoreScale2Rows(tileScale2, scale2Tile, rows);

    for (uint32_t colBegin = 0; colBegin < k2Total; colBegin += kSwiGluFloatTileCols) {
        const uint32_t cols = k2Total - colBegin < kSwiGluFloatTileCols ? k2Total - colBegin : kSwiGluFloatTileCols;
        gateHalfTile = SwigluHalfLoadTile(rows, cols);
        upHalfTile = SwigluHalfLoadTile(rows, cols);
        gateTile = SwiGluFloatTile(rows, cols);
        upTile = SwiGluFloatTile(rows, cols);
        negGateTile = SwiGluFloatTile(rows, cols);
        expTile = SwiGluFloatTile(rows, cols);
        denomTile = SwiGluFloatTile(rows, cols);
        sigmoidTile = SwiGluFloatTile(rows, cols);
        productTile = SwiGluFloatTile(rows, cols);
        scaledTile = SwiGluFloatTile(rows, cols);
        quantSrcTile = SwiGluQuantSrcTile(rows, kDequantInt8TileCols);
        quantTile = SwiGluQuantTile(rows, kDequantInt8TileCols);
        pto::TASSIGN(gateHalfTile, kSwFp16GateHalfOff);
        pto::TASSIGN(upHalfTile, kSwFp16UpHalfOff);
        pto::TASSIGN(gateTile, kSwiGluGateTileOffset);
        pto::TASSIGN(upTile, kSwiGluUpTileOffset);
        pto::TASSIGN(negGateTile, kSwiGluNegGateTileOffset);
        pto::TASSIGN(expTile, kSwiGluExpTileOffset);
        pto::TASSIGN(denomTile, kSwiGluDenomTileOffset);
        pto::TASSIGN(sigmoidTile, kSwiGluSigmoidTileOffset);
        pto::TASSIGN(productTile, kSwiGluProductTileOffset);
        pto::TASSIGN(scaledTile, kSwiGluScaledTileOffset);
        pto::TASSIGN(quantSrcTile, kSwiGluQuantTileOffset + 0x400);
        pto::TASSIGN(quantTile, kSwiGluQuantTileOffset);

        auto gateGlobal = MakeSwigluFp16InputView(tileGmm1, colBegin, rows, cols, gmm1N);
        auto upGlobal = MakeSwigluFp16InputView(tileGmm1, k2Total + colBegin, rows, cols, gmm1N);
        pto::TLOAD(gateHalfTile, gateGlobal);
        pto::TLOAD(upHalfTile, upGlobal);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        pto::TCVT(gateTile, gateHalfTile, pto::RoundMode::CAST_RINT);
        pipe_barrier(PIPE_V);
        pto::TCVT(upTile, upHalfTile, pto::RoundMode::CAST_RINT);
        pipe_barrier(PIPE_V);

        pto::TROWEXPANDMUL(gateTile, gateTile, scaleExpandedTile);
        pipe_barrier(PIPE_V);
        pto::TROWEXPANDMUL(upTile, upTile, scaleExpandedTile);
        pipe_barrier(PIPE_V);

        pto::TMULS(negGateTile, gateTile, -1.0f);
        pipe_barrier(PIPE_V);
        pto::TEXP(expTile, negGateTile);
        pipe_barrier(PIPE_V);
        pto::TADDS(denomTile, expTile, 1.0f);
        pipe_barrier(PIPE_V);
        pto::TDIVS(sigmoidTile, 1.0f, denomTile);
        pipe_barrier(PIPE_V);
        pto::TMUL(productTile, gateTile, sigmoidTile);
        pipe_barrier(PIPE_V);
        pto::TMUL(scaledTile, productTile, upTile);
        pipe_barrier(PIPE_V);

        pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();
        ExpandSwiGluQuantSrc(quantSrcTile.data(), scaledTile.data(), rows, cols);
        pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();
        pto::TQUANT<pto::QuantType::INT8_SYM>(quantTile, quantSrcTile, invScaleTile);
        pipe_barrier(PIPE_V);
        StoreSwiGluQuantRows(tileSwiGluQ, quantTile, rows, k2Total, colBegin, cols);
    }
}

V4_FORCE_INLINE_AICORE void RunSwiGluFp16TilePto(__gm__ half* tileGmm1,
                                                   __gm__ float* perTokenScale1,
                                                   __gm__ int8_t* tileSwiGluQ,
                                                   __gm__ float* tileScale2,
                                                   uint32_t rows,
                                                   uint32_t gmm1N)
{
    RunSwiGluFp16TilePtoBody(tileGmm1, perTokenScale1, tileSwiGluQ, tileScale2, rows, gmm1N);
}

}  // namespace v2_compute_vec

#endif
