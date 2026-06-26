/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_SWIGLU_FULL_ROW_IMPL_HPP
#define DISPATCH_MEGA_COMBINE_SWIGLU_FULL_ROW_IMPL_HPP

template <typename InputElement>
AICORE inline void Swiglu<InputElement>::ApplyFullRowPerTokenScale(uint32_t bufferId, float perTokenScale) const
{
    using TileFp32 = PtoVecTile<float, kSwigluVecTileElems>;
    const uint64_t ubCFp32Offset = SwigluCFp32Offset(bufferId);
    pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();
    pipe_barrier(PIPE_V);
    for (uint32_t offset = 0; offset < problemN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = problemN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : problemN_ - offset;
        TileFp32 dstTile(1, cur);
        TileFp32 srcTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(dstTile, ubCFp32Offset + elemOffset);
        pto::TASSIGN(srcTile, ubCFp32Offset + elemOffset);
        pto::TMULS(dstTile, srcTile, perTokenScale);
    }
    pipe_barrier(PIPE_V);
}

template <typename InputElement>
AICORE inline void Swiglu<InputElement>::ComputeFullRowSwiglu(uint32_t bufferId) const
{
    using TileFp32 = PtoVecTile<float, kSwigluVecTileElems>;
    const uint64_t ubCFp32Offset = SwigluCFp32Offset(bufferId);
    const uint64_t ubWorkOffset = SwigluWorkOffset(bufferId);
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 dstTile(1, cur);
        TileFp32 srcTile(1, cur);
        pto::TASSIGN(dstTile, ubWorkOffset + static_cast<uint64_t>(offset) * sizeof(float));
        pto::TASSIGN(srcTile, ubCFp32Offset + static_cast<uint64_t>(offset) * sizeof(float));
        pto::TMULS(dstTile, srcTile, -1.0f);
    }
    pipe_barrier(PIPE_V);
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 dstTile(1, cur);
        TileFp32 srcTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(dstTile, ubWorkOffset + elemOffset);
        pto::TASSIGN(srcTile, ubWorkOffset + elemOffset);
        pto::TEXP(dstTile, srcTile);
    }
    pipe_barrier(PIPE_V);
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 dstTile(1, cur);
        TileFp32 srcTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(dstTile, ubWorkOffset + elemOffset);
        pto::TASSIGN(srcTile, ubWorkOffset + elemOffset);
        pto::TADDS(dstTile, srcTile, 1.0f);
    }
    pipe_barrier(PIPE_V);
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 dstTile(1, cur);
        TileFp32 xTile(1, cur);
        TileFp32 denomTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(dstTile, ubWorkOffset + elemOffset);
        pto::TASSIGN(xTile, ubCFp32Offset + elemOffset);
        pto::TASSIGN(denomTile, ubWorkOffset + elemOffset);
        pto::TDIV(dstTile, xTile, denomTile);
    }
    pipe_barrier(PIPE_V);
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 dstTile(1, cur);
        TileFp32 siluTile(1, cur);
        TileFp32 gateTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(dstTile, ubWorkOffset + elemOffset);
        pto::TASSIGN(siluTile, ubWorkOffset + elemOffset);
        pto::TASSIGN(gateTile, ubCFp32Offset + static_cast<uint64_t>(outputN_ + offset) * sizeof(float));
        pto::TMUL(dstTile, siluTile, gateTile);
    }
    pipe_barrier(PIPE_V);
}

template <typename InputElement>
AICORE inline float Swiglu<InputElement>::ReduceFullRowMaxAbs(uint32_t bufferId) const
{
    using TileFp32 = PtoVecTile<float, kSwigluVecTileElems>;
    using RowMaxTile = pto::Tile<pto::TileType::Vec, float, 8, 1, pto::BLayout::ColMajor, -1, 1>;
    using ScalarTile = pto::Tile<pto::TileType::Vec, float, 1, 8, pto::BLayout::RowMajor, -1, -1>;
    const uint64_t ubWorkOffset = SwigluWorkOffset(bufferId);
    const uint64_t ubAbsOffset = SwigluAbsOffset(bufferId);
    const uint64_t ubMaxOffset = SwigluMaxOffset(bufferId);
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 absTile(1, cur);
        TileFp32 srcTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(absTile, ubAbsOffset + elemOffset);
        pto::TASSIGN(srcTile, ubWorkOffset + elemOffset);
        pto::TABS(absTile, srcTile);
    }
    pipe_barrier(PIPE_V);

    bool firstReduceChunk = true;
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 absTile(1, cur);
        TileFp32 tmpTile(1, cur);
        RowMaxTile rowMaxTile(1);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(absTile, ubAbsOffset + elemOffset);
        pto::TASSIGN(tmpTile, ubAbsOffset + elemOffset);
        pto::TASSIGN(rowMaxTile, firstReduceChunk ? ubMaxOffset : ubAbsOffset);
        pto::TROWMAX(rowMaxTile, absTile, tmpTile);
        pto::TSYNC<pto::Op::TROWMAX>();
        if (!firstReduceChunk) {
            ScalarTile accTile(1, 1);
            ScalarTile newTile(1, 1);
            ScalarTile dstTile(1, 1);
            pto::TASSIGN(accTile, ubMaxOffset);
            pto::TASSIGN(newTile, ubAbsOffset);
            pto::TASSIGN(dstTile, ubMaxOffset);
            pto::TMAX(dstTile, accTile, newTile);
            pto::TSYNC<pto::Op::TMAX>();
        }
        firstReduceChunk = false;
    }
    pipe_barrier(PIPE_V);
    pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();
    TileFp32 maxScalarTile(1, 1);
    pto::TASSIGN(maxScalarTile, ubMaxOffset);
    return maxScalarTile.GetValue(0);
}

template <typename InputElement>
AICORE inline void Swiglu<InputElement>::QuantizeFullRowOutput(uint32_t bufferId, float quantScale) const
{
    using TileFp32 = PtoVecTile<float, kSwigluVecTileElems>;
    using TileI32 = PtoVecTile<int32_t, kSwigluVecTileElems>;
    using TileHalf = PtoVecTile<half, kSwigluVecTileElems>;
    const uint64_t ubDOffset = SwigluDOffset(bufferId);
    const uint64_t ubWorkOffset = SwigluWorkOffset(bufferId);
    const uint64_t ubAbsOffset = SwigluAbsOffset(bufferId);
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 dstTile(1, cur);
        TileFp32 srcTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(dstTile, ubAbsOffset + elemOffset);
        pto::TASSIGN(srcTile, ubWorkOffset + elemOffset);
        pto::TMULS(dstTile, srcTile, quantScale);
    }
    pipe_barrier(PIPE_V);
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileI32 i32Tile(1, cur);
        TileFp32 srcTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset);
        pto::TASSIGN(i32Tile, ubAbsOffset + elemOffset * sizeof(int32_t));
        pto::TASSIGN(srcTile, ubAbsOffset + elemOffset * sizeof(float));
        pto::TCVT(i32Tile, srcTile, pto::RoundMode::CAST_RINT);
    }
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileHalf halfTile(1, cur);
        TileI32 i32Tile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset);
        pto::TASSIGN(halfTile, ubAbsOffset + elemOffset * sizeof(half));
        pto::TASSIGN(i32Tile, ubAbsOffset + elemOffset * sizeof(int32_t));
        pto::TCVT(halfTile, i32Tile, pto::RoundMode::CAST_RINT);
    }
    pipe_barrier(PIPE_V);
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        PtoVecTile<int8_t, kSwigluVecTileElems> dTile(1, cur);
        TileHalf halfTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset);
        pto::TASSIGN(dTile, ubDOffset + elemOffset * sizeof(int8_t));
        pto::TASSIGN(halfTile, ubAbsOffset + elemOffset * sizeof(half));
        pto::TCVT(dTile, halfTile, pto::RoundMode::CAST_RINT);
    }
}

template <typename InputElement>
AICORE inline void Swiglu<InputElement>::StoreFullRowOutput(uint32_t rowIdx, uint32_t bufferId) const
{
    using TileD = PtoVecTile<int8_t, kSwigluVecTileElems>;
    using VectorShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using VectorStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using DGlobal = pto::GlobalTensor<int8_t, VectorShape, VectorStride, pto::Layout::ND>;
    using BlockTileD = pto::Tile<pto::TileType::Vec, int8_t, kSwigluFullRowIoBlockChunks, kSwigluVecTileElems,
                                 pto::BLayout::RowMajor, -1, -1>;
    using BlockShape = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
    using BlockStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using DBlockGlobal = pto::GlobalTensor<int8_t, BlockShape, BlockStride, pto::Layout::ND>;
    const uint64_t ubDOffset = SwigluDOffset(bufferId);
    __gm__ int8_t *gmDRow = gmPermutedTokenPtr_ + static_cast<uint64_t>(rowIdx) * outputN_;
    uint32_t offset = 0;
    while (outputN_ - offset >= kSwigluVecTileElems) {
        const uint32_t fullChunks = (outputN_ - offset) / kSwigluVecTileElems;
        const uint32_t chunkRows = fullChunks > kSwigluFullRowIoBlockChunks ? kSwigluFullRowIoBlockChunks : fullChunks;
        BlockTileD dTile(chunkRows, kSwigluVecTileElems);
        pto::TASSIGN(dTile, ubDOffset + static_cast<uint64_t>(offset) * sizeof(int8_t));
        BlockShape dShape(chunkRows, kSwigluVecTileElems);
        BlockStride dStride(static_cast<int64_t>(chunkRows) * kSwigluVecTileElems,
                            static_cast<int64_t>(chunkRows) * kSwigluVecTileElems,
                            static_cast<int64_t>(chunkRows) * kSwigluVecTileElems, kSwigluVecTileElems);
        DBlockGlobal dGlobal(gmDRow + offset, dShape, dStride);
        pto::TSTORE(dGlobal, dTile);
        offset += chunkRows * kSwigluVecTileElems;
    }
    if (offset < outputN_) {
        const uint32_t cur = outputN_ - offset;
        TileD dTile(1, cur);
        pto::TASSIGN(dTile, ubDOffset + static_cast<uint64_t>(offset) * sizeof(int8_t));
        VectorShape dShape(cur);
        VectorStride dStride(cur, cur, cur, cur);
        DGlobal dGlobal(gmDRow + offset, dShape, dStride);
        pto::TSTORE(dGlobal, dTile);
    }
}

#endif // DISPATCH_MEGA_COMBINE_SWIGLU_FULL_ROW_IMPL_HPP
