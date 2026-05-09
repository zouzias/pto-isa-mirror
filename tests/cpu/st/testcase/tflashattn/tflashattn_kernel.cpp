/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#include <cmath>

#include <pto/pto-inst.hpp>

using namespace pto;

namespace {

constexpr int kSeqLen = 64;
constexpr int kHeadDim = 32;

using QPlain = Tile<TileType::Vec, float, kSeqLen, kHeadDim, BLayout::RowMajor, kSeqLen, kHeadDim, SLayout::NoneBox>;
using KPlain = Tile<TileType::Vec, float, kSeqLen, kHeadDim, BLayout::RowMajor, kSeqLen, kHeadDim, SLayout::NoneBox>;
using KTPlain = Tile<TileType::Vec, float, kHeadDim, kSeqLen, BLayout::RowMajor, kHeadDim, kSeqLen, SLayout::NoneBox>;
using VPlain = Tile<TileType::Vec, float, kSeqLen, kHeadDim, BLayout::RowMajor, kSeqLen, kHeadDim, SLayout::NoneBox>;
using ScoresPlain = Tile<TileType::Vec, float, kSeqLen, kSeqLen, BLayout::RowMajor, kSeqLen, kSeqLen, SLayout::NoneBox>;
using RowReducePlain =
    Tile<TileType::Vec, float, kSeqLen, kSeqLen, BLayout::ColMajor, kSeqLen, kSeqLen, SLayout::NoneBox>;
using LeftQ = TileLeft<float, kSeqLen, kHeadDim, kSeqLen, kHeadDim>;
using RightKT = TileRight<float, kHeadDim, kSeqLen, kHeadDim, kSeqLen>;
using AccScores = TileAcc<float, kSeqLen, kSeqLen, kSeqLen, kSeqLen>;
using LeftP = TileLeft<float, kSeqLen, kSeqLen, kSeqLen, kSeqLen>;
using RightV = TileRight<float, kSeqLen, kHeadDim, kSeqLen, kHeadDim>;
using AccOut = TileAcc<float, kSeqLen, kHeadDim, kSeqLen, kHeadDim>;

struct FlashAttnTiles {
    QPlain qTile;
    KPlain kTile;
    KTPlain ktTile;
    VPlain vTile;
    LeftQ qLeft;
    RightKT kRight;
    AccScores scoresAcc;
    ScoresPlain scores;
    RowReducePlain rowMax;
    ScoresPlain scoresCentered;
    ScoresPlain expScores;
    RowReducePlain rowSum;
    ScoresPlain probs;
    LeftP pLeft;
    RightV vRight;
    AccOut outAcc;
};

template <typename GlobalQ, typename GlobalK, typename GlobalV>
AICORE inline void LoadInputs(FlashAttnTiles &tiles, GlobalQ &qGlobal, GlobalK &kGlobal, GlobalV &vGlobal)
{
    TLOAD(tiles.qTile, qGlobal);
    TLOAD(tiles.kTile, kGlobal);
    TLOAD(tiles.vTile, vGlobal);
    TMOV(tiles.qLeft, tiles.qTile);
    TTRANS(tiles.ktTile, tiles.kTile, tiles.kTile);
    TMOV(tiles.kRight, tiles.ktTile);
}

AICORE inline void ComputeProbabilities(FlashAttnTiles &tiles, float scale)
{
    TMATMUL(tiles.scoresAcc, tiles.qLeft, tiles.kRight);
    TMOV(tiles.scores, tiles.scoresAcc);
    TMULS(tiles.scores, tiles.scores, scale);
    TROWMAX(tiles.rowMax, tiles.scores, tiles.scores);
    TROWEXPANDSUB(tiles.scoresCentered, tiles.scores, tiles.rowMax);
    TEXP(tiles.expScores, tiles.scoresCentered);
    TROWSUM(tiles.rowSum, tiles.expScores, tiles.expScores);
    TROWEXPANDDIV(tiles.probs, tiles.expScores, tiles.rowSum);
}

template <typename GlobalO>
AICORE inline void StoreOutput(FlashAttnTiles &tiles, GlobalO &oGlobal)
{
    TMOV(tiles.pLeft, tiles.probs);
    TMOV(tiles.vRight, tiles.vTile);
    TMATMUL(tiles.outAcc, tiles.pLeft, tiles.vRight);
    TSTORE(oGlobal, tiles.outAcc);
}

template <typename GlobalQ, typename GlobalK, typename GlobalV, typename GlobalO>
AICORE inline void RunFlashAttnCore(GlobalQ &qGlobal, GlobalK &kGlobal, GlobalV &vGlobal, GlobalO &oGlobal)
{
    FlashAttnTiles tiles;
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    LoadInputs(tiles, qGlobal, kGlobal, vGlobal);
    ComputeProbabilities(tiles, scale);
    StoreOutput(tiles, oGlobal);
}

} // namespace

__global__ AICORE void RunTFLASHATTN(__gm__ float *out, __gm__ float *q, __gm__ float *k, __gm__ float *v)
{
    using GlobalQ = GlobalTensor<float, Shape<1, 1, 1, kSeqLen, kHeadDim>,
                                 Stride<kSeqLen * kHeadDim, kSeqLen * kHeadDim, kSeqLen * kHeadDim, kHeadDim, 1>>;
    using GlobalK = GlobalTensor<float, Shape<1, 1, 1, kSeqLen, kHeadDim>,
                                 Stride<kSeqLen * kHeadDim, kSeqLen * kHeadDim, kSeqLen * kHeadDim, kHeadDim, 1>>;
    using GlobalV = GlobalTensor<float, Shape<1, 1, 1, kSeqLen, kHeadDim>,
                                 Stride<kSeqLen * kHeadDim, kSeqLen * kHeadDim, kSeqLen * kHeadDim, kHeadDim, 1>>;
    using GlobalO = GlobalTensor<float, Shape<1, 1, 1, kSeqLen, kHeadDim>,
                                 Stride<kSeqLen * kHeadDim, kSeqLen * kHeadDim, kSeqLen * kHeadDim, kHeadDim, 1>>;

    GlobalQ qGlobal(q);
    GlobalK kGlobal(k);
    GlobalV vGlobal(v);
    GlobalO oGlobal(out);

    RunFlashAttnCore(qGlobal, kGlobal, vGlobal, oGlobal);
    out = oGlobal.data();
}

void LaunchTFLASHATTN(float *out, float *q, float *k, float *v, void *stream)
{
    (void)stream;
    RunTFLASHATTN(out, q, k, v);
}
