/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

using namespace pto;

constexpr int MAX_NGRAM_SIZE = 3;
constexpr int NUM_NGRAM_LAYERS = 2;
constexpr int NUM_EMBED_TABLE_PER_NGRAM = 8;
constexpr int NUM_OUT_COLS = (MAX_NGRAM_SIZE - 1) * NUM_EMBED_TABLE_PER_NGRAM;

template <typename T, int kTRows_, int kTCols_, int numTokens, int numOutCols>
__global__ AICORE void runEngramHashLayer(
    __gm__ T __out__ *output,
    __gm__ T __in__ *ngram_token_ids,
    __gm__ T __in__ *multipliers,
    __gm__ T __in__ *vocab_sizes,
    __gm__ T __in__ *offsets)
{
    using GlobalTokenIds = GlobalTensor<T, Shape<1, 1, 1, numTokens, MAX_NGRAM_SIZE>, Stride<1, 1, 1, MAX_NGRAM_SIZE, 1>>;
    using GlobalMultipliers = GlobalTensor<T, Shape<1, 1, 1, 1, MAX_NGRAM_SIZE>, Stride<1, 1, 1, 1, 1>>;
    using GlobalVocabSizes = GlobalTensor<T, Shape<1, 1, 1, MAX_NGRAM_SIZE - 1, NUM_EMBED_TABLE_PER_NGRAM>, Stride<1, 1, 1, NUM_EMBED_TABLE_PER_NGRAM, 1>>;
    using GlobalOffsets = GlobalTensor<T, Shape<1, 1, 1, 1, numOutCols>, Stride<1, 1, 1, 1, 1>>;
    using GlobalOutput = GlobalTensor<T, Shape<1, 1, 1, numTokens, numOutCols>, Stride<1, 1, 1, numOutCols, 1>>;

    using TileTokenIds = Tile<TileType::Vec, T, kTRows_, MAX_NGRAM_SIZE, BLayout::RowMajor, -1, -1>;
    using TileMultipliers = Tile<TileType::Vec, T, 1, MAX_NGRAM_SIZE, BLayout::RowMajor, -1, -1>;
    using TileVocabSizes = Tile<TileType::Vec, T, MAX_NGRAM_SIZE - 1, NUM_EMBED_TABLE_PER_NGRAM, BLayout::RowMajor, -1, -1>;
    using TileOffsets = Tile<TileType::Vec, T, 1, numOutCols, BLayout::RowMajor, -1, -1>;
    using TileOutput = Tile<TileType::Vec, T, kTRows_, numOutCols, BLayout::RowMajor, -1, -1>;
    using TileHash = Tile<TileType::Vec, T, kTRows_, 1, BLayout::RowMajor, -1, -1>;
    using TileTmp = Tile<TileType::Vec, T, kTRows_, numOutCols, BLayout::RowMajor, -1, -1>;
    using TileProd = Tile<TileType::Vec, T, kTRows_, MAX_NGRAM_SIZE, BLayout::RowMajor, -1, -1>;

    GlobalTokenIds tokenIdsGlobal(ngram_token_ids);
    GlobalMultipliers multipliersGlobal(multipliers);
    GlobalVocabSizes vocabSizesGlobal(vocab_sizes);
    GlobalOffsets offsetsGlobal(offsets);
    GlobalOutput outputGlobal(output);

    TileTokenIds tokenIdsTile;
    TileMultipliers multipliersTile;
    TileVocabSizes vocabSizesTile;
    TileOffsets offsetsTile;
    TileOutput outputTile;
    TileHash hashTile;
    TileTmp tmpTile;
    TileProd prodTile;

    TASSIGN<0x0>(tokenIdsTile);
    TASSIGN<kTRows_ * MAX_NGRAM_SIZE * sizeof(T)>(multipliersTile);
    TASSIGN<kTRows_ * MAX_NGRAM_SIZE * sizeof(T) + MAX_NGRAM_SIZE * sizeof(T)>(vocabSizesTile);
    TASSIGN<kTRows_ * MAX_NGRAM_SIZE * sizeof(T) + MAX_NGRAM_SIZE * sizeof(T) + (MAX_NGRAM_SIZE - 1) * NUM_EMBED_TABLE_PER_NGRAM * sizeof(T)>(offsetsTile);
    TASSIGN<kTRows_ * MAX_NGRAM_SIZE * sizeof(T) + MAX_NGRAM_SIZE * sizeof(T) + (MAX_NGRAM_SIZE - 1) * NUM_EMBED_TABLE_PER_NGRAM * sizeof(T) + numOutCols * sizeof(T)>(outputTile);
    TASSIGN<kTRows_ * MAX_NGRAM_SIZE * sizeof(T) + MAX_NGRAM_SIZE * sizeof(T) + (MAX_NGRAM_SIZE - 1) * NUM_EMBED_TABLE_PER_NGRAM * sizeof(T) + numOutCols * sizeof(T) + kTRows_ * numOutCols * sizeof(T)>(hashTile);
    TASSIGN<kTRows_ * MAX_NGRAM_SIZE * sizeof(T) + MAX_NGRAM_SIZE * sizeof(T) + (MAX_NGRAM_SIZE - 1) * NUM_EMBED_TABLE_PER_NGRAM * sizeof(T) + numOutCols * sizeof(T) + kTRows_ * numOutCols * sizeof(T) + kTRows_ * sizeof(T)>(tmpTile);
    TASSIGN<kTRows_ * MAX_NGRAM_SIZE * sizeof(T) + MAX_NGRAM_SIZE * sizeof(T) + (MAX_NGRAM_SIZE - 1) * NUM_EMBED_TABLE_PER_NGRAM * sizeof(T) + numOutCols * sizeof(T) + kTRows_ * numOutCols * sizeof(T) + kTRows_ * sizeof(T) + kTRows_ * numOutCols * sizeof(T)>(prodTile);

    TLOAD(tokenIdsTile, tokenIdsGlobal);
    TLOAD(multipliersTile, multipliersGlobal);
    TLOAD(vocabSizesTile, vocabSizesGlobal);
    TLOAD(offsetsTile, offsetsGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    TSTORE(outputGlobal, outputTile);
    output = outputGlobal.data();
}

template <typename T, int kTRows_, int kTCols_, int numTokens, int numOutCols>
__global__ AICORE void runEngramHash(
    __gm__ T __out__ *output,
    __gm__ T __in__ *ngram_token_ids,
    __gm__ T __in__ *multipliers,
    __gm__ T __in__ *vocab_sizes,
    __gm__ T __in__ *offsets)
{
    runEngramHashLayer<T, kTRows_, kTCols_, numTokens, numOutCols>(
        output,
        ngram_token_ids,
        multipliers,
        vocab_sizes,
        offsets);
}

template <typename T, int kTRows_, int kTCols_, int numTokens, int numOutCols>
void LaunchEngramHashLayer(
    T *output,
    T *ngram_token_ids,
    T *multipliers,
    T *vocab_sizes,
    T *offsets,
    void *stream)
{
    runEngramHashLayer<T, kTRows_, kTCols_, numTokens, numOutCols>
        <<<1, nullptr, stream>>>(output, ngram_token_ids, multipliers, vocab_sizes, offsets);
}

template <typename T, int kTRows_, int kTCols_, int numTokens, int numOutCols>
void LaunchEngramHash(
    T *output,
    T *ngram_token_ids,
    T *multipliers,
    T *vocab_sizes,
    T *offsets,
    void *stream)
{
    runEngramHash<T, kTRows_, kTCols_, numTokens, numOutCols>
        <<<1, nullptr, stream>>>(output, ngram_token_ids, multipliers, vocab_sizes, offsets);
}

template void LaunchEngramHash<int32_t, 1, MAX_NGRAM_SIZE, 1, NUM_OUT_COLS>(
    int32_t *output, int32_t *ngram_token_ids, int32_t *multipliers,
    int32_t *vocab_sizes, int32_t *offsets, void *stream);

template void LaunchEngramHash<int32_t, 16, MAX_NGRAM_SIZE, 16, NUM_OUT_COLS>(
    int32_t *output, int32_t *ngram_token_ids, int32_t *multipliers,
    int32_t *vocab_sizes, int32_t *offsets, void *stream);

template void LaunchEngramHash<int32_t, 128, MAX_NGRAM_SIZE, 128, NUM_OUT_COLS>(
    int32_t *output, int32_t *ngram_token_ids, int32_t *multipliers,
    int32_t *vocab_sizes, int32_t *offsets, void *stream);

template void LaunchEngramHash<int64_t, 1, MAX_NGRAM_SIZE, 1, NUM_OUT_COLS>(
    int64_t *output, int64_t *ngram_token_ids, int64_t *multipliers,
    int64_t *vocab_sizes, int64_t *offsets, void *stream);

template void LaunchEngramHash<int64_t, 16, MAX_NGRAM_SIZE, 16, NUM_OUT_COLS>(
    int64_t *output, int64_t *ngram_token_ids, int64_t *multipliers,
    int64_t *vocab_sizes, int64_t *offsets, void *stream);

template void LaunchEngramHash<int64_t, 128, MAX_NGRAM_SIZE, 128, NUM_OUT_COLS>(
    int64_t *output, int64_t *ngram_token_ids, int64_t *multipliers,
    int64_t *vocab_sizes, int64_t *offsets, void *stream);