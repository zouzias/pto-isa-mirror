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
#include "textract_acc2mat_layout_cases.h"

using namespace pto;

template <int Key>
__global__ AICORE void RunTExtractLayout(
    __gm__ int32_t* out, __gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ int32_t* bias)
{
    using C = TExtractLayoutCase<Key>;
    using Input = std::conditional_t<C::KindValue == 2, int8_t, half>;
    using Acc = std::conditional_t<C::KindValue == 2, int32_t, float>;
    using Output = std::conditional_t<C::KindValue == 0, half, std::conditional_t<C::KindValue == 1, bfloat16_t, Acc>>;
    constexpr int M = C::MValue, K = C::KValue, N = C::NValue;
    constexpr int R = C::RowsValue, W = C::ColsValue;
    constexpr int VR = C::ValidRowsValue, VC = C::ValidColsValue;
    constexpr STPhase phase = static_cast<STPhase>(C::PhaseValue);
    constexpr int count = phase == STPhase::Partial ? 2 : 1;
    constexpr int tileBytes = R * W * sizeof(Output);
    constexpr int bytes = 256 + count * tileBytes;
    constexpr int words = bytes / sizeof(int32_t);
    constexpr int base = 0x20000;
    constexpr int start = base - 128;
#if defined(__DAV_CUBE__)
    using AGlobal = GlobalTensor<Input, Shape<1, 1, 1, M, K>, pto::Stride<M * K, M * K, M * K, K, 1>>;
    using BGlobal = GlobalTensor<Input, Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, 1, K>, Layout::DN>;
    using RawGlobal = GlobalTensor<int32_t, Shape<1, 1, 1, 1, words>, pto::Stride<words, words, words, words, 1>>;
    using RawMat = Tile<TileType::Mat, int32_t, 1, words, BLayout::RowMajor, 1, words>;
    RawMat initial;
    TASSIGN(initial, start);
    RawGlobal outputGlobal(out);
    TLOAD(initial, outputGlobal);
    Tile<TileType::Mat, Input, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor> am;
    Tile<TileType::Mat, Input, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor> bm;
    TASSIGN(am, 0);
    TASSIGN(bm, 0x10000);
    AGlobal ag(reinterpret_cast<__gm__ Input*>(a));
    BGlobal bg(reinterpret_cast<__gm__ Input*>(b));
    TLOAD(am, ag);
    TLOAD(bm, bg);
    using BiasGlobal = GlobalTensor<int32_t, Shape<1, 1, 1, 1, N>, pto::Stride<N, N, N, N, 1>>;
    Tile<TileType::Mat, int32_t, 1, N, BLayout::RowMajor, 1, N> biasMat;
    Tile<TileType::Bias, int32_t, 1, N, BLayout::RowMajor, 1, N> biasTile;
    TASSIGN(biasMat, 0x30000);
    TASSIGN(biasTile, 0);
    if constexpr (C::BiasValue) {
        BiasGlobal biasGlobal(bias);
        TLOAD(biasMat, biasGlobal);
    }
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    constexpr int step = C::SplitValue ? K / 2 : K;
    TileLeft<Input, M, step> at;
    TileRight<Input, step, N> bt;
    TileAcc<Acc, M, N> acc;
    TASSIGN(at, 0);
    TASSIGN(bt, 0);
    TASSIGN(acc, 0);
    TEXTRACT(at, am, 0, 0);
    TEXTRACT(bt, bm, 0, 0);
    if constexpr (C::BiasValue)
        TMOV(biasTile, biasMat);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    if constexpr (C::BiasValue) {
        TMATMUL_BIAS<AccPhase::Final>(acc, at, bt, biasTile);
    } else if constexpr (C::SplitValue) {
        TMATMUL<AccPhase::Partial>(acc, at, bt);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        TEXTRACT(at, am, 0, step);
        TEXTRACT(bt, bm, step, 0);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        TMATMUL_ACC<AccPhase::Final>(acc, at, bt);
    } else {
        TMATMUL<phase == STPhase::Unspecified ? AccPhase::Unspecified : AccPhase::Final>(acc, at, bt);
    }
    if constexpr (phase == STPhase::Unspecified) {
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    }
    using Dst = Tile<TileType::Mat, Output, R, W, BLayout::ColMajor, -1, -1, SLayout::RowMajor, C::FractalValue>;
    Dst dst(VR, VC);
    TASSIGN(dst, base);
    constexpr ReluPreMode relu = C::ReluValue ? ReluPreMode::NormalRelu : ReluPreMode::NoRelu;
    if constexpr (C::PlainValue) {
        static_assert(phase == STPhase::Unspecified && !C::ReluValue);
        TEXTRACT(dst, acc, C::RowValue, C::ColValue);
    } else {
        TEXTRACT<phase, Dst, decltype(acc), relu>(dst, acc, C::RowValue, C::ColValue);
    }
    if constexpr (phase == STPhase::Partial) {
        Dst last(VR, VC);
        TASSIGN(last, base + tileBytes);
        TEXTRACT<STPhase::Final, Dst, decltype(acc), relu>(last, acc, 0, 0);
    }
    set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    copy_cbuf_to_ubuf((__ubuf__ void*)0, (__cbuf__ void*)start, 0, 1, bytes / 32, 0, 0);
    set_intra_block(PIPE_MTE1, 0);
    set_intra_block(PIPE_MTE1, 16);
#endif
#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, 0);
    if (get_subblockid() == 0) {
        Tile<TileType::Vec, int32_t, 1, words> vec;
        TASSIGN(vec, 0);
        GlobalTensor<int32_t, Shape<1, 1, 1, 1, words>, pto::Stride<words, words, words, words, 1>> outputGlobal(out);
        TSTORE(outputGlobal, vec);
    }
#endif
}

template <int Key>
void launchTExtractLayout(uint8_t* out, uint8_t* a, uint8_t* b, uint8_t* bias, void* stream)
{
    RunTExtractLayout<Key>
        <<<1, nullptr, stream>>>(reinterpret_cast<int32_t*>(out), a, b, reinterpret_cast<int32_t*>(bias));
}

template void launchTExtractLayout<1>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractLayout<2>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractLayout<3>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractLayout<4>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractLayout<5>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractLayout<6>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractLayout<7>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractLayout<8>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractLayout<9>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractLayout<10>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractLayout<11>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractLayout<12>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractLayout<13>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractLayout<14>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractLayout<15>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractLayout<16>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractLayout<17>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractLayout<18>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractLayout<19>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractLayout<20>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractLayout<21>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractLayout<22>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractLayout<23>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractLayout<24>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractLayout<25>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractLayout<26>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
