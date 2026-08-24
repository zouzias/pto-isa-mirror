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

using pto::BLayout;
using pto::CmpMode;
using pto::GlobalTensor;
using pto::Shape;
using pto::Stride;
using pto::Tile;
using pto::TileType;

template <typename T, int Rows, int Cols, int ValidRows, int ValidCols, CmpMode cmpMode, bool isSrc1Tile>
__global__ AICORE void runTCmps(__gm__ uint8_t* out, __gm__ T* src0, __gm__ T* src1)
{
#include "oversize_bodies/run_tcmps_body.inl"
}

template <typename T, int Rows, int Cols, int ValidRows, int ValidCols, CmpMode cmpMode, bool isSrc1Tile>
void LaunchTCmps(uint8_t* out, T* src0, T* src1, void* stream)
{
    if constexpr (std::is_same_v<T, uint16_t>) {
        runTCmps<half, Rows, Cols, ValidRows, ValidCols, cmpMode, isSrc1Tile>
            <<<1, nullptr, stream>>>((out), (half*)(src0), (half*)(src1));
    } else {
        runTCmps<T, Rows, Cols, ValidRows, ValidCols, cmpMode, isSrc1Tile><<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template void LaunchTCmps<uint16_t, 32, 32, 32, 32, CmpMode::EQ, false>(
    uint8_t* out, uint16_t* src0, uint16_t* src1, void* stream);
template void LaunchTCmps<float, 8, 64, 8, 64, CmpMode::GT, true>(uint8_t* out, float* src0, float* src1, void* stream);
template void LaunchTCmps<int32_t, 4, 64, 4, 64, CmpMode::NE, false>(
    uint8_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTCmps<int32_t, 128, 128, 64, 64, CmpMode::LT, true>(
    uint8_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTCmps<int32_t, 64, 64, 32, 32, CmpMode::EQ, false>(
    uint8_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTCmps<int32_t, 16, 32, 16, 32, CmpMode::EQ, true>(
    uint8_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTCmps<float, 128, 128, 64, 64, CmpMode::LE, false>(
    uint8_t* out, float* src0, float* src1, void* stream);
template void LaunchTCmps<int32_t, 77, 80, 32, 32, CmpMode::EQ, true>(
    uint8_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTCmps<int32_t, 32, 32, 32, 32, CmpMode::EQ, false>(
    uint8_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTCmps<int16_t, 32, 32, 16, 32, CmpMode::EQ, true>(
    uint8_t* out, int16_t* src0, int16_t* src1, void* stream);
template void LaunchTCmps<int16_t, 77, 80, 32, 32, CmpMode::LE, false>(
    uint8_t* out, int16_t* src0, int16_t* src1, void* stream);
