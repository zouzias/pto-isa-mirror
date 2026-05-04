/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

template<typename SrcT, typename DstT, int SrcRows, int SrcCols, int DstRows, int DstCols, 
         int IdxRow, int IdxCol, bool UseRelu>
AICORE inline void RunTEXTRACTQuantScalar(__gm__ DstT* out, __gm__ SrcT* srcIn, __gm__ uint64_t* quantIn)
{
    using SrcShape = pto::Shape<1, 1, 1, SrcRows, SrcCols>;
    using SrcStride = pto::Stride<1 * SrcRows * SrcCols, 1 * SrcRows * SrcCols, SrcRows * SrcCols, SrcCols, 1>;
    using SrcGlobal = GlobalTensor<SrcT, SrcShape, SrcStride>;
    
    using DstShape = pto::Shape<1, 1, 1, DstRows, DstCols>;
    using DstStride = pto::Stride<1 * DstRows * DstCols, 1 * DstRows * DstCols, DstRows * DstCols, DstCols, 1>;
    using DstGlobal = GlobalTensor<DstT, DstShape, DstStride>;
    
    using SrcTile = Tile<TileType::Acc, SrcT, SrcRows, SrcCols, BLayout::ColMajor, SrcRows, SrcCols, SLayout::RowMajor, 512>;
    using DstTile = Tile<TileType::Vec, DstT, DstRows, DstCols, BLayout::RowMajor, DstRows, DstCols, SLayout::NoneBox, 512>;
    
    SrcGlobal srcGlobal(srcIn);
    DstGlobal outGlobal(out);
    
    SrcTile srcTile;
    DstTile dstTile;
    
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);
    
    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    
    uint64_t preQuantScalar = quantIn[0];
    constexpr ReluPreMode reluMode = UseRelu ? ReluPreMode::NormalRelu : ReluPreMode::NoRelu;
    TEXTRACT(dstTile, srcTile, preQuantScalar, IdxRow, IdxCol);
    
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    
    TSTORE(outGlobal, dstTile);
}

template<typename SrcT, typename DstT, int SrcRows, int SrcCols, int DstRows, int DstCols, 
         int IdxRow, int IdxCol, bool UseRelu>
AICORE inline void RunTEXTRACTQuantVector(__gm__ DstT* out, __gm__ SrcT* srcIn, __gm__ uint64_t* quantIn)
{
    using SrcShape = pto::Shape<1, 1, 1, SrcRows, SrcCols>;
    using SrcStride = pto::Stride<1 * SrcRows * SrcCols, 1 * SrcRows * SrcCols, SrcRows * SrcCols, SrcCols, 1>;
    using SrcGlobal = GlobalTensor<SrcT, SrcShape, SrcStride>;
    
    using DstShape = pto::Shape<1, 1, 1, DstRows, DstCols>;
    using DstStride = pto::Stride<1 * DstRows * DstCols, 1 * DstRows * DstCols, DstRows * DstCols, DstCols, 1>;
    using DstGlobal = GlobalTensor<DstT, DstShape, DstStride>;
    
    using QuantShape = pto::Shape<1, 1, 1, 1, DstCols>;
    using QuantStride = pto::Stride<1 * DstCols, DstCols, DstCols, DstCols, 1>;
    using QuantGlobal = GlobalTensor<uint64_t, QuantShape, QuantStride>;
    
    using SrcTile = Tile<TileType::Acc, SrcT, SrcRows, SrcCols, BLayout::ColMajor, SrcRows, SrcCols, SLayout::RowMajor, 512>;
    using DstTile = Tile<TileType::Vec, DstT, DstRows, DstCols, BLayout::RowMajor, DstRows, DstCols, SLayout::NoneBox, 512>;
    using QuantTile = Tile<TileType::Scaling, uint64_t, 1, DstCols, BLayout::RowMajor, 1, DstCols, SLayout::NoneBox, 512>;
    
    SrcGlobal srcGlobal(srcIn);
    DstGlobal outGlobal(out);
    QuantGlobal quantGlobal(quantIn);
    
    SrcTile srcTile;
    DstTile dstTile;
    QuantTile quantTile;
    
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);
    TASSIGN(quantTile, 0x20000);
    
    TLOAD(srcTile, srcGlobal);
    TLOAD(quantTile, quantGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    
    constexpr ReluPreMode reluMode = UseRelu ? ReluPreMode::NormalRelu : ReluPreMode::NoRelu;
    TEXTRACT_FP(dstTile, srcTile, quantTile, IdxRow, IdxCol);
    
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    
    TSTORE(outGlobal, dstTile);
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_1(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantScalar<int32_t, int8_t, 128, 64, 128, 64, 0, 0, false>(
        reinterpret_cast<__gm__ int8_t*>(out),
        reinterpret_cast<__gm__ int32_t*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_2(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantScalar<int32_t, int8_t, 128, 64, 96, 32, 0, 0, true>(
        reinterpret_cast<__gm__ int8_t*>(out),
        reinterpret_cast<__gm__ int32_t*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_3(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantScalar<int32_t, int8_t, 128, 128, 64, 64, 0, 0, false>(
        reinterpret_cast<__gm__ int8_t*>(out),
        reinterpret_cast<__gm__ int32_t*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_4(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantScalar<int32_t, int8_t, 256, 128, 128, 64, 0, 0, true>(
        reinterpret_cast<__gm__ int8_t*>(out),
        reinterpret_cast<__gm__ int32_t*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_5(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantVector<int32_t, int8_t, 128, 64, 64, 32, 8, 0, false>(
        reinterpret_cast<__gm__ int8_t*>(out),
        reinterpret_cast<__gm__ int32_t*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_6(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantVector<int32_t, int8_t, 96, 96, 64, 64, 0, 0, true>(
        reinterpret_cast<__gm__ int8_t*>(out),
        reinterpret_cast<__gm__ int32_t*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_7(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantVector<int32_t, int8_t, 128, 128, 96, 96, 0, 0, false>(
        reinterpret_cast<__gm__ int8_t*>(out),
        reinterpret_cast<__gm__ int32_t*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_8(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantVector<int32_t, int8_t, 256, 64, 128, 32, 0, 0, true>(
        reinterpret_cast<__gm__ int8_t*>(out),
        reinterpret_cast<__gm__ int32_t*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_9(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantScalar<int32_t, uint8_t, 128, 64, 128, 64, 0, 0, false>(
        reinterpret_cast<__gm__ uint8_t*>(out),
        reinterpret_cast<__gm__ int32_t*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_10(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantScalar<int32_t, uint8_t, 128, 64, 96, 32, 0, 0, true>(
        reinterpret_cast<__gm__ uint8_t*>(out),
        reinterpret_cast<__gm__ int32_t*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_11(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantScalar<int32_t, uint8_t, 128, 128, 64, 64, 0, 0, false>(
        reinterpret_cast<__gm__ uint8_t*>(out),
        reinterpret_cast<__gm__ int32_t*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_12(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantScalar<int32_t, uint8_t, 256, 128, 128, 64, 0, 0, true>(
        reinterpret_cast<__gm__ uint8_t*>(out),
        reinterpret_cast<__gm__ int32_t*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_13(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantVector<int32_t, uint8_t, 128, 64, 64, 32, 8, 0, false>(
        reinterpret_cast<__gm__ uint8_t*>(out),
        reinterpret_cast<__gm__ int32_t*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_14(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantVector<int32_t, uint8_t, 96, 96, 64, 64, 0, 0, true>(
        reinterpret_cast<__gm__ uint8_t*>(out),
        reinterpret_cast<__gm__ int32_t*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_15(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantVector<int32_t, uint8_t, 128, 128, 96, 96, 0, 0, false>(
        reinterpret_cast<__gm__ uint8_t*>(out),
        reinterpret_cast<__gm__ int32_t*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_16(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantVector<int32_t, uint8_t, 256, 64, 128, 32, 0, 0, true>(
        reinterpret_cast<__gm__ uint8_t*>(out),
        reinterpret_cast<__gm__ int32_t*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_17(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantScalar<int32_t, uint16_t, 128, 64, 128, 64, 0, 0, false>(
        reinterpret_cast<__gm__ uint16_t*>(out),
        reinterpret_cast<__gm__ int32_t*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_18(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantScalar<int32_t, uint16_t, 128, 64, 96, 32, 0, 0, true>(
        reinterpret_cast<__gm__ uint16_t*>(out),
        reinterpret_cast<__gm__ int32_t*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_19(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantScalar<int32_t, uint16_t, 128, 128, 64, 64, 0, 0, false>(
        reinterpret_cast<__gm__ uint16_t*>(out),
        reinterpret_cast<__gm__ int32_t*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_20(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantScalar<int32_t, uint16_t, 256, 128, 128, 64, 0, 0, true>(
        reinterpret_cast<__gm__ uint16_t*>(out),
        reinterpret_cast<__gm__ int32_t*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_21(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantVector<int32_t, uint16_t, 128, 64, 64, 32, 8, 0, false>(
        reinterpret_cast<__gm__ uint16_t*>(out),
        reinterpret_cast<__gm__ int32_t*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_22(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantVector<int32_t, uint16_t, 96, 96, 64, 64, 0, 0, true>(
        reinterpret_cast<__gm__ uint16_t*>(out),
        reinterpret_cast<__gm__ int32_t*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_23(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantVector<int32_t, uint16_t, 128, 128, 96, 96, 0, 0, false>(
        reinterpret_cast<__gm__ uint16_t*>(out),
        reinterpret_cast<__gm__ int32_t*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_24(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantVector<int32_t, uint16_t, 256, 64, 128, 32, 0, 0, true>(
        reinterpret_cast<__gm__ uint16_t*>(out),
        reinterpret_cast<__gm__ int32_t*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_25(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantScalar<float, int8_t, 128, 64, 128, 64, 0, 0, false>(
        reinterpret_cast<__gm__ int8_t*>(out),
        reinterpret_cast<__gm__ float*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_26(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantScalar<float, int8_t, 128, 64, 96, 32, 0, 0, true>(
        reinterpret_cast<__gm__ int8_t*>(out),
        reinterpret_cast<__gm__ float*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_27(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantScalar<float, int8_t, 128, 128, 64, 64, 0, 0, false>(
        reinterpret_cast<__gm__ int8_t*>(out),
        reinterpret_cast<__gm__ float*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_28(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantScalar<float, int8_t, 256, 128, 128, 64, 0, 0, true>(
        reinterpret_cast<__gm__ int8_t*>(out),
        reinterpret_cast<__gm__ float*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_29(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantVector<float, int8_t, 128, 64, 64, 32, 8, 0, false>(
        reinterpret_cast<__gm__ int8_t*>(out),
        reinterpret_cast<__gm__ float*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_30(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantVector<float, int8_t, 96, 96, 64, 64, 0, 0, true>(
        reinterpret_cast<__gm__ int8_t*>(out),
        reinterpret_cast<__gm__ float*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_31(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantVector<float, int8_t, 128, 128, 96, 96, 0, 0, false>(
        reinterpret_cast<__gm__ int8_t*>(out),
        reinterpret_cast<__gm__ float*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_32(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantVector<float, int8_t, 256, 64, 128, 32, 0, 0, true>(
        reinterpret_cast<__gm__ int8_t*>(out),
        reinterpret_cast<__gm__ float*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_33(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantScalar<float, uint8_t, 128, 64, 128, 64, 0, 0, false>(
        reinterpret_cast<__gm__ uint8_t*>(out),
        reinterpret_cast<__gm__ float*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_34(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantScalar<float, uint8_t, 128, 64, 96, 32, 0, 0, true>(
        reinterpret_cast<__gm__ uint8_t*>(out),
        reinterpret_cast<__gm__ float*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_35(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantScalar<float, uint8_t, 128, 128, 64, 64, 0, 0, false>(
        reinterpret_cast<__gm__ uint8_t*>(out),
        reinterpret_cast<__gm__ float*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_36(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantScalar<float, uint8_t, 256, 128, 128, 64, 0, 0, true>(
        reinterpret_cast<__gm__ uint8_t*>(out),
        reinterpret_cast<__gm__ float*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_37(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantVector<float, uint8_t, 128, 64, 64, 32, 8, 0, false>(
        reinterpret_cast<__gm__ uint8_t*>(out),
        reinterpret_cast<__gm__ float*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_38(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantVector<float, uint8_t, 96, 96, 64, 64, 0, 0, true>(
        reinterpret_cast<__gm__ uint8_t*>(out),
        reinterpret_cast<__gm__ float*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_39(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantVector<float, uint8_t, 128, 128, 96, 96, 0, 0, false>(
        reinterpret_cast<__gm__ uint8_t*>(out),
        reinterpret_cast<__gm__ float*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_40(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantVector<float, uint8_t, 256, 64, 128, 32, 0, 0, true>(
        reinterpret_cast<__gm__ uint8_t*>(out),
        reinterpret_cast<__gm__ float*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_41(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantScalar<float, uint16_t, 128, 64, 128, 64, 0, 0, false>(
        reinterpret_cast<__gm__ uint16_t*>(out),
        reinterpret_cast<__gm__ float*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_42(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantScalar<float, uint16_t, 128, 64, 96, 32, 0, 0, true>(
        reinterpret_cast<__gm__ uint16_t*>(out),
        reinterpret_cast<__gm__ float*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_43(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantScalar<float, uint16_t, 128, 128, 64, 64, 0, 0, false>(
        reinterpret_cast<__gm__ uint16_t*>(out),
        reinterpret_cast<__gm__ float*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_44(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantScalar<float, uint16_t, 256, 128, 128, 64, 0, 0, true>(
        reinterpret_cast<__gm__ uint16_t*>(out),
        reinterpret_cast<__gm__ float*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_45(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantScalar<float, uint16_t, 128, 64, 128, 64, 0, 0, false>(
        reinterpret_cast<__gm__ uint16_t*>(out),
        reinterpret_cast<__gm__ float*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_46(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantScalar<float, uint16_t, 128, 64, 96, 32, 0, 0, true>(
        reinterpret_cast<__gm__ uint16_t*>(out),
        reinterpret_cast<__gm__ float*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_47(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantScalar<float, uint16_t, 128, 128, 64, 64, 0, 0, false>(
        reinterpret_cast<__gm__ uint16_t*>(out),
        reinterpret_cast<__gm__ float*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}

extern "C" __global__ AICORE void launchTEXTRACTQuant_48(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* quantIn)
{
    RunTEXTRACTQuantScalar<float, uint16_t, 256, 128, 128, 64, 0, 0, true>(
        reinterpret_cast<__gm__ uint16_t*>(out),
        reinterpret_cast<__gm__ float*>(srcIn),
        reinterpret_cast<__gm__ uint64_t*>(quantIn));
}


template <int32_t testKey>
void launchTEXTRACTQuant(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream)
{
    if constexpr (testKey == 1) {
        launchTEXTRACTQuant_1<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 2) {
        launchTEXTRACTQuant_2<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 3) {
        launchTEXTRACTQuant_3<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 4) {
        launchTEXTRACTQuant_4<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 5) {
        launchTEXTRACTQuant_5<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 6) {
        launchTEXTRACTQuant_6<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 7) {
        launchTEXTRACTQuant_7<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 8) {
        launchTEXTRACTQuant_8<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 9) {
        launchTEXTRACTQuant_9<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 10) {
        launchTEXTRACTQuant_10<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 11) {
        launchTEXTRACTQuant_11<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 12) {
        launchTEXTRACTQuant_12<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 13) {
        launchTEXTRACTQuant_13<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 14) {
        launchTEXTRACTQuant_14<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 15) {
        launchTEXTRACTQuant_15<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 16) {
        launchTEXTRACTQuant_16<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 17) {
        launchTEXTRACTQuant_17<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 18) {
        launchTEXTRACTQuant_18<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 19) {
        launchTEXTRACTQuant_19<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 20) {
        launchTEXTRACTQuant_20<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 21) {
        launchTEXTRACTQuant_21<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 22) {
        launchTEXTRACTQuant_22<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 23) {
        launchTEXTRACTQuant_23<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 24) {
        launchTEXTRACTQuant_24<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 25) {
        launchTEXTRACTQuant_25<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 26) {
        launchTEXTRACTQuant_26<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 27) {
        launchTEXTRACTQuant_27<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 28) {
        launchTEXTRACTQuant_28<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 29) {
        launchTEXTRACTQuant_29<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 30) {
        launchTEXTRACTQuant_30<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 31) {
        launchTEXTRACTQuant_31<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 32) {
        launchTEXTRACTQuant_32<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 33) {
        launchTEXTRACTQuant_33<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 34) {
        launchTEXTRACTQuant_34<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 35) {
        launchTEXTRACTQuant_35<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 36) {
        launchTEXTRACTQuant_36<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 37) {
        launchTEXTRACTQuant_37<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 38) {
        launchTEXTRACTQuant_38<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 39) {
        launchTEXTRACTQuant_39<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 40) {
        launchTEXTRACTQuant_40<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 41) {
        launchTEXTRACTQuant_41<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 42) {
        launchTEXTRACTQuant_42<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 43) {
        launchTEXTRACTQuant_43<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 44) {
        launchTEXTRACTQuant_44<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 45) {
        launchTEXTRACTQuant_45<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 46) {
        launchTEXTRACTQuant_46<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 47) {
        launchTEXTRACTQuant_47<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    } else if constexpr (testKey == 48) {
        launchTEXTRACTQuant_48<<<1, nullptr, stream>>>(out, srcIn, quantIn);
    }
}

template void launchTEXTRACTQuant<1>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<2>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<3>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<4>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<5>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<6>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<7>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<8>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<9>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<10>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<11>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<12>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<13>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<14>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<15>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<16>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<17>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<18>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<19>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<20>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<21>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<22>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<23>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<24>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<25>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<26>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<27>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<28>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<29>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<30>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<31>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<32>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<33>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<34>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<35>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<36>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<37>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<38>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<39>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<40>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<41>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<42>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<43>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<44>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<45>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<46>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<47>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
template void launchTEXTRACTQuant<48>(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);
