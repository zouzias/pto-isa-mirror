/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <acl/acl.h>
#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename TileDataDst, typename TileDataSrc>
__tf__ PTO_INTERNAL void RunVCGMax(typename TileDataDst::TileDType __out__ dst,
                                   typename TileDataSrc::TileDType __in__ src, uint32_t validCols)
{
    __ubuf__ half *dstPtr = (__ubuf__ half *)__cce_get_tile_ptr(dst);
    __ubuf__ half *srcPtr = (__ubuf__ half *)__cce_get_tile_ptr(src);

    RegTensor<half> vreg_in;
    RegTensor<half> vreg_out;
    MaskReg preg_in = CreatePredicate<half>(validCols);
    MaskReg preg_out = pset_b16(PAT_VL8);

    vlds(vreg_in, srcPtr, 0, NORM);
    vcgmax((vector_f16 &)vreg_out, (vector_f16 &)vreg_in, preg_in);
    vsts((vector_f16 &)vreg_out, dstPtr, 0, NORM_B16, preg_out);
}

template <int srcCols, int srcValidCols, int dstCols, int dstValidCols>
PTO_INTERNAL void runVCGMax(__gm__ half *out, __gm__ half *src)
{
    using DynShape = Shape<1, 1, 1, -1, -1>;
    using DynStride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalDataSrc = GlobalTensor<half, DynShape, DynStride>;
    using GlobalDataDst = GlobalTensor<half, DynShape, DynStride>;

    GlobalDataSrc srcGlobal(src, DynShape(1, srcValidCols), DynStride(1, srcCols));
    GlobalDataDst dstGlobal(out, DynShape(1, dstValidCols), DynStride(1, dstCols));

    using SrcTileData = Tile<TileType::Vec, half, 1, srcCols, BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<TileType::Vec, half, 1, dstCols, BLayout::RowMajor, -1, -1>;

    SrcTileData srcTile(1, srcValidCols);
    DstTileData dstTile(1, dstValidCols);
    TASSIGN<0x0>(srcTile);
    TASSIGN<srcCols * sizeof(half)>(dstTile);

    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    __VEC_SCOPE__
    {
        RunVCGMax<DstTileData, SrcTileData>(dstTile.data(), srcTile.data(), srcValidCols);
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
}

extern "C" __global__ AICORE void launchVCGMAXCase1(__gm__ half *out, __gm__ half *src)
{
    runVCGMax<128, 16, 16, 8>(out, src);
}

template <uint32_t caseId>
void launchVCGMAXTestCase(void *out, void *src, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchVCGMAXCase1<<<1, nullptr, stream>>>((half *)out, (half *)src);
            break;
        }
        default: {
            break;
        }
    }
}

template void launchVCGMAXTestCase<1>(void *out, void *src, aclrtStream stream);
