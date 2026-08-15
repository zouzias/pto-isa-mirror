/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TROWPROD_HPP
#define TROWPROD_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "TRowReduceOps.hpp"

namespace pto {

template <typename T, typename TileDataOut, typename TileDataIn, typename TileDataTmp>
__tf__ PTO_INTERNAL void TRowProd(
    typename TileDataOut::TileDType __out__ dst, typename TileDataIn::TileDType __in__ src,
    typename TileDataTmp::TileDType __in__ tmp, int validRow, int validCol)
{
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    __ubuf__ T* tmpPtr = (__ubuf__ T*)__cce_get_tile_ptr(tmp);

    constexpr unsigned dstRowStride = TileDataOut::RowStride;
    constexpr unsigned srcRowStride = TileDataIn::RowStride;

    constexpr unsigned elemsPerRepeat = REPEAT_BYTE / sizeof(T);
    unsigned repeatsNum = validCol / elemsPerRepeat;
    unsigned repeatRemain = validCol % elemsPerRepeat;
    unsigned tmpRepeatsNum = TileDataTmp::RowStride / elemsPerRepeat;

    set_mask_count();

    for (unsigned row = 0; row < validRow; ++row, dstPtr += dstRowStride, srcPtr += srcRowStride) {
        for (unsigned i = 0; i < tmpRepeatsNum; i++) {
            vector_dup(tmpPtr + i * elemsPerRepeat, (T)1.0f, 1, 1, 1, 0, 0);
            pipe_barrier(PIPE_V);
        }
        for (unsigned i = 0; i < repeatsNum / 2; i++) {
            vmul(tmpPtr + i * elemsPerRepeat, srcPtr + row * srcRowStride + 2 * i * elemsPerRepeat,
                 srcPtr + row * srcRowStride + (2 * i + 1) * elemsPerRepeat, 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
        }
        unsigned tmpRepeatTime = repeatsNum / 2;
        unsigned j = 0;
        while (tmpRepeatTime > 0) {
            vmul(tmpPtr + j * elemsPerRepeat, tmpPtr + 2 * j * elemsPerRepeat, tmpPtr + (2 * j + 1) * elemsPerRepeat,
                 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            tmpRepeatTime /= 2;
            j++;
        }
        if (repeatsNum % 2 != 0) {
            vmul(tmpPtr, tmpPtr, srcPtr + row * srcRowStride + (repeatsNum - 1) * elemsPerRepeat, 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
        }
        if (repeatRemain > 0) {
            set_vector_mask(0, repeatRemain);
            vmul(tmpPtr, tmpPtr, srcPtr + row * srcRowStride + repeatsNum * elemsPerRepeat, 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            set_vector_mask(-1, -1);
        }
        unsigned tmpElement = validCol / elemsPerRepeat > 0 ? elemsPerRepeat : validCol;
        while (tmpElement > 0) {
            set_vector_mask(0, tmpElement / 2);
            vmul(tmpPtr, tmpPtr, tmpPtr + tmpElement / 2, 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            tmpElement /= 2;
            set_vector_mask(-1, -1);
        }

        PtoSetWaitFlag<PIPE_V, PIPE_S>();
        dstPtr[0] = tmpPtr[0];
        PtoSetWaitFlag<PIPE_S, PIPE_V>();
        set_vector_mask(-1, -1);
    }

    set_mask_norm();
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWPROD_IMPL(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp)
{
    using T = typename TileDataIn::DType;
    int validCol = src.GetValidCol();
    int validRow = src.GetValidRow();
    TRowReduceCheck<TileDataOut, TileDataIn>(validRow, validCol, dst.GetValidRow());

    TRowProd<T, TileDataOut, TileDataIn, TileDataTmp>(dst.data(), src.data(), tmp.data(), validRow, validCol);
}

} // namespace pto
#endif
