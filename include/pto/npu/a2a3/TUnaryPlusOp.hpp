/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_A2A3_TUNARYPLUSOP_HPP
#define PTO_NPU_A2A3_TUNARYPLUSOP_HPP

#include <type_traits>

#include "pto/npu/a2a3/TUnaryOp.hpp"

namespace pto {
  template <typename TileDataDst, typename TileDataSrc>
  PTO_INTERNAL void TUnaryPlusStaticCheck() {
    using T = typename TileDataDst::DType;
    static_assert(std::is_same_v<T, typename TileDataSrc::DType>,
                  "TUnaryPlusStaticCheck: The data type of dst must be consistent with src.");

    static_assert(std::is_same_v<T, float32_t> || std::is_same_v<T, float> ||
                  std::is_same_v<T, half> || std::is_same_v<T, float16_t>,
                  "TUnaryPlusStaticCheck: Invalid data type");

    static_assert(TileDataDst::Loc == TileType::Vec,
                  "TUnaryPlusStaticCheck: TileType of src and dst tiles must be TileType::Vec.");

    static_assert(TileDataDst::ValidCol <= TileDataDst::Cols,
                  "TUnaryPlusStaticCheck: Number of valid columns must not be greater than number of tile columns.");

    static_assert(TileDataDst::ValidRow <= TileDataDst::Rows,
                  "TUnaryPlusStaticCheck: Number of valid rows must not be greater than number of tile rows.");
  }

  template <typename TileDataDst, typename TileDataSrc>
  PTO_INTERNAL void TRSQRT_IMPL(TileDataDst &dst, TileDataSrc &src) {
    TUnaryPlusStaticCheck<TileDataDst, TileDataSrc>();
    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();
    PTO_ASSERT(dstValidRow == src.GetValidRow(), "TRSQRT: Number of rows of src and dst must be the same.");
    PTO_ASSERT(dstValidCol == src.GetValidCol(), "TRSQRT: Number of columns of src and dst must be the same.");
    if constexpr (std::is_same_v<TileDataDst, TileDataSrc>) {
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileDataDst::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataDst::DType);
        constexpr unsigned rowStride = TileDataDst::RowStride;
#ifdef ACCURATE_RSQRT
        TRsqrtCustom<TileDataDst>(dst.data(), src.data(), dstValidRow, dstValidCol);
#else
        TUnaryOp<TileDataDst, _vrsqrt, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), src.data(), dstValidRow, dstValidCol);
#endif
    } else {
        // Different dst/src strides: use the "+ (plus)" unary template.
        TUnaryPlusOp<TileDataDst, TileDataSrc, _vrsqrt>(dst.data(), src.data(), dstValidRow, dstValidCol);
    }
  }

  /* SQRT */
  template <typename TileDataDst, typename TileDataSrc>
  PTO_INTERNAL void TSQRT_IMPL(TileDataDst &dst, TileDataSrc &src) {
    TUnaryPlusStaticCheck<TileDataDst, TileDataSrc>();
    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();
    PTO_ASSERT(dstValidRow == src.GetValidRow(), "TSQRT: Number of rows of src and dst must be the same.");
    PTO_ASSERT(dstValidCol == src.GetValidCol(), "TSQRT: Number of columns of src and dst must be the same.");
    if constexpr (std::is_same_v<TileDataDst, TileDataSrc>) {
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileDataDst::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataDst::DType);
        constexpr unsigned rowStride = TileDataDst::RowStride;
        TUnaryOp<TileDataDst, _vsqrt, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), src.data(), dstValidRow, dstValidCol);
    } else {
        // Different dst/src strides: use the "+ (plus)" unary template.
        TUnaryPlusOp<TileDataDst, TileDataSrc, _vsqrt>(dst.data(), src.data(), dstValidRow, dstValidCol);
    }
  }

  /* EXP */
  template <typename TileDataDst, typename TileDataSrc>
  PTO_INTERNAL void TEXP_IMPL(TileDataDst &dst, TileDataSrc &src) {
    TUnaryPlusStaticCheck<TileDataDst, TileDataSrc>();
    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();
    PTO_ASSERT(dstValidRow == src.GetValidRow(), "TEXP: Number of rows of src and dst must be the same.");
    PTO_ASSERT(dstValidCol == src.GetValidCol(), "TEXP: Number of columns of src and dst must be the same.");
    if constexpr (std::is_same_v<TileDataDst, TileDataSrc>) {
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileDataDst::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataDst::DType);
        constexpr unsigned rowStride = TileDataDst::RowStride;
        TUnaryOp<TileDataDst, _vexp, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), src.data(), dstValidRow, dstValidCol);
    } else {
        // Different dst/src strides: use the "+ (plus)" unary template.
        TUnaryPlusOp<TileDataDst, TileDataSrc, _vexp>(dst.data(), src.data(), dstValidRow, dstValidCol);
    }
  }
}

#endif
