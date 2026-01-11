/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_A2A3_TUNARYOP_HPP
#define PTO_NPU_A2A3_TUNARYOP_HPP

#include <cstdint>
#include <type_traits>

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

// Consolidated A2/A3 templates (Binary/Unary/Reduce/...).
#include "pto/npu/a2a3/TAllTemplates.hpp"

namespace pto {

    /* RSQRT */

    template <typename TileData>
    __tf__ AICORE void TRsqrtCustom(typename TileData::TileDType __out__ dst,
                                        typename TileData::TileDType __in__ src,
                                        unsigned validRow,
                                        unsigned validCol) {
        __ubuf__ typename TileData::DType *dstPtr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
        __ubuf__ typename TileData::DType *srcPtr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(src);

        unsigned TShape0 = TileData::Rows;
        unsigned TShape1 = TileData::Cols;
        (void)TShape0;

        // ACCURATE_RSQRT path:
        // 1) compute `sqrt(x)` into dst
        // 2) compute `1 / sqrt(x)` into dst (using a constant-one buffer)
        __ubuf__ typename TileData::DType *ones = reinterpret_cast<__ubuf__ typename TileData::DType*>(static_cast<std::uintptr_t>(0x2fc00));
        vector_dup(ones, (typename TileData::DType)(1.0), 1, 1, 1, 8, 8);

        set_mask_count();
        set_vector_mask(0, validCol);
        for (uint32_t i = 0; i < validRow; ++i) {
            vsqrt((dstPtr + i * TShape1), (srcPtr + i * TShape1), 1, 1, 1, 8, 8);
        }
        pipe_barrier(PIPE_V);

        set_vector_mask(0, validCol);
        for (uint32_t i = 0; i < validRow; ++i) {
            vdiv((dstPtr + i * TShape1), (ones), (dstPtr + i * TShape1), 1, 1, 1, 1, 8, 0, 8); 
        }
        pipe_barrier(PIPE_V);

        set_mask_norm();
        set_vector_mask(-1, -1);
    }

    template<typename DataType>
    AICORE void _vrsqrt(__ubuf__ DataType* dst, __ubuf__ DataType* src,
                            uint8_t repeat, uint16_t dstBlockStride, uint16_t srcBlockStride,
                            uint8_t dstRepeatStride, uint8_t srcRepeatStride) {
        vrsqrt(dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }

    template <typename TileData>
    AICORE void TRSQRT_IMPL(TileData &dst, TileData &src) {
        static_assert(std::is_same<typename TileData::DType, float32_t>::value ||
                      std::is_same<typename TileData::DType, float>::value ||
                      std::is_same<typename TileData::DType, half>::value ||
                      std::is_same<typename TileData::DType, float16_t>::value,
                      "TRSQRT: Invalid data type");
        static_assert(TileData::Loc == TileType::Vec, "TRSQRT: TileType of src and dst tiles must be TileType::Vec.");
        static_assert(TileData::ValidCol <= TileData::Cols, "TRSQRT: Number of valid columns must not be greater than number of tile columns.");
        static_assert(TileData::ValidRow <= TileData::Rows, "TRSQRT: Number of valid rows must not be greater than number of tile rows.");
        static_assert(TileData::isRowMajor, "TRSQRT: Not supported Layout type");

        PTO_ASSERT(src.GetValidCol() == dst.GetValidCol(), "TRSQRT: Number of columns of src and dst must be the same.");
        PTO_ASSERT(src.GetValidRow() == dst.GetValidRow(), "TRSQRT: Number of rows of src and dst must be the same.");

        unsigned validCol = dst.GetValidCol();
        unsigned validRow = dst.GetValidRow();
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        constexpr unsigned rowStride = TileData::RowStride;

#ifdef ACCURATE_RSQRT
        TRsqrtCustom<TileData>(dst.data(), src.data(), validRow, validCol);
#else
        TUnaryOp<TileData, _vrsqrt, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), src.data(), validRow, validCol);
#endif
    }

    /* SQRT */

    template<typename DataType>
    AICORE void _vsqrt(__ubuf__ DataType* dst, __ubuf__ DataType* src, 
                            uint8_t repeat, uint16_t dstBlockStride, uint16_t srcBlockStride,
                            uint8_t dstRepeatStride, uint8_t srcRepeatStride) {
        vsqrt(dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }

    template <typename TileData>
    AICORE void TSQRT_IMPL(TileData &dst, TileData &src) {
        static_assert(std::is_same<typename TileData::DType, float32_t>::value ||
                      std::is_same<typename TileData::DType, float>::value ||
                      std::is_same<typename TileData::DType, half>::value ||
                      std::is_same<typename TileData::DType, float16_t>::value,
                      "TSQRT: Invalid data type");
        static_assert(TileData::Loc == TileType::Vec, "TSQRT: TileType of src and dst tiles must be TileType::Vec.");
        static_assert(TileData::ValidCol <= TileData::Cols, "TSQRT: Number of valid columns must not be greater than number of tile columns.");
        static_assert(TileData::ValidRow <= TileData::Rows, "TSQRT: Number of valid rows must not be greater than number of tile rows.");
        static_assert(TileData::isRowMajor, "TSQRT: Not supported Layout type");

        PTO_ASSERT(src.GetValidCol() == dst.GetValidCol(), "TSQRT: Number of columns of src and dst must be the same.");
        PTO_ASSERT(src.GetValidRow() == dst.GetValidRow(), "TSQRT: Number of rows of src and dst must be the same.");

        unsigned validCol = dst.GetValidCol();
        unsigned validRow = dst.GetValidRow();
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        constexpr unsigned rowStride = TileData::RowStride;
        TUnaryOp<TileData, _vsqrt, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), src.data(), validRow, validCol);
    }

    /* EXP */

    template<typename DataType>
    AICORE void _vexp(__ubuf__ DataType* dst, __ubuf__ DataType* src, 
                          uint8_t repeat, uint16_t dstBlockStride, uint16_t srcBlockStride,
                          uint8_t dstRepeatStride, uint8_t srcRepeatStride) {
        vexp(dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }

    template <typename TileData>
    AICORE void TEXP_IMPL(TileData &dst, TileData &src) {
        static_assert(std::is_same<typename TileData::DType, float32_t>::value ||
                      std::is_same<typename TileData::DType, float>::value ||
                      std::is_same<typename TileData::DType, half>::value ||
                      std::is_same<typename TileData::DType, float16_t>::value,
                      "TEXP: Invalid data type");
        static_assert(TileData::Loc == TileType::Vec, "TEXP: TileType of src and dst tiles must be TileType::Vec.");
        static_assert(TileData::ValidCol <= TileData::Cols, "TEXP: Number of valid columns must not be greater than number of tile columns.");
        static_assert(TileData::ValidRow <= TileData::Rows, "TEXP: Number of valid rows must not be greater than number of tile rows.");
        static_assert(TileData::isRowMajor, "TEXP: Not supported Layout type");

        PTO_ASSERT(src.GetValidCol() == dst.GetValidCol(), "TEXP: Number of columns of src and dst must be the same.");
        PTO_ASSERT(src.GetValidRow() == dst.GetValidRow(), "TEXP: Number of rows of src and dst must be the same.");

        unsigned validCol = dst.GetValidCol();
        unsigned validRow = dst.GetValidRow();
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        constexpr unsigned rowStride = TileData::RowStride;
        TUnaryOp<TileData, _vexp, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), src.data(), validRow, validCol);
    }

    /* ABS */

    template<typename DataType>
    AICORE void _vabs(__ubuf__ DataType* dst, __ubuf__ DataType* src, 
                          uint8_t repeat, uint16_t dstBlockStride, uint16_t srcBlockStride,
                          uint8_t dstRepeatStride, uint8_t srcRepeatStride) {
        vabs(dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }

    template <typename TileData>
    AICORE void TABS_IMPL(TileData &dst, TileData &src) {
        static_assert(std::is_same<typename TileData::DType, float32_t>::value ||
                      std::is_same<typename TileData::DType, float>::value ||
                      std::is_same<typename TileData::DType, half>::value ||
                      std::is_same<typename TileData::DType, float16_t>::value,
                      "TABS: Invalid data type");
        static_assert(TileData::Loc == TileType::Vec, "TABS: TileType of src and dst tiles must be TileType::Vec.");
        static_assert(TileData::ValidCol <= TileData::Cols, "TABS: Number of valid columns must not be greater than number of tile columns.");
        static_assert(TileData::ValidRow <= TileData::Rows, "TABS: Number of valid rows must not be greater than number of tile rows.");
        static_assert(TileData::isRowMajor, "TABS: Not supported Layout type");

        PTO_ASSERT(src.GetValidCol() == dst.GetValidCol(), "TABS: Number of columns of src and dst must be the same.");
        PTO_ASSERT(src.GetValidRow() == dst.GetValidRow(), "TABS: Number of rows of src and dst must be the same.");

        unsigned validCol = dst.GetValidCol();
        unsigned validRow = dst.GetValidRow();
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        constexpr unsigned rowStride = TileData::RowStride;
        TUnaryOp<TileData, _vabs, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), src.data(), validRow, validCol);
    }

    /* LOG */

    template<typename DataType>
    AICORE void _vlog(__ubuf__ DataType* dst, __ubuf__ DataType* src, 
                          uint8_t repeat, uint16_t dstBlockStride, uint16_t srcBlockStride,
                          uint8_t dstRepeatStride, uint8_t srcRepeatStride) {
        vln(dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }

    template <typename TileData>
    AICORE void TLOG_IMPL(TileData &dst, TileData &src) {
        static_assert(std::is_same<typename TileData::DType, float32_t>::value ||
                      std::is_same<typename TileData::DType, float>::value ||
                      std::is_same<typename TileData::DType, half>::value ||
                      std::is_same<typename TileData::DType, float16_t>::value,
                      "TLOG: Invalid data type");
        static_assert(TileData::Loc == TileType::Vec, "TLOG: TileType of src and dst tiles must be TileType::Vec.");
        static_assert(TileData::ValidCol <= TileData::Cols, "TLOG: Number of valid columns must not be greater than number of tile columns.");
        static_assert(TileData::ValidRow <= TileData::Rows, "TLOG: Number of valid rows must not be greater than number of tile rows.");
        static_assert(TileData::isRowMajor, "TLOG: Not supported Layout type");

        PTO_ASSERT(src.GetValidCol() == dst.GetValidCol(), "TLOG: Number of columns of src and dst must be the same.");
        PTO_ASSERT(src.GetValidRow() == dst.GetValidRow(), "TLOG: Number of rows of src and dst must be the same.");

        unsigned validCol = dst.GetValidCol();
        unsigned validRow = dst.GetValidRow();
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        constexpr unsigned rowStride = TileData::RowStride;
        TUnaryOp<TileData, _vlog, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), src.data(), validRow, validCol);
    }

    /* RECIP */

    template <typename TileData>
    __tf__ AICORE void TRecipCustom(typename TileData::TileDType __out__ dst,
                                        typename TileData::TileDType __in__ src,
                                        unsigned validRow,
                                        unsigned validCol) {
        __ubuf__ typename TileData::DType *dstPtr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
        __ubuf__ typename TileData::DType *srcPtr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(src);

        unsigned TShape0 = TileData::Rows;
        unsigned TShape1 = TileData::Cols;

        __ubuf__ typename TileData::DType *ones = reinterpret_cast<__ubuf__ typename TileData::DType*>(static_cast<std::uintptr_t>(0x2fc00));
        vector_dup(ones, (typename TileData::DType)(1.0), 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        set_mask_count();
        set_vector_mask(0, validCol);
        for (uint32_t i = 0; i < validRow; ++i) {
            vdiv((dstPtr + i * TShape1), (ones), (srcPtr + i * TShape1), 1, 1, 1, 1, 8, 0, 8); 
        }
        pipe_barrier(PIPE_V);

        set_mask_norm();
        set_vector_mask(-1, -1);
    }

    template<typename DataType>
    AICORE void _vrecip(__ubuf__ DataType* dst, __ubuf__ DataType* src, 
                          uint8_t repeat, uint16_t dstBlockStride, uint16_t srcBlockStride,
                          uint8_t dstRepeatStride, uint8_t srcRepeatStride) {
        vrec(dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }

    template <typename TileData>
    AICORE void TRECIP_IMPL(TileData &dst, TileData &src) {
        static_assert(std::is_same<typename TileData::DType, float32_t>::value ||
                      std::is_same<typename TileData::DType, float>::value ||
                      std::is_same<typename TileData::DType, half>::value ||
                      std::is_same<typename TileData::DType, float16_t>::value,
                      "TRECIP: Invalid data type");
        static_assert(TileData::Loc == TileType::Vec, "TRECIP: TileType of src and dst tiles must be TileType::Vec.");
        static_assert(TileData::ValidCol <= TileData::Cols, "TRECIP: Number of valid columns must not be greater than number of tile columns.");
        static_assert(TileData::ValidRow <= TileData::Rows, "TRECIP: Number of valid rows must not be greater than number of tile rows.");
        static_assert(TileData::isRowMajor, "TRECIP: Not supported Layout type");

        PTO_ASSERT(src.GetValidCol() == dst.GetValidCol(), "TRECIP: Number of columns of src and dst must be the same.");
        PTO_ASSERT(src.GetValidRow() == dst.GetValidRow(), "TRECIP: Number of rows of src and dst must be the same.");

        unsigned validCol = dst.GetValidCol();
        unsigned validRow = dst.GetValidRow();
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        constexpr unsigned rowStride = TileData::RowStride;
#ifdef ACCURATE_RECIP
        TRecipCustom<TileData>(dst.data(), src.data(), validRow, validCol);
#else
        TUnaryOp<TileData, _vrecip, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), src.data(), validRow, validCol);
#endif
    }
}

#endif
