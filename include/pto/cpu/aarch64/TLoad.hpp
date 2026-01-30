/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TLOAD_HPP
#define TLOAD_HPP

#include <unistd.h>
#include <cassert>
#include <algorithm>
#include <limits>
#include "pto/cpu/parallel.hpp"

namespace pto {
    template <typename TileData>
    AICORE constexpr typename TileData::DType getPadValue()
    {    
        switch (TileData::PadVal)
        {
            case PadValue::Null:
            case PadValue::Zero: return typename TileData::DType(0);
            case PadValue::Min:
                if constexpr(std::numeric_limits<typename TileData::DType>::has_infinity) {
                    return -std::numeric_limits<typename TileData::DType>::infinity();
                } else {
                    return std::numeric_limits<typename TileData::DType>::min();
                }
            case PadValue::Max:
                if constexpr(std::numeric_limits<typename TileData::DType>::has_infinity) {
                    return std::numeric_limits<typename TileData::DType>::infinity();
                } else {
                    return std::numeric_limits<typename TileData::DType>::max();
                }
        }
        return 0;
    }

    template <typename GlobalData, typename TileData>
    void LoadPackedA(typename TileData::DType* dst, typename GlobalData::DType* src, int m, int k, int stride0, int stride1) {
        int m_blocks = (m + PACKED_ROW - 1) / PACKED_ROW;

        for (size_t mb = 0; mb < m_blocks; ++mb) {
            int m_start = mb * PACKED_ROW;
            int m_len = std::min(PACKED_ROW, m - m_start);
            auto* dst_ptr = dst + mb * k * PACKED_ROW;

            for (int ki = 0; ki < k; ++ki) {
                for (int mi = 0; mi < PACKED_ROW; ++mi) {
                    if (mi < m_len) {
                        dst_ptr[ki * PACKED_ROW + mi] = src[(m_start + mi) * stride0 + ki * stride1];
                    } else {
                        dst_ptr[ki * PACKED_ROW + mi] = getPadValue<TileData>();
                    }
                }
            }
        }
    }

    template <typename GlobalData, typename TileData>
    void LoadPackedB(typename TileData::DType* dst, typename GlobalData::DType* src, int n, int k, int stride0, int stride1) {
        int n_blocks = (n + PACKED_COL - 1) / PACKED_COL;

        for (size_t nb = 0; nb < n_blocks; ++nb) {
            int n_start = nb * PACKED_COL;
            int n_len = std::min(PACKED_COL, n - n_start);
            auto* dst_ptr = dst + nb * k * PACKED_COL;

            for (int ki = 0; ki < k; ++ki) {
                for (int ni = 0; ni < PACKED_COL; ++ni) {
                    if (ni < n_len) {
                        dst_ptr[ki * PACKED_COL + ni] = src[ki * stride0 + (n_start + ni) * stride1];
                    } else {
                        dst_ptr[ki * PACKED_COL + ni] = getPadValue<TileData>();
                    }
                }
            }
        }
    }
    

    template <typename GlobalData, typename TileData>
    __tf__  PTO_INLINE void LoadGeneric(typename TileData::DType *dst, typename GlobalData::DType* src,
        int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
        int gStride3, int gStride4, int validRow, int validCol) {
        
        if constexpr (TileData::layout == Layout::ND || TileData::layout == Layout::DN) {
            for (int i = 0; i < validRow; ++i) {
                for (int j = 0; j < validCol; ++j) {
                    int64_t src_offset = 0;
                    if constexpr (GlobalData::layout == Layout::ND) {
                        src_offset = i * gStride3 + j * gStride4;
                    } else {
                        src_offset = j * gStride3 + i * gStride4;
                    }
                    dst[i * validCol + j] = src[src_offset];
                }
            }
        }
    }

    template <typename TileData, typename GlobalData>
    PTO_INTERNAL void TLOAD_IMPL(TileData &dst, GlobalData &src)
    {
        static_assert(sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
                      "Source dtype must be same with dst dtype");
        
        if constexpr (TileData::BFractal == BLayout::PackedA) {
            LoadPackedA<GlobalData, TileData>(dst.data(), src.data(),
            dst.GetValidRow(), dst.GetValidCol(),
            src.GetStride(pto::GlobalTensorDim::DIM_0),
            src.GetStride(pto::GlobalTensorDim::DIM_1));
        } else if constexpr (TileData::BFractal == BLayout::PackedB) {
            LoadPackedB<GlobalData, TileData>(dst.data(), src.data(),
            dst.GetValidCol(), dst.GetValidRow(),
            src.GetStride(pto::GlobalTensorDim::DIM_0),
            src.GetStride(pto::GlobalTensorDim::DIM_1));
        } else {
            if constexpr (GlobalData::layout == pto::Layout::ND || GlobalData::layout == pto::Layout::DN) {
                TLoad<TileData, GlobalData>(dst.data(),
                    src.data(),
                    src.GetShape(pto::GlobalTensorDim::DIM_0),
                    src.GetShape(pto::GlobalTensorDim::DIM_1),
                    src.GetShape(pto::GlobalTensorDim::DIM_2),
                    src.GetShape(pto::GlobalTensorDim::DIM_3),
                    src.GetShape(pto::GlobalTensorDim::DIM_4),
                    src.GetStride(pto::GlobalTensorDim::DIM_0),
                    src.GetStride(pto::GlobalTensorDim::DIM_1),
                    src.GetStride(pto::GlobalTensorDim::DIM_2),
                    src.GetStride(pto::GlobalTensorDim::DIM_3),
                    src.GetStride(pto::GlobalTensorDim::DIM_4),
                    dst.GetValidRow(),
                    dst.GetValidCol());
            }
        }
    }
}
#endif
