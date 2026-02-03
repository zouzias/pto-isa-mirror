/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TGET_HPP
#define TGET_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"

namespace pto{
    
    template<typename GlobalDstData, typename GlobalSrcData>
    void TGet_Impl(typename GlobalDstData::DType *dst,
                    typename GlobalSrcData::DType *src,
                    unsigned validRow, 
                    unsigned validCol
                ) {
                    for (size_t i = 0; i < validRow; i++)
                    {
                        for (size_t j = 0; j < validCol; j++)
                        {
                            dst[i * validCol + j] = src[i * validCol + j];
                        }
                    }
                }

    template <typename GlobalDstData, typename GlobalSrcData, typename TileData>
    PTO_INTERNAL void TGET_IMPL(GlobalDstData &dst, GlobalSrcData &src, TileData &src1) {
        unsigned row = dst.GetShape(3);
        unsigned col = dst.GetShape(4);
        TGet_Impl<GlobalDstData, GlobalSrcData>(dst.data(), src.data(), row, col);
    }
}
#endif
