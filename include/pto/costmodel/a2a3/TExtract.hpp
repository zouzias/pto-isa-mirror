/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TEXTRACT_COSTMODEL_HPP
#define TEXTRACT_COSTMODEL_HPP

#include <cstdint>
#include <pto/common/type.hpp>
#include "pto/costmodel/pto_isa_costmodel.hpp"

namespace pto {

// TEXTRACT: img2colv2_cbuf_to_ca / load_cbuf_to_ca_transpose / load_cbuf_to_cb (PIPE_M cube DMA).
// All overloads share the same Op.

// 1. Basic overload (no extra template param)
template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol)
{
    using T = typename SrcTileData::DType;
    auto stats = runExtractOp();
    dst.SetCycle(CostModel::GetInstance().PredictCycle<T>(stats));
}

// 2. With dstValidCol (5 runtime args)
template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol,
                                uint16_t dstValidCol)
{
    using T = typename SrcTileData::DType;
    auto stats = runExtractOp();
    dst.SetCycle(CostModel::GetInstance().PredictCycle<T>(stats));
}

// 3. With ReluPreMode (4 runtime args)
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol)
{
    using T = typename SrcTileData::DType;
    auto stats = runExtractOp();
    dst.SetCycle(CostModel::GetInstance().PredictCycle<T>(stats));
}

// 4. With ReluPreMode + preQuantScalar (5 runtime args)
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow,
                                uint16_t indexCol)
{
    using T = typename SrcTileData::DType;
    auto stats = runExtractOp();
    dst.SetCycle(CostModel::GetInstance().PredictCycle<T>(stats));
}

// 5. With FpTileData + ReluPreMode
template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow,
                                uint16_t indexCol)
{
    using T = typename SrcTileData::DType;
    auto stats = runExtractOp();
    dst.SetCycle(CostModel::GetInstance().PredictCycle<T>(stats));
}

} // namespace pto

#endif // TEXTRACT_COSTMODEL_HPP
