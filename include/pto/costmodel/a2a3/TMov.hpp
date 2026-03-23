/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TMOV_COSTMODEL_HPP
#define TMOV_COSTMODEL_HPP

#include <cstdint>
#include <pto/common/type.hpp>
#include "pto/costmodel/pto_isa_costmodel.hpp"

namespace pto {

// TMOV basic overload (Vec→Vec path):
//   NPU source calls TMovToVec → TCopy → copy_ubuf_to_ubuf (MTE1).
//   Costmodel replicates this by calling runCopyOp(dst, src).
template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src)
{
    using T = typename DstTileData::DType;
    auto stats = runMovVecOp(dst, src);
}

// TMOV with ReluPreMode (Acc→Mat path: copy_matrix_cc_to_cbuf, PIPE_M).
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src)
{
    using T = typename DstTileData::DType;
    auto stats = runMovCubeOp();
}

// TMOV with AccToVecMode (Acc→Vec path, PIPE_M).
template <typename DstTileData, typename SrcTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src)
{
    using T = typename DstTileData::DType;
    auto stats = runMovCubeOp();
}

// TMOV with FpTileData (vector-quant path: set_fpc + copy_matrix_cc_to_cbuf, PIPE_M).
template <typename DstTileData, typename SrcTileData, typename FpTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp)
{
    using T = typename DstTileData::DType;
    auto stats = runMovCubeOp();
}

// TMOV with FpTileData + AccToVecMode (vector-quant AccToVec, PIPE_M).
template <typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp)
{
    using T = typename DstTileData::DType;
    auto stats = runMovCubeOp();
}

// TMOV with scalar preQuantScalar (scalar-quant path: set_quant_pre + copy_matrix_cc_to_cbuf, PIPE_M).
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar)
{
    using T = typename DstTileData::DType;
    auto stats = runMovCubeOp();
}

// TMOV with AccToVecMode + scalar preQuantScalar (PIPE_M).
template <typename DstTileData, typename SrcTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar)
{
    using T = typename DstTileData::DType;
    auto stats = runMovCubeOp();
}

} // namespace pto

#endif // TMOV_COSTMODEL_HPP
