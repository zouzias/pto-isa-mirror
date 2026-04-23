/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TPREFETCH_L2_HPP
#define PTO_COMM_TPREFETCH_L2_HPP

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/comm/comm_types.hpp"
#include "pto/comm/async/async_types.hpp"
#include "pto/npu/comm/async/sdma/sdma_async_intrin.hpp"
#include "pto/npu/comm/async/sdma/sdma_cmo_intrin.hpp"

namespace pto {
namespace comm {

namespace detail {

template <typename GlobalData>
PTO_INTERNAL bool TPrefetchL2IsFlatContiguous1D(GlobalData &globalData)
{
    const int dim0 = globalData.GetShape(GlobalTensorDim::DIM_0);
    const int dim1 = globalData.GetShape(GlobalTensorDim::DIM_1);
    const int dim2 = globalData.GetShape(GlobalTensorDim::DIM_2);
    const int dim3 = globalData.GetShape(GlobalTensorDim::DIM_3);
    const int dim4 = globalData.GetShape(GlobalTensorDim::DIM_4);

    const int pitch0 = globalData.GetStride(GlobalTensorDim::DIM_0);
    const int pitch1 = globalData.GetStride(GlobalTensorDim::DIM_1);
    const int pitch2 = globalData.GetStride(GlobalTensorDim::DIM_2);
    const int pitch3 = globalData.GetStride(GlobalTensorDim::DIM_3);
    const int pitch4 = globalData.GetStride(GlobalTensorDim::DIM_4);

    const bool hasPackedLayout = (pitch4 == 1) && (pitch3 == dim4) && (pitch2 == dim3 * pitch3) &&
                                 (pitch1 == dim2 * pitch2) && (pitch0 == dim1 * pitch1);
    const bool isSingleLine = (dim0 == 1 && dim1 == 1 && dim2 == 1 && dim3 == 1);
    return hasPackedLayout && isSingleLine;
}

template <typename GlobalData>
PTO_INTERNAL uint64_t TPrefetchL2GetTotalBytes(GlobalData &globalData)
{
    const uint64_t d0 = static_cast<uint64_t>(globalData.GetShape(GlobalTensorDim::DIM_0));
    const uint64_t d1 = static_cast<uint64_t>(globalData.GetShape(GlobalTensorDim::DIM_1));
    const uint64_t d2 = static_cast<uint64_t>(globalData.GetShape(GlobalTensorDim::DIM_2));
    const uint64_t d3 = static_cast<uint64_t>(globalData.GetShape(GlobalTensorDim::DIM_3));
    const uint64_t d4 = static_cast<uint64_t>(globalData.GetShape(GlobalTensorDim::DIM_4));
    using T = typename GlobalData::RawDType;
    return (((d0 * d1) * d2) * d3) * d4 * sizeof(T);
}

template <typename GlobalData>
PTO_INTERNAL AsyncEvent TPREFETCH_L2_SDMA_IMPL(GlobalData &srcGlobalData, const sdma::SdmaExecContext &execCtx)
{
    if (srcGlobalData.data() == nullptr) {
        return AsyncEvent(0, DmaEngine::SDMA);
    }

    if (!TPrefetchL2IsFlatContiguous1D(srcGlobalData)) {
        return AsyncEvent(0, DmaEngine::SDMA);
    }

    const uint64_t totalBytes = TPrefetchL2GetTotalBytes(srcGlobalData);
    if (totalBytes == 0) {
        return AsyncEvent(0, DmaEngine::SDMA);
    }

    const uint64_t eventHandle =
        sdma::__sdma_cmo_prefetch(srcGlobalData.data(), totalBytes, execCtx);
    return AsyncEvent(eventHandle, DmaEngine::SDMA);
}

PTO_INTERNAL AsyncEvent TPREFETCH_L2_RAW_SDMA_IMPL(__gm__ void *src, uint64_t bytes,
                                                     const sdma::SdmaExecContext &execCtx)
{
    if (src == nullptr || bytes == 0) {
        return AsyncEvent(0, DmaEngine::SDMA);
    }

    const uint64_t eventHandle =
        sdma::__sdma_cmo_prefetch(reinterpret_cast<__gm__ uint8_t *>(src), bytes, execCtx);
    return AsyncEvent(eventHandle, DmaEngine::SDMA);
}

} // namespace detail

// GlobalTensor overload
template <typename GlobalData>
PTO_INTERNAL AsyncEvent TPREFETCH_L2_IMPL(GlobalData &srcGlobalData, const AsyncSession &session)
{
    return detail::TPREFETCH_L2_SDMA_IMPL(srcGlobalData, session.sdmaSession.execCtx);
}

// Raw pointer overload
PTO_INTERNAL AsyncEvent TPREFETCH_L2_IMPL(__gm__ void *src, uint64_t bytes, const AsyncSession &session)
{
    return detail::TPREFETCH_L2_RAW_SDMA_IMPL(src, bytes, session.sdmaSession.execCtx);
}

// Direct SdmaExecContext overloads (for users who manage context manually)
template <typename GlobalData>
PTO_INTERNAL AsyncEvent TPREFETCH_L2_IMPL(GlobalData &srcGlobalData, const sdma::SdmaExecContext &execCtx)
{
    return detail::TPREFETCH_L2_SDMA_IMPL(srcGlobalData, execCtx);
}

PTO_INTERNAL AsyncEvent TPREFETCH_L2_IMPL(__gm__ void *src, uint64_t bytes, const sdma::SdmaExecContext &execCtx)
{
    return detail::TPREFETCH_L2_RAW_SDMA_IMPL(src, bytes, execCtx);
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TPREFETCH_L2_HPP
