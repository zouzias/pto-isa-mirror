/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TGET_ASYNC_HPP
#define PTO_COMM_TGET_ASYNC_HPP

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/comm/comm_types.hpp"
#include "pto/comm/async/async_types.hpp"
#include "pto/npu/comm/async/sdma/sdma_async_intrin.hpp"

namespace pto {
namespace comm {

// ============================================================================
// TGET_ASYNC_IMPL: Asynchronous remote read operation implementation
//
// Directly transfers data from remote NPU's GM to local GM without UB staging.
// Returns AsyncEvent for synchronization with TSYNC.
//
// Data flow: srcGlobalData (remote GM) -> DMA Engine -> dstGlobalData (local GM)
// ============================================================================

namespace detail {

template <typename GlobalData>
PTO_INTERNAL bool TGetAsyncIsFlatContiguous1D(GlobalData &globalData)
{
    const int shape0 = globalData.GetShape(GlobalTensorDim::DIM_0);
    const int shape1 = globalData.GetShape(GlobalTensorDim::DIM_1);
    const int shape2 = globalData.GetShape(GlobalTensorDim::DIM_2);
    const int shape3 = globalData.GetShape(GlobalTensorDim::DIM_3);
    const int shape4 = globalData.GetShape(GlobalTensorDim::DIM_4);

    const int stride0 = globalData.GetStride(GlobalTensorDim::DIM_0);
    const int stride1 = globalData.GetStride(GlobalTensorDim::DIM_1);
    const int stride2 = globalData.GetStride(GlobalTensorDim::DIM_2);
    const int stride3 = globalData.GetStride(GlobalTensorDim::DIM_3);
    const int stride4 = globalData.GetStride(GlobalTensorDim::DIM_4);

    const bool isContiguous = (stride4 == 1) && (stride3 == shape4) && (stride2 == shape3 * stride3) &&
                              (stride1 == shape2 * stride2) && (stride0 == shape1 * stride1);
    const bool isLogical1D = (shape0 == 1 && shape1 == 1 && shape2 == 1 && shape3 == 1);
    return isContiguous && isLogical1D;
}

template <typename GlobalData>
PTO_INTERNAL uint32_t TGetAsyncGetTotalElemCount(GlobalData &globalData)
{
    return static_cast<uint32_t>(
        globalData.GetShape(GlobalTensorDim::DIM_0) * globalData.GetShape(GlobalTensorDim::DIM_1) *
        globalData.GetShape(GlobalTensorDim::DIM_2) * globalData.GetShape(GlobalTensorDim::DIM_3) *
        globalData.GetShape(GlobalTensorDim::DIM_4));
}

template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL bool TGetAsyncCheckTensorCompatibility()
{
    using SrcT = typename GlobalSrcData::RawDType;
    static_assert(std::is_same_v<SrcT, typename GlobalDstData::RawDType>, "TGET_ASYNC: src/dst element type mismatch");
    static_assert(GlobalSrcData::layout == GlobalDstData::layout, "TGET_ASYNC: src/dst layout mismatch");
    return true;
}

template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent TGET_ASYNC_SDMA_IMPL(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                                             const sdma::SdmaExecContext &execCtx)
{
    (void)TGetAsyncCheckTensorCompatibility<GlobalDstData, GlobalSrcData>();

    if (!TGetAsyncIsFlatContiguous1D(srcGlobalData)) {
        return AsyncEvent(0, DmaEngine::SDMA);
    }

    const uint32_t totalElems = TGetAsyncGetTotalElemCount(srcGlobalData);
    using T = typename GlobalSrcData::RawDType;
    const uint64_t eventHandle =
        sdma::__sdma_get_async(dstGlobalData.data(), srcGlobalData.data(), totalElems * sizeof(T), execCtx);
    return AsyncEvent(eventHandle, DmaEngine::SDMA);
}

} // namespace detail

// ============================================================================
// Main TGET_ASYNC_IMPL with DmaEngine template parameter
// ============================================================================

template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent TGET_ASYNC_IMPL(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                                        const AsyncSession &session)
{
    if constexpr (engine == DmaEngine::SDMA) {
        return detail::TGET_ASYNC_SDMA_IMPL(dstGlobalData, srcGlobalData, session.sdmaSession.execCtx);
    } else {
        PTO_ASSERT(false, "TGET_ASYNC: only SDMA engine is implemented currently");
        return AsyncEvent(0, engine);
    }
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TGET_ASYNC_HPP
