/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TPUT_ASYNC_COMMON_DETAIL_HPP
#define PTO_COMM_TPUT_ASYNC_COMMON_DETAIL_HPP

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/comm/async_common/TGetAsyncCommonDetail.hpp"

namespace pto {
namespace comm {
namespace detail {

template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_SDMA_IMPL(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                                             const sdma::SdmaExecContext &execCtx)
{
    (void)TCommAsyncCheckTensorCompatibility<GlobalDstData, GlobalSrcData>("TPUT_ASYNC");

    PTO_ASSERT(srcGlobalData.data() != nullptr && dstGlobalData.data() != nullptr,
               "TPUT_ASYNC: src and dst tensor pointers must not be null.");

    PTO_ASSERT(TCommAsyncIsFlatContiguous1D(srcGlobalData),
               "TPUT_ASYNC: src tensor must be flat contiguous 1D (packed layout, single logical line). "
               "Multi-dimensional or non-contiguous tensors are not supported by SDMA async path.");
    PTO_ASSERT(TCommAsyncIsFlatContiguous1D(dstGlobalData),
               "TPUT_ASYNC: dst tensor must be flat contiguous 1D (packed layout, single logical line). "
               "Multi-dimensional or non-contiguous tensors are not supported by SDMA async path.");

    const uint32_t dstElems = TCommAsyncGetTotalElemCount(dstGlobalData);
    const uint32_t srcElems = TCommAsyncGetTotalElemCount(srcGlobalData);
    PTO_ASSERT(dstElems >= srcElems, "TPUT_ASYNC SDMA: dst buffer too small for src data.");

    using T = typename GlobalSrcData::RawDType;
    const uint64_t eventHandle =
        sdma::__sdma_put_async(dstGlobalData.data(), srcGlobalData.data(), srcElems * sizeof(T), execCtx);
    return AsyncEvent(eventHandle, DmaEngine::SDMA);
}

} // namespace detail
} // namespace comm
} // namespace pto

#endif // PTO_COMM_TPUT_ASYNC_COMMON_DETAIL_HPP
