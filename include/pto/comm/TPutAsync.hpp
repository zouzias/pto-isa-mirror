/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TPUT_ASYNC_HPP
#define PTO_COMM_TPUT_ASYNC_HPP

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/comm/comm_types.hpp"
#include "pto/comm/sdma_async_intrin.hpp"

#if defined(CANN_SHMEM)
// Include SHMEM library's SDMA implementation for better compatibility
#include "shmemi_device_sdma.hpp"
#endif

namespace pto {
namespace comm {

// ============================================================================
// TPUT_ASYNC_IMPL: Asynchronous remote write operation implementation
// 
// Directly transfers data from local GM to remote NPU's GM without UB staging.
// Returns AsyncEvent for synchronization with TSYNC.
//
// Data flow: srcGlobalData (local GM) → DMA Engine → dstGlobalData (remote GM)
// ============================================================================

namespace detail {

// SDMA implementation
template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_SDMA_IMPL(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData)
{
    using T = typename GlobalSrcData::RawDType;
    
    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>,
        "TPUT_ASYNC: src/dst element type mismatch");
    static_assert(GlobalSrcData::layout == GlobalDstData::layout,
        "TPUT_ASYNC: src/dst layout mismatch");

    // Get transfer parameters from GlobalTensor
    const int gShape0 = srcGlobalData.GetShape(GlobalTensorDim::DIM_0);
    const int gShape1 = srcGlobalData.GetShape(GlobalTensorDim::DIM_1);
    const int gShape2 = srcGlobalData.GetShape(GlobalTensorDim::DIM_2);
    const int gShape3 = srcGlobalData.GetShape(GlobalTensorDim::DIM_3);
    const int gShape4 = srcGlobalData.GetShape(GlobalTensorDim::DIM_4);
    
    const uint32_t totalElems = gShape0 * gShape1 * gShape2 * gShape3 * gShape4;
    const uint64_t transferSize = totalElems * sizeof(T);
    
#if defined(CANN_SHMEM)
    // Use SHMEM library's SDMA implementation for better compatibility
    aclshmemi_sdma_post_send(
        reinterpret_cast<__gm__ uint8_t*>(dstGlobalData.data()),
        reinterpret_cast<__gm__ uint8_t*>(srcGlobalData.data()),
        0,  // opcode
        transferSize);
    uint64_t eventHandle = reinterpret_cast<uint64_t>(srcGlobalData.data());
#else
    uint64_t eventHandle = sdma::__sdma_put_async(dstGlobalData.data(), srcGlobalData.data(),
                                                  transferSize);
#endif
    
    return AsyncEvent(eventHandle, DmaEngine::SDMA);
}

// URMA implementation
template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_URMA_IMPL(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData)
{
    using T = typename GlobalSrcData::RawDType;
    
    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>,
        "TPUT_ASYNC: src/dst element type mismatch");
    static_assert(GlobalSrcData::layout == GlobalDstData::layout,
        "TPUT_ASYNC: src/dst layout mismatch");

    // Get transfer parameters from GlobalTensor
    const int gShape0 = srcGlobalData.GetShape(GlobalTensorDim::DIM_0);
    const int gShape1 = srcGlobalData.GetShape(GlobalTensorDim::DIM_1);
    const int gShape2 = srcGlobalData.GetShape(GlobalTensorDim::DIM_2);
    const int gShape3 = srcGlobalData.GetShape(GlobalTensorDim::DIM_3);
    const int gShape4 = srcGlobalData.GetShape(GlobalTensorDim::DIM_4);
    
    const uint32_t totalElems = gShape0 * gShape1 * gShape2 * gShape3 * gShape4;
    
    // TODO: Call actual URMA PUT intrinsic
    // uint64_t eventHandle = __urma_put_async(dstGlobalData.data(), srcGlobalData.data(), 
    //                                          totalElems * sizeof(T));
    uint64_t eventHandle = 1;  // Placeholder
    
    return AsyncEvent(eventHandle, DmaEngine::URMA);
}

} // namespace detail

// ============================================================================
// Main TPUT_ASYNC_IMPL with DmaEngine template parameter
// ============================================================================

template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_IMPL(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData)
{
    if constexpr (engine == DmaEngine::SDMA) {
        return detail::TPUT_ASYNC_SDMA_IMPL(dstGlobalData, srcGlobalData);
    } else {
        return detail::TPUT_ASYNC_URMA_IMPL(dstGlobalData, srcGlobalData);
    }
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TPUT_ASYNC_HPP
