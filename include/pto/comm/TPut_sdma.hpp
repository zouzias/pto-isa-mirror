/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TPUT_SDMA_HPP
#define PTO_COMM_TPUT_SDMA_HPP

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/common/pto_tile.hpp"
#include "pto/comm/comm_types.hpp"
#include "pto/comm/sdma/sdma.hpp"
#include "pto/comm/sdma/sdma_impl.hpp"
#include <cstdint>

namespace pto {
namespace comm {

// ============================================================================
// TPUT_SDMA: Asynchronous remote write operation using SDMA engine
// 
// Native implementation using Ascend SDMA intrinsics.
// Data flow: srcGlobal (local GM) -> SDMA -> dstGlobal (remote GM)
// 
// This is an asynchronous operation that returns immediately and completes
// in the background, allowing computation-communication overlap.
//
// Parameters:
//   - dstGlobal: Destination GlobalTensor on remote PE
//   - srcGlobal: Source GlobalTensor on local PE
//   - numRows: (optional) Number of rows to transfer
//   - numCols: (optional) Number of columns to transfer
//
// Returns:
//   - SdmaEvent: Event handle for synchronization
//
// Constraints:
//   - Element type must match between src and dst
//   - Layout must match between src and dst
//   - Element size must be 1, 2, 4, or 8 bytes
//   - Addresses should be 32-byte aligned for optimal performance
//   - No UB staging required (direct GM to GM transfer)
//
// Note: If numRows and numCols are not specified, the transfer size is
// determined by the GlobalTensor shape.
// ============================================================================

// TPUT_SDMA_IMPL: Implementation using GlobalTensor shape
template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL SdmaEvent TPUT_SDMA_IMPL(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal)
{
    // Delegate to SDMA class put method
    return sdma::SDMA::put(dstGlobal, srcGlobal);
}

// TPUT_SDMA_IMPL: Implementation with explicit size specification
template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL SdmaEvent TPUT_SDMA_IMPL(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal,
                                      uint32_t numRows, uint32_t numCols)
{
    using T = typename GlobalSrcData::RawDType;
    uint64_t transfer_size = static_cast<uint64_t>(numRows) * numCols * sizeof(T);
    
    // Delegate to SDMA class put method with explicit size
    return sdma::SDMA::put(dstGlobal, srcGlobal, transfer_size);
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TPUT_SDMA_HPP
