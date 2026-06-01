/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MOE_DISPATCH_COMBINE_A8W8_KERNEL_A3_INT8_BACKEND_HPP_
#define MOE_DISPATCH_COMBINE_A8W8_KERNEL_A3_INT8_BACKEND_HPP_

#include <pto/common/type.hpp>
#include <pto/pto-inst.hpp>

#include "moe_new_dispatch_combine_a8w8_types.hpp"

namespace moe_new_dispatch_combine_a8w8 {

struct ExpertSegment {
    uint32_t localExpert = 0;
    uint32_t rowBegin = 0;
    uint32_t rowCount = 0;
};

struct OwnerSegment {
    uint32_t tokenOwnerRank = 0;
    uint32_t srcRowBegin = 0;
    uint32_t rowCount = 0;
    uint64_t dstOffsetD = 0;
};

struct A3Int8Backend {
    PTO_INTERNAL void RunGmm1(ExpertSegment segment)
    {
        // M2 call site: TMATMUL/TMATMUL_ACC int8 x int8 -> int32 accumulator.
        (void)segment;
    }

    PTO_INTERNAL void RunActivationAndQuant(ExpertSegment segment)
    {
        // M2 call site: TCVT/TMUL/TADD for dequant/SwiGLU prep, then TQUANT to int8.
        (void)segment;
    }

    PTO_INTERNAL void RunGmm2(ExpertSegment segment)
    {
        // M2 call site: TMATMUL/TMATMUL_ACC over gmm2InputInt8 and int8 W2.
        (void)segment;
    }

    PTO_INTERNAL void RunGmm2EpilogueAndReturn(ExpertSegment segment, OwnerSegment owner)
    {
        // M2/M3 call site: TCVT/TMUL output cast followed by direct TPUT + TNOTIFY return.
        (void)segment;
        (void)owner;
    }
};

} // namespace moe_new_dispatch_combine_a8w8

#endif // MOE_DISPATCH_COMBINE_A8W8_KERNEL_A3_INT8_BACKEND_HPP_
