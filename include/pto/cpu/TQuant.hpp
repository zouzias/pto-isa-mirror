/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_TQUANT_HPP
#define PTO_CPU_TQUANT_HPP

#include <pto/common/cpu_stub.hpp>

namespace pto {

// CPU-SIM: TQUANT is target-specific (e.g., FP32->FP8 quantization). Provide a stub.
template <typename TileDataSrc, typename TileDataExp, typename TileDataOut,
          typename TileDataMax, int mode>
PTO_INTERNAL void TQUANT_IMPL(TileDataSrc &src,
                             TileDataExp &exp,
                             TileDataOut &dst,
                             TileDataMax &max,
                             TileDataSrc &scaling)
{
    (void)src;
    (void)exp;
    (void)dst;
    (void)max;
    (void)scaling;
    (void)mode;
    PTO_CPU_STUB_ASSERT(false && "TQUANT is not supported in CPU-SIM");
}

} // namespace pto

#endif
