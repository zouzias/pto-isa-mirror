/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_TIMG2COL_HPP
#define PTO_CPU_TIMG2COL_HPP

#include <cstdint>

#include <pto/common/cpu_stub.hpp>
#include <pto/common/type.hpp>

namespace pto {

// CPU-SIM: TIMG2COL is not implemented. Provide a stub so code compiles and fails fast if executed.
template <typename TileData, typename ConvTileData, SetFmatrixMode FmatrixMode, typename T>
PTO_INTERNAL void TIMG2COL_IMPL(TileData &dst, ConvTileData &src,
                               uint16_t posM, uint16_t posK,
                               const Img2colTileConfig<T> &cfg)
{
    (void)dst;
    (void)src;
    (void)posM;
    (void)posK;
    (void)cfg;
    PTO_CPU_STUB_ASSERT(false && "TIMG2COL is not supported in CPU-SIM");
}

} // namespace pto

#endif
