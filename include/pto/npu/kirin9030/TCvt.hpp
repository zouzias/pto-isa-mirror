/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/**
 * @file TCvt.hpp
 * @brief Type Conversion (TCVT) Implementation for NPU Kirin9030 Architecture
 */

#ifndef TCVT_HPP
#define TCVT_HPP

#include "pto/npu/kirin9030/common.hpp"
#include "pto/npu/kirin9030/utils.hpp"

#ifdef __DAV_VEC__

#include "pto/common/arch/register/tcvt_common.hpp"
namespace pto {
namespace kirin9030 {
using ::pto::TCVT_IMPL;
} // namespace kirin9030
} // namespace pto

#elif defined(__DAV_CUBE__)

namespace pto {
namespace kirin9030 {
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void TCVT_IMPL(TileDataD &dst, TileDataS &src, RoundMode mode, SaturationMode satMode,
                            bool needSetCtrl = true)
{}
template <typename TileDataD, typename TileDataS, typename TmpTileData>
PTO_INTERNAL void TCVT_IMPL(TileDataD &dst, TileDataS &src, TmpTileData &tmp, RoundMode mode, SaturationMode satMode,
                            bool needSetCtrl = true)
{}
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void TCVT_IMPL(TileDataD &dst, TileDataS &src, RoundMode mode, bool needSetCtrl = true)
{}

template <typename TileDataD, typename TileDataS, typename TmpTileData>
PTO_INTERNAL void TCVT_IMPL(TileDataD &dst, TileDataS &src, TmpTileData &tmp, RoundMode mode, bool needSetCtrl = true)
{}

} // namespace kirin9030
} // namespace pto
#endif // __DAV_VEC__

#endif // TCVT_HPP
