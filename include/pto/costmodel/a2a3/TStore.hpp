/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSTORE_HPP
#define TSTORE_HPP
#include "pto/common/type.hpp"

namespace pto {

template <typename TileData, typename GlobalData, AtomicType currentAtomicType = AtomicType::AtomicNone,
          STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TSTORE_IMPL(GlobalData &dst, TileData &src)
{
    return;
}

template <typename TileData, typename GlobalData, AtomicType currentAtomicType = AtomicType::AtomicNone,
          ReluPreMode reluPreMode, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TSTORE_IMPL(GlobalData &dst, TileData &src)
{
    return;
}

template <typename TileData, typename GlobalData, AtomicType currentAtomicType = AtomicType::AtomicNone,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TSTORE_IMPL(GlobalData &dst, TileData &src, uint64_t preQuantScalar)
{
    return;
}

template <typename TileData, typename GlobalData, typename FpTileData,
          AtomicType currentAtomicType = AtomicType::AtomicNone, ReluPreMode reluPreMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TSTORE_IMPL(GlobalData &dst, TileData &src, FpTileData &fp)
{
    return;
}
} // namespace pto
#endif
