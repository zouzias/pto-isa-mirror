/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#ifndef __PTO_RESHAPE_KIRIN9030__
#define __PTO_RESHAPE_KIRIN9030__

#include <type_traits>

#include "pto/common/pto_tile.hpp"
#include "pto/npu/kirin9030/TAssign.hpp"

namespace pto {

template <typename TileDataOut, typename TileDataIn>
PTO_INTERNAL void TRESHAPE_IMPL(TileDataOut &dst, TileDataIn &src)
{
#ifndef __PTO_AUTO__
    static_assert(is_tile_data_v<TileDataIn>, "input must be a Tile instance.");
    static_assert(is_tile_data_v<TileDataOut>, "output must be a Tile instance.");

    using DType = typename TileDataIn::DType;
    using NewElement = typename TileDataOut::DType;

    constexpr auto Loc = TileDataIn::Loc;
    constexpr auto NewLoc = TileDataOut::Loc;

    constexpr int Numel = TileDataIn::Numel;
    constexpr int NewNumel = TileDataOut::Numel;

    constexpr auto SFractal = TileDataIn::SFractal;
    constexpr auto NewSFractal = TileDataOut::SFractal;

    static_assert(Loc == NewLoc, "TRESHAPE: Source and target TileType must be identical.");
    static_assert(sizeof(DType) * Numel == sizeof(NewElement) * NewNumel, "TRESHAPE: Total byte size must match.");
    static_assert((SFractal == SLayout::NoneBox && NewSFractal == SLayout::NoneBox) ||
                      (SFractal != SLayout::NoneBox && NewSFractal != SLayout::NoneBox),
                  "TRESHAPE: Cannot reshape between boxed and non-boxed layouts.");

    TASSIGN_IMPL(dst, reinterpret_cast<uintptr_t>(src.data()));
#else
    __cce_alias(dst.data(), src.data(), 0);
#endif
}

} // namespace pto

#endif
