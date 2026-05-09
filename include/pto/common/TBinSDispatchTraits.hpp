/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_TBINS_DISPATCH_TRAITS_HPP
#define PTO_TBINS_DISPATCH_TRAITS_HPP

#include <pto/common/constants.hpp>

namespace pto {

template <typename TileDataDst, typename TileDataSrc, unsigned elementsPerRepeat>
struct TBinSDispatchTraits {
    static constexpr bool tileDataContinue =
        ((TileDataDst::Cols == TileDataDst::ValidCol) && (TileDataSrc::Cols == TileDataSrc::ValidCol)) ||
        ((TileDataDst::Rows == 1) && (TileDataSrc::Rows == 1));

    static constexpr unsigned totalRepeats =
        (TileDataDst::Rows * TileDataDst::Cols + elementsPerRepeat - 1) / elementsPerRepeat;
    static constexpr bool nonVLAligned =
        (((TileDataDst::Cols % elementsPerRepeat) != 0) && (TileDataDst::Cols > elementsPerRepeat));
    static constexpr bool enableCountMode = nonVLAligned || (totalRepeats > pto::REPEAT_MAX);
};

} // namespace pto

#endif // PTO_TBINS_DISPATCH_TRAITS_HPP
