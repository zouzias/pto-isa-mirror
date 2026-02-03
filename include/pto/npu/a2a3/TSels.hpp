/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSELS_HPP
#define TSELS_HPP

#include <pto/common/constants.hpp>

namespace pto {
    template <typename TileDataDst, typename TileDataMask, typename TileDataSrc>
    PTO_INTERNAL void TSELS_IMPL(TileDataDst &dst, TileDataMask &mask, TileDataSrc &src, TileDataDst::DType scalar) {
        TEXPANDS_IMPL(dst, scalar);
        pipe_barrier(PIPE_V);
        TSEL_IMPL(dst, mask, src, dst);
    }
}
#endif
