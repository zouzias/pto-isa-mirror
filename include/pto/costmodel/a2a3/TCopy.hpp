/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TCOPY_COSTMODEL_HPP
#define TCOPY_COSTMODEL_HPP

#include "pto/costmodel/pto_isa_costmodel.hpp"

namespace pto {

// TCOPY: copy_ubuf_to_ubuf (MTE1 pipeline).
// See TCopyOp.hpp for parameter derivation.
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void TCOPY_IMPL(TileDataD &dst, TileDataS &src)
{
    using T = typename TileDataD::DType;
    auto stats = runCopyOp(dst, src);
}

} // namespace pto

#endif // TCOPY_COSTMODEL_HPP
