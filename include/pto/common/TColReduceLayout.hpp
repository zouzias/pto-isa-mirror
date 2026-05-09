/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#ifndef PTO_TCOL_REDUCE_LAYOUT_HPP
#define PTO_TCOL_REDUCE_LAYOUT_HPP

#include <pto/common/constants.hpp>

namespace pto {

template <typename T, unsigned srcstride>
struct ColReduceLayout {
    static constexpr int DTypeSize = sizeof(T);
    static constexpr int blockSizeElem = BLOCK_BYTE_SIZE / DTypeSize;
    static constexpr int numBlockPerLine = (srcstride * DTypeSize + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;
    static constexpr int dupSrcStride = numBlockPerLine * blockSizeElem;
    static constexpr int elementsPerRepeat = REPEAT_BYTE / DTypeSize;
};

} // namespace pto

#endif // PTO_TCOL_REDUCE_LAYOUT_HPP
