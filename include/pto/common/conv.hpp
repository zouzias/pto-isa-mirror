/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

namespace pto {
template<typename T = uint64_t>
struct Img2colTileConfig{
    uint8_t padList[4] = {0};
    uint16_t fmapH = 0;
    uint16_t fmapW = 0;
    uint16_t filterH = 1;
    uint16_t filterW = 1;
    uint8_t dilationH = 1;
    uint8_t dilationW = 1;
    uint8_t strideH = 1;
    uint8_t strideW = 1;
    uint16_t channelSize = 0;
    T padValue = 0;
    bool transpose = false;
    bool smallChannel = false;

    AICORE Img2colTileConfig() = default;
};
} // namespace pto
#endif
