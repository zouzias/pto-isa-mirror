/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSET_IMG2COL_PADDING_HPP
#define TSET_IMG2COL_PADDING_HPP

namespace pto {
template <typename T>
PTO_INTERNAL void TSET_IMG2COL_PADDING_IMPL(const Img2colTileConfig<T> &cfg)
{
    uint32_t paddingValue = 0;
    constexpr uint16_t paddingValueShiftBit = 8;
    if constexpr (sizeof(T) == 1) {
        uint8_t u8Value = *reinterpret_cast<const uint8_t *>(&cfg.padValue);
        paddingValue = (static_cast<uint16_t>(u8Value) << paddingValueShiftBit) | u8Value;
    } else if constexpr (sizeof(T) == 2) {
        paddingValue = *reinterpret_cast<const uint16_t *>(&cfg.padValue);
    } else if constexpr (sizeof(T) == 4) {
        paddingValue = *reinterpret_cast<const uint32_t *>(&cfg.padValue);
    }
    set_padding(paddingValue);
}
} // namespace pto
#endif // TSET_IMG2COL_PADDING_HPP
