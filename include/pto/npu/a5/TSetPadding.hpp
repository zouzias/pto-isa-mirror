/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSETPADDING_HPP
#define TSETPADDING_HPP

namespace pto {
template <typename T, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL>
PTO_INTERNAL void TSETPADDING_IMPL(const Img2colTileConfig<T> &cfg)
{
    if constexpr (FmatrixMode == SetFmatrixMode::FMATRIX_A_MANUAL || FmatrixMode == SetFmatrixMode::FMATRIX_B_MANUAL) {
        uint32_t paddingValue = 0;
        uint64_t paddingConfig = 0;
        constexpr uint16_t padValueShiftBit = 8;
        constexpr uint32_t padModeShiftBit = 32;
        if constexpr (sizeof(T) == 1) {
            uint8_t u8Value = *reinterpret_cast<const uint8_t*>(&cfg.padValue);
            paddingValue = (static_cast<uint16_t>(u8Value) << padValueShiftBit) | u8Value;
        } else if constexpr (sizeof(T) == 2) {
            paddingValue = *reinterpret_cast<const uint16_t*>(&cfg.padValue);
        } else if constexpr (sizeof(T) == 4) {
            paddingValue = *reinterpret_cast<const uint32_t*>(&cfg.padValue);
        }
        paddingConfig |= uint64_t(paddingValue) << padModeShiftBit;
        if constexpr(FmatrixMode == SetFmatrixMode::FMATRIX_A_MANUAL) {
            set_padding(paddingConfig);
        } else if constexpr (FmatrixMode == SetFmatrixMode::FMATRIX_B_MANUAL) {
            set_padding_b(paddingConfig);
        }
    }
}
}  // namespace pto
#endif  // TSETFMATRIX_HPP
