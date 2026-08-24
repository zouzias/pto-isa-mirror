/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSORT32_SOFT_HPP
#define TSORT32_SOFT_HPP

/**
 * @file TSort32Soft.hpp
 * @brief Software VBS32 implementation for kirinDev0000 (__NPU_ARCH__ == 5101).
 *
 * kirinDev0000 (dav-l510) does not support the hardware VBS32 instruction.
 * This implementation uses scalar insertion sort. All __ubuf__ access is
 * performed through half pointers only, since the NPU backend does not support
 * scalar uint32 load/store to __ubuf__ memory.
 *
 * Output format matches hardware VBS32: each sorted element occupies 8 bytes
 * (half value + 2-byte padding + uint32 index), 256 bytes per 32-element repeat.
 */

#include <pto/common/constants.hpp>

namespace pto {

constexpr const uint32_t VBS32_BLOCK_SIZE = 32;
constexpr const uint32_t VBS32_HALF_DST_STRIDE = 128;

static AICORE inline void SoftVbsort32Half(
    __ubuf__ half* dst, __ubuf__ half* src, __ubuf__ uint32_t* idx, uint32_t repeatNum)
{
    __ubuf__ half* srcHalf = src;
    __ubuf__ half* idxHalf = reinterpret_cast<__ubuf__ half*>(idx);

    for (uint32_t r = 0; r < repeatNum; r++) {
        __ubuf__ half* srcBlock = srcHalf + r * VBS32_BLOCK_SIZE;
        __ubuf__ half* idxBlock = idxHalf + r * VBS32_BLOCK_SIZE * 2;
        __ubuf__ half* dstBlock = dst + r * VBS32_HALF_DST_STRIDE;

        half vals[VBS32_BLOCK_SIZE];
        uint32_t idxs[VBS32_BLOCK_SIZE];

        for (uint32_t i = 0; i < VBS32_BLOCK_SIZE; i++) {
            vals[i] = srcBlock[i];
            half loH = idxBlock[i * 2];
            half hiH = idxBlock[i * 2 + 1];
            uint16_t lo = *reinterpret_cast<uint16_t*>(&loH);
            uint16_t hi = *reinterpret_cast<uint16_t*>(&hiH);
            idxs[i] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
        }

        for (uint32_t i = 1; i < VBS32_BLOCK_SIZE; i++) {
            half keyVal = vals[i];
            uint32_t keyIdx = idxs[i];
            int32_t j = static_cast<int32_t>(i) - 1;
            while (j >= 0) {
                float vj = static_cast<float>(vals[j]);
                float kv = static_cast<float>(keyVal);
                bool vjNaN = (vj != vj);
                bool kvNaN = (kv != kv);
                bool shouldSwap = (vjNaN && !kvNaN) || (!vjNaN && !kvNaN && vj < kv);
                if (!shouldSwap) {
                    break;
                }
                vals[j + 1] = vals[j];
                idxs[j + 1] = idxs[j];
                j--;
            }
            vals[j + 1] = keyVal;
            idxs[j + 1] = keyIdx;
        }

        for (uint32_t i = 0; i < VBS32_BLOCK_SIZE; i++) {
            __ubuf__ half* elemDst = dstBlock + i * 4;
            elemDst[0] = vals[i];
            elemDst[1] = static_cast<half>(0);
            uint16_t lo = static_cast<uint16_t>(idxs[i] & 0xFFFF);
            uint16_t hi = static_cast<uint16_t>((idxs[i] >> 16) & 0xFFFF);
            elemDst[2] = *reinterpret_cast<half*>(&lo);
            elemDst[3] = *reinterpret_cast<half*>(&hi);
        }
    }
}

} // namespace pto
#endif // TSORT32_SOFT_HPP
