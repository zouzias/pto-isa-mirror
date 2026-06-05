/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMMON_NPU_DEDUP_GATHER_COMMON_HPP
#define PTO_COMMON_NPU_DEDUP_GATHER_COMMON_HPP

#include <pto/common/type.hpp>
#include <pto/common/constants.hpp>

namespace pto {

// Requires architecture common.hpp and utils.hpp to be included before this header for MaskReg and PSetWithType.
template <typename T>
PTO_INTERNAL void PIntlvWithType(MaskReg &dst0, MaskReg &dst1, MaskReg src0, MaskReg src1)
{
    if constexpr (sizeof(T) == sizeof(float)) {
        pintlv_b32(dst0, dst1, src0, src1);
    } else if constexpr (sizeof(T) == sizeof(half)) {
        pintlv_b16(dst0, dst1, src0, src1);
    } else if constexpr (sizeof(T) == sizeof(uint8_t)) {
        pintlv_b8(dst0, dst1, src0, src1);
    }
}

template <typename T, MaskPattern maskPattern>
PTO_INTERNAL MaskReg GetMaskVal()
{
    MaskReg pg0;
    MaskReg pg1;
    MaskReg dstPg0;
    MaskReg dstPg1;
    if constexpr (maskPattern == MaskPattern::P0101) {
        pg0 = PSetWithType<T>(PAT_ALL);
        pg1 = PSetWithType<T>(PAT_ALLF);
        PIntlvWithType<T>(dstPg0, dstPg1, pg0, pg1);
    } else if constexpr (maskPattern == MaskPattern::P1010) {
        pg0 = PSetWithType<T>(PAT_ALL);
        pg1 = PSetWithType<T>(PAT_ALLF);
        PIntlvWithType<T>(dstPg0, dstPg1, pg1, pg0);
    } else if constexpr (maskPattern == MaskPattern::P0001) {
        pg0 = PSetWithType<T>(PAT_ALL);
        pg1 = PSetWithType<T>(PAT_ALLF);
        PIntlvWithType<T>(dstPg0, dstPg1, pg0, pg1);
        PIntlvWithType<T>(dstPg0, dstPg1, dstPg0, pg1);
    } else if constexpr (maskPattern == MaskPattern::P0010) {
        pg0 = PSetWithType<T>(PAT_ALL);
        pg1 = PSetWithType<T>(PAT_ALLF);
        PIntlvWithType<T>(dstPg0, dstPg1, pg0, pg1);
        PIntlvWithType<T>(dstPg0, dstPg1, pg1, dstPg0);
    } else if constexpr (maskPattern == MaskPattern::P0100) {
        pg0 = PSetWithType<T>(PAT_ALL);
        pg1 = PSetWithType<T>(PAT_ALLF);
        PIntlvWithType<T>(dstPg0, dstPg1, pg1, pg0);
        PIntlvWithType<T>(dstPg0, dstPg1, dstPg0, pg1);
    } else if constexpr (maskPattern == MaskPattern::P1000) {
        pg0 = PSetWithType<T>(PAT_ALL);
        pg1 = PSetWithType<T>(PAT_ALLF);
        PIntlvWithType<T>(dstPg0, dstPg1, pg1, pg0);
        PIntlvWithType<T>(dstPg0, dstPg1, pg1, dstPg0);
    } else if constexpr (maskPattern == MaskPattern::P1111) {
        dstPg0 = PSetWithType<T>(PAT_ALL);
    }
    return dstPg0;
}

} // namespace pto

#endif
