/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMMON_NPU_DEDUP_REM_COMMON_HPP
#define PTO_COMMON_NPU_DEDUP_REM_COMMON_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {

// RegTensor<T> and MaskReg are provided by the architecture-specific common.hpp included before this header.
template <typename T>
PTO_INTERNAL void RemFloatCommon(RegTensor<T> &dst, RegTensor<T> &src0, RegTensor<T> &src1, MaskReg &preg)
{
    using U = std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, uint16_t>;
    static const U inf = sizeof(T) == sizeof(int32_t) ? 0x7F800000 : 0x7C00;
    static const U abs_mask = sizeof(T) == sizeof(int32_t) ? 0x7FFFFFFF : 0x7FFF;
    static const U nan = sizeof(T) == sizeof(int32_t) ? 0x7FC00000 : 0x7E00;

    MaskReg infMask, diffSignMask;
    RegTensor<T> diffSign, src0Abs, nanReg, absReg;
    vdiv(dst, src0, src1, preg, MODE_ZEROING);
    vtrc(dst, dst, ROUND_F, preg);
    vmul(dst, dst, src1, preg, MODE_ZEROING);
    vsub(dst, src0, dst, preg, MODE_ZEROING);

    vmul(diffSign, src1, dst, preg, MODE_ZEROING);
    vcmps_lt(diffSignMask, diffSign, 0.0f, preg);
    vadd(diffSign, dst, src1, diffSignMask, MODE_MERGING);

    vdup((RegTensor<U> &)absReg, abs_mask, preg, MODE_ZEROING);
    vand((RegTensor<U> &)src0Abs, (RegTensor<U> &)src0, (RegTensor<U> &)absReg, preg);
    vcmps_eq(infMask, (RegTensor<U> &)src0Abs, inf, preg);
    vdup((RegTensor<U> &)nanReg, nan, infMask, MODE_ZEROING);
    vsel(dst, nanReg, dst, infMask);
}

} // namespace pto

#endif
