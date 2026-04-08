/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
// Implementation of interface adaptation layer for device-side and cloud-side compatibility
#ifndef ARCH_CCE_INTRINSIC_HPP
#define ARCH_CCE_INTRINSIC_HPP
#include <pto/common/arch_macro.hpp>

PTO_INTERNAL void pto_copy_ubuf_to_ubuf(__ubuf__ void *dst, __ubuf__ void *src, uint16_t nBurst, uint16_t lenBurst,
                                        uint16_t srcGap, uint16_t dstGap)
{
#if defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_KIRINX90)
    copy_ubuf_to_ubuf(dst, src, 0, nBurst, lenBurst, srcGap, dstGap);
#elif defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_A5)
    copy_ubuf_to_ubuf(dst, src, nBurst, lenBurst, srcGap, dstGap);
#endif
}

#endif // ARCH_CCE_INTRINSIC_HPP