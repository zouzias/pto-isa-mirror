/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file kernel_reg_compute_struct_intf.h
 * \brief
 */

#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_REG_COMPUTE_STRUCT_INTF_H__
#endif

#ifndef ASCENDC_MODULE_REG_COMPUTE_STRUCT_INTERFACE_H
#define ASCENDC_MODULE_REG_COMPUTE_STRUCT_INTERFACE_H

#include "../kernel_reg_compute_datatype_impl.h"
#include "reg_compute/kernel_reg_compute_vec_binary_impl.h"
#include "reg_compute/kernel_reg_compute_vec_cmpsel_impl.h"
#include "reg_compute/kernel_reg_compute_vec_duplicate_impl.h"

namespace AscendC {
namespace Reg {

struct RegTrait {
    int REG_NUM = 1;
};

constexpr RegTrait RegTraitNumOne = {1};
constexpr RegTrait RegTraitNumTwo = {2};

template <typename T, const RegTrait& regTrait = RegTraitNumOne>
struct RegTensor {
    __simd_callee__ inline RegTensor(){};
    using ActualT = T;
    static constexpr RegTrait trait = regTrait;
    static constexpr int REG_NUM = trait.REG_NUM;
    using RegType = typename TypeGet<T>::T;
    RegType reg[trait.REG_NUM];

    __simd_callee__ inline operator RegType& ()
    {
        // only process one reg, two registers require explicit call
        return reg[0];
    }
    __simd_callee__ void Print() const;
};

} // namespace Reg
} // namespace AscendC

#include "../../impl/basic_api/reg_compute/kernel_reg_compute_struct_intf_impl.h"
#endif // ASCENDC_MODULE_REG_COMPUTE_STRUCT_INTERFACE_H

#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_REG_COMPUTE_STRUCT_INTF_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_REG_COMPUTE_STRUCT_INTF_H__
#endif

