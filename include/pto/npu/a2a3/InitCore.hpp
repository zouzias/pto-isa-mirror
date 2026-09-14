/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef INIT_CORE_HPP
#define INIT_CORE_HPP

namespace pto {

PTO_INTERNAL void INIT_CORE_IMPL()
{
    set_atomic_none();
#ifdef __DAV_CUBE__
    set_l1_3d_size(static_cast<uint64_t>(0));
    set_padding(static_cast<uint64_t>(0));
#endif
#ifdef __DAV_VEC__
    set_mask_norm();
    set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
#endif
}
} // namespace pto
#endif
