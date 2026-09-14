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
    set_mask_norm();
    constexpr uint64_t ctrlBits = 0x1000000000000;
    constexpr uint64_t preCtrlBit = 0x1000000000000008;
    uint64_t prevCtrl = get_ctrl() & ctrlBits;
    uint64_t val = preCtrlBit | prevCtrl;
    set_ctrl(val);

#ifdef __DAV_CUBE__
    constexpr uint64_t padValue = 0;
    set_padding(padValue);
#endif
#ifdef __DAV_VEC__
    set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    uint64_t loopSizePara = (1uL << 21) | 1uL;
    set_loop_size_ubtoout(loopSizePara);
    set_loop_size_outtoub(loopSizePara);
#endif
    set_st_atomic_cfg(0b00100100);
}
} // namespace pto
#endif
