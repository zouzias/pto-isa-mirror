/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_A2A3_L2_CACHE_HINT_HPP
#define PTO_NPU_A2A3_L2_CACHE_HINT_HPP

#include <cstdint>
#include <pto/common/type.hpp>

namespace pto {

// A2A3-only: NotAlloc TLOAD remaps GM via runtime L2 disable offset.
AICORE constexpr bool IsNotAllocTLoadL2Hint(TLoadL2Hint hint)
{
    return hint == TLoadL2Hint::NotAllocKeep || hint == TLoadL2Hint::NotAllocClean || hint == TLoadL2Hint::NotAllocDrop;
}

} // namespace pto

// Match AscendC: global PtoOpSystemRunCfg + g_opL2CacheHintCfg so runtime can fill
// the L2 disable GM offset for A2A3 NotAlloc TLOAD.
#ifndef PTO_OP_SYSTEM_RUN_CFG_DEFINED
struct PtoOpSystemRunCfg {
    uint64_t l2Cacheoffset;
};
#define PTO_OP_SYSTEM_RUN_CFG_DEFINED
#endif

#if defined(PTO_NPU_ARCH_A2A3) && defined(__CCE_AICORE__) && !defined(__CPU_SIM) && !defined(__COSTMODEL)
// AscendC already emits symbol+meta under L2_CACHE_HINT + __NPU_DEVICE__.
#if !(defined(L2_CACHE_HINT) && defined(__NPU_DEVICE__))
struct PtoBinaryMetaAscFeature {
    uint16_t type;
    uint16_t len;
    uint32_t feature; // PRINT=1, FFTS=2, L2CACHE=3
};
// CCE aicore may not define __NPU_DEVICE__; still emit like AscendC device path.
inline __gm__ PtoOpSystemRunCfg g_opL2CacheHintCfg = {0};
static const PtoBinaryMetaAscFeature __pto_asc_feature_l2cache__
    __attribute__((used, section(".ascend.meta"))) = {4, 4, 3};
#endif
#elif defined(PTO_NPU_ARCH_A2A3)
// CPU/costmodel/host: zero stub so references compile; Apply ignores offset.
inline PtoOpSystemRunCfg g_opL2CacheHintCfg = {0};
#endif

namespace pto {

template <TLoadL2Hint l2Control, typename T>
PTO_INTERNAL T* ApplyTLoadL2HintAddr(T* addr)
{
#if defined(PTO_NPU_ARCH_A2A3) && !defined(__CPU_SIM) && !defined(__COSTMODEL)
    if constexpr (IsNotAllocTLoadL2Hint(l2Control)) {
        // Runtime-filled only (AscendC-compatible). CA/sim leaves 0 => addr+0.
        // NPU fills 0x100000000000 for real not-alloc. No hardcoded fallback.
        return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(addr) + g_opL2CacheHintCfg.l2Cacheoffset);
    } else {
        return addr;
    }
#else
    (void)addr;
    return addr;
#endif
}

} // namespace pto

#endif // PTO_NPU_A2A3_L2_CACHE_HINT_HPP
