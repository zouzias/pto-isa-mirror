/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Minimal AIC-only probes that isolate whether the A5 AIC (dav-c310 cube) core
// can execute plain GM scalar store, st_atomic and ld_dev at runtime. Each probe
// is launched on a single AIC block via dav-c310-cube single chevron; the stream
// sync return code (0 vs 507015) and the read-back GM contents form the truth
// table interpreted by main.cpp.

#include <pto/pto-inst.hpp>
#include "acl/acl.h"

using namespace pto;

constexpr int32_t kProbeBlockCount = 1;
constexpr int32_t kProbeCacheLine = 8;

// Probe A: plain AIC scalar GM store baseline. out[0] preset 0, expect 42.
extern "C" __global__ AICORE void RunAicProbeStore_4001(__gm__ int32_t __out__* out)
{
#if defined(__DAV_CUBE__)
    out[0] = 42;
    dcci(static_cast<__gm__ void*>(out), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
#else
    (void)out;
#endif
}

// Probe B: AIC st_atomic add. out[0] preset 10, expect 11 if supported.
extern "C" __global__ AICORE void RunAicProbeStAtomic_4002(__gm__ int32_t __out__* out)
{
#if defined(__DAV_CUBE__)
    set_st_atomic_cfg(ATOMIC_S32, ATOMIC_SUM);
    dcci(static_cast<__gm__ void*>(out), SINGLE_CACHE_LINE);
    st_atomic<int32_t>(1, out);
    dcci(static_cast<__gm__ void*>(out), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
#else
    (void)out;
#endif
}

// Probe C: AIC ld_dev read. out[8] preset 1234, out[0] preset 0; read out[8]
// via ld_dev then publish it to out[0] (plain store, interpret with Probe A).
extern "C" __global__ AICORE void RunAicProbeLdDev_4003(__gm__ int32_t __out__* out)
{
#if defined(__DAV_CUBE__)
    __gm__ int32_t* src = out + kProbeCacheLine;
    dcci(static_cast<__gm__ void*>(src), SINGLE_CACHE_LINE);
    const int32_t value = static_cast<int32_t>(ld_dev(reinterpret_cast<__gm__ uint32_t*>(src), 0));
    out[0] = value;
    dcci(static_cast<__gm__ void*>(out), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
#else
    (void)out;
#endif
}

void LaunchAicProbeStore(int32_t* out, void* stream)
{
    RunAicProbeStore_4001<<<kProbeBlockCount, nullptr, stream>>>(out);
}

void LaunchAicProbeStAtomic(int32_t* out, void* stream)
{
    RunAicProbeStAtomic_4002<<<kProbeBlockCount, nullptr, stream>>>(out);
}

void LaunchAicProbeLdDev(int32_t* out, void* stream)
{
    RunAicProbeLdDev_4003<<<kProbeBlockCount, nullptr, stream>>>(out);
}
