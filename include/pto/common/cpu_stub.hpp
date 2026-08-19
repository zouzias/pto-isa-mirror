/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPUSTUB_HPP
#define PTO_CPUSTUB_HPP

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <algorithm>
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>
#include <type_traits>
#include <dlfcn.h>
#include <string>
#include "type.hpp"

#include <pto/cpu/MXTypes.hpp>
#include <pto/cpu/Hifloat8.hpp>

#define __global__
#define AICORE
#define __aicore__
#define __gm__
#define __out__
#define __in__
#define __ubuf__
#define __cbuf__
#define __ca__
#define __cb__
#define __cc__
#define __fbuf__
#define __biasbuf__
#define __tf__

typedef void* aclrtStream;
typedef int pipe_t;
using event_t = int;
const pipe_t PIPE_S = 0;
const pipe_t PIPE_V = 1;
const pipe_t PIPE_MTE1 = 2;
const pipe_t PIPE_MTE2 = 3;
const pipe_t PIPE_MTE3 = 4;
const pipe_t PIPE_M = 5;
const pipe_t PIPE_ALL = 6;
const pipe_t PIPE_FIX = 7;
inline void pipe_barrier(pipe_t pipe) { (void)pipe; }

#define aclFloat16ToFloat(x) ((float)(x))

#if !defined(__COSTMODEL)
enum {
    ACL_MEM_MALLOC_HUGE_FIRST = 0,
    ACL_MEMCPY_HOST_TO_DEVICE = 0,
    ACL_MEMCPY_DEVICE_TO_HOST = 1,
    ACL_MEMCPY_DEVICE_TO_DEVICE = 2,
};

static inline int aclrtMallocHost(void** p, size_t sz)
{
    assert(sz != 0 && "[PTO][CA] Constraint violated. Condition: %s. Hint: see docs/coding/debug.md\n");
    *p = calloc(1, sz);
    return 0;
}

inline int aclrtMalloc(void** p, size_t sz, int) { return aclrtMallocHost(p, sz); }

inline int aclrtMemcpy(void* dst, size_t szDst, const void* src, size_t szSrc, int)
{
    std::memcpy(dst, src, std::min(szDst, szSrc));
    return 0;
}

inline int aclrtMemset(void* dst, size_t dstSize, int value, size_t count)
{
    constexpr int aclSuccess = 0;
    constexpr int aclErrorParamInvalid = 145000;

    if (count == 0) {
        return aclSuccess;
    }
    if (dst == nullptr || count > dstSize) {
        return aclErrorParamInvalid;
    }
    std::fill_n(reinterpret_cast<uint8_t*>(dst), count, static_cast<uint8_t>(value));
    return aclSuccess;
}

inline int aclrtSynchronizeStream(aclrtStream) { return 0; }

inline int aclrtFree(void* p)
{
    free(p);
    return 0;
}

inline int aclrtFreeHost(void* p)
{
    free(p);
    return 0;
}

inline int aclrtDestroyStream(aclrtStream) { return 0; }
inline int aclrtResetDevice(int) { return 0; }
inline int aclFinalize() { return 0; }
#endif

inline void set_flag(pipe_t, pipe_t, int) {}
inline void wait_flag(pipe_t, pipe_t, int) {}
#if !defined(__COSTMODEL)
using mem_dsb_t = int;
struct cache_line_t {
    static constexpr int SINGLE_CACHE_LINE = 0;
    static constexpr int ENTIRE_DATA_CACHE = 0;
    static constexpr int CACHELINE_OUT = 0;
};
struct dcci_dst_t {
    static constexpr int CACHELINE_OUT = 0;
};
inline constexpr mem_dsb_t DSB_DDR = 0;
inline constexpr mem_dsb_t DSB_ALL = 0;
inline constexpr mem_dsb_t DSB_UB = 0;
inline void dcci(const volatile void*, int) {}
inline void dcci(const volatile void*, int, int) {}
inline void dsb(mem_dsb_t) {}
inline uint64_t& cpu_ctrl_register()
{
    static thread_local uint64_t ctrl = 0;
    return ctrl;
}

inline uint64_t get_ctrl() { return cpu_ctrl_register(); }
inline void set_ctrl(uint64_t value) { cpu_ctrl_register() = value; }
inline uint64_t sbitset1(uint64_t value, int bit) { return value | (1ULL << bit); }
inline uint64_t sbitset0(uint64_t value, int bit) { return value & ~(1ULL << bit); }
#endif
#define __cce_get_tile_ptr(x) x
#define set_mask_norm(...)
#define set_vector_mask(...)

inline uint32_t get_block_idx();

#include <pto/cpu/trace.hpp>

/* <Hccl> */
#define HcclHostBarrier(x, y)
#define CommMpiInit(x, y) (true)
#define CommMpiFinalize()
#define SKIP_IF_RANKS_LT(n)
static constexpr uint32_t HCCL_MAX_RANK_NUM = 64;

static constexpr uint32_t QUANT_SCALAR_REG_OFFSET = 0;
static constexpr uint32_t QUANT_VECTOR_REG_OFFSET = 1;

struct HcclRootInfo {};

struct CommDeviceContext {
    uint64_t workSpace;
    uint64_t workSpaceSize;

    uint32_t rankId;
    uint32_t rankNum;
    uint64_t winSize;
    uint64_t windowsIn[HCCL_MAX_RANK_NUM];
    uint64_t windowsOut[HCCL_MAX_RANK_NUM];
};
/* </Hccl> */

#define EVENT_ID0 0
#define EVENT_ID1 1
#define EVENT_ID2 2
#define EVENT_ID3 3
#define EVENT_ID4 4
#define EVENT_ID5 5
#define EVENT_ID6 6
#define EVENT_ID7 7

#define F16_MAX 65504.0f

#include <pto/common/cpu_stub_runtime_inl.hpp>

#endif
