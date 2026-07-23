/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_CCU_LOOPGROUP_UTILS_HPP
#define PTO_COMM_ASYNC_CCU_LOOPGROUP_UTILS_HPP

#if defined(__CCE_KT_TEST__)
#error "ccu_loopgroup_utils.hpp is a host-only header and cannot be included in device code."
#endif

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace pto {
namespace comm {
namespace ccu {

// cann-9.2 public CCU MS constants (avoid internal microcode headers).
inline constexpr uint32_t CCU_MS_SIZE = 4096;
inline constexpr uint32_t CCU_MS_INTERLEAVE = 8;
inline constexpr uint32_t CCU_MS_DEFAULT_LOOP_COUNT = 64;

struct GoConfig {
    uint32_t loopCount{16};
    uint32_t msInterleave{CCU_MS_INTERLEAVE};
    uint64_t memSlice{CCU_MS_SIZE};
};

struct GoSize {
    uint64_t addrOffset{0};
    uint64_t loopIterNum{0};
    uint64_t parallelParam{0};
    uint64_t residual{0};
};

namespace detail {

constexpr uint64_t SetBits(uint16_t end) { return ((uint64_t(1) << (end + 1)) - uint64_t(1)); }

} // namespace detail

inline uint64_t GetMaxLoopIterNum()
{
    constexpr uint16_t loopNumBitNum = 12;
    return detail::SetBits(loopNumBitNum);
}

inline uint64_t EncodeLoopParam(uint64_t loopCtxId, uint64_t gsaOffset, uint64_t loopIterNum)
{
    constexpr uint16_t ctxIdBitNum = 8;
    constexpr uint16_t ctxIdShiftBit = 45;
    constexpr uint16_t gsaBitNum = 32;
    constexpr uint16_t gsaShiftBit = 13;
    constexpr uint16_t loopNumBitNum = 13;
    constexpr uint16_t loopNumShiftBit = 0;
    return ((loopCtxId & detail::SetBits(ctxIdBitNum)) << ctxIdShiftBit) |
           ((gsaOffset & detail::SetBits(gsaBitNum)) << gsaShiftBit) |
           ((loopIterNum & detail::SetBits(loopNumBitNum)) << loopNumShiftBit);
}

inline uint64_t EncodeParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
{
    constexpr uint16_t repeatBitNum = 7;
    constexpr uint16_t repeatNumShiftBit = 55;
    constexpr uint16_t repeatLoopBitNum = 7;
    constexpr uint16_t repeatLoopShiftBit = 48;
    constexpr uint16_t totalLoopBitNum = 7;
    constexpr uint16_t totalLoopShiftBit = 41;
    return ((repeatNum & detail::SetBits(repeatBitNum)) << repeatNumShiftBit) |
           ((repeatLoopIndex & detail::SetBits(repeatLoopBitNum)) << repeatLoopShiftBit) |
           ((totalLoopNum & detail::SetBits(totalLoopBitNum)) << totalLoopShiftBit);
}

inline uint64_t EncodeOffsetParam(uint64_t gsaOffset, uint64_t msOffset, uint64_t ckeOffset)
{
    constexpr uint16_t gsaBitNum = 32;
    constexpr uint16_t gsaShiftBit = 21;
    constexpr uint16_t msBitNum = 11;
    constexpr uint16_t msShiftBit = 10;
    constexpr uint16_t ckeBitNum = 10;
    constexpr uint16_t ckeShiftBit = 0;
    return ((gsaOffset & detail::SetBits(gsaBitNum)) << gsaShiftBit) |
           ((msOffset & detail::SetBits(msBitNum)) << msShiftBit) |
           ((ckeOffset & detail::SetBits(ckeBitNum)) << ckeShiftBit);
}

// EncodeParallelParam.repeatNum is 7-bit: loopCount - 1 ≤ 127 → hard max 128.
inline constexpr uint32_t kMaxLoopCountByEncoding = 128;
inline constexpr uint32_t kDefaultMaxLoopCount = CCU_MS_DEFAULT_LOOP_COUNT;
inline constexpr uint32_t kMaxMsBufsPerKernel = kDefaultMaxLoopCount * CCU_MS_INTERLEAVE;
inline constexpr uint32_t kMaxLoopCountByMsBudget = kMaxMsBufsPerKernel / CCU_MS_INTERLEAVE;
// LoopGroup crashes with loopCount=1 (repeatNum=0 triggers invalid CcuBuf access).
inline constexpr uint32_t kMinLoopCount = 2;

inline uint32_t CalcLoopCount(uint64_t payloadBytes, uint64_t memSlice = CCU_MS_SIZE)
{
    const char* env = std::getenv("PTO_CCU_LOOP_COUNT");
    if (env != nullptr && *env != '\0') {
        int v = std::atoi(env);
        if (v > 0) {
            uint32_t loopCount = std::clamp(static_cast<uint32_t>(v), kMinLoopCount, kMaxLoopCountByEncoding);
            loopCount = std::min(loopCount, kMaxLoopCountByMsBudget);
            while (loopCount > kMinLoopCount && payloadBytes == static_cast<uint64_t>(loopCount) * memSlice) {
                --loopCount;
            }
            return loopCount;
        }
    }

    uint32_t chunksNeeded = static_cast<uint32_t>(std::max(uint64_t{1}, (payloadBytes + memSlice - 1) / memSlice));

    uint32_t loopCount = std::clamp(chunksNeeded, kMinLoopCount, kDefaultMaxLoopCount);
    loopCount = std::min(loopCount, kMaxLoopCountByMsBudget);
    while (loopCount > kMinLoopCount && payloadBytes == static_cast<uint64_t>(loopCount) * memSlice) {
        --loopCount;
    }
    return loopCount;
}

inline GoSize CalcGoSize(uint64_t totalBytes, const GoConfig& cfg)
{
    GoSize result;
    uint64_t loopSize = cfg.loopCount * cfg.memSlice;

    uint64_t m = totalBytes / loopSize;
    uint64_t n = (totalBytes - m * loopSize) / cfg.memSlice;
    uint64_t p = totalBytes - m * loopSize - n * cfg.memSlice;

    uint64_t maxSize = loopSize * (GetMaxLoopIterNum() + 1);
    if (totalBytes == maxSize) {
        m = GetMaxLoopIterNum();
        n = cfg.loopCount - 1;
        p = cfg.memSlice;
    }

    result.addrOffset = cfg.memSlice * cfg.loopCount * m;
    result.loopIterNum = m;

    if (n == 0 && p == 0) {
        result.parallelParam = 0;
        result.residual = 0;
    } else if (n != 0 && p == 0) {
        result.parallelParam = EncodeParallelParam(n - 1, 0, 1);
        result.residual = cfg.memSlice;
    } else if (n == 0 && p != 0) {
        result.parallelParam = EncodeParallelParam(0, 0, 1);
        result.residual = p;
    } else {
        result.parallelParam = EncodeParallelParam(n - 1, 1, 2);
        result.residual = p;
    }

    return result;
}

} // namespace ccu
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_CCU_LOOPGROUP_UTILS_HPP
