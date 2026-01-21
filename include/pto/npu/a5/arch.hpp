/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
// Implementation of interface adaptation layer for device-side and cloud-side compatibility
#ifndef ARCH_HPP
#define ARCH_HPP
#include "../../common/type.hpp"

namespace pto {
  template <typename T>
  PTO_INTERNAL void pto_copy_gm_to_ubuf_align_v2(__ubuf__ T *dst, __gm__ T *src, uint32_t sid, uint32_t nBurst,
    uint32_t lenBurst, uint32_t leftPaddingCount, uint32_t rightPaddingCount, bool dataSelectBit, uint32_t l2CacheCtrl,
    uint64_t srcBurstStride, uint32_t dstBurstStride) {
    #if __NPU_ARCH__ == 3113
      copy_gm_to_ubuf_align_v2(dst, src, sid, nBurst, lenBurst, leftPaddingCount, rightPaddingCount, dataSelectBit,
        srcBurstStride, dstBurstStride);
    #else
      copy_gm_to_ubuf_align_v2(dst, src, sid, nBurst, lenBurst, leftPaddingCount, rightPaddingCount, dataSelectBit,
        l2CacheCtrl, srcBurstStride, dstBurstStride);
    #endif
  }

  template <typename T>
  PTO_INTERNAL void pto_copy_gm_to_cbuf_align_v2(__cbuf__ T *dst, __gm__ T *src, uint32_t sid, uint32_t nBurst,
    uint32_t lenBurst, uint32_t leftPaddingCount, uint32_t rightPaddingCount, bool dataSelectBit, uint32_t l2CacheCtrl,
    uint64_t srcBurstStride, uint32_t dstBurstStride) {
    #if __NPU_ARCH__ == 3113
      copy_gm_to_cbuf_align_v2(dst, src, sid, nBurst, lenBurst, leftPaddingCount, rightPaddingCount, dataSelectBit,
        srcBurstStride, dstBurstStride);
    #else
      copy_gm_to_cbuf_align_v2(dst, src, sid, nBurst, lenBurst, leftPaddingCount, rightPaddingCount, dataSelectBit,
        l2CacheCtrl, srcBurstStride, dstBurstStride);
    #endif
  }

  template <typename T>
  PTO_INTERNAL void pto_copy_gm_to_cbuf_multi_nd2nz(__cbuf__ T *dst, __gm__ T *src, uint32_t sid,
    uint64_t loop1SrcStride, uint32_t l2CacheCtrl, uint32_t nValue, uint32_t dValue, bool loop4SrcStride = 0,
    bool smallC0En = false, bool antiqEn = false) {
    #if __NPU_ARCH__ == 3113
      copy_gm_to_cbuf_multi_nd2nz(dst, src, sid, loop1SrcStride, nValue, dValue, loop4SrcStride, smallC0En, antiqEn);
    #else
      copy_gm_to_cbuf_multi_nd2nz(dst, src, sid, loop1SrcStride, l2CacheCtrl, nValue, dValue, loop4SrcStride,
        smallC0En);
    #endif
  }

  PTO_INTERNAL void pto_copy_ubuf_to_gm_align_v2(__gm__ void *dst, __ubuf__ void *src, uint32_t sid, uint32_t nBurst,
    uint32_t lenBurst, uint32_t l2CacheCtrl, uint64_t srcBurstStride, uint32_t dstBurstStride) {
    #if __NPU_ARCH__ == 3113
      copy_ubuf_to_gm_align_v2(dst, src, sid, nBurst, lenBurst, srcBurstStride, dstBurstStride);
    #else
      copy_ubuf_to_gm_align_v2(dst, src, sid, nBurst, lenBurst, l2CacheCtrl, srcBurstStride, dstBurstStride);
    #endif
  }

  PTO_INTERNAL void pto_set_loop2_stride_outtoub(uint64_t loop2DstStride, uint64_t loop2SrcStride) {
    #if __NPU_ARCH__ == 9999999
    #else
      set_loop2_stride_outtoub(loop2DstStride << 40 | loop2SrcStride);
    #endif
  }

  PTO_INTERNAL void pto_set_loop1_stride_outtoub(uint64_t loop1DstStride, uint64_t loop1SrcStride) {
    #if __NPU_ARCH__ == 9999999
    #else
      set_loop1_stride_outtoub(loop1DstStride << 40 | loop1SrcStride);
    #endif
  }

  PTO_INTERNAL void pto_set_loop_size_outtoub(uint64_t loop2, uint64_t loop1) {
    #if __NPU_ARCH__ == 9999999
    #else
      set_loop_size_outtoub(loop2 << 21 | loop1);
    #endif
  }

  PTO_INTERNAL void pto_set_mov_pad_val(uint64_t config) {
    #if __NPU_ARCH__ == 3113
      set_pad_val_outtoub(config);
    #else
      set_mov_pad_val(config);
    #endif
  }

  template <typename CT, typename AT, typename BT>
  PTO_INTERNAL void pto_mad(__cc__ CT *c, __ca__ AT *a, __cb__ BT *b, uint16_t m, uint16_t k, uint16_t n,
    uint16_t uintFlag, bool gemvCtrl, bool ctrlMatrixC, bool initMatrixC) {
    #if __NPU_ARCH__ == 3113
      mad(c, a, b, m, k, n, 0, 0, uintFlag, 0, ctrlMatrixC, initMatrixC);
    #else
      mad(c, a, b, m, k, n, uintFlag, gemvCtrl, ctrlMatrixC, initMatrixC);
    #endif
  }

#if defined (__DAV_VEC__)
#if (__NPU_ARCH__ != 3113)
    // Import rounding type definitions from __cce_simd namespace for TCvt.hpp
    using __cce_simd::RoundRType;
    using __cce_simd::RoundAType;
    using __cce_simd::RoundFType;
    using __cce_simd::RoundCType;
    using __cce_simd::RoundZType;
    using __cce_simd::RoundOType;
#else
    using ::RoundRType;
    using ::RoundAType;
    using ::RoundFType;
    using ::RoundCType;
    using ::RoundZType;
    using ::RoundOType;
#endif
#endif
}

#endif