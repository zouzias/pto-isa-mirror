/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMATMUL_HPP
#define TMATMUL_HPP

#include "pto/cpu/tile_offsets.hpp"
#include "pto/common/pto_tile.hpp"
#include "pto/common/constants.hpp"
#include <arm_neon.h>

#if defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_SME)
#include <arm_sve.h>
#endif
#if defined(__ARM_FEATURE_SME)
#include <arm_sme.h>
#endif
#include <iostream>
#include <algorithm>

namespace pto {
    template <typename TileAcc, typename TileLeft, typename TileRight>
    void TMatmul_Generic(typename TileAcc::TileDType dst,
                        typename TileLeft::TileDType src0,
                        typename TileRight::TileDType src1,
                        uint16_t M, uint16_t N, uint16_t K,
                        bool accumulate)
    {
        for (size_t i = 0; i < M; ++i) {
            for (uint16_t j = 0; j < N; j++) {
                typename TileAcc::DType mul_acc = 0;

                for (uint16_t k = 0; k < K; k++) {
                    size_t src0Idx = GetTileElementOffset<TileLeft>(i, k);
                    size_t src1Idx = GetTileElementOffset<TileRight>(k, j);
                    mul_acc += static_cast<typename TileAcc::DType>(src0[src0Idx]) *
                               static_cast<typename TileAcc::DType>(src1[src1Idx]);
                }

                size_t dstIdx = GetTileElementOffset<TileAcc>(i, j);
                if (accumulate) {
                    dst[dstIdx] += mul_acc;
                } else {
                    dst[dstIdx] = mul_acc;
                }
            }
        }
    }
    // SVE Intrinsic Impl
#if defined(__ARM_FEATURE_SVE)
    inline void Kernel32x32_SVE(float* c_ptr, int strideC, const float* a_ptr, const float* b_ptr, int k_size,
                                int m_start, int n_start, int m_valid, int n_valid, bool accumulate) {
        constexpr int MR = PACKED_ROW;
        constexpr int NR = PACKED_COL;

        for (int n_off = 0; n_off < NR; n_off += svcntw()) {
            int n_block_remain = NR -n_off;
            int n_global_remain = n_valid - n_start - n_off;

            int limit = n_block_remain < n_global_remain ? n_block_remain : n_global_remain;
            if (limit <= 0) break;

            svbool_t pg = svwhilelt_b32(0, limit);

            for (int m_off = 0; m_off < MR; m_off +=4) {
                if (m_start + m_off >= m_valid) break;

                svfloat32_t c0, c1, c2, c3;

                if (accumulate) {
                    if (m_start + m_off + 0 < m_valid) c0 = svld1_f32(pg, c_ptr + (m_off + 0) * strideC + n_off);
                    else c0 = svdup_f32(0.0f);
                    if (m_start + m_off + 1 < m_valid) c1 = svld1_f32(pg, c_ptr + (m_off + 1) * strideC + n_off);
                    else c1 = svdup_f32(0.0f);
                    if (m_start + m_off + 2 < m_valid) c2 = svld1_f32(pg, c_ptr + (m_off + 2) * strideC + n_off);
                    else c2 = svdup_f32(0.0f);
                    if (m_start + m_off + 3 < m_valid) c3 = svld1_f32(pg, c_ptr + (m_off + 3) * strideC + n_off);
                    else c3 = svdup_f32(0.0f);
                } else {
                    c0 = svdup_f32(0.0f);
                    c1 = svdup_f32(0.0f);
                    c2 = svdup_f32(0.0f);
                    c3 = svdup_f32(0.0f);
                }

                const float* ap = a_ptr + m_off;
                const float* bp = b_ptr + n_off;

                for (int k = 0; k < k_size; ++k) {
                    svfloat32_t b_vec = svld1_f32(pg, bp);
                    bp += PACKED_COL;

                    c0 = svmla_n_f32_x(pg, c0, b_vec, ap[0]);
                    c1 = svmla_n_f32_x(pg, c1, b_vec, ap[1]);
                    c2 = svmla_n_f32_x(pg, c2, b_vec, ap[2]);
                    c3 = svmla_n_f32_x(pg, c3, b_vec, ap[3]);

                    ap += PACKED_ROW;
                }

                if (m_start + m_off + 0 < m_valid) svst1_f32(pg, c_ptr + (m_off + 0) * strideC + n_off, c0);
                if (m_start + m_off + 1 < m_valid) svst1_f32(pg, c_ptr + (m_off + 1) * strideC + n_off, c1);
                if (m_start + m_off + 2 < m_valid) svst1_f32(pg, c_ptr + (m_off + 2) * strideC + n_off, c2);
                if (m_start + m_off + 3 < m_valid) svst1_f32(pg, c_ptr + (m_off + 3) * strideC + n_off, c3);
            }
        }
    }
    #endif

    inline void Kernel32x32(float* c_ptr, int strideC, const float* a_ptr, const float* b_ptr, int k_size,
                            int m_start, int n_start, int m_valid, int n_valid, bool accumulate = false) {

#if defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_SME)
        Kernel32x32_SVE(c_ptr, strideC, a_ptr, b_ptr, k_size, m_start, n_start, m_valid, n_valid, accumulate);
#else
        constexpr int MR = PACKED_ROW;
        constexpr int NR = PACKED_COL;

        for (int m_off = 0; m_off < MR; m_off +=4) {
            if (m_start + m_off >= m_valid) break;

            {
                float32x4_t c[4][4];
                if (accumulate && n_start < n_valid) {
                    for (int i =0; i < 4; ++i) {
                        int cur_row = m_start + m_off + i;
                        if (cur_row >= m_valid) {
                            for (int j = 0; j < 4; ++j) c[i][j] = vdupq_n_f32(0.0f);
                            continue;
                        }
                        float* dst_row = c_ptr + (m_off + i) * strideC;
                        if (n_start + 16 <= n_valid) {
                            c[i][0] = vld1q_f32(dst_row);
                            c[i][1] = vld1q_f32(dst_row + 4);
                            c[i][2] = vld1q_f32(dst_row + 8);
                            c[i][3] = vld1q_f32(dst_row + 12);
                        } else {
                            float tmp[16] = {0};
                            for (int j = 0; j < 16; ++j) if (n_start + j < n_valid) tmp[j] = dst_row[j];
                            c[i][0] = vld1q_f32(tmp);
                            c[i][1] = vld1q_f32(tmp + 4);
                            c[i][2] = vld1q_f32(tmp + 8);
                            c[i][3] = vld1q_f32(tmp + 12);
                        }
                    }
                } else {
                    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) c[i][j] = vdupq_n_f32(0.0f);
                }

                const float* ap = a_ptr + m_off;
                const float* bp = b_ptr;

                for (int k = 0; k < k_size; ++k) {
                    float32x4_t b0 = vld1q_f32(bp);
                    float32x4_t b1 = vld1q_f32(bp + 4);
                    float32x4_t b2 = vld1q_f32(bp + 8);
                    float32x4_t b3 = vld1q_f32(bp + 12);
                    bp += PACKED_COL;

                    float32x4_t a = vld1q_f32(ap);
                    ap += PACKED_ROW;

                    c[0][0] = vfmaq_laneq_f32(c[0][0], b0, a, 0);
                    c[0][1] = vfmaq_laneq_f32(c[0][1], b1, a, 0);
                    c[0][2] = vfmaq_laneq_f32(c[0][2], b2, a, 0);
                    c[0][3] = vfmaq_laneq_f32(c[0][3], b3, a, 0);
                    c[1][0] = vfmaq_laneq_f32(c[1][0], b0, a, 1);
                    c[1][1] = vfmaq_laneq_f32(c[1][1], b1, a, 1);
                    c[1][2] = vfmaq_laneq_f32(c[1][2], b2, a, 1);
                    c[1][3] = vfmaq_laneq_f32(c[1][3], b3, a, 1);
                    c[2][0] = vfmaq_laneq_f32(c[2][0], b0, a, 2);
                    c[2][1] = vfmaq_laneq_f32(c[2][1], b1, a, 2);
                    c[2][2] = vfmaq_laneq_f32(c[2][2], b2, a, 2);
                    c[2][3] = vfmaq_laneq_f32(c[2][3], b3, a, 2);
                    c[3][0] = vfmaq_laneq_f32(c[3][0], b0, a, 3);
                    c[3][1] = vfmaq_laneq_f32(c[3][1], b1, a, 3);
                    c[3][2] = vfmaq_laneq_f32(c[3][2], b2, a, 3);
                    c[3][3] = vfmaq_laneq_f32(c[3][3], b3, a, 3);
                }

                for (int i = 0; i < 4; ++i) {
                    int cur_row = m_start + m_off + i;
                    if (cur_row >= m_valid) break;
                    float* dst_row = c_ptr + (m_off + i) * strideC;
                    if (n_start + 16 <= n_valid) {
                        vst1q_f32(dst_row, c[i][0]);
                        vst1q_f32(dst_row + 4, c[i][1]);
                        vst1q_f32(dst_row + 8, c[i][2]);
                        vst1q_f32(dst_row + 12, c[i][3]);
                    } else {
                        float tmp[16];
                        vst1q_f32(tmp, c[i][0]);
                        vst1q_f32(tmp + 4, c[i][1]);
                        vst1q_f32(tmp + 8, c[i][2]);
                        vst1q_f32(tmp + 12, c[i][3]);
                        for (int j = 0; j < 16; ++j) if (n_start + j < n_valid) dst_row[j] = tmp[j];
                    }
                }
            }

            if constexpr (PACKED_COL > 16) {
                if (n_start + 16 < n_valid){
                    float32x4_t c[4][4];
                    if (accumulate && n_start + 16 < n_valid) {
                        for (int i =0; i < 4; ++i) {
                            int cur_row = m_start + m_off + i;
                            if (cur_row >= m_valid) {
                                for (int j = 0; j < 4; ++j) c[i][j] = vdupq_n_f32(0.0f);
                                continue;
                            }
                            float* dst_row = c_ptr + (m_off + i) * strideC + 16;
                            if (n_start + 32 <= n_valid) {
                                c[i][0] = vld1q_f32(dst_row);
                                c[i][1] = vld1q_f32(dst_row + 4);
                                c[i][2] = vld1q_f32(dst_row + 8);
                                c[i][3] = vld1q_f32(dst_row + 12);
                            } else {
                                float tmp[16] = {0};
                                for (int j = 0; j < 16; ++j) if (n_start + 16 + j < n_valid) tmp[j] = dst_row[j];
                                c[i][0] = vld1q_f32(tmp);
                                c[i][1] = vld1q_f32(tmp + 4);
                                c[i][2] = vld1q_f32(tmp + 8);
                                c[i][3] = vld1q_f32(tmp + 12);
                            }
                        }
                    } else {
                        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) c[i][j] = vdupq_n_f32(0.0f);
                    }

                    const float* ap = a_ptr + m_off;
                    const float* bp = b_ptr + 16;

                    for (int k = 0; k < k_size; ++k) {
                        float32x4_t b0 = vld1q_f32(bp);
                        float32x4_t b1 = vld1q_f32(bp + 4);
                        float32x4_t b2 = vld1q_f32(bp + 8);
                        float32x4_t b3 = vld1q_f32(bp + 12);
                        bp += PACKED_COL;

                        float32x4_t a = vld1q_f32(ap);
                        ap += PACKED_ROW;

                        c[0][0] = vfmaq_laneq_f32(c[0][0], b0, a, 0);
                        c[0][1] = vfmaq_laneq_f32(c[0][1], b1, a, 0);
                        c[0][2] = vfmaq_laneq_f32(c[0][2], b2, a, 0);
                        c[0][3] = vfmaq_laneq_f32(c[0][3], b3, a, 0);
                        c[1][0] = vfmaq_laneq_f32(c[1][0], b0, a, 1);
                        c[1][1] = vfmaq_laneq_f32(c[1][1], b1, a, 1);
                        c[1][2] = vfmaq_laneq_f32(c[1][2], b2, a, 1);
                        c[1][3] = vfmaq_laneq_f32(c[1][3], b3, a, 1);
                        c[2][0] = vfmaq_laneq_f32(c[2][0], b0, a, 2);
                        c[2][1] = vfmaq_laneq_f32(c[2][1], b1, a, 2);
                        c[2][2] = vfmaq_laneq_f32(c[2][2], b2, a, 2);
                        c[2][3] = vfmaq_laneq_f32(c[2][3], b3, a, 2);
                        c[3][0] = vfmaq_laneq_f32(c[3][0], b0, a, 3);
                        c[3][1] = vfmaq_laneq_f32(c[3][1], b1, a, 3);
                        c[3][2] = vfmaq_laneq_f32(c[3][2], b2, a, 3);
                        c[3][3] = vfmaq_laneq_f32(c[3][3], b3, a, 3);
                    }

                    for (int i = 0; i < 4; ++i) {
                        int cur_row = m_start + m_off + i;
                        if (cur_row >= m_valid) break;
                        float* dst_row = c_ptr + (m_off + i) * strideC + 16;
                        if (n_start + 32 <= n_valid) {
                            vst1q_f32(dst_row, c[i][0]);
                            vst1q_f32(dst_row + 4, c[i][1]);
                            vst1q_f32(dst_row + 8, c[i][2]);
                            vst1q_f32(dst_row + 12, c[i][3]);
                        } else {
                            float tmp[16];
                            vst1q_f32(tmp, c[i][0]);
                            vst1q_f32(tmp + 4, c[i][1]);
                            vst1q_f32(tmp + 8, c[i][2]);
                            vst1q_f32(tmp + 12, c[i][3]);
                            for (int j = 0; j < 16; ++j) if (n_start + 16 + j < n_valid) dst_row[j] = tmp[j];
                        }
                    }
                }
            }
        }
#endif
    }

    template <AccPhase Phase = AccPhase::Unspecified, typename TileAcc, typename TileLeft, typename TileRight>
    PTO_INTERNAL void TMATMUL_IMPL_COMMON(TileAcc &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, bool accumulate)
    {
        using T = typename TileAcc::DType;
        using A_T = typename TileLeft::DType;
        using B_T = typename TileRight::DType;

        if constexpr (TileLeft::BFractal == BLayout::PackedA &&
                    TileRight::BFractal == BLayout::PackedB &&
                    std::is_same_v<T, float> &&
                    std::is_same_v<A_T, float> &&
                    std::is_same_v<B_T, float>) {
                        int k_aligned = TileLeft::Cols;
                        uint16_t m = aMatrix.GetValidRow();
                        uint16_t k = aMatrix.GetValidCol();
                        uint16_t n = bMatrix.GetValidCol();

                        const float* packed_a = (const float*)aMatrix.data();
                        const float* packed_b = (const float*)bMatrix.data();
                        float* c_data = (float*)cMatrix.data();

                        int strideC = TileAcc::RowStride;

                        int m_blocks = (m + PACKED_ROW - 1) / PACKED_ROW;
                        int n_blocks = (n + PACKED_COL - 1) / PACKED_COL;

                        for (int mb = 0; mb < m_blocks; ++mb) {
                            int m_curr = mb * PACKED_ROW;
                            for (int nb = 0; nb < n_blocks; ++nb) {
                                int n_curr = nb * PACKED_COL;

                                const float* ptr_a = packed_a + mb * k_aligned * PACKED_ROW;
                                const float* ptr_b = packed_b + nb * k_aligned * PACKED_COL;
                                float* ptr_c = c_data + m_curr * strideC + n_curr;
                                Kernel32x32(ptr_c, strideC, ptr_a, ptr_b, k, m_curr, n_curr, m, n, accumulate);
                }
            }
        } else {
            uint16_t m = aMatrix.GetValidRow();
            uint16_t k = aMatrix.GetValidCol();
            uint16_t n = bMatrix.GetValidCol();

            TMatmul_Generic<TileAcc, TileLeft, TileRight>(cMatrix.data(), aMatrix.data(), bMatrix.data(), m, n, k, accumulate);
        }
    }

    template <AccPhase Phase = AccPhase::Unspecified, typename TileAcc, typename TileLeft, typename TileRight>
    PTO_INTERNAL void TMATMUL_IMPL(TileAcc &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
    {
        TMATMUL_IMPL_COMMON(cMatrix, aMatrix, bMatrix, false);
    }

    template <AccPhase Phase = AccPhase::Unspecified, typename TileAcc, typename TileLeft, typename TileRight>
    PTO_INTERNAL void TMATMUL_ACC_IMPL(TileAcc &cOutMatrix, TileAcc &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix)
    {
        if (&cOutMatrix != &cInMatrix) {
            std::copy(cInMatrix.data(), cInMatrix.data() + cInMatrix.Numel, cOutMatrix.data());
        }
        TMATMUL_IMPL_COMMON(cOutMatrix, aMatrix, bMatrix, true);
    }

    template <AccPhase Phase = AccPhase::Unspecified, typename TileAcc, typename TileLeft, typename TileRight, typename TileBias>
    PTO_INTERNAL void TMATMUL_BIAS_IMPL(TileAcc &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasMatrix)
    {
        TMATMUL_IMPL_COMMON(cMatrix, aMatrix, bMatrix, false);

        int m = cMatrix.GetValidRow();
        int n = cMatrix.GetValidCol();

        for(size_t r=0; r<m; r++) {
            for(size_t c=0; c<n; c++) {
                size_t out_idx = GetTileElementOffset<TileAcc>(r,c);
                size_t bias_idx = GetTileElementOffset<TileBias>(0,c);
                cMatrix.data()[out_idx] += static_cast<typename TileAcc::DType>(biasMatrix.data()[bias_idx]);
            }
        }
    }
}
#endif
