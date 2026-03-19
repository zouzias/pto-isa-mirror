/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_ISA_COSTMODEL_OP_STRUCT_HPP
#define PTO_ISA_COSTMODEL_OP_STRUCT_HPP

#include <iostream>
#include "pto/costmodel/costmodel_types.hpp"

#ifndef B16_REPEAT_MAX
#define B16_REPEAT_MAX 65535
#endif

namespace pto {

inline int sum_repeat_times;

// BinOp
struct AddOp {
    PTO_INTERNAL static void BinInstr(std::vector<CostModelStats>& stats, uint8_t repeats)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "AddOp, repeats:  " << static_cast<int>(repeats) << std::endl;
    }

    PTO_INTERNAL static void BinInstr(std::vector<CostModelStats>& stats, uint8_t repeats, uint8_t dstRepeatStride,
                                      uint8_t src0RepeatStride, uint8_t src1RepeatStride)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "AddOp, repeats:  " << static_cast<int>(repeats) << std::endl;
    }
};

struct MulOp {
    PTO_INTERNAL static void BinInstr(std::vector<CostModelStats>& stats, uint8_t repeats)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "MulOp, repeats: " << static_cast<int>(repeats) << std::endl;
    }

    PTO_INTERNAL static void BinInstr(std::vector<CostModelStats>& stats, uint8_t repeats, uint8_t dstRepeatStride,
                                      uint8_t src0RepeatStride, uint8_t src1RepeatStride)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "MulOp, repeats: " << static_cast<int>(repeats) << std::endl;
    }
};

struct SubOp {
    PTO_INTERNAL static void BinInstr(std::vector<CostModelStats>& stats, uint8_t repeats)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "SubOp, repeats: " << static_cast<int>(repeats) << std::endl;
    }

    PTO_INTERNAL static void BinInstr(std::vector<CostModelStats>& stats, uint8_t repeats, uint8_t dstRepeatStride,
                                      uint8_t src0RepeatStride, uint8_t src1RepeatStride)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "SubOp, repeats: " << static_cast<int>(repeats) << std::endl;
    }
};

// BinSOp
struct AddSOp {
    PTO_INTERNAL static void BinSInstr(std::vector<CostModelStats>& stats, uint8_t repeats)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "AddSOp, repeats: " << static_cast<int>(repeats) << std::endl;
    }

    PTO_INTERNAL static void BinSInstr(std::vector<CostModelStats>& stats, uint8_t repeats, uint8_t dstRepeatStride,
                                       uint8_t srcRepeatStride)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "AddSOp, repeats: " << static_cast<int>(repeats) << std::endl;
    }
};

struct MulSOp {
    PTO_INTERNAL static void BinSInstr(std::vector<CostModelStats>& stats, uint8_t repeats)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "MulSOp, repeats: " << static_cast<int>(repeats) << std::endl;
    }

    PTO_INTERNAL static void BinSInstr(std::vector<CostModelStats>& stats, uint8_t repeats, uint8_t dstRepeatStride,
                                       uint8_t srcRepeatStride)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "MulSOp, repeats: " << static_cast<int>(repeats) << std::endl;
    }
};


struct MinSOp {
    PTO_INTERNAL static void BinSInstr(std::vector<CostModelStats>& stats, uint8_t repeats)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "MinSOp, repeats: " << static_cast<int>(repeats) << std::endl;
    }

    PTO_INTERNAL static void BinSInstr(std::vector<CostModelStats>& stats, uint8_t repeats, uint8_t dstRepeatStride,
                                       uint8_t srcRepeatStride)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "MinSOp, repeats: " << static_cast<int>(repeats) << std::endl;
    }
};


struct SDivOp {
    PTO_INTERNAL static void BinSInstr(std::vector<CostModelStats>& stats, uint8_t repeats)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "SDivOp, repeats: " << static_cast<int>(repeats) << std::endl;
    }

    PTO_INTERNAL static void BinSInstr(std::vector<CostModelStats>& stats, uint8_t repeats, uint8_t dstRepeatStride,
                                       uint8_t srcRepeatStride)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "SDivOp, repeats: " << static_cast<int>(repeats) << std::endl;
    }
};

struct DivSOp {
    PTO_INTERNAL static void BinSInstr(std::vector<CostModelStats>& stats, uint8_t repeats)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "DivSOp, repeats: " << static_cast<int>(repeats) << std::endl;
    }

    PTO_INTERNAL static void BinSInstr(std::vector<CostModelStats>& stats, uint8_t repeats, uint8_t dstRepeatStride,
                                       uint8_t srcRepeatStride)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "DivSOp, repeats: " << static_cast<int>(repeats) << std::endl;
    }
};


// UnaryOp
struct AbsOp {
    PTO_INTERNAL static void UnaryInstr(std::vector<CostModelStats>& stats, uint8_t repeats,
                                        uint8_t dstStride = BLOCK_MAX_PER_REPEAT,
                                        uint8_t srcStride = BLOCK_MAX_PER_REPEAT)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "AbsOp, repeats: " << static_cast<int>(repeats) << std::endl;
    }
};

struct ExpOp {
    PTO_INTERNAL static void UnaryInstr(std::vector<CostModelStats>& stats, uint8_t repeats,
                                        uint8_t dstStride = BLOCK_MAX_PER_REPEAT,
                                        uint8_t srcStride = BLOCK_MAX_PER_REPEAT)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "ExpOp, repeats: " << static_cast<int>(repeats) << std::endl;
    }
};

struct SqrtOp {
    PTO_INTERNAL static void UnaryInstr(std::vector<CostModelStats>& stats, uint8_t repeats,
                                        uint8_t dstStride = BLOCK_MAX_PER_REPEAT,
                                        uint8_t srcStride = BLOCK_MAX_PER_REPEAT)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "SqrtOp, repeats: " << static_cast<int>(repeats) << std::endl;
    }
};

template <typename InstrOp>
struct TRowReduceOp {
    PTO_INTERNAL static void BinInstr(std::vector<CostModelStats>& stats, uint8_t rptTimes,
                                      uint16_t dstRptStride, uint16_t src0RptStride, uint16_t src1RptStride)
    {
        InstrOp::BinInstrImpl(stats, rptTimes, dstRptStride, src0RptStride, src1RptStride);

    }

    PTO_INTERNAL static void ReduceInstr(std::vector<CostModelStats>& stats, uint8_t rptTimes, uint16_t dstRptStride,
                                         uint16_t srcBlkStride, uint16_t srcRptStride)
    {
        InstrOp::ReduceInstrImpl(stats, rptTimes, dstRptStride, srcBlkStride, srcRptStride);
    }

    template <int Rows, int ValidRow, int Cols, int ValidCol>
    PTO_INTERNAL static void ReduceOptFP32_64x128(std::vector<CostModelStats>& stats)
    {
        //static_assert(std::is_same_v<T, float>, "This optimization is only for float type.");
        //static_assert(Rows == 64 && ValidRow == 64 && Cols == 128 && ValidCol == 128,
        //              "This optimization is only for [64, 128] input.");
        // [64, 128] -> [64, 16]
        InstrOp::GroupReduceInstrImpl(stats, ValidRow * 2, 1, 1, 8);
        //pipe_barrier(PIPE_V);
        // [64, 16] -> [64, 8]
        InstrOp::BinInstrImpl(stats, ValidRow / 8, 8, 16, 16, 1, 2, 2);
        //pipe_barrier(PIPE_V);
        // [64, 8] -> [64, 1]
        InstrOp::GroupReduceInstrImpl(stats, ValidRow / 8, 1, 1, 8);
        //pipe_barrier(PIPE_V);
        return;
    }

    template <int Rows, int ValidRow, int Cols, int ValidCol>
    PTO_INTERNAL static void ReduceOptFP32_32x256(std::vector<CostModelStats>& stats)
    {
       // static_assert(std::is_same_v<T, float>, "This optimization is only for float type.");
        //static_assert(Rows == 32 && ValidRow == 32 && Cols == 256 && ValidCol == 256,
        //              "This optimization is only for [32, 256] input.");
        // [32, 256] -> [32, 32]
        InstrOp::GroupReduceInstrImpl(stats, ValidRow * 4, 1, 1, 8);
        //pipe_barrier(PIPE_V);
        // [32, 32] -> [32, 16]
        InstrOp::BinInstrImpl(stats, ValidRow / 4, 8, 16, 16, 1, 2, 2);
        //pipe_barrier(PIPE_V);
        // [32, 16] -> [32, 8]
        InstrOp::BinInstrImpl(stats, ValidRow / 8, 8, 16, 16, 1, 2, 2);
        //pipe_barrier(PIPE_V);
        // [32, 8] -> [32, 1]
        InstrOp::GroupReduceInstrImpl(stats, ValidRow / 8, 1, 1, 8);
        //pipe_barrier(PIPE_V);
        return;
    }

    template <int Rows, int ValidRow, int Cols, int ValidCol>
    PTO_INTERNAL static void ReduceOptFP32_16x512(std::vector<CostModelStats>& stats)
    {
        //static_assert(std::is_same_v<T, float>, "This optimization is only for float type.");
        //static_assert(Rows == 16 && ValidRow == 16 && Cols == 512 && ValidCol == 512,
        //              "This optimization is only for [16, 512] input.");
        // [16, 512] -> [16, 64]
        InstrOp::GroupReduceInstrImpl(stats, ValidRow * 8, 1, 1, 8);
        //pipe_barrier(PIPE_V);
        // [16, 64] -> [16, 8]
        InstrOp::GroupReduceInstrImpl(stats, ValidRow, 1, 1, 8);
        //pipe_barrier(PIPE_V);
        // [16, 8] -> [16, 1]
        InstrOp::GroupReduceInstrImpl(stats, ValidRow / 8, 1, 1, 8);
        //pipe_barrier(PIPE_V);
        return;
    }

    template <int Rows, int ValidRow, int Cols, int ValidCol>
    PTO_INTERNAL static void ReduceOptFP32_8x1024(std::vector<CostModelStats>& stats)
    {
        //static_assert(std::is_same_v<T, float>, "This optimization is only for float type.");
        //static_assert(Rows == 8 && ValidRow == 8 && Cols == 1024 && ValidCol == 1024,
        //              "This optimization is only for [8, 1024] input.");
        // [8, 1024] -> [8, 128]
        InstrOp::GroupReduceInstrImpl(stats, ValidRow * 16, 1, 1, 8);
        //pipe_barrier(PIPE_V);
        // [8, 128] -> [8, 16]
        InstrOp::GroupReduceInstrImpl(stats, ValidRow * 2, 1, 1, 8);
        //pipe_barrier(PIPE_V);
        // [8, 16] -> [8, 8]
        InstrOp::BinInstrImpl(stats, ValidRow / 8, 8, 16, 16, 1, 2, 2);
        //pipe_barrier(PIPE_V);
        // [8, 8] -> [8, 1]
        InstrOp::GroupReduceInstrImpl(stats, ValidRow / 8, 1, 1, 8);
        //pipe_barrier(PIPE_V);
        return;
    }

    template <bool CntModeEn, int Cols, uint32_t DstStride, uint32_t SrcStride, uint8_t ElemPerRpt>
    PTO_INTERNAL static void ReduceInstrByMode(std::vector<CostModelStats>& stats, unsigned rptTimes)
    {
        if constexpr (DstStride > B16_REPEAT_MAX) {
            for (int i = 0; i < rptTimes; i++) {
                ReduceInstr(stats, 1, 0, 1, 0);
            }
        } else if constexpr (CntModeEn) {
            //set_mask_count();
            //set_vector_mask(0, (uint32_t)rptTimes * ElemPerRpt);
            ReduceInstr(stats, 0, DstStride, 1, SrcStride);
            //set_mask_norm();
            //set_vector_mask(-1, -1);
        } else {
            ReduceInstr(stats, rptTimes, DstStride, 1, SrcStride);
        }
    }

    template <bool CntModeEn, int DstCols, int Src0Cols, int Src1Cols, uint32_t DstStride, uint32_t Src0RptStride,
              uint32_t Src1RptStride, uint8_t ElemPerRpt>
    PTO_INTERNAL static void BinInstrByMode(std::vector<CostModelStats>& stats, unsigned rptTimes)
    {
        if constexpr (DstStride > REPEAT_MAX || Src0RptStride > REPEAT_MAX || Src1RptStride > REPEAT_MAX) {
            for (int i = 0; i < rptTimes; i++) {
                BinInstr(stats, 1, 0, 0, 0);
            }
        } else if constexpr (CntModeEn) {
            //set_mask_count();
            //set_vector_mask(0, rptTimes * ElemPerRpt);
            BinInstr(stats, 0, DstStride, Src0RptStride, Src1RptStride);
            //set_mask_norm();
            //set_vector_mask(-1, -1);
        } else {
            BinInstr(stats, rptTimes, DstStride, Src0RptStride, Src1RptStride);
        }
    }

    template <int TmpCols, int SrcCols, uint32_t TmpStride, uint32_t SrcStride, uint8_t ElemPerRpt>
    PTO_INTERNAL static void FillTmp(std::vector<CostModelStats>& stats, int srcRptPerRow, int validRow, int validCol)
    {
        if (validCol >= 2 * ElemPerRpt) {
            // validcol大于等于2次repeat，将完整的2次repeat比较后写入tmp
            BinInstrByMode<true, TmpCols, SrcCols, SrcCols, TmpStride, SrcStride, SrcStride, ElemPerRpt>(
                stats, validRow);
            //pipe_barrier(PIPE_V);
        }
    }

    template <int TmpCols, int SrcCols, uint32_t TmpStride, uint32_t SrcStride, uint8_t ElemPerRpt>
    PTO_INTERNAL static void TmpProc(std::vector<CostModelStats>& stats, int srcRptPerRow, int validRow)
    {
        for (int i = 2; i < srcRptPerRow; ++i) {
            BinInstrByMode<true, TmpCols, TmpCols, SrcCols, TmpStride, TmpStride, SrcStride, ElemPerRpt>(
                stats, validRow);
            //pipe_barrier(PIPE_V);
        }
    }
};

struct TRowMaxOp : TRowReduceOp<TRowMaxOp> {
    PTO_INTERNAL static void BinInstrImpl(std::vector<CostModelStats>& stats, uint8_t rptTimes, uint16_t dstRptStride,
                                          uint16_t src0RptStride, uint16_t src1RptStride, uint8_t dstBlockStride = 1,
                                          uint8_t src0BlockStride = 1, uint8_t src1BlockStride = 1)
    {
        //vmax(dst, src0, src1, rptTimes, dstBlockStride, src0BlockStride, src1BlockStride, dstRptStride, src0RptStride,
             //src1RptStride);
        stats.emplace_back("vmax", rptTimes, dstBlockStride, src0BlockStride, src1BlockStride, dstRptStride,
                           src0RptStride, src1RptStride, 0);
    }

    PTO_INTERNAL static void ReduceInstrImpl(std::vector<CostModelStats>& stats, uint8_t rptTimes, uint16_t dstRptStride,
                                             uint16_t srcBlkStride, uint16_t srcRptStride)
    {
        //vcmax(dst, src, rptTimes, dstRptStride, srcBlkStride, srcRptStride, ONLY_VALUE);
        stats.emplace_back("vcmax", rptTimes, dstRptStride, srcBlkStride, srcRptStride, 2);
    }

    PTO_INTERNAL static void GroupReduceInstrImpl(std::vector<CostModelStats>& stats, uint8_t rptTimes,
                                                  uint16_t dstRptStride, uint16_t srcBlkStride, uint16_t srcRptStride)
    {
        //vcgmax(dst, src, rptTimes, dstRptStride, src0Stride, src1Stride);
        stats.emplace_back("vcgmax", rptTimes, dstRptStride, srcBlkStride, srcRptStride, 0);
    }
};

} // namespace pto


#endif // PTO_ISA_COSTMODEL_OP_STRUCT_HPP
