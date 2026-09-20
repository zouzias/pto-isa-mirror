/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TEXTRACT_ACC2MAT_LAYOUT_CASES_H_
#define TEXTRACT_ACC2MAT_LAYOUT_CASES_H_

template <int Key>
struct TExtractLayoutCase;
#define LAYOUT_CASE(                                                                                          \
    key, Kind, Fractal, M, K, N, Rows, Cols, ValidRows, ValidCols, Row, Col, Phase, Split, Bias, Relu, Plain) \
    template <>                                                                                               \
    struct TExtractLayoutCase<key> {                                                                          \
        static constexpr int KindValue = Kind;                                                                \
        static constexpr int FractalValue = Fractal;                                                          \
        static constexpr int MValue = M;                                                                      \
        static constexpr int KValue = K;                                                                      \
        static constexpr int NValue = N;                                                                      \
        static constexpr int RowsValue = Rows;                                                                \
        static constexpr int ColsValue = Cols;                                                                \
        static constexpr int ValidRowsValue = ValidRows;                                                      \
        static constexpr int ValidColsValue = ValidCols;                                                      \
        static constexpr int RowValue = Row;                                                                  \
        static constexpr int ColValue = Col;                                                                  \
        static constexpr int PhaseValue = Phase;                                                              \
        static constexpr int SplitValue = Split;                                                              \
        static constexpr int ReluValue = Relu;                                                                \
        static constexpr int PlainValue = Plain;                                                              \
        static constexpr int BiasValue = Bias;                                                                \
    };

// key, dtype (half/bf16/int32/float), fractal, source M/K/N, physical/valid destination, offset, phase, split, bias,
// relu, plain
LAYOUT_CASE(1, 0, 1024, 32, 96, 64, 16, 64, 16, 64, 16, 0, 3, 0, 0, 0, 0)
LAYOUT_CASE(2, 1, 1024, 32, 96, 64, 16, 64, 16, 64, 16, 0, 3, 0, 0, 0, 0)
LAYOUT_CASE(3, 2, 512, 32, 96, 64, 16, 48, 16, 48, 16, 16, 3, 0, 0, 0, 0)
LAYOUT_CASE(4, 0, 512, 32, 96, 64, 16, 64, 16, 64, 16, 0, 3, 0, 0, 0, 0)
LAYOUT_CASE(5, 1, 512, 32, 96, 64, 16, 64, 16, 64, 16, 0, 3, 0, 0, 0, 0)
LAYOUT_CASE(6, 2, 1024, 32, 96, 64, 16, 48, 16, 48, 16, 16, 3, 0, 0, 0, 0)
LAYOUT_CASE(7, 3, 512, 32, 96, 64, 16, 48, 16, 48, 16, 16, 3, 0, 0, 0, 0)
LAYOUT_CASE(8, 0, 1024, 32, 96, 64, 16, 64, 16, 64, 16, 0, 2, 0, 0, 0, 0)
LAYOUT_CASE(9, 1, 1024, 32, 96, 64, 16, 64, 16, 64, 16, 0, 2, 0, 0, 0, 0)
LAYOUT_CASE(10, 2, 512, 32, 96, 64, 16, 48, 16, 48, 16, 16, 2, 0, 0, 0, 0)
LAYOUT_CASE(11, 0, 1024, 32, 96, 64, 16, 64, 16, 64, 16, 0, 0, 0, 0, 0, 0)
LAYOUT_CASE(12, 2, 512, 32, 96, 64, 16, 48, 16, 48, 16, 16, 0, 0, 0, 0, 0)
LAYOUT_CASE(13, 0, 1024, 32, 96, 64, 16, 64, 16, 64, 16, 0, 3, 1, 0, 0, 0)
LAYOUT_CASE(14, 2, 512, 32, 128, 64, 16, 48, 16, 48, 16, 16, 3, 1, 0, 0, 0)
LAYOUT_CASE(15, 0, 1024, 64, 96, 96, 32, 64, 16, 48, 16, 16, 3, 0, 0, 0, 0)
LAYOUT_CASE(16, 2, 512, 64, 96, 96, 32, 64, 16, 40, 16, 16, 3, 0, 0, 0, 0)
LAYOUT_CASE(17, 2, 512, 32, 96, 64, 16, 48, 16, 48, 16, 16, 3, 0, 1, 0, 0)
LAYOUT_CASE(18, 2, 512, 32, 96, 64, 16, 48, 16, 48, 16, 16, 2, 0, 1, 0, 0)
LAYOUT_CASE(19, 0, 1024, 64, 96, 128, 32, 96, 32, 96, 16, 16, 3, 0, 0, 0, 0)
LAYOUT_CASE(20, 0, 1024, 32, 96, 64, 16, 32, 7, 32, 5, 16, 3, 0, 0, 0, 0)
LAYOUT_CASE(21, 0, 1024, 32, 96, 64, 16, 32, 16, 16, 16, 48, 3, 0, 0, 0, 0)

LAYOUT_CASE(22, 0, 1024, 32, 96, 64, 16, 64, 16, 64, 16, 0, 3, 0, 0, 1, 0)
LAYOUT_CASE(23, 1, 1024, 32, 96, 64, 16, 64, 16, 64, 16, 0, 2, 0, 0, 1, 0)
LAYOUT_CASE(24, 2, 512, 32, 96, 64, 16, 48, 16, 48, 16, 16, 0, 0, 1, 0, 1)
LAYOUT_CASE(25, 0, 1024, 32, 96, 64, 16, 32, 16, 16, 16, 48, 0, 0, 0, 0, 1)
LAYOUT_CASE(26, 1, 1024, 32, 96, 64, 16, 64, 16, 64, 16, 0, 0, 0, 0, 0, 1)

#undef LAYOUT_CASE
#endif
