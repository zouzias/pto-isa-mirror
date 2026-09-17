/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TEXTRACT_SMALL_M_CASES_H_
#define TEXTRACT_SMALL_M_CASES_H_

// Shared case parameters for host buffer sizing and kernel template instantiation.
template <int Key>
struct TExtractSmallMCase;

template <>
struct TExtractSmallMCase<1> {
    static constexpr bool Bf16 = false;
    static constexpr bool Compact = false;
    static constexpr bool Dynamic = false;
    static constexpr int Rows = 16, Cols = 64, M = 4, K = 64, Row = 0, Col = 0, Groups = 4;
};
template <>
struct TExtractSmallMCase<2> {
    static constexpr bool Bf16 = true;
    static constexpr bool Compact = false;
    static constexpr bool Dynamic = false;
    static constexpr int Rows = 16, Cols = 64, M = 4, K = 64, Row = 0, Col = 0, Groups = 4;
};
template <>
struct TExtractSmallMCase<3> {
    static constexpr bool Bf16 = false;
    static constexpr bool Compact = true;
    static constexpr bool Dynamic = true;
    static constexpr int Rows = 32, Cols = 128, M = 4, K = 63, Row = 15, Col = 16, Groups = 1;
};
template <>
struct TExtractSmallMCase<4> {
    static constexpr bool Bf16 = true;
    static constexpr bool Compact = true;
    static constexpr bool Dynamic = true;
    static constexpr int Rows = 32, Cols = 128, M = 4, K = 63, Row = 15, Col = 16, Groups = 1;
};
template <>
struct TExtractSmallMCase<5> {
    static constexpr bool Bf16 = false;
    static constexpr bool Compact = false;
    static constexpr bool Dynamic = false;
    static constexpr int Rows = 32, Cols = 64, M = 15, K = 64, Row = 17, Col = 16, Groups = 1;
};
template <>
struct TExtractSmallMCase<6> {
    static constexpr bool Bf16 = true;
    static constexpr bool Compact = true;
    static constexpr bool Dynamic = false;
    static constexpr int Rows = 32, Cols = 64, M = 8, K = 17, Row = 24, Col = 16, Groups = 1;
};
template <>
struct TExtractSmallMCase<7> {
    static constexpr bool Bf16 = false;
    static constexpr bool Compact = false;
    static constexpr bool Dynamic = false;
    static constexpr int Rows = 16, Cols = 64, M = 1, K = 1, Row = 15, Col = 0, Groups = 1;
};
template <>
struct TExtractSmallMCase<8> {
    static constexpr bool Bf16 = true;
    static constexpr bool Compact = false;
    static constexpr bool Dynamic = false;
    static constexpr int Rows = 16, Cols = 64, M = 2, K = 64, Row = 13, Col = 0, Groups = 1;
};
template <>
struct TExtractSmallMCase<9> {
    static constexpr bool Bf16 = false;
    static constexpr bool Compact = false;
    static constexpr bool Dynamic = true;
    static constexpr int Rows = 32, Cols = 64, M = 16, K = 64, Row = 16, Col = 16, Groups = 1;
};
template <>
struct TExtractSmallMCase<10> {
    static constexpr bool Bf16 = true;
    static constexpr bool Compact = false;
    static constexpr bool Dynamic = false;
    static constexpr int Rows = 16, Cols = 512, M = 4, K = 512, Row = 12, Col = 0, Groups = 1;
};
template <>
struct TExtractSmallMCase<11> {
    static constexpr bool Bf16 = false;
    static constexpr bool Compact = true;
    static constexpr bool Dynamic = false;
    static constexpr int Rows = 16, Cols = 256, M = 4, K = 256, Row = 12, Col = 0, Groups = 1;
};
template <>
struct TExtractSmallMCase<12> {
    static constexpr bool Bf16 = false;
    static constexpr bool Compact = false;
    static constexpr bool Dynamic = true;
    static constexpr int Rows = 32, Cols = 128, M = 4, K = 63, Row = 28, Col = 16, Groups = 1;
};

template <>
struct TExtractSmallMCase<13> {
    static constexpr bool Bf16 = false;
    static constexpr bool Compact = true;
    static constexpr bool Dynamic = true;
    static constexpr int Rows = 32, Cols = 128, M = 16, K = 63, Row = 16, Col = 16, Groups = 1;
};

#endif // TEXTRACT_SMALL_M_CASES_H_
