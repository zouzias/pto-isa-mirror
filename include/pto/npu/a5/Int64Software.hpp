/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef INT64_SOFTWARE_HPP
#define INT64_SOFTWARE_HPP

#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>

namespace pto {

enum class Int64BinaryOp { Add, Sub, Mul, Div, Max, Min, Rem, Shl, Shr };

template <typename T>
constexpr bool IsInt64SoftwareType = std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>;

template <typename T>
PTO_INTERNAL bool Int64CompareWordsSoftwareCalc(
    uint32_t lhsLow, uint32_t lhsHigh, uint32_t rhsLow, uint32_t rhsHigh, CmpMode mode)
{
    static_assert(IsInt64SoftwareType<T>, "Int64 software simulation only supports int64_t and uint64_t.");
    bool equal = lhsHigh == rhsHigh && lhsLow == rhsLow;
    bool less;
    if constexpr (std::is_same_v<T, int64_t>) {
        int32_t lhsSignedHigh = static_cast<int32_t>(lhsHigh);
        int32_t rhsSignedHigh = static_cast<int32_t>(rhsHigh);
        less = lhsSignedHigh < rhsSignedHigh || (lhsSignedHigh == rhsSignedHigh && lhsLow < rhsLow);
    } else {
        less = lhsHigh < rhsHigh || (lhsHigh == rhsHigh && lhsLow < rhsLow);
    }
    switch (mode) {
        case CmpMode::EQ:
            return equal;
        case CmpMode::NE:
            return !equal;
        case CmpMode::LT:
            return less;
        case CmpMode::GT:
            return !equal && !less;
        case CmpMode::GE:
            return !less;
        case CmpMode::LE:
            return equal || less;
        default:
            PTO_ASSERT(false, "Int64 software comparison received an invalid comparison mode.");
            return false;
    }
}

template <typename T, unsigned DstRowBytes, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64CompareSoftware(
    __ubuf__ uint8_t* dst, __ubuf__ T* src0, __ubuf__ T* src1, CmpMode mode, unsigned validRows, unsigned validCols)
{
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
#endif
    __ubuf__ uint32_t* dstWords = (__ubuf__ uint32_t*)dst;
    __ubuf__ uint32_t* src0Words = (__ubuf__ uint32_t*)src0;
    __ubuf__ uint32_t* src1Words = (__ubuf__ uint32_t*)src1;
    constexpr unsigned dstRowWords = DstRowBytes / sizeof(uint32_t);
    unsigned validWords = (validCols + 31U) / 32U;
    for (unsigned i = 0; i < validRows; ++i) {
        for (unsigned j = 0; j < validWords; ++j) {
            uint32_t packed = 0;
            for (unsigned bit = 0; bit < 32U; ++bit) {
                unsigned col = j * 32U + bit;
                unsigned src0Word = (i * Src0Cols + col) * 2U;
                unsigned src1Word = (i * Src1Cols + col) * 2U;
                if (col < validCols && Int64CompareWordsSoftwareCalc<T>(
                                           src0Words[src0Word], src0Words[src0Word + 1], src1Words[src1Word],
                                           src1Words[src1Word + 1], mode)) {
                    packed |= 1U << bit;
                }
            }
            dstWords[i * dstRowWords + j] = packed;
        }
    }
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
#endif
}

template <typename T, unsigned DstRowBytes, unsigned SrcCols>
PTO_INTERNAL void Int64CompareScalarSoftware(
    __ubuf__ uint8_t* dst, __ubuf__ T* src, T scalar, CmpMode mode, unsigned validRows, unsigned validCols)
{
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
#endif
    __ubuf__ uint32_t* dstWords = (__ubuf__ uint32_t*)dst;
    __ubuf__ uint32_t* srcWords = (__ubuf__ uint32_t*)src;
    uint64_t scalarBits = static_cast<uint64_t>(scalar);
    uint32_t scalarLow = static_cast<uint32_t>(scalarBits);
    uint32_t scalarHigh = static_cast<uint32_t>(scalarBits >> 32);
    constexpr unsigned dstRowWords = DstRowBytes / sizeof(uint32_t);
    unsigned validWords = (validCols + 31U) / 32U;
    for (unsigned i = 0; i < validRows; ++i) {
        for (unsigned j = 0; j < validWords; ++j) {
            uint32_t packed = 0;
            for (unsigned bit = 0; bit < 32U; ++bit) {
                unsigned col = j * 32U + bit;
                unsigned srcWord = (i * SrcCols + col) * 2U;
                if (col < validCols && Int64CompareWordsSoftwareCalc<T>(
                                           srcWords[srcWord], srcWords[srcWord + 1], scalarLow, scalarHigh, mode)) {
                    packed |= 1U << bit;
                }
            }
            dstWords[i * dstRowWords + j] = packed;
        }
    }
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
#endif
}

template <typename T, Int64BinaryOp Op>
PTO_INTERNAL T Int64BinarySoftwareCalc(T lhs, T rhs)
{
    static_assert(IsInt64SoftwareType<T>, "Int64 software simulation only supports int64_t and uint64_t.");
    using U = uint64_t;
    if constexpr (Op == Int64BinaryOp::Add) {
        return static_cast<T>(static_cast<U>(lhs) + static_cast<U>(rhs));
    } else if constexpr (Op == Int64BinaryOp::Sub) {
        return static_cast<T>(static_cast<U>(lhs) - static_cast<U>(rhs));
    } else if constexpr (Op == Int64BinaryOp::Mul) {
        return static_cast<T>(static_cast<U>(lhs) * static_cast<U>(rhs));
    } else if constexpr (Op == Int64BinaryOp::Max) {
        return lhs > rhs ? lhs : rhs;
    } else if constexpr (Op == Int64BinaryOp::Min) {
        return lhs < rhs ? lhs : rhs;
    } else if constexpr (Op == Int64BinaryOp::Div) {
        if (rhs == 0) {
            return 0;
        }
        if constexpr (std::is_same_v<T, int64_t>) {
            if (lhs == static_cast<T>(static_cast<U>(1) << 63) && rhs == -1) {
                return lhs;
            }
        }
        return lhs / rhs;
    } else if constexpr (Op == Int64BinaryOp::Rem) {
        if (rhs == 0) {
            return 0;
        }
        if constexpr (std::is_same_v<T, int64_t>) {
            if (lhs == static_cast<T>(static_cast<U>(1) << 63) && rhs == -1) {
                return 0;
            }
        }
        return lhs % rhs;
    } else if constexpr (Op == Int64BinaryOp::Shl) {
        return static_cast<T>(static_cast<U>(lhs) << (static_cast<U>(rhs) & 63U));
    } else {
        return static_cast<T>(lhs >> (static_cast<U>(rhs) & 63U));
    }
}

template <Int64BinaryOp Op, typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64ScalarSoftware(
    __ubuf__ T* dst, __ubuf__ T* src, T scalar, unsigned validRows, unsigned validCols)
{
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
#endif
    for (unsigned i = 0; i < validRows; ++i) {
        for (unsigned j = 0; j < validCols; ++j) {
            dst[i * DstCols + j] = Int64BinarySoftwareCalc<T, Op>(src[i * SrcCols + j], scalar);
        }
    }
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
#endif
}

template <typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64ReverseDivSoftware(
    __ubuf__ T* dst, __ubuf__ T* src, T scalar, unsigned validRows, unsigned validCols)
{
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
#endif
    for (unsigned i = 0; i < validRows; ++i) {
        for (unsigned j = 0; j < validCols; ++j) {
            dst[i * DstCols + j] = Int64BinarySoftwareCalc<T, Int64BinaryOp::Div>(scalar, src[i * SrcCols + j]);
        }
    }
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
#endif
}

template <typename T, unsigned DstRows, unsigned DstCols>
PTO_INTERNAL void Int64FillSoftware(__ubuf__ T* dst, T scalar, unsigned validRows, unsigned validCols, bool rowMajor)
{
    static_assert(IsInt64SoftwareType<T>, "Int64 software simulation only supports int64_t and uint64_t.");
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
#endif
    for (unsigned i = 0; i < validRows; ++i) {
        for (unsigned j = 0; j < validCols; ++j) {
            unsigned offset = rowMajor ? i * DstCols + j : j * DstRows + i;
            dst[offset] = scalar;
        }
    }
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
#endif
}

template <typename T, unsigned DstCols>
PTO_INTERNAL void Int64TriSoftware(__ubuf__ T* dst, unsigned validRows, unsigned validCols, int diagonal, bool upper)
{
    static_assert(IsInt64SoftwareType<T>, "Int64 software simulation only supports int64_t and uint64_t.");
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
#endif
    for (unsigned i = 0; i < validRows; ++i) {
        for (unsigned j = 0; j < validCols; ++j) {
            bool one = upper ? static_cast<int>(j) - static_cast<int>(i) >= diagonal :
                               static_cast<int>(j) - static_cast<int>(i) <= diagonal;
            dst[i * DstCols + j] = one ? static_cast<T>(1) : static_cast<T>(0);
        }
    }
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
#endif
}

template <typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64RowExpandSoftware(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols)
{
    static_assert(IsInt64SoftwareType<T>, "Int64 software simulation only supports int64_t and uint64_t.");
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
#endif
    for (unsigned i = 0; i < validRows; ++i) {
        T value = src[i * SrcCols];
        for (unsigned j = 0; j < validCols; ++j) {
            dst[i * DstCols + j] = value;
        }
    }
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
#endif
}

template <typename T, unsigned DstCols>
PTO_INTERNAL void Int64ColExpandSoftware(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols)
{
    static_assert(IsInt64SoftwareType<T>, "Int64 software simulation only supports int64_t and uint64_t.");
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
#endif
    for (unsigned i = 0; i < validRows; ++i) {
        for (unsigned j = 0; j < validCols; ++j) {
            dst[i * DstCols + j] = src[j];
        }
    }
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
#endif
}

template <typename T, unsigned DstCols, unsigned MaskRowBytes, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64SelectSoftware(
    __ubuf__ T* dst, __ubuf__ uint8_t* mask, __ubuf__ T* src0, __ubuf__ T* src1, unsigned validRows, unsigned validCols)
{
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
#endif
    for (unsigned i = 0; i < validRows; ++i) {
        for (unsigned j = 0; j < validCols; ++j) {
            bool selectSrc0 = (mask[i * MaskRowBytes + j / 8U] & (1U << (j % 8U))) != 0;
            dst[i * DstCols + j] = selectSrc0 ? src0[i * Src0Cols + j] : src1[i * Src1Cols + j];
        }
    }
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
#endif
}

template <typename T, unsigned DstCols, unsigned MaskRowBytes, unsigned SrcCols>
PTO_INTERNAL void Int64SelectScalarSoftware(
    __ubuf__ T* dst, __ubuf__ uint8_t* mask, __ubuf__ T* src, T scalar, unsigned validRows, unsigned validCols)
{
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
#endif
    for (unsigned i = 0; i < validRows; ++i) {
        for (unsigned j = 0; j < validCols; ++j) {
            bool selectSrc = (mask[i * MaskRowBytes + j / 8U] & (1U << (j % 8U))) != 0;
            dst[i * DstCols + j] = selectSrc ? src[i * SrcCols + j] : scalar;
        }
    }
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
#endif
}

template <typename T, typename I, unsigned DstNumel, unsigned SrcCols, unsigned IdxCols>
PTO_INTERNAL void Int64ScatterSoftware(
    __ubuf__ T* dst, __ubuf__ T* src, __ubuf__ I* index, unsigned validRows, unsigned validCols)
{
    static_assert(IsInt64SoftwareType<T>, "Int64 software simulation only supports int64_t and uint64_t.");
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
#endif
    for (unsigned i = 0; i < DstNumel; ++i) {
        dst[i] = 0;
    }
    for (unsigned i = 0; i < validRows; ++i) {
        for (unsigned j = 0; j < validCols; ++j) {
            unsigned srcOffset = i * SrcCols + j;
            unsigned dstOffset = static_cast<unsigned>(index[i * IdxCols + j]);
            // As with the hardware path, indexes are element offsets and must be in range.
            dst[dstOffset] = src[srcOffset];
        }
    }
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
#endif
}

template <typename T, typename I, unsigned DstCols, unsigned IdxCols>
PTO_INTERNAL void Int64GatherSoftware(
    __ubuf__ T* dst, __ubuf__ T* src, __ubuf__ I* index, unsigned validRows, unsigned validCols)
{
    static_assert(IsInt64SoftwareType<T>, "Int64 software simulation only supports int64_t and uint64_t.");
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
#endif
    for (unsigned i = 0; i < validRows; ++i) {
        for (unsigned j = 0; j < validCols; ++j) {
            // As with the hardware path, indexes are element offsets and must be in range.
            unsigned srcOffset = static_cast<unsigned>(index[i * IdxCols + j]);
            dst[i * DstCols + j] = src[srcOffset];
        }
    }
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
#endif
}

template <MaskPattern Pattern>
PTO_INTERNAL constexpr unsigned Int64MaskPatternOffset()
{
    if constexpr (Pattern == MaskPattern::P1010 || Pattern == MaskPattern::P0010) {
        return 1;
    } else if constexpr (Pattern == MaskPattern::P0100) {
        return 2;
    } else if constexpr (Pattern == MaskPattern::P1000) {
        return 3;
    } else {
        return 0;
    }
}

template <MaskPattern Pattern, auto Axis, typename T, unsigned DstNumel, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64ScatterPatternSoftware(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols)
{
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
#endif
    constexpr unsigned times = GetTimesByMask<Pattern>();
    constexpr unsigned offset = Int64MaskPatternOffset<Pattern>();
    for (unsigned i = 0; i < DstNumel; ++i) {
        dst[i] = 0;
    }
    for (unsigned i = 0; i < validRows; ++i) {
        for (unsigned j = 0; j < validCols; ++j) {
            if constexpr (Axis == ScatterAxis::SCATTER_COL) {
                dst[(i * times + offset) * DstCols + j] = src[i * SrcCols + j];
            } else {
                dst[i * DstCols + j * times + offset] = src[i * SrcCols + j];
            }
        }
    }
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
#endif
}

template <MaskPattern Pattern, auto Axis, typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64GatherPatternSoftware(__ubuf__ T* dst, __ubuf__ T* src, unsigned srcRows, unsigned srcCols)
{
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
#endif
    constexpr unsigned times = GetTimesByMask<Pattern>();
    constexpr unsigned offset = Int64MaskPatternOffset<Pattern>();
    if constexpr (Axis == GatherAxis::GATHER_COL) {
        unsigned dstRows = srcRows / times;
        for (unsigned i = 0; i < dstRows; ++i) {
            for (unsigned j = 0; j < srcCols; ++j) {
                dst[i * DstCols + j] = src[(i * times + offset) * SrcCols + j];
            }
        }
    } else {
        unsigned dstCols = srcCols / times;
        for (unsigned i = 0; i < srcRows; ++i) {
            for (unsigned j = 0; j < dstCols; ++j) {
                dst[i * DstCols + j] = src[i * SrcCols + j * times + offset];
            }
        }
    }
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
#endif
}

template <Int64BinaryOp Op, typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64RowReduceSoftware(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols)
{
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
#endif
    for (unsigned i = 0; i < validRows; ++i) {
        T value = src[i * SrcCols];
        for (unsigned j = 1; j < validCols; ++j) {
            value = Int64BinarySoftwareCalc<T, Op>(value, src[i * SrcCols + j]);
        }
        dst[i * DstCols] = value;
    }
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
#endif
}

template <Int64BinaryOp Op, typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64ColReduceSoftware(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols)
{
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
#endif
    for (unsigned j = 0; j < validCols; ++j) {
        T value = src[j];
        for (unsigned i = 1; i < validRows; ++i) {
            value = Int64BinarySoftwareCalc<T, Op>(value, src[i * SrcCols + j]);
        }
        dst[j] = value;
    }
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
#endif
}

template <Int64BinaryOp Op, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64BinarySoftware(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned validRows, unsigned validCols)
{
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
#endif
    for (unsigned i = 0; i < validRows; ++i) {
        for (unsigned j = 0; j < validCols; ++j) {
            dst[i * DstCols + j] = Int64BinarySoftwareCalc<T, Op>(src0[i * Src0Cols + j], src1[i * Src1Cols + j]);
        }
    }
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
#endif
}

template <Int64BinaryOp Op, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64PartSoftware(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned src0Rows, unsigned src0Cols, unsigned src1Rows,
    unsigned src1Cols, unsigned dstRows, unsigned dstCols)
{
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
#endif
    for (unsigned i = 0; i < dstRows; ++i) {
        for (unsigned j = 0; j < dstCols; ++j) {
            bool inSrc0 = i < src0Rows && j < src0Cols;
            bool inSrc1 = i < src1Rows && j < src1Cols;
            if (inSrc0 && inSrc1) {
                dst[i * DstCols + j] = Int64BinarySoftwareCalc<T, Op>(src0[i * Src0Cols + j], src1[i * Src1Cols + j]);
            } else if (inSrc0) {
                dst[i * DstCols + j] = src0[i * Src0Cols + j];
            } else if (inSrc1) {
                dst[i * DstCols + j] = src1[i * Src1Cols + j];
            }
        }
    }
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
#endif
}

} // namespace pto

#endif
