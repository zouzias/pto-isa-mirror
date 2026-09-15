/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#pragma once

#include <string_view>
#include <pto/costmodel/lightweight_costmodel.hpp>

namespace pto::mocker::lightweight {

// clang-format off
#define PTO_COSTMODEL_OPCODE_LIST                                                                                      \
    X(TADD)                                                                                                           \
    X(TSUB)                                                                                                           \
    X(TMUL)                                                                                                           \
    X(TDIV)                                                                                                           \
    X(TRECIP)                                                                                                         \
    X(TADDS)                                                                                                          \
    X(TSUBS)                                                                                                          \
    X(TMULS)                                                                                                          \
    X(TDIVS) X(TMINS) X(TMAXS) X(TABS) X(TNEG) X(TEXP) X(TSQRT) X(TRSQRT) X(TLOG) X(TRELU) X(TLRELU) X(TNOT)          \
        X(TROWSUM) X(TROWMAX) X(TROWMIN) X(TROWPROD) X(TCOLSUM) X(TCOLMAX) X(TCOLMIN) X(TCOLPROD) X(TMATMUL) X(TGEMV) \
            X(TCVT) X(TMOV) X(TLOAD) X(TSTORE) X(TTRANS) X(TPREFETCH) X(TSORT32) X(TMRGSORT) X(TSEL) X(TSCATTER)      \
                X(TEXTRACT) X(TINSERT) X(TROWEXPAND) X(TCOLEXPAND) X(TLOADCONV)
// clang-format on

inline bool TryMapOpcode(std::string_view opcode, ::pto::mocker::lightweight::PtoOpcode& out)
{
#define X(name)                                            \
    if (opcode == #name) {                                 \
        out = ::pto::mocker::lightweight::PtoOpcode::name; \
        return true;                                       \
    }
    PTO_COSTMODEL_OPCODE_LIST
#undef X
    return false;
}

// ── String → lightweight::DType mapping (X-macro, paired str↔enum) ──

#define PTO_COSTMODEL_DTYPE_LIST \
    X("fp32", Float)            \
    X("fp16", Half)             \
    X("int8", Int8)             \
    X("int16", Int16)           \
    X("int32", Int32)           \
    X("uint8", Uint8)           \
    X("uint16", Uint16)         \
    X("uint32", Uint32)         \
    X("bf16", BFloat16)         \
    X("fp8_e4m3", Float8E4M3)   \
    X("fp8_e5m2", Float8E5M2)   \
    X("hif8", HFloat8)          \
    X("fp4_e1m2", Float4E1M2)   \
    X("fp4_e2m1", Float4E2M1)

inline bool TryMapDType(std::string_view dtype, ::pto::mocker::lightweight::DType& out)
{
#define X(str, enum_val)                                     \
    if (dtype == str) {                                     \
        out = ::pto::mocker::lightweight::DType::enum_val;  \
        return true;                                        \
    }
    PTO_COSTMODEL_DTYPE_LIST
#undef X
    return false;
}


#undef PTO_COSTMODEL_OPCODE_LIST
#undef PTO_COSTMODEL_DTYPE_LIST

} // namespace pto::mocker::lightweight
