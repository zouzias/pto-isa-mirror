/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "../tmov_ub2l1/ub2l1_nd2nz_kernel.h"
#include "nd2nz_cases.h"

using namespace pto;

template <int32_t TestKey>
void launchTExtractNd2Nz(uint64_t* out, uint64_t* src, void* stream);

#define LAUNCH_ND2NZ(                                                                                                \
    TestKey, Name, T, ElementBits, SrcRows, SrcCols, DstRows, DstCols, ValidRows, ValidCols, Operation, IndexRow,    \
    IndexCol, Dynamic, SrcValidRows, SrcValidCols)                                                                   \
    template <>                                                                                                      \
    void launchTExtractNd2Nz<TestKey>(uint64_t * out, uint64_t * src, void* stream)                                  \
    {                                                                                                                \
        PtoTestCommon::runUbToL1Nd2Nz<                                                                               \
            T, ElementBits, SrcRows, SrcCols, DstRows, DstCols, ValidRows, ValidCols, Operation, IndexRow, IndexCol, \
            Dynamic, SrcValidRows, SrcValidCols><<<1, nullptr, stream>>>(                                            \
            reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), ValidRows, ValidCols, SrcValidRows,    \
            SrcValidCols);                                                                                           \
    }
TEXTRACT_UB2L1_ND2NZ_CASES(LAUNCH_ND2NZ)
#undef LAUNCH_ND2NZ
