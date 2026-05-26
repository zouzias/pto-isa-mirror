/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MOE_DISPATCH_GOLDEN_H_
#define MOE_DISPATCH_GOLDEN_H_

#include "common.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace moe_dispatch {

struct MoeDispatchHostData {
    std::vector<uint16_t> inputA;
    std::vector<int32_t> expertIdx;
};

struct MoeDispatchGolden {
    std::vector<int32_t> tokenPerExpert;
    std::vector<int32_t> prefixPerExpert;
    std::vector<int32_t> expandedRowIdx;
    std::vector<uint16_t> packedA;
};

inline uint32_t ExpertOwnerRank(uint32_t expert, const MoeDispatchShape &shape)
{
    return expert / shape.expertPerRank;
}

inline MoeDispatchHostData GenerateHostData(const MoeDispatchShape &shape, uint32_t rank, uint32_t seed)
{
    MoeDispatchHostData data;
    data.inputA.resize(static_cast<size_t>(shape.m) * shape.k);
    data.expertIdx.resize(static_cast<size_t>(shape.m) * shape.topK);
    for (uint32_t row = 0; row < shape.m; ++row) {
        for (uint32_t col = 0; col < shape.k; ++col) {
            data.inputA[static_cast<size_t>(row) * shape.k + col] =
                static_cast<uint16_t>((seed + rank * 257 + row * 17 + col) & 0xFFFFU);
        }
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            data.expertIdx[static_cast<size_t>(row) * shape.topK + slot] =
                static_cast<int32_t>((row * shape.topK + slot + rank) % shape.expertNum);
        }
    }
    return data;
}

inline MoeDispatchGolden BuildDispatchGolden(const MoeDispatchShape &shape, const MoeDispatchHostData &data)
{
    const size_t inputElems = static_cast<size_t>(shape.m) * shape.k;
    const size_t routeElems = static_cast<size_t>(shape.m) * shape.topK;
    if (data.inputA.size() < inputElems || data.expertIdx.size() < routeElems) {
        throw std::invalid_argument("host data is smaller than moe_dispatch shape");
    }

    MoeDispatchGolden golden;
    golden.tokenPerExpert.assign(shape.expertNum, 0);
    golden.prefixPerExpert.assign(shape.expertNum, 0);
    golden.expandedRowIdx.assign(routeElems, -1);
    golden.packedA.assign(static_cast<size_t>(shape.maxOutputSize) * shape.k, 0);

    for (size_t route = 0; route < routeElems; ++route) {
        const int32_t expert = data.expertIdx[route];
        if (expert < 0 || static_cast<uint32_t>(expert) >= shape.expertNum) {
            throw std::invalid_argument("expertIdx contains out-of-range expert id");
        }
        ++golden.tokenPerExpert[expert];
    }

    int32_t prefix = 0;
    for (uint32_t expert = 0; expert < shape.expertNum; ++expert) {
        golden.prefixPerExpert[expert] = prefix;
        prefix += golden.tokenPerExpert[expert];
    }
    if (prefix > static_cast<int32_t>(shape.maxOutputSize)) {
        throw std::runtime_error("maxOutputSize is smaller than expanded routed rows");
    }

    std::vector<int32_t> cursor(shape.expertNum, 0);
    for (uint32_t token = 0; token < shape.m; ++token) {
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            const size_t route = static_cast<size_t>(token) * shape.topK + slot;
            const uint32_t expert = static_cast<uint32_t>(data.expertIdx[route]);
            const int32_t packedRow = golden.prefixPerExpert[expert] + cursor[expert]++;
            golden.expandedRowIdx[route] = packedRow;
            std::copy_n(data.inputA.data() + static_cast<size_t>(token) * shape.k, shape.k,
                        golden.packedA.data() + static_cast<size_t>(packedRow) * shape.k);
        }
    }
    return golden;
}

} // namespace moe_dispatch

#endif // MOE_DISPATCH_GOLDEN_H_
