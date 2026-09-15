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

#include <type_traits>

#include <pto/costmodel/trace.hpp>

template <typename CType, typename AType, typename BType>
inline void mad(
    CType c, AType a, BType b, auto m, auto k, auto n, auto phase, auto gemvCtrl, auto cmatrixSource,
    auto cmatrixInitVal)
{
    using AElement = std::remove_pointer_t<AType>;
    const uint64_t mTiles = (static_cast<uint64_t>(m) + 15) / 16;
    const uint64_t kTiles = (static_cast<uint64_t>(k) + 32 / sizeof(AElement) - 1) / (32 / sizeof(AElement));
    const uint64_t nTiles = (static_cast<uint64_t>(n) + 15) / 16;
    const uint64_t slope = std::is_same_v<AElement, float> ? 2 : 1;
    uint64_t cycles = slope * mTiles * kTiles * nTiles;
    if (::pto::mocker::IsPipeQueueEmpty(::pto::mocker::evaluator::PipeKey::CUBE))
        cycles += 6;
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::CUBE, "mad", cycles, c, a, b, m, k, n, phase, gemvCtrl, cmatrixSource,
        cmatrixInitVal);
}

template <typename CType, typename AType, typename BType>
inline void mad_mx(
    CType c, AType a, BType b, auto m, auto k, auto n, auto phase, auto gemvCtrl, auto biasBufferCtrl,
    auto cmatrixInitVal)
{
    mad(c, a, b, m, k, n, phase, gemvCtrl, biasBufferCtrl, cmatrixInitVal);
}
