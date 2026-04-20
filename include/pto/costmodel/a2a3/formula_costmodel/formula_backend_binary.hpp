/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MOCKER_FORMULA_BACKEND_ELEMENTWISE_HPP
#define PTO_MOCKER_FORMULA_BACKEND_ELEMENTWISE_HPP

#include <pto/costmodel/a2a3/formula_costmodel/formula_backend_utils.hpp>

namespace pto::mocker::fit {

template <typename T>
inline bool TryEstimateTSUBCycles(uint64_t rows, uint64_t cols, uint64_t &cycles)
{
    return TryEstimateFormulaCycles<T>("TSUB", rows, cols, cycles);
}

template <typename T>
inline bool TryEstimateTMULCycles(uint64_t rows, uint64_t cols, uint64_t &cycles)
{
    return TryEstimateFormulaCycles<T>("TMUL", rows, cols, cycles);
}

template <typename T>
inline bool TryEstimateTADDSCycles(uint64_t rows, uint64_t cols, uint64_t &cycles)
{
    return TryEstimateFormulaCycles<T>("TADDS", rows, cols, cycles);
}

template <typename T>
inline bool TryEstimateTDIVSCycles(uint64_t rows, uint64_t cols, uint64_t &cycles)
{
    return TryEstimateFormulaCycles<T>("TDIVS", rows, cols, cycles);
}

template <typename T>
inline bool TryEstimateTMULSCycles(uint64_t rows, uint64_t cols, uint64_t &cycles)
{
    return TryEstimateFormulaCycles<T>("TMULS", rows, cols, cycles);
}

template <typename T>
inline bool TryEstimateTMINSCycles(uint64_t rows, uint64_t cols, uint64_t &cycles)
{
    return TryEstimateFormulaCycles<T>("TMINS", rows, cols, cycles);
}

} // namespace pto::mocker::fit

#endif
