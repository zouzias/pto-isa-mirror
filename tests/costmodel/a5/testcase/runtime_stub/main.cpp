/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "mocker_test_common.hpp"

#include <pto/pto-inst.hpp>

int main()
{
    std::array<float, 64> src0{};
    std::array<float, 64> src1{};
    std::array<float, 64> dst{};
    uint32_t count = 31;

    pto::RegTensor<float> reg0;
    pto::RegTensor<float> reg1;
    pto::RegTensor<float> reg2;
    auto mask = pto::CreatePredicate<float>(count);

    pto::mocker::ResetTrace();
    vlds(reg0, src0.data(), 0, NORM);
    vlds(reg1, src1.data(), 16, NORM, POST_UPDATE);
    vadd(reg2, reg0, reg1, mask, MODE_ZEROING);
    vsts(reg2, dst.data(), 0, pto::DistVST::DIST_NORM_B32, mask);
    pto::mocker::test::PrintTrace("runtime_stub raw calls", pto::mocker::GetTrace());
    return 0;
}
