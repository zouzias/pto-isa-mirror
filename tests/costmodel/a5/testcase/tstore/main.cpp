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

void LaunchTStore(float *dst);

int main()
{
    constexpr std::size_t kElemCount = 16u * 31u;
    std::array<float, kElemCount> dst{};
    for (std::size_t i = 0; i < dst.size(); ++i) {
        dst[i] = static_cast<float>(i);
    }

    pto::mocker::ResetTrace();
    LaunchTStore(dst.data());
    pto::mocker::test::PrintTrace("tstore", pto::mocker::GetTrace());
    return 0;
}
