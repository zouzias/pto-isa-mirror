/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef VFSIM_API_NATIVE_RUNTIME_VALUE_H
#define VFSIM_API_NATIVE_RUNTIME_VALUE_H

#include <cstdint>
#include <string>
#include <vector>

namespace vfsim {

enum class ValueStorageKind { Register, UB, Scalar };

struct ValueInfo {
    std::string valueId;
    ValueStorageKind storage = ValueStorageKind::Register;
    std::string dtype;
    std::vector<int64_t> shape;
};

ValueStorageKind inferValueStorage(const std::string& valueId);
std::string valueStorageName(ValueStorageKind storage);

} // namespace vfsim

#endif // VFSIM_API_NATIVE_RUNTIME_VALUE_H
