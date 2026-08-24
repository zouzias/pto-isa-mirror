// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef VFSIM_API_NATIVE_JSON_VF_INFO_ADAPTER_H_
#define VFSIM_API_NATIVE_JSON_VF_INFO_ADAPTER_H_

#include <filesystem>

#include "api/native/VfInfo.h"

namespace vfsim {

VfInfo loadJsonVfInfo(const std::filesystem::path& path);

} // namespace vfsim

#endif // VFSIM_API_NATIVE_JSON_VF_INFO_ADAPTER_H_
