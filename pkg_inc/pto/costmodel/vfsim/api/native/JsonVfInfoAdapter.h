// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: CANN-1.0

#ifndef VFSIM_API_NATIVE_JSON_VF_INFO_ADAPTER_H_
#define VFSIM_API_NATIVE_JSON_VF_INFO_ADAPTER_H_

#include <filesystem>

#include "api/native/VfInfo.h"

namespace vfsim {

VfInfo loadJsonVfInfo(const std::filesystem::path& path);

} // namespace vfsim

#endif // VFSIM_API_NATIVE_JSON_VF_INFO_ADAPTER_H_
