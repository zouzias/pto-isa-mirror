/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_RDMA_BACKEND_CONFIG_HPP
#define PTO_COMM_ASYNC_RDMA_BACKEND_CONFIG_HPP

#if defined(__CCE_KT_TEST__)
#error "rdma_backend_config.hpp is a host-only header and cannot be included in device code."
#endif

#include "pto/comm/rdma_backend.hpp"

namespace pto {
namespace comm {
namespace rdma {

constexpr const char* kHns1825BackendName = "HNS_1825";
constexpr const char* kNoRdmaBackendName = "NONE";

struct BackendSelection {
    RdmaBackend backend{RdmaBackend::NONE};
    const char* name{kNoRdmaBackendName};
};

// PTO_RDMA_BACKEND is resolved by the build system. Runtime code only observes
// the resulting compile definition, so every rank in one binary has the same
// backend and no environment lookup occurs here.
inline const BackendSelection& GetBackendSelection()
{
#ifdef PTO_RDMA_BACKEND_HNS_1825_SUPPORTED
    static constexpr BackendSelection selection{RdmaBackend::HNS_1825, kHns1825BackendName};
#else
    static constexpr BackendSelection selection{};
#endif
    return selection;
}

} // namespace rdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_RDMA_BACKEND_CONFIG_HPP
