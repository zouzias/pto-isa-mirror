#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
tmp_dir=$(mktemp -d)
trap 'rm -rf "${tmp_dir}"' EXIT

cxx=${CXX:-g++}
cxxflags=(-std=c++17 -Wall -Wextra -Werror -I"${repo_root}/include" -I"${repo_root}/pkg_inc")

cat >"${tmp_dir}/workspace_contract_no_engines.cpp" <<'CPP'
#include <cassert>
#include <cstdint>

#define PTO_COMM_WORKSPACE_SDMA_SUPPORTED 0
#define PTO_COMM_WORKSPACE_URMA_SUPPORTED 0
#define PTO_COMM_WORKSPACE_RDMA_SUPPORTED 0
#include "pto/comm/workspace.hpp"

int main()
{
    using namespace pto::comm;

    static_assert(static_cast<int32_t>(WorkspaceStatus::Ok) == 0, "Ok must stay ABI-stable at zero");

    WorkspaceRequest req{};
    Workspace ws{};
    assert(CreateWorkspace(DmaEngine::SDMA, req, nullptr) == WorkspaceStatus::InvalidArgument);
    assert(CreateWorkspace(DmaEngine::SDMA, req, &ws) == WorkspaceStatus::Unsupported);
    assert(ws.addr == nullptr && ws.bytes == 0 && ws.impl == nullptr);

    Workspace occupied{};
    occupied.addr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x1));
    assert(CreateWorkspace(DmaEngine::SDMA, req, &occupied) == WorkspaceStatus::InvalidArgument);

    Workspace abandoned{};
    abandoned.addr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x2));
    abandoned.bytes = 123;
    abandoned.impl = reinterpret_cast<void*>(static_cast<uintptr_t>(0x3));
    abandoned.engine = DmaEngine::RDMA;
    AbandonWorkspace(&abandoned);
    assert(abandoned.addr == nullptr && abandoned.bytes == 0 && abandoned.impl == nullptr);

    DestroyWorkspace(nullptr);
    DestroyWorkspace(&abandoned);
    assert(abandoned.addr == nullptr && abandoned.bytes == 0 && abandoned.impl == nullptr);
    return 0;
}
CPP

"${cxx}" "${cxxflags[@]}" "${tmp_dir}/workspace_contract_no_engines.cpp" -o "${tmp_dir}/workspace_contract_no_engines"
"${tmp_dir}/workspace_contract_no_engines"

cat >"${tmp_dir}/workspace_sdma_size.cpp" <<'CPP'
#include <cstdint>

#include "pto/comm/async_common/sdma_constants.hpp"

int main()
{
    using namespace pto::comm::sdma;
    static_assert(kSdmaWorkspaceBytes == kSdmaContextWorkspaceBytes +
                                             kSdmaMaxChannelGroups * kSdmaFlagPayloadBytesPerGroup +
                                             kSdmaSignalValueSlotsBytes,
                  "SDMA workspace size must include the STARS context, every group payload, and the signal slots");
    static_assert(kSdmaWorkspaceBytes == 53248U, "SDMA workspace size changed; update callers and tests together");
    return 0;
}
CPP

"${cxx}" "${cxxflags[@]}" "${tmp_dir}/workspace_sdma_size.cpp" -o "${tmp_dir}/workspace_sdma_size"
"${tmp_dir}/workspace_sdma_size"

if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    cann_flags=(
        -I"${ASCEND_HOME_PATH}/include"
        -I"${ASCEND_HOME_PATH}/include/hccl"
        -I"${ASCEND_HOME_PATH}/pkg_inc"
        -I"${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime"
    )

    cat >"${tmp_dir}/workspace_default_header.cpp" <<'CPP'
#include "pto/comm/workspace.hpp"

int main()
{
    pto::comm::Workspace ws{};
    pto::comm::AbandonWorkspace(&ws);
    return ws.addr == nullptr ? 0 : 1;
}
CPP
    "${cxx}" "${cxxflags[@]}" "${cann_flags[@]}" -c "${tmp_dir}/workspace_default_header.cpp" \
        -o "${tmp_dir}/workspace_default_header.o"

    cat >"${tmp_dir}/workspace_urma_size.cpp" <<'CPP'
#include <cstdint>
#include <type_traits>
#include <utility>

#include "pto/comm/async/urma/urma_workspace_manager.hpp"

int main()
{
    using namespace pto::comm::urma;

    // The manager derives its size from the resolved layout (per-peer vs shared pool), so the
    // formula is no longer a compile-time constant a caller can restate here. Pin the contract
    // callers actually use instead: the accessor exists, is const, and yields uint64_t.
    static_assert(
        std::is_same_v<decltype(std::declval<const UrmaWorkspaceManager&>().GetWorkspaceSize()), uint64_t>,
        "UrmaWorkspaceManager must report its workspace size as uint64_t");
    return 0;
}
CPP
    "${cxx}" "${cxxflags[@]}" "${cann_flags[@]}" "${tmp_dir}/workspace_urma_size.cpp" \
        -o "${tmp_dir}/workspace_urma_size"
    "${tmp_dir}/workspace_urma_size"
else
    echo "[SKIP] ASCEND_HOME_PATH is not set; skipped default workspace.hpp and URMA size compile checks."
fi

echo "comm workspace header checks passed"
