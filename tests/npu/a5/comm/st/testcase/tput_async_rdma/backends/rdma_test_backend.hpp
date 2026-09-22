/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_TESTS_NPU_A5_COMM_ST_RDMA_TEST_BACKEND_HPP
#define PTO_TESTS_NPU_A5_COMM_ST_RDMA_TEST_BACKEND_HPP

#include <cstdint>
#include <cstdlib>
#include <acl/acl.h>

#ifdef PTO_RDMA_BACKEND_HNS_1825_SUPPORTED
#include "hns_1825/hns_1825_bootstrap.hpp"
#include "pto/comm/async/rdma/backends/hns_1825/hns_1825_types.hpp"
#else
#error "The RDMA ST has no adapter for the configured backend."
#endif

namespace pto {
namespace comm {
namespace rdma {
namespace test {

#ifdef PTO_RDMA_BACKEND_HNS_1825_SUPPORTED
using BackendBootstrap = hns_1825::bootstrap::BootstrapConfig;

struct BackendQueueSnapshot {
    uint32_t sqDepth{0};
    uint32_t cqDepth{0};
    hns_1825::QueueState state{};
};

// Read only while the test stream is idle; the driver owns both ring depths.
inline bool ReadBackendQueueSnapshot(const void* workspace, uint32_t peer, BackendQueueSnapshot& snapshot)
{
    RdmaInfo info{};
    if (aclrtMemcpy(&info, sizeof(info), workspace, sizeof(info), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        peer >= info.rankCount || info.qpNum != 1 || info.sqPtr == 0 || info.scqPtr == 0) {
        return false;
    }
    hns_1825::SqContext sq{};
    hns_1825::CqContext cq{};
    if (aclrtMemcpy(
            &sq, sizeof(sq), reinterpret_cast<const void*>(info.sqPtr + peer * sizeof(sq)), sizeof(sq),
            ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        aclrtMemcpy(
            &cq, sizeof(cq), reinterpret_cast<const void*>(info.scqPtr + peer * sizeof(cq)), sizeof(cq),
            ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        sq.stateAddr == 0) {
        return false;
    }
    snapshot.sqDepth = sq.depth;
    snapshot.cqDepth = cq.depth;
    return aclrtMemcpy(
               &snapshot.state, sizeof(snapshot.state), reinterpret_cast<const void*>(sq.stateAddr),
               sizeof(snapshot.state), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
}

inline bool BackendVerboseEnabled()
{
    const char* verbose = std::getenv("PTO_ROCE_VERBOSE");
    return verbose != nullptr && verbose[0] == '1';
}

inline const char* DescribeBackendCompletionStatus(uint32_t status)
{
    switch (status) {
        case hns_1825::kHns1825PollCqTimeoutError:
            return "queue_progress_timeout";
        case hns_1825::kHns1825InvalidArgumentError:
            return "invalid_transfer";
        case hns_1825::kHns1825InvalidContextError:
            return "invalid_context";
        case hns_1825::kHns1825CqeError:
            return "cqe_error_without_syndrome";
        case hns_1825::kQueueIndexExhaustedError:
            return "queue_index_exhausted";
        default:
            return nullptr;
    }
}
#endif

} // namespace test
} // namespace rdma
} // namespace comm
} // namespace pto

#endif // PTO_TESTS_NPU_A5_COMM_ST_RDMA_TEST_BACKEND_HPP
