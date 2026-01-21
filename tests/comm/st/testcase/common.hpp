// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <iostream>

#if defined(ASCEND_SHMEM)
#include "shmem_api.h"
#include "host/shmem_host_sync.h"
#elif defined(CANN_SHMEM)
#include "shmem.h"
#endif
#include "pto/comm/context_manager.hpp"


struct ShmemEnv {
    int rank {0};
    int size {1};
    const char *ipPort {nullptr};
    uint64_t heapBytes {8ULL * 1024 * 1024};
};

inline bool LoadEnv(ShmemEnv &env)
{
    const char *rankEnv = std::getenv("SHMEM_RANK");
    const char *sizeEnv = std::getenv("SHMEM_SIZE");
    const char *ipEnv   = std::getenv("SHMEM_IP_PORT");
    const char *heapEnv = std::getenv("SHMEM_HEAP_BYTES");

    if (rankEnv == nullptr || sizeEnv == nullptr || ipEnv == nullptr) {
        std::cerr << "[ERROR] SHMEM_RANK/SHMEM_SIZE/SHMEM_IP_PORT must be set.\n";
        return false;
    }
    env.rank = std::atoi(rankEnv);
    env.size = std::atoi(sizeEnv);
    env.ipPort = ipEnv;
    if (heapEnv != nullptr) {
        env.heapBytes = std::strtoull(heapEnv, nullptr, 10);
    }
    return true;
}

inline bool ShmemInitFromEnv(ShmemEnv &env)
{
    pto::comm::InitOptions opts;
    opts.backend = pto::comm::BackendKind::Shmem;
    opts.rank = env.rank;
    opts.size = env.size;
    opts.symmetricHeapBytes = env.heapBytes;
    opts.ipPort = env.ipPort;
    const int ret = pto::comm::ContextManager::Init(opts);
    return (ret == 0);
}

inline void ShmemFinalize()
{
    pto::comm::ContextManager::Finalize();
}

