// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <cstring>

// Shmem API headers for symmetric heap memory initialization
#if defined(ASCEND_SHMEM)
#include "shmem_api.h"
#elif defined(CANN_SHMEM)
#include "shmem.h"
#endif

// ============================================================================
// ShmemEnv: Environment configuration for shmem initialization
// ============================================================================
struct ShmemEnv {
    int rank {0};
    int size {1};
    const char *ipPort {nullptr};
    uint64_t heapBytes {8ULL * 1024 * 1024};  // Default 8MB symmetric heap
};

// ============================================================================
// LoadEnv: Load shmem environment from environment variables
// ============================================================================
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

// ============================================================================
// ShmemInit: Initialize shmem symmetric heap with given options
// Adapted from ShmemBackend::Init in shmem_backend.hpp
// ============================================================================
inline int ShmemInit(const ShmemEnv &env)
{
#if defined(ASCEND_SHMEM)
    shmem_init_attr_t *attr = nullptr;
    const int ret = shmem_set_attr(env.rank, env.size, env.heapBytes, env.ipPort, &attr);
    if (ret != 0) {
        std::cerr << "[ERROR] shmem_set_attr failed with code: " << ret << std::endl;
        return ret;
    }
    int initRet = shmem_init_attr(attr);
    if (initRet != 0) {
        std::cerr << "[ERROR] shmem_init_attr failed with code: " << initRet << std::endl;
    }
    return initRet;
#elif defined(CANN_SHMEM)
    aclshmemx_init_attr_t attributes;
    
    attributes.my_pe = env.rank;
    attributes.n_pes = env.size;
    attributes.local_mem_size = env.heapBytes;
    
    // Copy IP:port string
    size_t ipLen = 0;
    if (env.ipPort != nullptr) {
        for (; ipLen < ACLSHMEM_MAX_IP_PORT_LEN - 1 && env.ipPort[ipLen] != '\0'; ++ipLen) {
            attributes.ip_port[ipLen] = env.ipPort[ipLen];
        }
    }
    attributes.ip_port[ipLen] = '\0';
    
    // Set option attributes
    constexpr int attrVersion = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    constexpr int DEFAULT_TIMEOUT = 120;  // seconds
    attributes.option_attr = {attrVersion, ACLSHMEM_DATA_OP_SDMA, 
                              DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, -1};
    
    // Use default unique ID
    aclshmemx_uniqueid_t defaultUid = ACLSHMEM_UNIQUEID_INITIALIZER;
    attributes.comm_args = reinterpret_cast<void *>(&defaultUid);
    
    int initRet = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    if (initRet != 0) {
        std::cerr << "[ERROR] aclshmemx_init_attr failed with code: " << initRet << std::endl;
    }
    return initRet;
#else
    std::cerr << "[ERROR] No shmem backend defined (ASCEND_SHMEM or CANN_SHMEM)" << std::endl;
    return -1;
#endif
}

// ============================================================================
// ShmemInitFromEnv: Initialize shmem from ShmemEnv structure
// ============================================================================
inline bool ShmemInitFromEnv(ShmemEnv &env)
{
    const int ret = ShmemInit(env);
    return (ret == 0);
}

// ============================================================================
// ShmemFinalize: Finalize shmem
// ============================================================================
inline void ShmemFinalize()
{
#if defined(ASCEND_SHMEM) || defined(CANN_SHMEM)
    shmem_finalize();
#endif
}

// ============================================================================
// ShmemMalloc: Allocate symmetric heap memory
// ============================================================================
inline void* ShmemMalloc(size_t bytes)
{
#if defined(ASCEND_SHMEM) || defined(CANN_SHMEM)
    return shmem_malloc(bytes);
#else
    return nullptr;
#endif
}

// ============================================================================
// ShmemFree: Free symmetric heap memory
// ============================================================================
inline void ShmemFree(void *ptr)
{
#if defined(ASCEND_SHMEM) || defined(CANN_SHMEM)
    shmem_free(ptr);
#endif
}

// ============================================================================
// ShmemBarrierAll: Global barrier synchronization
// ============================================================================
inline void ShmemBarrierAll()
{
#if defined(ASCEND_SHMEM)
    shmem_barrier_all();
#elif defined(CANN_SHMEM)
    aclshmem_barrier_all();
#endif
}

// ============================================================================
// ShmemQuiet: Ensure all pending operations complete
// Note: aclshmem_quiet() is device-only in CANN_SHMEM, so we use barrier instead
// ============================================================================
inline void ShmemQuiet()
{
#if defined(ASCEND_SHMEM)
    shmem_quiet();
#elif defined(CANN_SHMEM)
    // aclshmem_quiet() is device-only, use barrier for host synchronization
    aclshmem_barrier_all();
#endif
}

// ============================================================================
// ShmemMyPe: Get current rank ID
// ============================================================================
inline int ShmemMyPe()
{
#if defined(ASCEND_SHMEM) || defined(CANN_SHMEM)
    return shmem_my_pe();
#else
    return 0;
#endif
}

// ============================================================================
// ShmemNPes: Get total number of ranks
// ============================================================================
inline int ShmemNPes()
{
#if defined(ASCEND_SHMEM) || defined(CANN_SHMEM)
    return shmem_n_pes();
#else
    return 1;
#endif
}

// ============================================================================
// ShmemSetConfStoreTls: Configure TLS for shmem operations
// Note: CANN_SHMEM API changed - now takes (bool, const char*, uint32_t)
// ============================================================================
inline int ShmemSetConfStoreTls(bool enable, const char *tlsInfo, uint32_t tlsInfoLen)
{
#if defined(ASCEND_SHMEM)
    return shmem_set_conf_store_tls(enable, tlsInfo, tlsInfoLen);
#elif defined(CANN_SHMEM)
    return shmem_set_conf_store_tls(enable, tlsInfo, tlsInfoLen);
#else
    (void)enable;
    (void)tlsInfo;
    (void)tlsInfoLen;
    return 0;
#endif
}

// ============================================================================
// ShmemPtr: Get remote PE's address mapping for symmetric memory (Device)
//
// Converts a local symmetric heap address to the actual remote address that
// can be used to access memory on the specified PE. This is essential for
// remote memory operations (PUT/GET).
//
// Parameters:
//   - localPtr: Local symmetric heap address
//   - pe: Target PE (rank) number
//
// Returns: Mapped address for accessing memory on remote PE
// ============================================================================
template <typename T>
AICORE inline __gm__ T* ShmemPtr(__gm__ T *localPtr, int pe)
{
#if defined(ASCEND_SHMEM)
    return (__gm__ T *)shmem_ptr(localPtr, pe);
#elif defined(CANN_SHMEM)
    return (__gm__ T *)aclshmem_ptr(localPtr, pe);
#else
    (void)pe;
    return localPtr;
#endif
}

// ============================================================================
// ShmemDeviceBarrierAll: Global barrier synchronization (Device)
//
// Synchronizes all PEs in the device kernel. All PEs must call this function,
// and no PE will proceed until all have arrived.
// ============================================================================
AICORE inline void ShmemDeviceBarrierAll()
{
#if defined(ASCEND_SHMEM)
    shmem_barrier_all();
#elif defined(CANN_SHMEM)
    aclshmem_barrier_all();
#endif
}

// ============================================================================
// ShmemDeviceQuiet: Ensure all pending remote operations complete (Device)
//
// Ensures that all previously issued remote memory operations (PUT, GET, etc.)
// from this PE have completed before proceeding.
// ============================================================================
AICORE inline void ShmemDeviceQuiet()
{
#if defined(ASCEND_SHMEM)
    shmem_quiet();
#elif defined(CANN_SHMEM)
    aclshmem_quiet();
#endif
}
