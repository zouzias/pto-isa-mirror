/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_COMBINE_TILE_COMM_MPI_H_
#define DISPATCH_COMBINE_TILE_COMM_MPI_H_

#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <cstddef>
#include <cstdint>

namespace dispatch_combine_tile {

struct MpiContext {
    int rank = 0;
    int size = 1;
    bool initialized = false;
};

using DispatchTileMpiComm = int;
using DispatchTileMpiDatatype = int;

static constexpr DispatchTileMpiComm kDispatchTileMpiCommWorld = static_cast<DispatchTileMpiComm>(0x44000000);
static constexpr DispatchTileMpiDatatype kDispatchTileMpiChar = static_cast<DispatchTileMpiDatatype>(0x4c000101);

using MpiInitFunc = int (*)(int *, char ***);
using MpiCommSizeFunc = int (*)(DispatchTileMpiComm, int *);
using MpiCommRankFunc = int (*)(DispatchTileMpiComm, int *);
using MpiBcastFunc = int (*)(void *, int, DispatchTileMpiDatatype, int, DispatchTileMpiComm);
using MpiBarrierFunc = int (*)(DispatchTileMpiComm);
using MpiFinalizeFunc = int (*)();

inline void *&MpiLibraryHandle()
{
    static void *handle = nullptr;
    return handle;
}

inline bool LooksLikeMpiLibrary(const char *path)
{
    const char *slash = std::strrchr(path, '/');
    const char *base = slash == nullptr ? path : slash + 1;
    return (std::strncmp(base, "libmpi", 6) == 0 || std::strncmp(base, "libmpich", 8) == 0) &&
           std::strstr(base, ".so") != nullptr;
}

inline void *LoadMpiLibrary()
{
    void *&handle = MpiLibraryHandle();
    if (handle != nullptr) {
        return handle;
    }

    const char *envPath = std::getenv("MPI_LIB_PATH");
    if (envPath != nullptr && envPath[0] != '\0') {
        if (envPath[0] != '/') {
            std::cerr << "[MPI] Ignoring MPI_LIB_PATH because it is not absolute: " << envPath << "\n";
        } else if (!LooksLikeMpiLibrary(envPath)) {
            std::cerr << "[MPI] Ignoring MPI_LIB_PATH because it does not look like libmpi*.so*: " << envPath << "\n";
        } else {
            handle = dlopen(envPath, RTLD_NOW);
            if (handle != nullptr) {
                std::cerr << "[MPI] Loaded from MPI_LIB_PATH: " << envPath << "\n";
                return handle;
            }
            std::cerr << "[MPI] dlopen failed for MPI_LIB_PATH=" << envPath << ": " << dlerror() << "\n";
        }
    }

    static const char *candidates[] = {"libmpi.so",
                                       "libmpich.so",
                                       "/usr/local/mpich/lib/libmpi.so",
                                       "/lib/aarch64-linux-gnu/libmpich.so",
                                       "/lib/x86_64-linux-gnu/libmpich.so",
                                       "/usr/lib/libmpi.so",
                                       "/usr/lib/libmpich.so",
                                       nullptr};
    for (uint32_t i = 0; candidates[i] != nullptr; ++i) {
        handle = dlopen(candidates[i], RTLD_NOW);
        if (handle != nullptr) {
            std::cerr << "[MPI] Loaded: " << candidates[i] << "\n";
            return handle;
        }
    }

    return nullptr;
}

template <typename Func>
inline Func GetMpiFunc(const char *name)
{
    void *handle = LoadMpiLibrary();
    if (handle == nullptr) {
        throw std::runtime_error("Cannot find MPI library. Set MPI_LIB_PATH to the path of libmpi.so");
    }
    void *func = dlsym(handle, name);
    if (func == nullptr) {
        throw std::runtime_error(std::string("Cannot find MPI symbol: ") + name);
    }
    return reinterpret_cast<Func>(func);
}

// Header-only adaptation of kernels/manual/a2a3/gemm_ar/comm_mpi.h.
inline MpiContext InitMpiAndRank(int *argc, char ***argv)
{
    auto init = GetMpiFunc<MpiInitFunc>("MPI_Init");
    int ret = init(argc, argv);
    if (ret != 0) {
        throw std::runtime_error("MPI_Init failed: " + std::to_string(ret));
    }

    MpiContext context;
    context.initialized = true;
    auto commRank = GetMpiFunc<MpiCommRankFunc>("MPI_Comm_rank");
    auto commSize = GetMpiFunc<MpiCommSizeFunc>("MPI_Comm_size");
    ret = commRank(kDispatchTileMpiCommWorld, &context.rank);
    if (ret != 0) {
        throw std::runtime_error("MPI_Comm_rank failed: " + std::to_string(ret));
    }
    ret = commSize(kDispatchTileMpiCommWorld, &context.size);
    if (ret != 0) {
        throw std::runtime_error("MPI_Comm_size failed: " + std::to_string(ret));
    }
    return context;
}

inline void MpiBroadcast(MpiContext *context, void *data, size_t bytes, int root)
{
    if (context == nullptr || !context->initialized) {
        return;
    }
    if (bytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("MPI broadcast payload is too large");
    }
    auto bcast = GetMpiFunc<MpiBcastFunc>("MPI_Bcast");
    int ret = bcast(data, static_cast<int>(bytes), kDispatchTileMpiChar, root, kDispatchTileMpiCommWorld);
    if (ret != 0) {
        throw std::runtime_error("MPI_Bcast failed: " + std::to_string(ret));
    }
}

inline void MpiBarrier(MpiContext *context)
{
    if (context == nullptr || !context->initialized) {
        return;
    }
    auto barrier = GetMpiFunc<MpiBarrierFunc>("MPI_Barrier");
    int ret = barrier(kDispatchTileMpiCommWorld);
    if (ret != 0) {
        throw std::runtime_error("MPI_Barrier failed: " + std::to_string(ret));
    }
}

inline void FinalizeMpi(MpiContext *context)
{
    if (context == nullptr || !context->initialized) {
        return;
    }
    auto finalize = GetMpiFunc<MpiFinalizeFunc>("MPI_Finalize");
    (void)finalize();
    context->initialized = false;
}

} // namespace dispatch_combine_tile

#endif // DISPATCH_COMBINE_TILE_COMM_MPI_H_
