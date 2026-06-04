/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "moe_token_permute_body.hpp"
#include "acl/acl.h"
#include "runtime/rt.h"

#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <vector>

using namespace moe_token_permute;

#if defined(SYNCALL_MIX_BUILD_AIC)
PTO_SYNCALL_MIX_AIC_KERNEL_META(RunMoeTokenPermute_2101_mix_aic, 1, 2);

extern "C" __global__ AICORE void RunMoeTokenPermute_2101_mix_aic(__gm__ uint64_t __in__ *fftsAddrGm,
                                                                  __gm__ half __in__ *tokensGm,
                                                                  __gm__ int32_t __in__ *indicesGm,
                                                                  __gm__ half __out__ *permOutGm,
                                                                  __gm__ int32_t __out__ *sioOutGm,
                                                                  __gm__ int32_t __out__ *workspaceGm)
{
    (void)tokensGm;
    (void)indicesGm;
    (void)permOutGm;
    (void)sioOutGm;
    (void)workspaceGm;
    RunMoeTokenPermuteAicOnly(fftsAddrGm);
}
#endif

#if defined(SYNCALL_MIX_BUILD_AIV)
PTO_SYNCALL_MIX_AIC_KERNEL_META(RunMoeTokenPermute_2101_mix_aiv, 1, 2);

extern "C" __global__ AICORE void RunMoeTokenPermute_2101_mix_aiv(__gm__ uint64_t __in__ *fftsAddrGm,
                                                                  __gm__ half __in__ *tokensGm,
                                                                  __gm__ int32_t __in__ *indicesGm,
                                                                  __gm__ half __out__ *permOutGm,
                                                                  __gm__ int32_t __out__ *sioOutGm,
                                                                  __gm__ int32_t __out__ *workspaceGm)
{
    RunMoeTokenPermuteAivBody(tokensGm, indicesGm, permOutGm, sioOutGm, workspaceGm,
                             reinterpret_cast<uint64_t>(fftsAddrGm));
}
#endif

#if defined(SYNCALL_MIX_BUILD_AIC) && !defined(SYNCALL_MIX_REGISTER_BUILD)
namespace {
const char *GetCurrentSharedObjectPath(const void *anchor)
{
#if defined(SYNCALL_MIX_REGISTER_OBJECT_PATH)
    (void)anchor;
    return SYNCALL_MIX_REGISTER_OBJECT_PATH;
#else
    Dl_info info{};
    if (dladdr(anchor, &info) == 0 || info.dli_fname == nullptr) {
        std::fprintf(stderr, "dladdr failed for moe_token_permute register kernel\n");
        std::abort();
    }
    return info.dli_fname;
#endif
}

std::vector<char> ReadCurrentSharedObject(const char *path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::fprintf(stderr, "failed to open moe_token_permute register kernel binary: %s\n", path);
        std::abort();
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        std::fprintf(stderr, "invalid moe_token_permute register kernel binary size: %s\n", path);
        std::abort();
    }
    std::vector<char> data(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(data.data(), size)) {
        std::fprintf(stderr, "failed to read moe_token_permute register kernel binary: %s\n", path);
        std::abort();
    }
    return data;
}

void LaunchRegisterMixKernel(const void *anchor, uint64_t tilingKey, uint8_t *ffts, aclFloat16 *tokens, int32_t *indices,
                             aclFloat16 *permOut, int32_t *sioOut, int32_t *workspace, void *stream)
{
    const char *path = GetCurrentSharedObjectPath(anchor);
    static const std::vector<char> kernelBinary = ReadCurrentSharedObject(path);
    rtDevBinary_t binary{RT_DEV_BINARY_MAGIC_ELF, 0, kernelBinary.data(), kernelBinary.size()};
    void *handle = nullptr;
    rtError_t ret = rtRegisterAllKernel(&binary, &handle);
    if (ret != RT_ERROR_NONE || handle == nullptr) {
        ret = rtBinaryLoadWithoutTilingKey(kernelBinary.data(), kernelBinary.size(), &handle);
        if (ret != RT_ERROR_NONE || handle == nullptr) {
            std::fprintf(stderr, "register moe_token_permute mix kernel failed, path=%s, size=%zu, ret=%d\n", path,
                         kernelBinary.size(), ret);
            std::abort();
        }
    }

    void *args[] = {reinterpret_cast<uint64_t *>(ffts), reinterpret_cast<half *>(tokens),
                    indices, reinterpret_cast<half *>(permOut), sioOut, workspace};
    rtArgsEx_t argsInfo{};
    argsInfo.args = args;
    argsInfo.argsSize = sizeof(args);
    rtTaskCfgInfo_t cfgInfo{};
    ret = rtKernelLaunchWithHandleV2(handle, tilingKey, kMoeNumCores, &argsInfo, nullptr, stream, &cfgInfo);
    if (ret != RT_ERROR_NONE) {
        std::fprintf(stderr, "rtKernelLaunchWithHandleV2 failed for moe_token_permute, ret=%d\n", ret);
        std::abort();
    }
}
} // namespace

void LaunchMoeTokenPermuteRegister(aclFloat16 *tokensGm, int32_t *indicesGm, aclFloat16 *permOutGm, int32_t *sioOutGm,
                                   int32_t *workspaceGm, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    LaunchRegisterMixKernel(reinterpret_cast<const void *>(&LaunchMoeTokenPermuteRegister), kMoeRegisterTilingKey,
                            reinterpret_cast<uint8_t *>(fftsAddr), tokensGm, indicesGm, permOutGm, sioOutGm,
                            workspaceGm, stream);
}
#endif
