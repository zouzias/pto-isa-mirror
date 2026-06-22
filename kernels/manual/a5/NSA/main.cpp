/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <acl/acl.h>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "test_common.h"
#include "runtime/rt.h"
#include "fa_performance_kernel.h"
#include "generated_cases.h"

using namespace std;
using namespace PtoTestCommon;

static thread_local std::string g_case_name;
static int g_chip_id = 0;

static std::string GetGoldenDir()
{
    return "./" + g_case_name;
}

template <typename T, int S0, int HEAD_SIZE, int S1, int CUBE_S0, int CUBE_S1, int TILE_S1, int QK_PRELOAD,
          bool CAUSAL_MASK>
bool run_branch(const char *prefix, aclrtStream stream, uint64_t ffts, aclFloat16 *qDevice, std::vector<float> &o_out)
{
    constexpr int tile_factor = TILE_S1 / CUBE_S1;
    constexpr size_t qk_fifo_stride = static_cast<size_t>(kFaCvFifoSize) * static_cast<size_t>(CUBE_S0) *
                                      static_cast<size_t>(tile_factor) * static_cast<size_t>(CUBE_S1);
    constexpr size_t p_fifo_stride = qk_fifo_stride;
    constexpr size_t p_max_fifo_stride = static_cast<size_t>(kFaCvFifoSize) * static_cast<size_t>(CUBE_S0);
    constexpr size_t pv_fifo_stride =
        static_cast<size_t>(kFaCvFifoSize) * static_cast<size_t>(CUBE_S0) * static_cast<size_t>(HEAD_SIZE);
    const size_t block_rows = S0 / CUBE_S0;
    constexpr size_t cv_comm_bytes =
        (static_cast<size_t>(S0) / static_cast<size_t>(CUBE_S0)) * kFaCvCommSlotBytes;
    constexpr size_t profile_bytes_per_block = kFaProfileBytesPerBlock;
    const size_t profile_bytes = profile_bytes_per_block * block_rows;

    size_t kSize = HEAD_SIZE * S1 * sizeof(aclFloat16);
    size_t vSize = S1 * HEAD_SIZE * sizeof(aclFloat16);
    const size_t qk_fifo_bytes = qk_fifo_stride * block_rows * sizeof(T);
    const size_t p_fifo_bytes_half = p_fifo_stride * block_rows * sizeof(aclFloat16);
    const size_t p_fifo_bytes_float = p_max_fifo_stride * block_rows * sizeof(float);
    const size_t pv_fifo_bytes = pv_fifo_stride * block_rows * sizeof(T);
    const size_t oSize = S0 * HEAD_SIZE * sizeof(T);
    const int num_tiles = S1 / TILE_S1;
    const size_t gsumSize = static_cast<size_t>(S0) * static_cast<size_t>(num_tiles) * sizeof(float);
    const size_t oPartsTotalSize = oSize * num_tiles;

    aclFloat16 *kHost = nullptr;
    aclFloat16 *vHost = nullptr;
    aclFloat16 *kDevice = nullptr;
    aclFloat16 *vDevice = nullptr;
    T *qkDevice = nullptr;
    aclFloat16 *pDevice = nullptr;
    float *expMaxIfifoDevice = nullptr;
    float *gSumDevice = nullptr;
    float *expMaxDevice = nullptr;
    T *oDevice = nullptr;
    T *oPartsDevice = nullptr;
    T *pvDevice = nullptr;
    uint8_t *profileDevice = nullptr;
    uint8_t *cvCommDevice = nullptr;

    aclrtMallocHost((void **)(&kHost), kSize);
    aclrtMallocHost((void **)(&vHost), vSize);
    ReadFile(GetGoldenDir() + "/" + std::string(prefix) + "_kt.bin", kSize, kHost, kSize);
    ReadFile(GetGoldenDir() + "/" + std::string(prefix) + "_v.bin", vSize, vHost, vSize);

    aclrtMalloc((void **)&kDevice, kSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&vDevice, vSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&qkDevice, qk_fifo_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&pDevice, p_fifo_bytes_half, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&expMaxIfifoDevice, p_fifo_bytes_float, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&gSumDevice, gsumSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&expMaxDevice, gsumSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&oDevice, oSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&oPartsDevice, oPartsTotalSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&pvDevice, pv_fifo_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&profileDevice, profile_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&cvCommDevice, cv_comm_bytes, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(kDevice, kSize, kHost, kSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(vDevice, vSize, vHost, vSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTFA<S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, kFaCvFifoSize, false, CAUSAL_MASK,
              kFaCvFifoConsSyncPeriod>(
        (uint16_t *)ffts, (aclFloat16 *)qDevice, (aclFloat16 *)kDevice, (aclFloat16 *)vDevice, (aclFloat16 *)pDevice,
        (float *)expMaxIfifoDevice, (float *)gSumDevice, (float *)expMaxDevice, (float *)oDevice, (float *)oPartsDevice,
        (float *)qkDevice, (float *)pvDevice, profileDevice, stream, cvCommDevice);

    aclrtSynchronizeStream(stream);

    o_out.resize(S0 * HEAD_SIZE);
    aclrtMemcpy(o_out.data(), oSize, oDevice, oSize, ACL_MEMCPY_DEVICE_TO_HOST);

    size_t goldenBranchSize = oSize;
    std::vector<float> golden_branch(S0 * HEAD_SIZE);
    ReadFile(GetGoldenDir() + "/o_" + std::string(prefix) + ".bin", goldenBranchSize, golden_branch.data(),
             goldenBranchSize);
    std::cout << "[CHECK] branch " << prefix << " output compare" << std::endl;
    const bool branch_ok = ResultCmp<float>(golden_branch, o_out, 0.002f);
    std::cout << "[SUMMARY] branch " << prefix << " status: " << (branch_ok ? "OK" : "FAIL") << std::endl;

    aclrtFree(kDevice);
    aclrtFree(vDevice);
    aclrtFree(qkDevice);
    aclrtFree(pDevice);
    aclrtFree(expMaxIfifoDevice);
    aclrtFree(gSumDevice);
    aclrtFree(expMaxDevice);
    aclrtFree(oDevice);
    aclrtFree(oPartsDevice);
    aclrtFree(pvDevice);
    aclrtFree(profileDevice);
    aclrtFree(cvCommDevice);
    aclrtFreeHost(kHost);
    aclrtFreeHost(vHost);
    return branch_ok;
}

template <typename T, int S0, int HEAD_SIZE, int S1_CMP, int S1_SLC, int S1_WIN, int CUBE_S0, int CUBE_S1,
          int TILE_S1, int QK_PRELOAD, bool CAUSAL_MASK>
void run_nsa()
{
    size_t qSize = S0 * HEAD_SIZE * sizeof(aclFloat16);
    size_t oSize = S0 * HEAD_SIZE * sizeof(float);
    size_t gateSize = S0 * 3 * sizeof(float);

    aclInit(nullptr);
    aclrtSetDevice(g_chip_id);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    aclFloat16 *qHost = nullptr;
    aclFloat16 *qDevice = nullptr;
    aclrtMallocHost((void **)(&qHost), qSize);
    aclrtMalloc((void **)&qDevice, qSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ReadFile(GetGoldenDir() + "/q.bin", qSize, qHost, qSize);
    aclrtMemcpy(qDevice, qSize, qHost, qSize, ACL_MEMCPY_HOST_TO_DEVICE);

    uint64_t ffts{0};
    uint32_t fftsLen{0};
    rtGetC2cCtrlAddr(&ffts, &fftsLen);

    std::vector<float> o_cmp, o_slc, o_win;
    const bool cmp_ok =
        run_branch<T, S0, HEAD_SIZE, S1_CMP, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, CAUSAL_MASK>(
            "cmp", stream, ffts, qDevice, o_cmp);
    const bool slc_ok = run_branch<T, S0, HEAD_SIZE, S1_SLC, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, false>(
        "slc", stream, ffts, qDevice, o_slc);
    const bool win_ok = run_branch<T, S0, HEAD_SIZE, S1_WIN, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, false>(
        "win", stream, ffts, qDevice, o_win);

    std::vector<float> gates(S0 * 3);
    ReadFile(GetGoldenDir() + "/gates.bin", gateSize, gates.data(), gateSize);

    std::vector<float> o_fused(S0 * HEAD_SIZE);
    for (int r = 0; r < S0; ++r) {
        const float g0 = gates[r * 3 + 0];
        const float g1 = gates[r * 3 + 1];
        const float g2 = gates[r * 3 + 2];
        for (int d = 0; d < HEAD_SIZE; ++d) {
            const size_t idx = static_cast<size_t>(r) * HEAD_SIZE + d;
            o_fused[idx] = g0 * o_cmp[idx] + g1 * o_slc[idx] + g2 * o_win[idx];
        }
    }

    WriteFile(GetGoldenDir() + "/o_out.bin", o_fused.data(), oSize);

    std::vector<float> golden_o(S0 * HEAD_SIZE);
    ReadFile(GetGoldenDir() + "/o.bin", oSize, golden_o.data(), oSize);
    std::cout << "[CHECK] NSA gated output compare" << std::endl;
    const bool ok = ResultCmp<float>(golden_o, o_fused, 0.002f);
    std::cout << (ok ? "test success" : "test failed") << std::endl;
    std::cout << "[SUMMARY] o_out status: " << (ok ? "OK" : "FAIL") << std::endl;

    aclrtFree(qDevice);
    aclrtFreeHost(qHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(g_chip_id);
    aclFinalize();

    if (!cmp_ok || !slc_ok || !win_ok || !ok) {
        std::exit(1);
    }
}

template <typename T, int S0, int HEAD_SIZE, int S1_CMP, int S1_SLC, int S1_WIN, int CUBE_S0, int CUBE_S1,
          int TILE_S1, int QK_PRELOAD, bool CAUSAL_MASK>
void run_nsa_case(const std::string &case_name)
{
    g_case_name = case_name;
    run_nsa<T, S0, HEAD_SIZE, S1_CMP, S1_SLC, S1_WIN, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, CAUSAL_MASK>();
}

int main(int argc, char **argv)
{
    struct CaseEntry {
        std::string name;
        std::function<void()> run;
    };

    std::vector<CaseEntry> cases = {
#define TNSA_CASE_ENTRY(S0, HEAD, S1, S1_CMP, S1_SLC, S1_WIN, BLOCK, NSEL, WIN, CUBE_S0, CUBE_S1, TILE_S1,          \
                      QK_PRELOAD, CAUSAL_MASK)                                                                        \
    {"case_float_H_" #HEAD "_S0_" #S0 "_S1_" #S1 "_B" #BLOCK "_N" #NSEL "_W" #WIN, []() {                            \
         run_nsa_case<float, S0, HEAD, S1_CMP, S1_SLC, S1_WIN, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, CAUSAL_MASK>( \
             "case_float_H_" #HEAD "_S0_" #S0 "_S1_" #S1 "_B" #BLOCK "_N" #NSEL "_W" #WIN);                          \
     }},
        TNSA_FOR_EACH_CASE(TNSA_CASE_ENTRY)
#undef TNSA_CASE_ENTRY
    };

    std::string filter_arg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--case=", 0) == 0 || arg.rfind("--cases=", 0) == 0) {
            filter_arg = arg.substr(arg.find('=') + 1);
        } else if ((arg == "--case" || arg == "--cases") && i + 1 < argc) {
            filter_arg = argv[++i];
        } else if (arg.rfind("--npu=", 0) == 0) {
            g_chip_id = std::stoi(arg.substr(6));
        }
    }

    auto should_run = [&](const std::string &name) {
        if (filter_arg.empty())
            return true;
        return name.find(filter_arg) != std::string::npos || name == filter_arg;
    };

    int ran = 0;
    for (const auto &c : cases) {
        if (should_run(c.name)) {
            std::cout << "[DEBUG] Running " << c.name << std::endl;
            c.run();
            ++ran;
        }
    }
    if (ran == 0) {
        std::cerr << "[WARN] No cases matched" << std::endl;
        return 1;
    }
    return 0;
}
