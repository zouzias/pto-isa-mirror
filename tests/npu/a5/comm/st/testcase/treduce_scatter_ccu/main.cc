/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// CCU-based TReduceScatter end-to-end ST (GTest).
//
// ReduceScatter semantics: each rank has N elements of input. After the op,
// rank r holds the reduced slice [r*sliceSize .. (r+1)*sliceSize), where
// sliceSize = N / nranks. Each element in rank r's output is the sum of
// all ranks' inputs at that position within the slice.
//
// Run:  mpirun -n 2 ./treduce_scatter_ccu
//       mpirun -n 4 ./treduce_scatter_ccu

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <dlfcn.h>
#include <unistd.h>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_rank_graph.h"
#include "hcomm/ccu/hccl_ccu_res.h"
#include "hcomm/ccu/ccu_assist_pub.h"

#include "pto/npu/comm/async/ccu/ccu_types.hpp"
#include "pto/npu/comm/async/ccu/ccu_gate_registry.hpp"
#include "pto/npu/comm/async/ccu/ccu_reduce_scatter_kernel.hpp"

#include <gtest/gtest.h>
#include "../comm_mpi.h"
#include "../ccu_test_main.hpp"

extern "C" int32_t rtSetDevice(int32_t deviceId);

extern "C" int treduce_scatter_ccu_trigger_launch(void *stream, uint64_t ckeVA, uint32_t mask);

extern "C" int treduce_scatter_ccu_fused_launch(void *stream, uint64_t inputVa, uint64_t outputVa, uint32_t selfIdx,
                                                 int nranks, float fillValue, uint64_t ckeVA, uint32_t mask);

namespace {

static constexpr size_t kMaxElements = 1024;
static constexpr size_t kMaxPayload = kMaxElements * sizeof(float);

#define ACL_OK(expr)                                                                                                  \
    do {                                                                                                              \
        aclError _r = (expr);                                                                                         \
        if (_r != ACL_SUCCESS) {                                                                                      \
            std::fprintf(stderr, "[TRS_CCU] ACL FAIL %s = %d (%s:%d)\n", #expr, static_cast<int>(_r), __FILE__,       \
                         __LINE__);                                                                                    \
            return false;                                                                                              \
        }                                                                                                             \
    } while (0)

#define HCCL_OK(expr)                                                                                                  \
    do {                                                                                                               \
        HcclResult _r = (expr);                                                                                        \
        if (_r != HCCL_SUCCESS) {                                                                                      \
            std::fprintf(stderr, "[TRS_CCU] HCCL FAIL %s = %d (%s:%d)\n", #expr, static_cast<int>(_r), __FILE__,       \
                         __LINE__);                                                                                     \
            return false;                                                                                               \
        }                                                                                                              \
    } while (0)

struct CcuEnv {
    HcclComm comm = nullptr;
    ThreadHandle threadHandle = 0;
    aclrtStream stream = nullptr;
    aclrtStream aivStream = nullptr;
    std::vector<ChannelHandle> channels;
    int rankId = -1;
    int nRanks = 0;
    int devId = -1;

    void *inputDev = nullptr;
    void *outputDev = nullptr;
    uint64_t inputVa = 0;
    uint64_t outputVa = 0;
    uint64_t token = 0;

    uint64_t mmioAddr = 0;
    uint32_t gateMask = 0;

    bool ready = false;
    bool gateResolved = false;
};

static CcuEnv g_env;
static uint64_t g_seqNo = 0;

bool SetupChannelsForCcu(HcclComm comm, int rankId, int nRanks, std::vector<ChannelHandle> &channels)
{
    std::vector<HcclChannelDesc> requests;
    for (int peer = 0; peer < nRanks; ++peer) {
        if (peer == rankId)
            continue;
        uint32_t netLayer = 0, listSize = 0;
        CommLink *linkList = nullptr;
        HcclResult rc = HcclRankGraphGetLinks(comm, netLayer, static_cast<uint32_t>(rankId),
                                              static_cast<uint32_t>(peer), &linkList, &listSize);
        if (rc != HCCL_SUCCESS)
            return false;

        bool found = false;
        for (uint32_t i = 0; i < listSize; ++i) {
            if (linkList[i].linkAttr.linkProtocol != COMM_PROTOCOL_UBC_CTP)
                continue;
            HcclChannelDesc desc;
            HcclChannelDescInit(&desc, 1);
            desc.remoteRank = static_cast<uint32_t>(peer);
            desc.notifyNum = 4;
            desc.channelProtocol = linkList[i].linkAttr.linkProtocol;
            desc.localEndpoint = linkList[i].srcEndpointDesc;
            desc.remoteEndpoint = linkList[i].dstEndpointDesc;
            requests.push_back(desc);
            found = true;
            break;
        }
        if (!found) {
            std::fprintf(stderr, "[TRS_CCU] rank=%d no UBC_CTP link to peer=%d\n", rankId, peer);
            return false;
        }
    }
    channels.resize(requests.size());
    if (!requests.empty()) {
        HcclResult rc = HcclChannelAcquire(comm, COMM_ENGINE_CCU, requests.data(),
                                           static_cast<uint32_t>(requests.size()), channels.data());
        if (rc != HCCL_SUCCESS)
            return false;
    }
    std::fprintf(stderr, "[TRS_CCU] rank=%d acquired %zu CCU channel(s)\n", rankId, channels.size());
    return true;
}

bool EnsureEnvReady()
{
    if (g_env.ready)
        return true;

    g_env.rankId = CommMpiRank();
    g_env.nRanks = CommMpiSize();
    g_env.devId = g_env.rankId;

    HcclRootInfo rootInfo{};
    if (g_env.rankId == 0) {
        rtSetDevice(0);
        aclrtSetDevice(0);
        if (HcclGetRootInfo(&rootInfo) != HCCL_SUCCESS)
            return false;
    }
    CommMpiBcast(&rootInfo, HCCL_ROOT_INFO_BYTES, COMM_MPI_CHAR, 0);
    CommMpiBarrier();

    ACL_OK(aclrtSetDevice(g_env.devId));
    ACL_OK(aclrtCreateStream(&g_env.stream));
    ACL_OK(aclrtCreateStream(&g_env.aivStream));

    HCCL_OK(HcclCommInitRootInfo(static_cast<uint32_t>(g_env.nRanks), &rootInfo, static_cast<uint32_t>(g_env.rankId),
                                 &g_env.comm));

    constexpr uint32_t kNotifyNum = 1;
    HCCL_OK(HcclThreadAcquireWithStream(g_env.comm, COMM_ENGINE_CCU, g_env.stream, kNotifyNum, &g_env.threadHandle));

    if (!SetupChannelsForCcu(g_env.comm, g_env.rankId, g_env.nRanks, g_env.channels))
        return false;

    aclrtMallocAttrValue modVal{};
    modVal.moduleId = 3;
    aclrtMallocAttribute attr{ACL_RT_MEM_ATTR_MODULE_ID, modVal};
    aclrtMallocConfig cfg{&attr, 1};
    ACL_OK(aclrtMallocWithCfg(&g_env.inputDev, kMaxPayload, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg));
    ACL_OK(aclrtMallocWithCfg(&g_env.outputDev, kMaxPayload, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg));

    g_env.inputVa = reinterpret_cast<uint64_t>(g_env.inputDev);
    g_env.outputVa = reinterpret_cast<uint64_t>(g_env.outputDev);

    const uint64_t spanBase = (g_env.inputVa < g_env.outputVa) ? g_env.inputVa : g_env.outputVa;
    const uint64_t spanEnd =
        (g_env.inputVa < g_env.outputVa) ? (g_env.outputVa + kMaxPayload) : (g_env.inputVa + kMaxPayload);
    g_env.token = hcomm::CcuRep::GetTokenInfo(spanBase, spanEnd - spanBase);

    g_env.ready = true;
    return true;
}

void CleanupEnv()
{
    if (!g_env.ready)
        return;
    if (g_env.outputDev)
        aclrtFree(g_env.outputDev);
    if (g_env.inputDev)
        aclrtFree(g_env.inputDev);
    if (g_env.comm)
        HcclCommDestroy(g_env.comm);
    if (g_env.aivStream)
        aclrtDestroyStream(g_env.aivStream);
    if (g_env.stream)
        aclrtDestroyStream(g_env.stream);
    aclrtResetDevice(g_env.devId);
    g_env.ready = false;
}

static bool DiscoverGateDescriptor(pto::comm::ccu::CcuGateDescriptor &gateDesc)
{
    for (int retry = 0; retry < 200; ++retry) {
        if (pto::comm::ccu::TryGet(static_cast<uint32_t>(g_env.rankId), gateDesc)) {
            return true;
        }
        usleep(10000);
    }
    std::fprintf(stderr, "[TRS_CCU] rank=%d TryGet FAILED after retries\n", g_env.rankId);
    return false;
}

static bool ResolveCkeMmio(const pto::comm::ccu::CcuGateDescriptor &gateDesc)
{
    constexpr int kRT_PROCESS_CP1 = 0;
    constexpr int kRT_RES_TYPE_CCU_CKE = 3;
    struct rtDevResInfo_t {
        uint32_t dieId;
        int procType;
        int resType;
        uint32_t resId;
        uint32_t flag;
    };
    struct rtDevResAddrInfo_t {
        uint64_t *resAddress;
        uint32_t *len;
    };
    using rtGetFn = int (*)(rtDevResInfo_t *, rtDevResAddrInfo_t *);

    void *rt = dlopen("libruntime.so", RTLD_NOW | RTLD_GLOBAL);
    if (!rt) {
        std::fprintf(stderr, "dlopen libruntime.so: %s\n", dlerror());
        return false;
    }
    auto rtGet = reinterpret_cast<rtGetFn>(dlsym(rt, "rtGetDevResAddress"));
    if (!rtGet) {
        std::fprintf(stderr, "dlsym: %s\n", dlerror());
        return false;
    }

    rtDevResInfo_t in{};
    in.dieId = gateDesc.dieId;
    in.procType = kRT_PROCESS_CP1;
    in.resType = kRT_RES_TYPE_CCU_CKE;
    in.resId = gateDesc.ckeId;
    in.flag = 0;
    uint64_t addr = 0;
    uint32_t len = 0;
    rtDevResAddrInfo_t out{&addr, &len};
    int qrc = rtGet(&in, &out);
    if (qrc != 0 || addr == 0) {
        std::fprintf(stderr, "[TRS_CCU] rank=%d rtGetDevResAddress FAIL rc=%d\n", g_env.rankId, qrc);
        return false;
    }
    g_env.mmioAddr = addr;
    g_env.gateMask = gateDesc.mask;
    return true;
}

bool ResolveGateOnce()
{
    if (g_env.gateResolved)
        return true;
    pto::comm::ccu::CcuGateDescriptor gateDesc{};
    if (!DiscoverGateDescriptor(gateDesc))
        return false;
    if (!ResolveCkeMmio(gateDesc))
        return false;
    g_env.gateResolved = true;
    return true;
}

// ReduceScatter: each rank's input[offset..offset+sliceSize] is the data
// that contributes to rank r's output. The output for each rank is
// sum(input_peer[offset..offset+sliceSize]) across all peers.
static bool PrepareReduceScatterBuffers(int rankId, int nRanks, size_t totalElements)
{
    // Input: fill entire buffer with (rankId + 1) — CCU reads from [offset..offset+slice]
    std::vector<float> inputHost(totalElements, static_cast<float>(rankId + 1));
    std::vector<float> zeroHost(kMaxElements, 0.0f);
    size_t payloadSize = totalElements * sizeof(float);
    ACL_OK(aclrtMemcpy(g_env.inputDev, kMaxPayload, inputHost.data(), payloadSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemcpy(g_env.outputDev, kMaxPayload, zeroHost.data(), kMaxPayload, ACL_MEMCPY_HOST_TO_DEVICE));
    return true;
}

static bool RegisterAndLaunchReduceScatterCcu(int rankId, int nRanks, size_t sliceSize, uint64_t offset, uint64_t seq)
{
    pto::comm::ccu::CcuReduceScatterKernelArg karg{
        static_cast<uint32_t>(rankId),
        static_cast<uint32_t>(nRanks),
        HcclDataType::HCCL_DATA_TYPE_FP32,
        HcclReduceOp::HCCL_REDUCE_SUM,
        sliceSize + seq,
    };
    karg.channels = g_env.channels;
    std::fprintf(stderr, "[TRS_CCU] rank=%d karg ready, channels=%zu\n", rankId, g_env.channels.size());

    hcomm::KernelCreator creator = pto::comm::ccu::MakeCcuReduceScatterCreator();
    CcuKernelHandle kHandle = 0;

    std::fprintf(stderr, "[TRS_CCU] rank=%d -> HcclCcuKernelRegister...\n", rankId);
    HCCL_OK(HcclCcuKernelRegister(g_env.comm, &kHandle, &creator, &karg));
    std::fprintf(stderr, "[TRS_CCU] rank=%d <- HcclCcuKernelRegister OK handle=%llu\n", rankId,
                 (unsigned long long)kHandle);

    std::fprintf(stderr, "[TRS_CCU] rank=%d -> HcclCcuKernelRegisterFinish...\n", rankId);
    HCCL_OK(HcclCcuKernelRegisterFinish(g_env.comm));
    std::fprintf(stderr, "[TRS_CCU] rank=%d <- HcclCcuKernelRegisterFinish OK\n", rankId);

    pto::comm::ccu::CcuReduceScatterTaskArg targ{g_env.inputVa, g_env.outputVa, offset, sliceSize, g_env.token};
    std::fprintf(stderr, "[TRS_CCU] rank=%d -> HcclCcuKernelLaunch...\n", rankId);
    HCCL_OK(HcclCcuKernelLaunch(g_env.comm, g_env.threadHandle, kHandle, &targ));
    std::fprintf(stderr, "[TRS_CCU] rank=%d <- HcclCcuKernelLaunch OK\n", rankId);
    return true;
}

static bool TriggerAndSyncReduceScatterCcu(int rankId)
{
    if (!ResolveGateOnce())
        return false;
    std::fprintf(stderr, "[TRS_CCU] rank=%d gate resolved mmio=0x%llx mask=0x%x\n", rankId,
                 (unsigned long long)g_env.mmioAddr, g_env.gateMask);

    usleep(200000);

    std::fprintf(stderr, "[TRS_CCU] rank=%d triggering CKE via TREDUCE_SCATTER<CCU>...\n", rankId);
    int rc = treduce_scatter_ccu_trigger_launch(g_env.aivStream, g_env.mmioAddr, g_env.gateMask);
    if (rc != 0)
        return false;
    std::fprintf(stderr, "[TRS_CCU] rank=%d trigger returned, syncing aivStream...\n", rankId);

    ACL_OK(aclrtSynchronizeStream(g_env.aivStream));
    std::fprintf(stderr, "[TRS_CCU] rank=%d aivStream synced, syncing ccuStream...\n", rankId);
    ACL_OK(aclrtSynchronizeStream(g_env.stream));
    std::fprintf(stderr, "[TRS_CCU] rank=%d ccuStream synced\n", rankId);
    return true;
}

static bool TriggerAndSyncReduceScatterCcuFused(int rankId, int nRanks, float fillValue)
{
    if (!ResolveGateOnce())
        return false;
    std::fprintf(stderr, "[TRS_CCU/fused] rank=%d gate resolved mmio=0x%llx mask=0x%x fill=%f\n", rankId,
                 (unsigned long long)g_env.mmioAddr, g_env.gateMask, fillValue);

    usleep(200000);

    std::fprintf(stderr, "[TRS_CCU/fused] rank=%d launching fused AIV kernel...\n", rankId);
    int rc = treduce_scatter_ccu_fused_launch(g_env.aivStream, g_env.inputVa, g_env.outputVa,
                                              static_cast<uint32_t>(rankId), nRanks, fillValue, g_env.mmioAddr,
                                              g_env.gateMask);
    if (rc != 0)
        return false;

    ACL_OK(aclrtSynchronizeStream(g_env.aivStream));
    std::fprintf(stderr, "[TRS_CCU/fused] rank=%d aivStream synced\n", rankId);

    {
        float probe[4] = {-999.0f, -999.0f, -999.0f, -999.0f};
        aclrtMemcpy(probe, sizeof(probe), g_env.inputDev, sizeof(probe), ACL_MEMCPY_DEVICE_TO_HOST);
        std::fprintf(stderr, "[TRS_CCU/fused] rank=%d inputDev probe (first 4): %f %f %f %f (expected %f)\n", rankId,
                     probe[0], probe[1], probe[2], probe[3], fillValue);
    }

    std::fprintf(stderr, "[TRS_CCU/fused] rank=%d syncing ccuStream...\n", rankId);
    ACL_OK(aclrtSynchronizeStream(g_env.stream));
    std::fprintf(stderr, "[TRS_CCU/fused] rank=%d ccuStream synced\n", rankId);
    return true;
}

// Verify ReduceScatter result. Each rank's output slice should be
// sum_{r=0..nRanks-1}(r+1) = nRanks*(nRanks+1)/2 at every position.
static bool VerifyReduceScatterResult(int rankId, int nRanks, size_t sliceElements)
{
    size_t sliceBytes = sliceElements * sizeof(float);
    std::vector<float> outputHost(sliceElements);
    ACL_OK(aclrtMemcpy(outputHost.data(), sliceBytes, g_env.outputDev, sliceBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    const float expected = static_cast<float>(nRanks * (nRanks + 1) / 2);

    std::fprintf(stderr, "[TRS_CCU] rank=%d sample (first 4 elems): %f %f %f %f (expected %f)\n", rankId,
                 outputHost[0], (sliceElements > 1 ? outputHost[1] : 0.0f),
                 (sliceElements > 2 ? outputHost[2] : 0.0f), (sliceElements > 3 ? outputHost[3] : 0.0f), expected);

    int mismatch = 0;
    for (size_t i = 0; i < sliceElements; ++i) {
        const float actual = outputHost[i];
        if (!std::isfinite(actual) || std::fabs(actual - expected) > 1e-3f) {
            if (mismatch < 8)
                std::fprintf(stderr, "[TRS_CCU] mismatch [%zu]: got=%f expected=%f\n", i, actual, expected);
            ++mismatch;
        }
    }
    bool pass = (mismatch == 0);
    std::fprintf(stderr, "[TRS_CCU] rank=%d %s: %zu elements\n", rankId, pass ? "PASS" : "FAIL", sliceElements);
    return pass;
}

bool RunReduceScatterCcu(size_t totalElements)
{
    CommMpiBarrier();
    if (!EnsureEnvReady())
        return false;

    const int rankId = g_env.rankId;
    const int nRanks = g_env.nRanks;
    const size_t sliceElements = totalElements / static_cast<size_t>(nRanks);
    const size_t sliceBytes = sliceElements * sizeof(float);
    const uint64_t offset = static_cast<uint64_t>(rankId) * sliceBytes;

    if (!PrepareReduceScatterBuffers(rankId, nRanks, totalElements))
        return false;

    uint64_t seq = g_seqNo++;
    std::fprintf(stderr, "[TRS_CCU] rank=%d RunReduceScatterCcu seq=%llu total=%zu slice=%zu offset=%llu\n", rankId,
                 (unsigned long long)seq, totalElements, sliceElements, (unsigned long long)offset);

    if (!RegisterAndLaunchReduceScatterCcu(rankId, nRanks, sliceBytes, offset, seq))
        return false;
    if (!TriggerAndSyncReduceScatterCcu(rankId))
        return false;

    bool pass = VerifyReduceScatterResult(rankId, nRanks, sliceElements);
    CommMpiBarrier();
    return pass;
}

bool RunReduceScatterCcuFused(size_t totalElements)
{
    CommMpiBarrier();
    if (!EnsureEnvReady())
        return false;

    const int rankId = g_env.rankId;
    const int nRanks = g_env.nRanks;
    const size_t sliceElements = totalElements / static_cast<size_t>(nRanks);
    const size_t sliceBytes = sliceElements * sizeof(float);
    const uint64_t offset = static_cast<uint64_t>(rankId) * sliceBytes;

    // Zero output; fill input with NaN sentinels (AIV sole producer)
    {
        std::vector<float> zeroHost(kMaxElements, 0.0f);
        ACL_OK(aclrtMemcpy(g_env.outputDev, kMaxPayload, zeroHost.data(), kMaxPayload, ACL_MEMCPY_HOST_TO_DEVICE));
        std::vector<float> sentinelHost(kMaxElements, std::nanf(""));
        ACL_OK(aclrtMemcpy(g_env.inputDev, kMaxPayload, sentinelHost.data(), kMaxPayload, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    uint64_t seq = g_seqNo++;
    std::fprintf(stderr, "[TRS_CCU/fused] rank=%d RunReduceScatterCcuFused seq=%llu total=%zu slice=%zu\n", rankId,
                 (unsigned long long)seq, totalElements, sliceElements);

    if (!RegisterAndLaunchReduceScatterCcu(rankId, nRanks, sliceBytes, offset, seq))
        return false;

    const float fillValue = static_cast<float>(rankId + 1);
    if (!TriggerAndSyncReduceScatterCcuFused(rankId, nRanks, fillValue))
        return false;

    bool pass = VerifyReduceScatterResult(rankId, nRanks, sliceElements);
    CommMpiBarrier();
    return pass;
}

void ResetEnv()
{
    CleanupEnv();
    g_env = CcuEnv{};
    g_seqNo = 0;
    std::fprintf(stderr, "[TRS_CCU] === env reset for next test ===\n");
}

class TReduceScatterCcuTest : public ::testing::Test {
protected:
    void TearDown() override
    {
        ResetEnv();
    }
};

TEST_F(TReduceScatterCcuTest, Float_1024_Sum_2Ranks)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunReduceScatterCcu(1024));
}

TEST_F(TReduceScatterCcuTest, Float_1024_Sum_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunReduceScatterCcu(1024));
}

TEST_F(TReduceScatterCcuTest, Fused_Float_1024_Sum_2Ranks)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunReduceScatterCcuFused(1024));
}

TEST_F(TReduceScatterCcuTest, Fused_Float_1024_Sum_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunReduceScatterCcuFused(1024));
}

} // namespace

int main(int argc, char **argv)
{
    return ::pto::comm::ccu::st::RunCcuStMain(argc, argv, &CleanupEnv);
}
