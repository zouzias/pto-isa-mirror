/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cmath>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <dlfcn.h>
#include <unistd.h>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_rank_graph.h"

#include "ccu_offline_adapter_c.h"

#include <gtest/gtest.h>
#include "../comm_mpi.h"
#include "../ccu_test_main.hpp"

extern "C" int32_t rtSetDevice(int32_t deviceId);
extern "C" int tgather_ccu_trigger_launch(void *stream, uint64_t ckeVA, uint32_t mask);

namespace {

static constexpr uint32_t kChannelKey = 0x1001;
static constexpr uint32_t kRootRank = 0;
static constexpr size_t kElements = 4;
static constexpr size_t kPayload = kElements * sizeof(float);
static constexpr size_t kDslStagingHbmRankStride = 4096;

#define ACL_OK(expr)                                                                                                    \
    do {                                                                                                                \
        aclError _r = (expr);                                                                                           \
        if (_r != ACL_SUCCESS) {                                                                                        \
            std::fprintf(stderr, "[TGATHER_CCU_DSL] ACL FAIL %s = %d (%s:%d)\n", #expr, static_cast<int>(_r), __FILE__, \
                         __LINE__);                                                                                     \
            return false;                                                                                               \
        }                                                                                                               \
    } while (0)

#define HCCL_OK(expr)                                                                                                    \
    do {                                                                                                                 \
        HcclResult _r = (expr);                                                                                          \
        if (_r != HCCL_SUCCESS) {                                                                                        \
            std::fprintf(stderr, "[TGATHER_CCU_DSL] HCCL FAIL %s = %d (%s:%d)\n", #expr, static_cast<int>(_r), __FILE__, \
                         __LINE__);                                                                                      \
            return false;                                                                                                \
        }                                                                                                                \
    } while (0)

struct CcuDslEnv {
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
    uint64_t adapterHbmVa = 0;

    uint64_t mmioAddr = 0;
    uint32_t gateMask = 0;

    bool ready = false;
    bool gateResolved = false;

    CcuOfflineAdapter *adapter = nullptr;
    CcuOfflineAdapterArtifact *artifact = nullptr;
    uint64_t kernelHandle = 0;
};

static CcuDslEnv g_env;

std::string ReadTextFile(const char *path)
{
    std::ifstream ifs(path);
    if (!ifs) {
        return "";
    }
    std::stringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

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
            std::fprintf(stderr, "[TGATHER_CCU_DSL] rank=%d no UBC_CTP link to peer=%d\n", rankId, peer);
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

    ACL_OK(aclrtMalloc(&g_env.inputDev, kPayload, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&g_env.outputDev, g_env.nRanks * kPayload, ACL_MEM_MALLOC_HUGE_FIRST));
    g_env.inputVa = reinterpret_cast<uint64_t>(g_env.inputDev);
    g_env.outputVa = reinterpret_cast<uint64_t>(g_env.outputDev);

    g_env.ready = true;
    return true;
}

void CleanupEnv()
{
    if (g_env.adapter && g_env.kernelHandle != 0) {
        (void)CcuOfflineAdapterUnregister(g_env.adapter, g_env.kernelHandle);
    }
    CcuOfflineAdapterDestroy(g_env.adapter);
    CcuOfflineAdapterArtifactDestroy(g_env.artifact);
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
    if (g_env.devId >= 0)
        aclrtResetDevice(g_env.devId);
    g_env = CcuDslEnv{};
}

bool ResolveCkeMmio(const CcuOfflineAdapterPublishedCkeDescriptor &gateDesc)
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
        std::fprintf(stderr, "dlsym rtGetDevResAddress: %s\n", dlerror());
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
        std::fprintf(stderr, "[TGATHER_CCU_DSL] rank=%d rtGetDevResAddress FAIL rc=%d\n", g_env.rankId, qrc);
        return false;
    }
    g_env.mmioAddr = addr;
    g_env.gateMask = gateDesc.mask;
    return true;
}

bool PrepareBuffers()
{
    std::vector<float> input(kElements);
    for (size_t i = 0; i < kElements; ++i) {
        input[i] = static_cast<float>(i + g_env.rankId * 10000);
    }
    ACL_OK(aclrtMemcpy(g_env.inputDev, kPayload, input.data(), kPayload, ACL_MEMCPY_HOST_TO_DEVICE));

    std::vector<float> output(static_cast<size_t>(g_env.nRanks) * kElements, -1.0f);
    ACL_OK(aclrtMemcpy(g_env.outputDev, g_env.nRanks * kPayload, output.data(), g_env.nRanks * kPayload,
                       ACL_MEMCPY_HOST_TO_DEVICE));

    if (g_env.rankId == static_cast<int>(kRootRank)) {
        ACL_OK(aclrtMemcpy(g_env.outputDev, kPayload, g_env.inputDev, kPayload, ACL_MEMCPY_DEVICE_TO_DEVICE));
    }
    return true;
}

struct AdapterChannelLayoutBuffers {
    CcuOfflineAdapterChannelLayout layout{};
    std::vector<uint64_t> rmtHbmSlotAddrs;
    std::vector<uint64_t> rmtHbmSlotTokens;
    std::vector<uint64_t> locHbmSlotTokens;
};

void RefreshLayoutPointers(AdapterChannelLayoutBuffers &buffers)
{
    buffers.layout.rmtHbmSlotAddrs = buffers.rmtHbmSlotAddrs.empty() ? nullptr : buffers.rmtHbmSlotAddrs.data();
    buffers.layout.rmtHbmSlotAddrsCount = buffers.rmtHbmSlotAddrs.size();
    buffers.layout.rmtHbmSlotTokens = buffers.rmtHbmSlotTokens.empty() ? nullptr : buffers.rmtHbmSlotTokens.data();
    buffers.layout.rmtHbmSlotTokensCount = buffers.rmtHbmSlotTokens.size();
    buffers.layout.locHbmSlotTokens = buffers.locHbmSlotTokens.empty() ? nullptr : buffers.locHbmSlotTokens.data();
    buffers.layout.locHbmSlotTokensCount = buffers.locHbmSlotTokens.size();
}

bool FillHbmChannelLayout(AdapterChannelLayoutBuffers &buffers)
{
    constexpr size_t kMaxHbmSlots = 8;
    CcuOfflineAdapterHbmSlotInfo localSlots[kMaxHbmSlots]{};
    size_t localSlotCount = 0;
    CcuOfflineAdapterStatus slotRet = CcuOfflineAdapterGetLocalHbmSlots(
        g_env.adapter, g_env.kernelHandle, 0, localSlots, kMaxHbmSlots, &localSlotCount);
    if (slotRet != CCU_OFFLINE_ADAPTER_OK) {
        std::fprintf(stderr, "[TGATHER_CCU_DSL] rank=%d cannot query local HBM slots: status=%d error=%s\n",
                     g_env.rankId, static_cast<int>(slotRet), CcuOfflineAdapterGetLastError(g_env.adapter));
        return false;
    }

    struct WireSlot {
        uint64_t present;
        uint64_t slot;
        uint64_t va;
        uint64_t token;
        uint64_t bytes;
    };
    static_assert(sizeof(WireSlot) == 5 * sizeof(uint64_t), "WireSlot must stay byte-copyable");

    std::array<WireSlot, kMaxHbmSlots> sendSlots{};
    for (size_t i = 0; i < localSlotCount; ++i) {
        const auto &slot = localSlots[i];
        if (slot.slot >= kMaxHbmSlots) {
            std::fprintf(stderr, "[TGATHER_CCU_DSL] rank=%d local HBM slot %u exceeds exchange limit %zu\n",
                         g_env.rankId, slot.slot, kMaxHbmSlots);
            return false;
        }
        sendSlots[slot.slot] = WireSlot{1, slot.slot, slot.va, slot.token, slot.bytes};
        if (buffers.locHbmSlotTokens.size() <= slot.slot) {
            buffers.locHbmSlotTokens.resize(slot.slot + 1);
        }
        buffers.locHbmSlotTokens[slot.slot] = slot.token;
        if (slot.slot == 0) {
            g_env.adapterHbmVa = slot.va;
        }
    }

    std::vector<WireSlot> allSlots(static_cast<size_t>(g_env.nRanks) * kMaxHbmSlots);
    if (CommMpiAllgather(sendSlots.data(), static_cast<int>(sizeof(WireSlot) * kMaxHbmSlots),
                         allSlots.data(), static_cast<int>(sizeof(WireSlot) * kMaxHbmSlots)) != 0) {
        std::fprintf(stderr, "[TGATHER_CCU_DSL] rank=%d MPI allgather HBM slots failed\n", g_env.rankId);
        return false;
    }

    const int peerRank = (g_env.nRanks == 2) ? (1 - g_env.rankId) : -1;
    if (peerRank < 0) {
        std::fprintf(stderr, "[TGATHER_CCU_DSL] HBM layout exchange currently expects 2 ranks, got %d\n", g_env.nRanks);
        return false;
    }

    const WireSlot *peerSlots = allSlots.data() + static_cast<size_t>(peerRank) * kMaxHbmSlots;
    for (size_t i = 0; i < kMaxHbmSlots; ++i) {
        if (peerSlots[i].present == 0) {
            continue;
        }
        const uint64_t slot = peerSlots[i].slot;
        if (slot >= kMaxHbmSlots) {
            return false;
        }
        const size_t idx = static_cast<size_t>(slot);
        if (buffers.rmtHbmSlotAddrs.size() <= idx) {
            buffers.rmtHbmSlotAddrs.resize(idx + 1);
        }
        if (buffers.rmtHbmSlotTokens.size() <= idx) {
            buffers.rmtHbmSlotTokens.resize(idx + 1);
        }
        buffers.rmtHbmSlotAddrs[idx] = peerSlots[i].va;
        buffers.rmtHbmSlotTokens[idx] = peerSlots[i].token;
        std::fprintf(stderr,
                     "[TGATHER_CCU_DSL] rank=%d peer=%d hbmSlot=%zu rmtVa=0x%lx rmtToken=0x%lx bytes=%lu\n",
                     g_env.rankId, peerRank, idx, peerSlots[i].va, peerSlots[i].token, peerSlots[i].bytes);
    }
    RefreshLayoutPointers(buffers);
    return true;
}

bool PrepareDslStagingHbm()
{
    if (g_env.adapterHbmVa == 0) {
        std::fprintf(stderr, "[TGATHER_CCU_DSL] rank=%d adapter HBM slot0 is missing\n", g_env.rankId);
        return false;
    }
    // The current DSL staging microcode uses offset 4096 for rx(root) and offset 0 for tx(peer).
    const uint64_t localOffset = (g_env.rankId == static_cast<int>(kRootRank)) ? kDslStagingHbmRankStride : 0;
    const uint64_t dst = g_env.adapterHbmVa + localOffset;
    ACL_OK(aclrtMemcpy(reinterpret_cast<void *>(dst), kPayload, g_env.inputDev, kPayload, ACL_MEMCPY_DEVICE_TO_DEVICE));
    std::array<float, kElements> probe{};
    ACL_OK(aclrtMemcpy(probe.data(), kPayload, reinterpret_cast<void *>(dst), kPayload, ACL_MEMCPY_DEVICE_TO_HOST));
    std::fprintf(stderr, "[TGATHER_CCU_DSL_PROBE] rank=%d phase=after-init offset=%lu values=%f,%f,%f,%f\n",
                 g_env.rankId, localOffset, probe[0], probe[1], probe[2], probe[3]);
    return true;
}

bool CopyDslStagingResultToOutput()
{
    if (g_env.rankId != static_cast<int>(kRootRank)) {
        return true;
    }
    if (g_env.adapterHbmVa == 0) {
        std::fprintf(stderr, "[TGATHER_CCU_DSL] rank=%d adapter HBM slot0 is missing before result copy\n", g_env.rankId);
        return false;
    }
    for (int rank = 0; rank < g_env.nRanks; ++rank) {
        const uint64_t resultOffset = (rank == static_cast<int>(kRootRank)) ? kDslStagingHbmRankStride : 0;
        const uint64_t src = g_env.adapterHbmVa + resultOffset;
        const uint64_t dst = g_env.outputVa + static_cast<uint64_t>(rank) * kPayload;
        std::array<float, kElements> probe{};
        ACL_OK(aclrtMemcpy(probe.data(), kPayload, reinterpret_cast<void *>(src), kPayload, ACL_MEMCPY_DEVICE_TO_HOST));
        std::fprintf(stderr,
                     "[TGATHER_CCU_DSL_PROBE] rank=%d phase=before-copy-out outRank=%d offset=%lu values=%f,%f,%f,%f\n",
                     g_env.rankId, rank, resultOffset, probe[0], probe[1], probe[2], probe[3]);
        ACL_OK(aclrtMemcpy(reinterpret_cast<void *>(dst), kPayload, reinterpret_cast<void *>(src), kPayload,
                           ACL_MEMCPY_DEVICE_TO_DEVICE));
    }
    return true;
}

bool CompileAndLaunchDsl()
{
    const bool isRoot = (g_env.rankId == static_cast<int>(kRootRank));
    const char *microcodePath = isRoot ? PTO_CCU_DSL_RX_MICROCODE : PTO_CCU_DSL_TX_MICROCODE;
    std::string microcode = ReadTextFile(microcodePath);
    if (microcode.empty()) {
        std::fprintf(stderr, "[TGATHER_CCU_DSL] rank=%d cannot read microcode: %s\n", g_env.rankId, microcodePath);
        return false;
    }

    CcuOfflineAdapterCompileInput src{};
    src.opName = isRoot ? "tgather_ccu_dsl_rx" : "tgather_ccu_dsl_tx";
    src.taskArgCount = 3;
    src.channelKeyHints[0] = kChannelKey;
    src.microcodePerDie[0] = microcode.c_str();
    src.prependAivGate = 1;

    CcuOfflineAdapterStatus status = CcuOfflineAdapterCompile(&src, &g_env.artifact);
    if (status != CCU_OFFLINE_ADAPTER_OK) {
        std::fprintf(stderr, "[TGATHER_CCU_DSL] compile error status=%d: %s\n",
                     static_cast<int>(status), CcuOfflineAdapterArtifactGetLastError(g_env.artifact));
        return false;
    }

    status = CcuOfflineAdapterCreate(/*insHandle*/ 0, g_env.devId, &g_env.adapter);
    if (status != CCU_OFFLINE_ADAPTER_OK || g_env.adapter == nullptr) {
        std::fprintf(stderr, "[TGATHER_CCU_DSL] create adapter failed status=%d\n", static_cast<int>(status));
        return false;
    }
    status = CcuOfflineAdapterRegister(g_env.adapter, g_env.artifact, &g_env.kernelHandle);
    if (status != CCU_OFFLINE_ADAPTER_OK) {
        std::fprintf(stderr, "[TGATHER_CCU_DSL] register adapter failed status=%d error=%s\n",
                     static_cast<int>(status), CcuOfflineAdapterGetLastError(g_env.adapter));
        return false;
    }

    AdapterChannelLayoutBuffers layoutBuffers;
    layoutBuffers.layout.channelKey = kChannelKey;
    if (!g_env.channels.empty()) {
        layoutBuffers.layout.channel = static_cast<uint64_t>(g_env.channels[0]);
    }
    if (!FillHbmChannelLayout(layoutBuffers)) {
        return false;
    }
    if (!PrepareDslStagingHbm()) {
        return false;
    }
    status = CcuOfflineAdapterApplyHcommChannelLayouts(g_env.adapter, g_env.kernelHandle,
                                                       &layoutBuffers.layout, 1);
    if (status != CCU_OFFLINE_ADAPTER_OK) {
        std::fprintf(stderr, "[TGATHER_CCU_DSL] apply adapter layout failed status=%d error=%s\n",
                     static_cast<int>(status), CcuOfflineAdapterGetLastError(g_env.adapter));
        return false;
    }

    CcuOfflineAdapterPublishedCkeDescriptor gate{};
    status = CcuOfflineAdapterGetPublishedCkeDescriptor(g_env.adapter, g_env.kernelHandle, 0, 0, &gate);
    if (status != CCU_OFFLINE_ADAPTER_OK) {
        std::fprintf(stderr, "[TGATHER_CCU_DSL] get published CKE failed status=%d error=%s\n",
                     static_cast<int>(status), CcuOfflineAdapterGetLastError(g_env.adapter));
        return false;
    }
    if (!ResolveCkeMmio(gate)) {
        return false;
    }

    const uint64_t dataVa = isRoot ? (g_env.outputVa + kPayload) : g_env.inputVa;
    std::vector<uint64_t> taskArgs{dataVa, dataVa, dataVa};
    status = CcuOfflineAdapterLaunch(g_env.adapter, g_env.kernelHandle,
                                     static_cast<uint64_t>(g_env.threadHandle),
                                     taskArgs.data(), taskArgs.size());
    if (status != CCU_OFFLINE_ADAPTER_OK) {
        std::fprintf(stderr, "[TGATHER_CCU_DSL] launch failed status=%d error=%s\n",
                     static_cast<int>(status), CcuOfflineAdapterGetLastError(g_env.adapter));
        return false;
    }
    return true;
}

bool TriggerAndSync()
{
    usleep(200000);
    int rc = tgather_ccu_trigger_launch(g_env.aivStream, g_env.mmioAddr, g_env.gateMask);
    if (rc != 0)
        return false;
    ACL_OK(aclrtSynchronizeStream(g_env.aivStream));
    ACL_OK(aclrtSynchronizeStream(g_env.stream));
    return true;
}

bool VerifyGatherResult()
{
    if (g_env.rankId != static_cast<int>(kRootRank))
        return true;

    std::vector<float> output(static_cast<size_t>(g_env.nRanks) * kElements);
    ACL_OK(aclrtMemcpy(output.data(), g_env.nRanks * kPayload, g_env.outputDev, g_env.nRanks * kPayload,
                       ACL_MEMCPY_DEVICE_TO_HOST));
    int mismatch = 0;
    for (int r = 0; r < g_env.nRanks; ++r) {
        for (size_t i = 0; i < kElements; ++i) {
            const float expected = static_cast<float>(i + r * 10000);
            const size_t idx = static_cast<size_t>(r) * kElements + i;
            if (std::fabs(output[idx] - expected) <= 1e-3f)
                continue;
            if (mismatch < 8) {
                std::fprintf(stderr, "[TGATHER_CCU_DSL] mismatch [r=%d i=%zu]: got=%f expected=%f\n", r, i,
                             output[idx], expected);
            }
            ++mismatch;
        }
    }
    return mismatch == 0;
}

bool RunGatherDsl()
{
    CommMpiBarrier();
    if (!EnsureEnvReady())
        return false;
    if (g_env.nRanks != 2) {
        std::fprintf(stderr, "[TGATHER_CCU_DSL] first version requires exactly 2 ranks, got %d\n", g_env.nRanks);
        return false;
    }
    if (!PrepareBuffers())
        return false;
    if (!CompileAndLaunchDsl())
        return false;
    if (!TriggerAndSync())
        return false;
    if (!CopyDslStagingResultToOutput())
        return false;
    bool pass = VerifyGatherResult();
    CommMpiBarrier();
    return pass;
}

class TGatherCcuDslTest : public ::testing::Test {
protected:
    void TearDown() override
    {
        CleanupEnv();
    }
};

TEST_F(TGatherCcuDslTest, Float_4_2Ranks)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunGatherDsl());
}

} // namespace

int main(int argc, char **argv)
{
    return ::pto::comm::ccu::st::RunCcuStMain(argc, argv, &CleanupEnv);
}
