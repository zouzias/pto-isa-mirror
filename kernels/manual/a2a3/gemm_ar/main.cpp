/**
 * GEMM AllReduce Demo - Main Entry Point with Integrated Data Generation
 * HCCL backend — launched via mpirun
 *
 * Generates random input matrices (fp16), computes golden reference (fp32 CPU GEMM),
 * then runs multi-card GEMM with AllReduce.
 *
 * Usage:
 *   mpirun -n <NRANKS> ./gemm_allreduce [--first-device ID]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <thread>
#include <chrono>
#include <algorithm>
#include <string>
#include <iostream>
#include <iomanip>

#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_types.h"
#include "comm_mpi.h"

#include "hccl_context.h"

#ifndef __CCE_KT_TEST__
#define __CCE_KT_TEST__
#define __CCE_KT_TEST_DEFINED_HERE__
#endif
#include "ready_queue.hpp"
#ifdef __CCE_KT_TEST_DEFINED_HERE__
#undef __CCE_KT_TEST__
#undef __CCE_KT_TEST_DEFINED_HERE__
#endif

#include "gemm_ar_config.h"

// Internal HCCL APIs
extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void *stream, void *mc2Tiling, void **commContext);
extern "C" HcclResult HcomGetCommHandleByGroup(const char *group, HcclComm *commHandle);

using CommTopo = uint32_t;
extern "C" HcclResult HcomGetL0TopoTypeEx(const char *group, CommTopo *topoType, uint32_t isSetDevice);
static constexpr uint32_t COMM_IS_NOT_SET_DEVICE = 0;
static constexpr uint32_t COMM_TOPO_MESH = 0b1u;

using rtError_t = int32_t;
using rtStream_t = void *;
static constexpr int32_t RT_STREAM_PRIORITY_DEFAULT = 0;
extern "C" rtError_t rtSetDevice(int32_t device);
extern "C" rtError_t rtStreamCreate(rtStream_t *stream, int32_t priority);
extern "C" rtError_t rtStreamDestroy(rtStream_t stream);

// ============================================================================
// V2 tiling structures for HcclAllocComResourceByTiling
// ============================================================================
namespace gemm_ar_tiling {

static constexpr uint32_t TILING_MAX_CC_NUM = 8U;
static constexpr uint32_t TILING_GROUP_NAME_SIZE = 128U;
static constexpr uint32_t TILING_ALG_CONFIG_SIZE = 128U;

struct Mc2InitTilingInner {
    uint32_t version;
    uint32_t mc2HcommCnt;
    uint32_t offset[TILING_MAX_CC_NUM];
    uint8_t debugMode;
    uint8_t preparePosition;
    uint16_t queueNum;
    uint16_t commBlockNum;
    uint8_t devType;
    char reserved[17];
};

struct Mc2cCTilingInner {
    uint8_t skipLocalRankCopy;
    uint8_t skipBufferWindowCopy;
    uint8_t stepSize;
    uint8_t version;
    char reserved[9];
    uint8_t commEngine;
    uint8_t srcDataType;
    uint8_t dstDataType;
    char groupName[TILING_GROUP_NAME_SIZE];
    char algConfig[TILING_ALG_CONFIG_SIZE];
    uint32_t opType;
    uint32_t reduceType;
};

struct Mc2CommConfigV2 {
    Mc2InitTilingInner init;
    Mc2cCTilingInner inner;
};

} // namespace gemm_ar_tiling

// ============================================================================
// HcclOpResParam compat structs for RING topology
// ============================================================================
namespace hccl_compat {

struct HcclSignalInfo {
    uint64_t resId;
    uint64_t addr;
    uint32_t devId;
    uint32_t tsId;
    uint32_t rankId;
    uint32_t flag;
};

struct HcclStreamInfo {
    int32_t streamIds;
    uint32_t sqIds;
    uint32_t cqIds;
    uint32_t logicCqids;
};

struct ListCommon {
    uint64_t nextHost;
    uint64_t preHost;
    uint64_t nextDevice;
    uint64_t preDevice;
};

static constexpr uint32_t COMPAT_LOCAL_NOTIFY_MAX_NUM = 64;
static constexpr uint32_t COMPAT_LOCAL_STREAM_MAX_NUM = 19;
static constexpr uint32_t COMPAT_AICPU_OP_NOTIFY_MAX_NUM = 2;

struct LocalResInfoV2 {
    uint32_t streamNum;
    uint32_t signalNum;
    HcclSignalInfo localSignals[COMPAT_LOCAL_NOTIFY_MAX_NUM];
    HcclStreamInfo streamInfo[COMPAT_LOCAL_STREAM_MAX_NUM];
    HcclStreamInfo mainStreamInfo;
    HcclSignalInfo aicpuOpNotify[COMPAT_AICPU_OP_NOTIFY_MAX_NUM];
    ListCommon nextTagRes;
};

struct AlgoTopoInfo {
    uint32_t userRank;
    uint32_t userRankSize;
    int32_t deviceLogicId;
    bool isSingleMeshAggregation;
    uint32_t deviceNumPerAggregation;
    uint32_t superPodNum;
    uint32_t devicePhyId;
    uint32_t topoType;
    uint32_t deviceType;
    uint32_t serverNum;
    uint32_t meshAggregationRankSize;
    uint32_t multiModuleDiffDeviceNumMode;
    uint32_t multiSuperPodDiffServerNumMode;
    uint32_t realUserRank;
    bool isDiffDeviceModule;
    bool isDiffDeviceType;
    uint32_t gcdDeviceNumPerAggregation;
    uint32_t moduleNum;
    uint32_t isUsedRdmaRankPairNum;
    uint64_t isUsedRdmaRankPair;
    uint32_t pairLinkCounterNum;
    uint64_t pairLinkCounter;
    uint32_t nicNum;
    uint64_t nicList;
    uint64_t complanRankLength;
    uint64_t complanRank;
    uint64_t bridgeRankNum;
    uint64_t bridgeRank;
    uint64_t serverAndsuperPodRankLength;
    uint64_t serverAndsuperPodRank;
};

struct HcclOpConfig {
    uint8_t deterministic;
    uint8_t retryEnable;
    uint8_t highPerfEnable;
    uint8_t padding[5];
    uint8_t linkTimeOut[8];
    uint64_t notifyWaitTime;
    uint32_t retryHoldTime;
    uint32_t retryIntervalTime;
    bool interXLinkDisable;
    uint32_t floatOverflowMode;
    uint32_t multiQpThreshold;
};

struct HDCommunicateParams {
    uint64_t hostAddr;
    uint64_t deviceAddr;
    uint64_t readCacheAddr;
    uint32_t devMemSize;
    uint32_t buffLen;
    uint32_t flag;
};

struct RemoteResPtr {
    uint64_t nextHostPtr;
    uint64_t nextDevicePtr;
};

struct HcclMC2WorkSpace {
    uint64_t workspace;
    uint64_t workspaceSize;
};

struct HcclRankRelationResV2 {
    uint32_t remoteUsrRankId;
    uint32_t remoteWorldRank;
    uint64_t windowsIn;
    uint64_t windowsOut;
    uint64_t windowsExp;
    ListCommon nextTagRes;
};

struct HcclOpResParamHead {
    uint32_t localUsrRankId;
    uint32_t rankSize;
    uint64_t winSize;
    uint64_t localWindowsIn;
    uint64_t localWindowsOut;
    char hcomId[128];
    uint64_t winExpSize;
    uint64_t localWindowsExp;
};

struct HcclOpResParam {
    HcclMC2WorkSpace mc2WorkSpace;
    uint32_t localUsrRankId;
    uint32_t rankSize;
    uint64_t winSize;
    uint64_t localWindowsIn;
    uint64_t localWindowsOut;
    char hcomId[128];
    uint64_t winExpSize;
    uint64_t localWindowsExp;
    uint32_t rWinStart;
    uint32_t rWinOffset;
    uint64_t version;
    LocalResInfoV2 localRes;
    AlgoTopoInfo topoInfo;
    HcclOpConfig config;
    uint64_t hostStateInfo;
    uint64_t aicpuStateInfo;
    uint64_t lockAddr;
    uint32_t rsv[16];
    uint32_t notifysize;
    uint32_t remoteResNum;
    RemoteResPtr remoteRes[1];
};

} // namespace hccl_compat

// ============================================================================
// Host-side helpers
// ============================================================================
inline void HcclHostBarrier(HcclComm comm, aclrtStream stream)
{
    HcclBarrier(comm, stream);
    aclrtSynchronizeStream(stream);
}

inline void *WindowAlloc(uint64_t windowBase, size_t &offset, size_t bytes)
{
    void *ptr = reinterpret_cast<void *>(windowBase + offset);
    offset += bytes;
    return ptr;
}

// ============================================================================
// Extern kernel launcher declarations (defined in comm_kernel.cpp / gemm_compute_kernel.cpp)
// ============================================================================
extern void launchGemmCommRS(uint8_t *gemm_output, uint8_t *recv_buffers,
                             uint8_t *queue_set, uint8_t *hcclCtx,
                             int rank, int nranks, void *stream, int num_compute_blocks);

extern void launchGemmCommReduce(uint8_t *gemm_output, uint8_t *recv_buffers, uint8_t *reduced_output,
                                 uint8_t *hcclCtx, int rank, int nranks, void *stream);

extern void launchGemmCommAG(uint8_t *reduced_output, uint8_t *hcclCtx,
                             int rank, int nranks, void *stream);

extern void launchGemmCompute(uint8_t *gemm_output, uint8_t *src0, uint8_t *src1,
                              uint8_t *queue_set, int rank, void *stream, int block_num, uint32_t k_per_rank);

// ============================================================================
// Helpers
// ============================================================================

struct PerfStats {
    double avg, min_val, max_val, std_dev;
};

static PerfStats calcStats(const std::vector<double>& times)
{
    double sum = 0.0, mn = times[0], mx = times[0];
    for (double t : times) {
        sum += t;
        if (t < mn) mn = t;
        if (t > mx) mx = t;
    }
    double avg = sum / times.size();
    double var = 0.0;
    for (double t : times) var += (t - avg) * (t - avg);
    return {avg, mn, mx, std::sqrt(var / times.size())};
}

// ============================================================================
// HCCL context initialization (MESH or RING)
// ============================================================================
struct GemmHcclContext {
    HcclComm comm{nullptr};
    HcclDeviceContext *deviceCtx{nullptr};
    HcclDeviceContext hostCtx{};
    bool ownsDeviceCtx{false};

    bool Init(int rankId, int nRanks, int deviceId, const HcclRootInfo *rootInfo, rtStream_t hcclStream)
    {
        constexpr int kMaxRetries = 3;
        HcclResult hret = HCCL_SUCCESS;
        for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
            hret = HcclCommInitRootInfo(static_cast<uint32_t>(nRanks), rootInfo,
                                        static_cast<uint32_t>(rankId), &comm);
            if (hret == HCCL_SUCCESS) break;
            std::cerr << "[WARN] Rank " << rankId << ": HcclCommInitRootInfo failed: " << hret
                      << " (attempt " << (attempt + 1) << "/" << kMaxRetries
                      << "), retrying in 5s..." << std::endl;
            sleep(5);
        }
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": HcclCommInitRootInfo failed after "
                      << kMaxRetries << " attempts: " << hret << std::endl;
            return false;
        }

        char group[128] = {};
        hret = HcclGetCommName(comm, group);
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": HcclGetCommName failed: " << hret << std::endl;
            return false;
        }

        CommTopo topoRet = 0;
        hret = HcomGetL0TopoTypeEx(group, &topoRet, COMM_IS_NOT_SET_DEVICE);
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": HcomGetL0TopoTypeEx failed: " << hret << std::endl;
            return false;
        }

        HcclComm commHandle = nullptr;
        hret = HcomGetCommHandleByGroup(group, &commHandle);
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": HcomGetCommHandleByGroup failed: " << hret << std::endl;
            return false;
        }

        CommMpiBarrier();

        gemm_ar_tiling::Mc2CommConfigV2 tiling{};
        memset(&tiling, 0, sizeof(tiling));

        tiling.init.version = 100U;
        tiling.init.mc2HcommCnt = 1U;
        tiling.init.commBlockNum = 48U;
        tiling.init.devType = 4U;
        tiling.init.offset[0] =
            static_cast<uint32_t>(reinterpret_cast<uint64_t>(&tiling.inner) - reinterpret_cast<uint64_t>(&tiling.init));

        tiling.inner.opType = 18U;
        tiling.inner.commEngine = 3U;
        tiling.inner.version = 1U;
        strncpy(tiling.inner.groupName, group, gemm_ar_tiling::TILING_GROUP_NAME_SIZE - 1);
        strncpy(tiling.inner.algConfig, "BatchWrite=level0:fullmesh", gemm_ar_tiling::TILING_ALG_CONFIG_SIZE - 1);

        void *ctxPtr = nullptr;
        hret = HcclAllocComResourceByTiling(commHandle, hcclStream, &tiling, &ctxPtr);
        if (hret != HCCL_SUCCESS || ctxPtr == nullptr) {
            std::cerr << "[ERROR] Rank " << rankId << ": HcclAllocComResourceByTiling failed: " << hret << std::endl;
            return false;
        }

        if (topoRet == COMM_TOPO_MESH) {
            return InitMeshPath(rankId, ctxPtr);
        }
        return InitRingPath(rankId, nRanks, ctxPtr);
    }

    void Finalize()
    {
        if (ownsDeviceCtx && deviceCtx != nullptr) {
            aclrtFree(deviceCtx);
            deviceCtx = nullptr;
        }
        if (comm != nullptr) {
            HcclCommDestroy(comm);
            comm = nullptr;
        }
    }

private:
    bool InitMeshPath(int rankId, void *ctxPtr)
    {
        deviceCtx = reinterpret_cast<HcclDeviceContext *>(ctxPtr);
        aclError aRet = aclrtMemcpy(&hostCtx, sizeof(hostCtx), deviceCtx, sizeof(hostCtx), ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": aclrtMemcpy(deviceCtx) failed: " << (int)aRet << std::endl;
            return false;
        }
        if (rankId == 0) {
            std::cout << "[INFO] HCCL MESH init OK"
                      << " rankId=" << hostCtx.rankId << " rankNum=" << hostCtx.rankNum
                      << " winSize=" << hostCtx.winSize << std::endl;
        }
        return true;
    }

    bool InitRingPath(int rankId, int nRanks, void *ctxPtr)
    {
        using namespace hccl_compat;
        auto *rawCtx = reinterpret_cast<uint8_t *>(ctxPtr);

        HcclOpResParamHead head{};
        const size_t headOff = offsetof(HcclOpResParam, localUsrRankId);
        aclError aRet = aclrtMemcpy(&head, sizeof(head), rawCtx + headOff, sizeof(head), ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": read HcclOpResParam head failed\n";
            return false;
        }

        if (head.rankSize == 0 || head.rankSize > HCCL_MAX_RANK_NUM) {
            std::cerr << "[ERROR] Rank " << rankId << ": invalid rankSize=" << head.rankSize << std::endl;
            return false;
        }

        const size_t remoteResOff = offsetof(HcclOpResParam, remoteRes);
        const size_t remoteResBytes = head.rankSize * sizeof(RemoteResPtr);
        std::vector<RemoteResPtr> remoteResArr(head.rankSize);

        aRet = aclrtMemcpy(remoteResArr.data(), remoteResBytes, rawCtx + remoteResOff, remoteResBytes,
                           ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": read remoteRes failed\n";
            return false;
        }

        memset(&hostCtx, 0, sizeof(hostCtx));

        uint64_t wsFields[2] = {0, 0};
        aRet = aclrtMemcpy(wsFields, sizeof(wsFields), rawCtx, sizeof(wsFields), ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet == ACL_SUCCESS) {
            hostCtx.workSpace = wsFields[0];
            hostCtx.workSpaceSize = wsFields[1];
        }

        hostCtx.rankId = head.localUsrRankId;
        hostCtx.rankNum = head.rankSize;
        hostCtx.winSize = head.winSize;

        for (uint32_t i = 0; i < head.rankSize; ++i) {
            if (i == head.localUsrRankId) {
                hostCtx.windowsIn[i] = head.localWindowsIn;
                continue;
            }

            uint64_t devPtr = remoteResArr[i].nextDevicePtr;
            if (devPtr == 0) {
                std::cerr << "[ERROR] Rank " << rankId << ": remoteRes[" << i << "].nextDevicePtr is null\n";
                return false;
            }

            HcclRankRelationResV2 remoteInfo{};
            aRet = aclrtMemcpy(&remoteInfo, sizeof(remoteInfo), reinterpret_cast<void *>(devPtr), sizeof(remoteInfo),
                               ACL_MEMCPY_DEVICE_TO_HOST);
            if (aRet != ACL_SUCCESS) {
                std::cerr << "[ERROR] Rank " << rankId << ": read remote rank " << i << " info failed\n";
                return false;
            }

            hostCtx.windowsIn[i] = remoteInfo.windowsIn;
        }

        void *newDevMem = nullptr;
        aRet = aclrtMalloc(&newDevMem, sizeof(HcclDeviceContext), ACL_MEM_MALLOC_HUGE_FIRST);
        if (aRet != ACL_SUCCESS || newDevMem == nullptr) {
            std::cerr << "[ERROR] Rank " << rankId << ": aclrtMalloc for RING deviceCtx failed\n";
            return false;
        }

        aRet = aclrtMemcpy(newDevMem, sizeof(HcclDeviceContext), &hostCtx, sizeof(HcclDeviceContext),
                           ACL_MEMCPY_HOST_TO_DEVICE);
        if (aRet != ACL_SUCCESS) {
            aclrtFree(newDevMem);
            std::cerr << "[ERROR] Rank " << rankId << ": copy RING deviceCtx to device failed\n";
            return false;
        }

        deviceCtx = reinterpret_cast<HcclDeviceContext *>(newDevMem);
        ownsDeviceCtx = true;

        if (rankId == 0) {
            std::cout << "[INFO] HCCL RING init OK"
                      << " rankId=" << hostCtx.rankId << " rankNum=" << hostCtx.rankNum
                      << " winSize=" << hostCtx.winSize << std::endl;
        }
        return true;
    }
};

// ============================================================================
// Per-rank execution logic
// ============================================================================
static bool RunGemmAllReducePerRank(int rank_id, int n_ranks, int device_id,
                                    const uint16_t *a_data, size_t a_bytes,
                                    const uint16_t *b_data, size_t b_bytes,
                                    const float *golden, size_t golden_bytes,
                                    const HcclRootInfo *rootInfo)
{
    int status = 0;
    aclrtStream computeStream = nullptr;
    aclrtStream commStream = nullptr;

    status |= aclrtCreateStream(&computeStream);
    status |= aclrtCreateStream(&commStream);

    // ------ HCCL init ------
    rtStream_t hcclStream = nullptr;
    rtStreamCreate(&hcclStream, RT_STREAM_PRIORITY_DEFAULT);

    GemmHcclContext hctx;
    if (!hctx.Init(rank_id, n_ranks, device_id, rootInfo, hcclStream)) {
        std::cerr << "[ERROR] Rank " << rank_id << ": HCCL init failed!\n";
        return false;
    }

    // ------ Allocate memory ------
    // Only buffers that are written remotely (via HcclRemotePtr / TPUT) must
    // reside in the HCCL RDMA window.  gemm_output is only read locally by the
    // comm kernel, so it can live in normal device memory.
    //
    //   recv_buffers   — remote-written in Phase 1 (ReduceScatter)
    //   reduced_output — remote-written in Phase 3 (AllGather)
    //   gemm_output    — local-only (compute kernel writes, comm kernel reads)
    size_t outputSize = static_cast<size_t>(G_M) * G_N * sizeof(float);
    size_t recvBuffersSize = static_cast<size_t>(n_ranks) * outputSize;

    void *gemm_output = nullptr;
    aclrtMalloc(&gemm_output, outputSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!gemm_output) {
        std::cerr << "[ERROR] Rank " << rank_id << ": aclrtMalloc gemm_output failed\n";
        return false;
    }

    uint64_t windowBase = hctx.hostCtx.windowsIn[hctx.hostCtx.rankId];
    size_t winOffset = 0;
    void *recv_buffers = WindowAlloc(windowBase, winOffset, recvBuffersSize);
    void *reduced_output = WindowAlloc(windowBase, winOffset, outputSize);

    if (rank_id == 0) {
        std::cout << "[INFO] HCCL window: winSize=" << hctx.hostCtx.winSize
                  << " used=" << winOffset
                  << " (recv=" << (recvBuffersSize / (1024 * 1024)) << "MB"
                  << ", reduced=" << (outputSize / (1024 * 1024)) << "MB)"
                  << "  gemm_output=" << (outputSize / (1024 * 1024)) << "MB (device mem)" << std::endl;
    }

    if (winOffset > hctx.hostCtx.winSize) {
        std::cerr << "[ERROR] Rank " << rank_id << ": HCCL window too small! need=" << winOffset
                  << " have=" << hctx.hostCtx.winSize << std::endl;
        aclrtFree(gemm_output);
        return false;
    }

    aclrtMemset(gemm_output, outputSize, 0, outputSize);
    aclrtMemset(recv_buffers, recvBuffersSize, 0, recvBuffersSize);
    aclrtMemset(reduced_output, outputSize, 0, outputSize);

    uint32_t k_per_rank = G_K / n_ranks;
    if (G_K % n_ranks != 0) {
        std::cerr << "[ERROR] K=" << G_K << " not divisible by nranks=" << n_ranks << "\n";
        return false;
    }

    size_t aSize = G_M * k_per_rank * sizeof(uint16_t);
    size_t bSize = k_per_rank * G_N * sizeof(uint16_t);
    void *src0_dev = nullptr, *src1_dev = nullptr;
    aclrtMalloc(&src0_dev, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&src1_dev, bSize, ACL_MEM_MALLOC_HUGE_FIRST);

    int tiles_per_block = (G_NUM_TILES + COMPUTE_BLOCK_NUM - 1) / COMPUTE_BLOCK_NUM;
    size_t queueSetSize = MultiBlockQueueSetSize(COMPUTE_BLOCK_NUM, tiles_per_block);
    void *queueSet_dev = nullptr;
    aclrtMalloc(&queueSet_dev, queueSetSize, ACL_MEM_MALLOC_HUGE_FIRST);

    MultiBlockQueueSet *queueSet_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void **>(&queueSet_host), queueSetSize);
    MultiBlockQueueSetInit(queueSet_host, COMPUTE_BLOCK_NUM, G_NUM_TILES);
    aclrtMemcpy(queueSet_dev, queueSetSize, queueSet_host, queueSetSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtFreeHost(queueSet_host);

    // ------ Upload input data to device ------
    if (rank_id == 0) {
        std::cout << "[INFO] Data Parallel: A_part[" << G_M << "x" << k_per_rank
                  << "], B_part[" << k_per_rank << "x" << G_N << "]" << std::endl;
    }

    aclrtMemcpy(src0_dev, aSize, a_data, a_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1_dev, bSize, b_data, b_bytes, ACL_MEMCPY_HOST_TO_DEVICE);

    HcclHostBarrier(hctx.comm, commStream);

    // ------ Helpers ------
    MultiBlockQueueSet *queueSet_reset_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void **>(&queueSet_reset_host), queueSetSize);

    auto resetState = [&]() {
        MultiBlockQueueSetInit(queueSet_reset_host, COMPUTE_BLOCK_NUM, G_NUM_TILES);
        aclrtMemcpy(queueSet_dev, queueSetSize, queueSet_reset_host, queueSetSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemset(gemm_output, outputSize, 0, outputSize);
        aclrtMemset(recv_buffers, recvBuffersSize, 0, recvBuffersSize);
        aclrtMemset(reduced_output, outputSize, 0, outputSize);
    };

    auto launchCompute = [&](aclrtStream s) {
        launchGemmCompute(
            reinterpret_cast<uint8_t *>(gemm_output),
            reinterpret_cast<uint8_t *>(src0_dev),
            reinterpret_cast<uint8_t *>(src1_dev),
            reinterpret_cast<uint8_t *>(queueSet_dev),
            rank_id, s, COMPUTE_BLOCK_NUM, k_per_rank);
    };

    uint8_t *hcclCtxPtr = reinterpret_cast<uint8_t *>(hctx.deviceCtx);

    auto launchComm = [&](aclrtStream s) {
        launchGemmCommRS(
            reinterpret_cast<uint8_t *>(gemm_output),
            reinterpret_cast<uint8_t *>(recv_buffers),
            reinterpret_cast<uint8_t *>(queueSet_dev),
            hcclCtxPtr,
            rank_id, n_ranks, s, COMPUTE_BLOCK_NUM);
        HcclHostBarrier(hctx.comm, s);

        launchGemmCommReduce(
            reinterpret_cast<uint8_t *>(gemm_output),
            reinterpret_cast<uint8_t *>(recv_buffers),
            reinterpret_cast<uint8_t *>(reduced_output),
            hcclCtxPtr,
            rank_id, n_ranks, s);
        HcclHostBarrier(hctx.comm, s);

        launchGemmCommAG(
            reinterpret_cast<uint8_t *>(reduced_output),
            hcclCtxPtr,
            rank_id, n_ranks, s);
        HcclHostBarrier(hctx.comm, s);
    };

    auto syncAll = [&]() {
        aclrtSynchronizeStream(computeStream);
        aclrtSynchronizeStream(commStream);
        HcclHostBarrier(hctx.comm, commStream);
    };

    // ------ Warmup ------
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        resetState();
        syncAll();
        launchCompute(computeStream);
        launchComm(commStream);
        syncAll();
    }

    // ------ Compute-only measurement ------
    std::vector<double> compute_times_us;
    for (int iter = 0; iter < COMPUTE_ONLY_ITERS; ++iter) {
        resetState();
        aclrtSynchronizeStream(computeStream);
        HcclHostBarrier(hctx.comm, commStream);

        auto t0 = std::chrono::high_resolution_clock::now();
        launchCompute(computeStream);
        aclrtSynchronizeStream(computeStream);
        auto t1 = std::chrono::high_resolution_clock::now();
        compute_times_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        HcclHostBarrier(hctx.comm, commStream);
    }

    // ------ Sequential measurement (compute -> comm, no overlap) ------
    std::vector<double> sequential_times_us, seq_compute_us, seq_comm_us;
    for (int iter = 0; iter < MEASURE_ITERS; ++iter) {
        resetState();
        syncAll();

        auto t0 = std::chrono::high_resolution_clock::now();
        launchCompute(computeStream);
        aclrtSynchronizeStream(computeStream);
        auto t1 = std::chrono::high_resolution_clock::now();
        launchComm(commStream);
        auto t2 = std::chrono::high_resolution_clock::now();

        seq_compute_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        seq_comm_us.push_back(std::chrono::duration<double, std::micro>(t2 - t1).count());
        sequential_times_us.push_back(std::chrono::duration<double, std::micro>(t2 - t0).count());
        HcclHostBarrier(hctx.comm, commStream);
    }

    // ------ Pipelined measurement (compute || comm, with overlap) ------

    aclrtEvent evComputeStart = nullptr, evComputeEnd = nullptr;
    aclrtCreateEvent(&evComputeStart);
    aclrtCreateEvent(&evComputeEnd);

    std::vector<double> pipelined_times_us, pipe_compute_us, pipe_comm_us;
    for (int iter = 0; iter < MEASURE_ITERS; ++iter) {
        resetState();
        syncAll();

        auto t0 = std::chrono::high_resolution_clock::now();

        aclrtRecordEvent(evComputeStart, computeStream);
        launchCompute(computeStream);
        aclrtRecordEvent(evComputeEnd, computeStream);

        launchComm(commStream);
        aclrtSynchronizeStream(computeStream);

        auto t1 = std::chrono::high_resolution_clock::now();

        float compute_ms = 0.0f;
        aclrtEventElapsedTime(&compute_ms, evComputeStart, evComputeEnd);

        double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

        pipe_compute_us.push_back((double)compute_ms * 1000.0);
        pipe_comm_us.push_back(total_us);
        pipelined_times_us.push_back(total_us);
    }

    aclrtDestroyEvent(evComputeStart);
    aclrtDestroyEvent(evComputeEnd);

    // ------ Final run for verification ------
    resetState();
    syncAll();
    launchCompute(computeStream);
    launchComm(commStream);
    syncAll();

    // ------ Verification ------
    float *output_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void **>(&output_host), outputSize);
    aclrtMemcpy(output_host, outputSize, reduced_output, outputSize, ACL_MEMCPY_DEVICE_TO_HOST);

    bool is_ok = true;
    if (rank_id == 0) {
        const float eps = 0.001f;
        const size_t valid_elements = (size_t)G_ORIG_M * G_ORIG_N;
        float max_diff = 0.0f, max_diff_ratio = 0.0f;
        size_t err_count = 0, zero_count = 0;
        const size_t err_threshold = static_cast<size_t>(valid_elements * eps);
        const size_t zero_threshold = 0x1000;

        for (size_t row = 0; row < G_ORIG_M; ++row) {
            for (size_t col = 0; col < G_ORIG_N; ++col) {
                size_t padded_idx = row * G_N + col;
                size_t golden_idx = row * G_N + col;
                float exp_val = golden[golden_idx], act_val = output_host[padded_idx];
                float diff = std::abs(exp_val - act_val);
                float rel = (std::abs(exp_val) > 1e-6f) ? (diff / std::abs(exp_val)) : diff;
                if (diff > max_diff) max_diff = diff;
                if (rel > max_diff_ratio) max_diff_ratio = rel;
                if (std::abs(act_val) <= 1e-6f && std::abs(exp_val) > 1e-6f) zero_count++;
                if ((diff > eps && rel > eps) || zero_count > zero_threshold) err_count++;
            }
        }

        is_ok = (err_count <= err_threshold) && (zero_count <= zero_threshold);
        std::cout << "[VERIFY] valid_region=" << G_ORIG_M << "x" << G_ORIG_N
                  << " max_diff=" << max_diff << " max_ratio=" << max_diff_ratio
                  << " err=" << err_count << "/" << err_threshold
                  << " zeros=0x" << std::hex << zero_count << std::dec
                  << " -> " << (is_ok ? "PASS" : "FAIL") << std::endl;
    }

    aclrtFreeHost(output_host);
    aclrtFreeHost(queueSet_reset_host);

    // ------ Performance report ------
    if (rank_id == 0) {
        PerfStats comp_s = calcStats(compute_times_us);
        PerfStats seq_s  = calcStats(sequential_times_us);
        PerfStats pipe_s = calcStats(pipelined_times_us);
        PerfStats seq_comp_s = calcStats(seq_compute_us);
        PerfStats seq_comm_s = calcStats(seq_comm_us);
        PerfStats pipe_comp_s = calcStats(pipe_compute_us);
        PerfStats pipe_comm_s = calcStats(pipe_comm_us);

        double flops_per_rank = 2.0 * G_ORIG_M * (double)k_per_rank * G_ORIG_N;
        double flops_total    = 2.0 * G_ORIG_M * (double)G_K * G_ORIG_N;
        auto gflops = [](double flops, double us) { return (us > 0) ? (flops / (us * 1e-6) / 1e9) : 0.0; };

        size_t tileBytes = static_cast<size_t>(G_BASE_M) * G_BASE_N * sizeof(float);
        int tiles_per_owner = (G_NUM_TILES + n_ranks - 1) / n_ranks;
        double rs_bytes = static_cast<double>(G_NUM_TILES - tiles_per_owner) * tileBytes;
        double ag_bytes = static_cast<double>(tiles_per_owner) * (n_ranks - 1) * tileBytes;
        double data_gb = (rs_bytes + ag_bytes) / (1024.0 * 1024.0 * 1024.0);
        auto bw_gbs = [&](double us) { return (us > 0) ? ((rs_bytes + ag_bytes) / (us * 1e-6) / (1024.0*1024.0*1024.0)) : 0.0; };

        double speedup = seq_s.avg / pipe_s.avg;
        double overlap_time = (seq_comp_s.avg + seq_comm_s.avg) - pipe_s.avg;
        double overlap_eff = (overlap_time > 0) ? (overlap_time / std::min(seq_comp_s.avg, seq_comm_s.avg) * 100.0) : 0.0;

        std::cout << std::fixed << std::setprecision(1);
        std::cout << "\n================================================================" << std::endl;
        std::cout << (is_ok ? "[SUCCESS]" : "[FAILED]") << " GEMM AllReduce (HCCL)" << std::endl;
        std::cout << "  M=" << G_ORIG_M << " K=" << G_K << " N=" << G_ORIG_N;
        if (G_M != G_ORIG_M || G_N != G_ORIG_N)
            std::cout << "  (padded " << G_M << "x" << G_K << "x" << G_N << ")";
        std::cout << "  ranks=" << n_ranks
                  << "  compute_blocks=" << COMPUTE_BLOCK_NUM
                  << "  comm_blocks=" << COMM_BLOCK_NUM << std::endl;
        std::cout << "  tiles=" << G_NUM_TILES << " (" << G_M_TILES << "x" << G_N_TILES << ")"
                  << "  comm_data=" << std::setprecision(3) << data_gb << " GB/rank" << std::endl;

        std::cout << "\n  Compute-only:   " << std::setprecision(1) << comp_s.avg << " us"
                  << "  (" << std::setprecision(0) << gflops(flops_per_rank, comp_s.avg) << " GFLOPS)" << std::endl;

        std::cout << "\n  Sequential:     " << std::setprecision(1) << seq_s.avg << " us" << std::endl;
        std::cout << "    compute:      " << seq_comp_s.avg << " us"
                  << "  (" << std::setprecision(0) << gflops(flops_per_rank, seq_comp_s.avg) << " GFLOPS)" << std::endl;
        std::cout << "    comm:         " << std::setprecision(1) << seq_comm_s.avg << " us"
                  << "  (" << std::setprecision(1) << bw_gbs(seq_comm_s.avg) << " GB/s)" << std::endl;

        std::cout << "\n  Pipelined:      " << std::setprecision(1) << pipe_s.avg << " us" << std::endl;
        std::cout << "    compute done: " << pipe_comp_s.avg << " us"
                  << "  (" << std::setprecision(0) << gflops(flops_per_rank, pipe_comp_s.avg) << " GFLOPS, "
                  << std::setprecision(1) << (gflops(flops_per_rank, pipe_comp_s.avg) / gflops(flops_per_rank, comp_s.avg) * 100.0)
                  << "% of pure)" << std::endl;
        std::cout << "    comm done:    " << std::setprecision(1) << pipe_comm_s.avg << " us"
                  << "  (" << std::setprecision(1) << bw_gbs(pipe_comm_s.avg) << " GB/s)" << std::endl;

        std::cout << "\n  Speedup:        " << std::setprecision(3) << speedup << "x" << std::endl;
        std::cout << "  Time saved:     " << std::setprecision(1) << (seq_s.avg - pipe_s.avg) << " us"
                  << " (" << std::setprecision(1) << ((seq_s.avg - pipe_s.avg) / seq_s.avg * 100.0) << "%)" << std::endl;
        std::cout << "  Overlap eff:    " << std::setprecision(1) << overlap_eff << "%" << std::endl;
        std::cout << "  Throughput:     " << std::setprecision(0) << gflops(flops_total, pipe_s.avg) << " GFLOPS (total)" << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }

    // ------ Cleanup ------
    aclrtFree(gemm_output);
    aclrtFree(src0_dev);
    aclrtFree(src1_dev);
    aclrtFree(queueSet_dev);

    hctx.Finalize();
    if (hcclStream) rtStreamDestroy(hcclStream);

    status |= aclrtDestroyStream(computeStream);
    status |= aclrtDestroyStream(commStream);

    return (status == 0) && is_ok;
}

// ============================================================================
// MPI-based multi-process launcher
// ============================================================================
static bool RunGemmAllReduce(int n_ranks, int first_device_id,
                             const uint16_t *a_parts,
                             const uint16_t *b_parts,
                             const float *golden)
{
    if (n_ranks <= 0 || n_ranks > 8) {
        std::cerr << "[ERROR] Invalid n_ranks: " << n_ranks << " (must be 1-8)\n";
        return false;
    }

    int mpiRank = CommMpiRank();

    uint32_t k_per_rank = G_K / n_ranks;
    size_t a_rank_bytes = (size_t)G_M * k_per_rank * sizeof(uint16_t);
    size_t b_rank_bytes = (size_t)G_N * k_per_rank * sizeof(uint16_t);
    size_t golden_bytes = (size_t)G_M * G_N * sizeof(float);
    size_t a_rank_elems = (size_t)G_M * k_per_rank;
    size_t b_rank_elems = (size_t)G_N * k_per_rank;

    if (mpiRank == 0) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "  GEMM AllReduce (ReduceScatter + AllGather) — HCCL backend" << std::endl;
        std::cout << "  M=" << G_ORIG_M << " K=" << G_K << " N=" << G_ORIG_N;
        if (G_M != G_ORIG_M || G_N != G_ORIG_N)
            std::cout << "  (padded " << G_M << "x" << G_N << ")";
        std::cout << "  tile=" << G_BASE_M << "x" << G_BASE_K << "x" << G_BASE_N
                  << "  tiles=" << G_NUM_TILES << std::endl;
        std::cout << "  ranks=" << n_ranks
                  << "  devices=[" << first_device_id << "," << (first_device_id + n_ranks) << ")"
                  << "  compute_blocks=" << COMPUTE_BLOCK_NUM
                  << "  comm_blocks=" << COMM_BLOCK_NUM << std::endl;
        std::cout << "================================================================" << std::endl;
    }

    int device_id = mpiRank % n_ranks + first_device_id;

    constexpr int kAclRepeatInit = 100002;
    aclError aRet = aclInit(nullptr);
    if (aRet != ACL_SUCCESS && static_cast<int>(aRet) != kAclRepeatInit) {
        std::cerr << "[ERROR] Rank " << mpiRank << ": aclInit failed: " << (int)aRet << std::endl;
        return false;
    }

    if (mpiRank == 0) {
        rtSetDevice(device_id);
    }

    aRet = aclrtSetDevice(device_id);
    if (aRet != ACL_SUCCESS) {
        std::cerr << "[ERROR] Rank " << mpiRank << ": aclrtSetDevice(" << device_id << ") failed\n";
        return false;
    }

    HcclRootInfo rootInfo{};
    if (mpiRank == 0) {
        constexpr int kMaxRetries = 3;
        HcclResult hret = HCCL_SUCCESS;
        for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
            hret = HcclGetRootInfo(&rootInfo);
            if (hret == HCCL_SUCCESS) break;
            std::cerr << "[WARN] HcclGetRootInfo failed: " << hret
                      << " (attempt " << (attempt + 1) << "/" << kMaxRetries
                      << "), retrying in 5s..." << std::endl;
            sleep(5);
        }
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] HcclGetRootInfo failed after " << kMaxRetries
                      << " attempts: " << hret << std::endl;
            return false;
        }
    }

    CommMpiBcast(&rootInfo, HCCL_ROOT_INFO_BYTES, COMM_MPI_CHAR, 0);
    CommMpiBarrier();

    const uint16_t *a_rank = a_parts + (size_t)mpiRank * a_rank_elems;
    const uint16_t *b_rank = b_parts + (size_t)mpiRank * b_rank_elems;

    bool ok = RunGemmAllReducePerRank(mpiRank, n_ranks, device_id,
                                      a_rank, a_rank_bytes,
                                      b_rank, b_rank_bytes,
                                      golden, golden_bytes,
                                      &rootInfo);

    CommMpiBarrier();

    aclrtResetDevice(device_id);
    aclFinalize();

    return ok;
}

// ============================================================================
// Data generation helpers
// ============================================================================

static uint16_t floatToHalf(float f)
{
    union { float f; uint32_t u; } bits;
    bits.f = f;
    uint32_t x = bits.u;
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x007FFFFF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static void computeGolden(const float *A, const float *B, float *C, int M, int K, int N)
{
    std::memset(C, 0, (size_t)M * N * sizeof(float));

    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    unsigned n_threads = std::min(hw, 64u);

    int rows_per = (M + (int)n_threads - 1) / (int)n_threads;
    std::vector<std::thread> threads;

    for (unsigned t = 0; t < n_threads; t++) {
        int r0 = (int)t * rows_per;
        int r1 = std::min(r0 + rows_per, M);
        if (r0 >= M) break;

        threads.emplace_back([A, B, C, K, N, r0, r1]() {
            constexpr int BLK = 64;
            for (int i = r0; i < r1; i++)
                for (int kk = 0; kk < K; kk += BLK) {
                    int kEnd = std::min(kk + BLK, K);
                    for (int jj = 0; jj < N; jj += BLK) {
                        int jEnd = std::min(jj + BLK, N);
                        for (int k = kk; k < kEnd; k++) {
                            float aik = A[(size_t)i * K + k];
                            for (int j = jj; j < jEnd; j++)
                                C[(size_t)i * N + j] += aik * B[(size_t)k * N + j];
                        }
                    }
                }
        });
    }
    for (auto &th : threads) th.join();
}

static bool generateData(int nranks,
                          std::vector<uint16_t> &a_parts,
                          std::vector<uint16_t> &b_parts,
                          std::vector<float> &golden)
{
    int k_per_rank = G_K / nranks;
    if (G_K % nranks != 0) {
        fprintf(stderr, "[ERROR] K=%d not divisible by nranks=%d\n", G_K, nranks);
        return false;
    }

    printf("Data Parallel: K=%d split into %d ranks, %d per rank\n", G_K, nranks, k_per_rank);
    if (G_M != G_ORIG_M || G_N != G_ORIG_N) {
        printf("  Padded: M %d->%d, N %d->%d (tile alignment)\n",
               G_ORIG_M, G_M, G_ORIG_N, G_N);
    }

    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(1, 4);

    size_t A_orig_elems = (size_t)G_ORIG_M * G_K;
    size_t B_orig_elems = (size_t)G_K * G_ORIG_N;
    std::vector<float> A_fp32(A_orig_elems), B_fp32(B_orig_elems);

    for (auto &v : A_fp32) v = (float)dist(gen);
    for (auto &v : B_fp32) v = (float)dist(gen);

    printf("  Computing golden reference (CPU GEMM %d×%d×%d)...\n", G_ORIG_M, G_K, G_ORIG_N);
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<float> golden_orig((size_t)G_ORIG_M * G_ORIG_N);
    computeGolden(A_fp32.data(), B_fp32.data(), golden_orig.data(), G_ORIG_M, G_K, G_ORIG_N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    printf("  Golden computed in %.1f s\n", secs);

    golden.assign((size_t)G_M * G_N, 0.0f);
    for (int i = 0; i < (int)G_ORIG_M; i++)
        std::memcpy(&golden[(size_t)i * G_N],
                    &golden_orig[(size_t)i * G_ORIG_N],
                    G_ORIG_N * sizeof(float));

    size_t a_rank_elems = (size_t)G_M * k_per_rank;
    size_t b_rank_elems = (size_t)G_N * k_per_rank;
    a_parts.assign((size_t)nranks * a_rank_elems, 0);
    b_parts.assign((size_t)nranks * b_rank_elems, 0);

    for (int r = 0; r < nranks; r++) {
        int k_start = r * k_per_rank;
        uint16_t *a_dst = a_parts.data() + (size_t)r * a_rank_elems;
        uint16_t *b_dst = b_parts.data() + (size_t)r * b_rank_elems;

        for (int i = 0; i < (int)G_ORIG_M; i++)
            for (int j = 0; j < k_per_rank; j++)
                a_dst[(size_t)i * k_per_rank + j] =
                    floatToHalf(A_fp32[(size_t)i * G_K + k_start + j]);

        for (int i = 0; i < (int)G_ORIG_N; i++)
            for (int j = 0; j < k_per_rank; j++)
                b_dst[(size_t)i * k_per_rank + j] =
                    floatToHalf(B_fp32[(size_t)(k_start + j) * G_ORIG_N + i]);

        printf("  Rank %d: A[%d×%d] cols[%d:%d], B[%d×%d] rows[%d:%d]\n",
               r, G_M, k_per_rank, k_start, k_start + k_per_rank,
               k_per_rank, G_N, k_start, k_start + k_per_rank);
    }

    double gsum = 0.0;
    for (auto v : golden_orig) gsum += v;
    printf("  Golden: shape=(%d, %d), sum=%.2f\n", G_ORIG_M, G_ORIG_N, gsum);
    return true;
}

// ============================================================================
// Entry point
// ============================================================================

static int parseFirstDevice(int argc, char *argv[])
{
    const char *envVal = getenv("GEMM_ALLREDUCE_FIRST_DEVICE");
    if (envVal != nullptr) {
        int val = atoi(envVal);
        if (val >= 0) return val;
    }
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "--first-device") == 0) {
            int val = atoi(argv[i + 1]);
            if (val >= 0) return val;
        }
    }
    return 0;
}

int main(int argc, char *argv[])
{
    if (!CommMpiInit(&argc, &argv)) {
        fprintf(stderr, "[ERROR] MPI_Init failed. Launch with: mpirun -n <NRANKS> ./gemm_allreduce\n");
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: mpirun -n <NRANKS> %s [--first-device ID]\n", argv[0]);
            CommMpiFinalize();
            return 0;
        }
    }

    int n_ranks = CommMpiSize();
    int first_device_id = parseFirstDevice(argc, argv);

    if (CommMpiRank() == 0) {
        printf("GEMM AllReduce (HCCL): ranks=%d, devices=[%d, %d)\n\n",
               n_ranks, first_device_id, first_device_id + n_ranks);
    }

    int k_per_rank = G_K / n_ranks;
    size_t a_total = (size_t)n_ranks * G_M * k_per_rank;
    size_t b_total = (size_t)n_ranks * G_N * k_per_rank;
    size_t g_total = (size_t)G_M * G_N;

    std::vector<uint16_t> a_parts(a_total), b_parts(b_total);
    std::vector<float> golden(g_total);

    int data_ok = 0;
    if (CommMpiRank() == 0) {
        if (!generateData(n_ranks, a_parts, b_parts, golden)) {
            data_ok = 1;
        }
    }

    CommMpiBcast(&data_ok, 1, COMM_MPI_INT, 0);
    if (data_ok != 0) {
        CommMpiFinalize();
        return 1;
    }

    CommMpiBcast(a_parts.data(), (int)(a_total * sizeof(uint16_t)), COMM_MPI_CHAR, 0);
    CommMpiBcast(b_parts.data(), (int)(b_total * sizeof(uint16_t)), COMM_MPI_CHAR, 0);
    CommMpiBcast(golden.data(), (int)(g_total * sizeof(float)), COMM_MPI_CHAR, 0);

    if (CommMpiRank() == 0) {
        printf("  Broadcast input data to all %d ranks.\n", n_ranks);
    }

    bool ok = RunGemmAllReduce(n_ranks, first_device_id,
                               a_parts.data(), b_parts.data(), golden.data());

    if (CommMpiRank() == 0) {
        printf(ok ? "\nGEMM AllReduce demo completed successfully.\n"
                  : "\nGEMM AllReduce demo FAILED.\n");
    }

    CommMpiFinalize();
    return ok ? 0 : 1;
}
