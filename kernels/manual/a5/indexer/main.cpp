/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "acl/acl.h"
#include "test_common.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>
using namespace std;
using namespace PtoTestCommon;

#ifndef INDEXER_TEST_N
#define INDEXER_TEST_N 2048
#endif
#ifndef INDEXER_TOPK
#define INDEXER_TOPK 512
#endif

void LaunchIndexerMxfp8(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3, float *postScale,
                        float *scoreOut, uint16_t *scoreOutBf16, uint32_t *outIdx, void *stream);
void LaunchIndexerMatmul(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3, float *postScale,
                         float *scoreOut, uint16_t *scoreOutBf16, uint32_t *outIdx, void *stream);
void LaunchIndexerPostProcess(float *matmulOut, float *postScale, float *scoreOut, uint16_t *scoreOutBf16, void *stream);
void LaunchIndexerTopK(uint16_t *scoreOutBf16, uint32_t *outIdx, void *stream);

enum class RunProcess : uint32_t {
    kAll = 0,
    kMatmul = 1,
    kPostProcess = 2,
    kTopK = 3,
};

RunProcess ParseProcess(int argc, char **argv)
{
    if (argc <= 1) {
        return RunProcess::kAll;
    }
    string arg = argv[1];
    if (arg == "all" || arg == "0" || arg == "--process=all" || arg == "--process=0") {
        return RunProcess::kAll;
    }
    if (arg == "1" || arg == "matmul" || arg == "--process=1" || arg == "--process=matmul") {
        return RunProcess::kMatmul;
    }
    if (arg == "2" || arg == "postprocess" || arg == "--process=2" || arg == "--process=postprocess") {
        return RunProcess::kPostProcess;
    }
    if (arg == "3" || arg == "topk" || arg == "--process=3" || arg == "--process=topk") {
        return RunProcess::kTopK;
    }
    printf("Unknown process arg: %s\n", arg.c_str());
    printf("Usage: ./indexer [all|1|2|3|matmul|postprocess|topk|--process=<...>]\n");
    return RunProcess::kAll;
}

template <typename T>
void VerifyMatmul(size_t matmulFileSize)
{
    std::vector<T> golden(matmulFileSize / sizeof(T));
    std::vector<T> dev(matmulFileSize / sizeof(T));
    ReadFile("../output/golden_matmul.bin", matmulFileSize, golden.data(), matmulFileSize);
    ReadFile("../output/output_z.bin", matmulFileSize, dev.data(), matmulFileSize);
    bool ret = ResultCmp(golden, dev, 0.05f);
    if (ret) {
        printf("matmul test success\n");
    } else {
        printf("matmul test failed\n");
    }
}

template <typename T>
void VerifyScore(size_t scoreFileSize)
{
    std::vector<T> golden(scoreFileSize / sizeof(T));
    std::vector<T> devFinal(scoreFileSize / sizeof(T));
    ReadFile("../output/golden_score.bin", scoreFileSize, golden.data(), scoreFileSize);
    ReadFile("../output/output_score.bin", scoreFileSize, devFinal.data(), scoreFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.05f);
    if (ret) {
        printf("score test success\n");
    } else {
        printf("score test failed\n");
    }
}

template <typename T>
uint32_t OrderedFloatKey(T v)
{
    static_assert(sizeof(T) == sizeof(uint32_t), "OrderedFloatKey expects 32-bit float-like type.");
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    if ((bits & 0x80000000u) != 0u) {
        return ~bits;
    }
    return bits ^ 0x80000000u;
}

template <typename T>
uint16_t FloatToBf16Bits(T v)
{
    static_assert(sizeof(T) == sizeof(uint32_t), "FloatToBf16Bits expects 32-bit float-like type.");
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

template <typename T>
void BuildScoreOrderedKeys(const T *score, uint32_t *scoreKey, uint32_t batch, uint32_t length)
{
    for (uint32_t b = 0; b < batch; ++b) {
        const T *scorePtr = score + static_cast<uint64_t>(b) * length;
        uint32_t *keyPtr = scoreKey + static_cast<uint64_t>(b) * length;
        for (uint32_t i = 0; i < length; ++i) {
            keyPtr[i] = OrderedFloatKey(scorePtr[i]);
        }
    }
}

template <typename T>
void TopKIndices(const T *score, uint32_t *outIdx, uint32_t length, uint32_t topk)
{
    std::vector<uint32_t> indices(length);
    std::iota(indices.begin(), indices.end(), 0u);
    auto cmp = [&](uint32_t lhs, uint32_t rhs) {
        uint32_t kL = OrderedFloatKey(score[lhs]);
        uint32_t kR = OrderedFloatKey(score[rhs]);
        if (kL != kR) {
            return kL > kR;
        }
        return lhs < rhs;
    };
    std::nth_element(indices.begin(), indices.begin() + topk, indices.end(), cmp);
    std::sort(indices.begin(), indices.begin() + topk, cmp);
    indices.resize(topk);
    for (uint32_t i = 0; i < topk; ++i) {
        outIdx[i] = indices[i];
    }
}

template <typename T>
uint32_t BuildTopkValueMultiset(const T *score, const uint32_t *idx, T *values, uint32_t topk, uint32_t length)
{
    uint32_t invalidIdxCount = 0;
    for (uint32_t i = 0; i < topk; ++i) {
        uint32_t id = idx[i];
        if (id >= length) {
            values[i] = static_cast<T>(0);
            ++invalidIdxCount;
        } else {
            values[i] = score[id];
        }
    }
    std::sort(values, values + topk);
    return invalidIdxCount;
}

template <typename T>
void VerifyTopkMultiset(size_t multisetFileSize)
{
    std::vector<T> golden(multisetFileSize / sizeof(T));
    std::vector<T> dev(multisetFileSize / sizeof(T));
    ReadFile("../output/golden_topk_multiset.bin", multisetFileSize, golden.data(), multisetFileSize);
    ReadFile("../output/output_topk_multiset.bin", multisetFileSize, dev.data(), multisetFileSize);

    bool ret = ResultCmp(golden, dev, 0.05f);
    if (ret) {
        printf("topk multiset test success\n");
    } else {
        printf("topk multiset test failed\n");
    }
}

void VerifyTopkIndex(size_t idxFileSize)
{
    std::vector<uint32_t> golden(idxFileSize / sizeof(uint32_t));
    std::vector<uint32_t> dev(idxFileSize / sizeof(uint32_t));
    ReadFile("../output/golden_topk_idx.bin", idxFileSize, golden.data(), idxFileSize);
    ReadFile("../output/output_idx.bin", idxFileSize, dev.data(), idxFileSize);

    bool rawOrderOk = (golden == dev);
    if (rawOrderOk) {
        printf("topk idx test success\n");
        return;
    }

    // Order-insensitive compare: sort indices within each batch, then compare.
    constexpr uint32_t kBatch = 2;
    constexpr uint32_t kTopK = INDEXER_TOPK;
    if (idxFileSize != static_cast<size_t>(kBatch) * kTopK * sizeof(uint32_t)) {
        printf("topk idx test failed (unexpected idx file size for sorted-compare)\n");
        return;
    }

    std::vector<uint32_t> goldenSorted = golden;
    std::vector<uint32_t> devSorted = dev;
    for (uint32_t b = 0; b < kBatch; ++b) {
        uint32_t *gBegin = goldenSorted.data() + static_cast<uint64_t>(b) * kTopK;
        uint32_t *dBegin = devSorted.data() + static_cast<uint64_t>(b) * kTopK;
        std::sort(gBegin, gBegin + kTopK);
        std::sort(dBegin, dBegin + kTopK);
    }

    bool sortedOk = (goldenSorted == devSorted);
    if (sortedOk) {
        printf("topk idx order differs, but sorted idx set matches\n");
    } else {
        printf("topk idx test failed\n");
    }
}

void VerifyScoreKey(size_t keyFileSize)
{
    std::vector<uint32_t> golden(keyFileSize / sizeof(uint32_t));
    std::vector<uint32_t> dev(keyFileSize / sizeof(uint32_t));
    ReadFile("../output/golden_score_key.bin", keyFileSize, golden.data(), keyFileSize);
    ReadFile("../output/output_score_key.bin", keyFileSize, dev.data(), keyFileSize);

    bool ok = (golden == dev);
    if (ok) {
        printf("score ordered-key test success\n");
    } else {
        printf("score ordered-key test failed\n");
    }
}

template <typename T, typename U, typename X, uint32_t blockDim, uint32_t m, uint32_t k, uint32_t n,
          uint32_t singleCoreM, uint32_t singleCoreK, uint32_t singleCoreN, uint32_t baseM, uint32_t baseK,
          uint32_t baseN, uint32_t stepM, uint32_t stepKa, uint32_t stepKb, uint32_t stepN>
void MxMatmul(RunProcess runProcess)
{
    size_t aFileSize = m * k * sizeof(U); // uint8_t represent fp8
    size_t bFileSize = k * n * sizeof(U);
    int sacleFactor = 32;
    size_t aScaleFileSize = m * k / sacleFactor * sizeof(X);
    size_t bScaleFileSize = k / sacleFactor * n * sizeof(X);
    size_t cFileSize = m * n * sizeof(T);
    constexpr uint32_t kBatch = 2;
    size_t postScaleFileSize = m * sizeof(float);       // [2, 64]
    size_t scoreFileSize = kBatch * n * sizeof(T);      // [2, n]
    size_t scoreBf16FileSize = kBatch * n * sizeof(uint16_t);
    constexpr uint32_t kTopK = INDEXER_TOPK;
    size_t postScaleDeviceBytes = postScaleFileSize;
    size_t scoreKeyFileSize = kBatch * n * sizeof(uint32_t);
    size_t outIdxFileSize = kBatch * kTopK * sizeof(uint32_t);
    size_t topkMultisetFileSize = kBatch * kTopK * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host, *src2Host, *src3Host, *postScaleHost;
    uint8_t *dstDevice, *src0Device, *src1Device, *src2Device, *src3Device;
    float *postScaleDevice = nullptr;
    float *scoreDevice = nullptr;
    uint16_t *scoreBf16Device = nullptr;
    uint32_t *outIdxDevice = nullptr;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);
    aclrtMallocHost((void **)(&src2Host), aScaleFileSize);
    aclrtMallocHost((void **)(&src3Host), bScaleFileSize);
    aclrtMallocHost((void **)(&postScaleHost), postScaleFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src2Device, aScaleFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src3Device, bScaleFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&postScaleDevice, postScaleDeviceBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&scoreDevice, scoreFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&scoreBf16Device, scoreBf16FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&outIdxDevice, outIdxFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    std::vector<T> scoreHost(scoreFileSize / sizeof(T));
    std::vector<uint16_t> scoreBf16Host(scoreBf16FileSize / sizeof(uint16_t));
    std::vector<uint32_t> outIdxHost(outIdxFileSize / sizeof(uint32_t));
    std::vector<uint32_t> scoreKeyHost(scoreKeyFileSize / sizeof(uint32_t));
    std::vector<T> topkMultisetHost(topkMultisetFileSize / sizeof(T));

    if (runProcess == RunProcess::kAll || runProcess == RunProcess::kMatmul) {
        ReadFile("../input/x1_gm.bin", aFileSize, src0Host, aFileSize);
        ReadFile("../input/x2_gm.bin", bFileSize, src1Host, bFileSize);
        ReadFile("../input/x1_scale_gm.bin", aScaleFileSize, src2Host, aScaleFileSize);
        ReadFile("../input/x2_scale_gm.bin", bScaleFileSize, src3Host, bScaleFileSize);
        ReadFile("../input/post_scale.bin", postScaleFileSize, postScaleHost, postScaleFileSize);

        aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(src2Device, aScaleFileSize, src2Host, aScaleFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(src3Device, bScaleFileSize, src3Host, bScaleFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(postScaleDevice, postScaleDeviceBytes, postScaleHost, postScaleDeviceBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }

    if (runProcess == RunProcess::kPostProcess) {
        ReadFile("../output/golden_matmul.bin", cFileSize, dstHost, cFileSize);
        ReadFile("../input/post_scale.bin", postScaleFileSize, postScaleHost, postScaleFileSize);
        aclrtMemcpy(dstDevice, cFileSize, dstHost, cFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(postScaleDevice, postScaleDeviceBytes, postScaleHost, postScaleDeviceBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }

    aclrtMemset(scoreDevice, scoreFileSize, 0, scoreFileSize);
    aclrtMemset(scoreBf16Device, scoreBf16FileSize, 0, scoreBf16FileSize);
    aclrtMemset(outIdxDevice, outIdxFileSize, 0, outIdxFileSize);

    if (runProcess == RunProcess::kAll) {
        LaunchIndexerMxfp8(dstDevice, src0Device, src1Device, src2Device, src3Device, postScaleDevice, scoreDevice,
                           scoreBf16Device, outIdxDevice, stream);
    } else if (runProcess == RunProcess::kMatmul) {
        LaunchIndexerMatmul(dstDevice, src0Device, src1Device, src2Device, src3Device, postScaleDevice, scoreDevice,
                            scoreBf16Device, outIdxDevice, stream);
    } else if (runProcess == RunProcess::kPostProcess) {
        LaunchIndexerPostProcess(reinterpret_cast<float *>(dstDevice), postScaleDevice, scoreDevice, scoreBf16Device, stream);
    } else {
        ReadFile("../output/golden_score.bin", scoreFileSize, scoreHost.data(), scoreFileSize);
        for (size_t i = 0; i < scoreHost.size(); ++i) {
            scoreBf16Host[i] = FloatToBf16Bits(scoreHost[i]);
        }
        aclrtMemcpy(scoreBf16Device, scoreBf16FileSize, scoreBf16Host.data(), scoreBf16FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        LaunchIndexerTopK(scoreBf16Device, outIdxDevice, stream);
    }

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (runProcess != RunProcess::kTopK) {
        aclrtMemcpy(scoreHost.data(), scoreFileSize, scoreDevice, scoreFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    }
    aclrtMemcpy(outIdxHost.data(), outIdxFileSize, outIdxDevice, outIdxFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (runProcess == RunProcess::kAll || runProcess == RunProcess::kPostProcess || runProcess == RunProcess::kTopK) {
        BuildScoreOrderedKeys(scoreHost.data(), scoreKeyHost.data(), kBatch, n);
    }
    if (runProcess == RunProcess::kAll || runProcess == RunProcess::kTopK) {
        uint32_t invalidIdxTotal = 0;
        for (uint32_t b = 0; b < kBatch; ++b) {
            const T *scorePtr = scoreHost.data() + static_cast<uint64_t>(b) * n;
            uint32_t *idxPtr = outIdxHost.data() + static_cast<uint64_t>(b) * kTopK;
            T *multisetPtr = topkMultisetHost.data() + static_cast<uint64_t>(b) * kTopK;
            invalidIdxTotal += BuildTopkValueMultiset(scorePtr, idxPtr, multisetPtr, kTopK, n);
        }
        if (invalidIdxTotal > 0) {
            printf("warning: detected %u out-of-range topk indices on host side\n", invalidIdxTotal);
        }
    }

    if (runProcess == RunProcess::kAll || runProcess == RunProcess::kMatmul) {
        WriteFile("../output/output_z.bin", dstHost, cFileSize);
    }
    if (runProcess == RunProcess::kAll || runProcess == RunProcess::kPostProcess || runProcess == RunProcess::kTopK) {
        WriteFile("../output/output_score.bin", scoreHost.data(), scoreFileSize);
        WriteFile("../output/output_score_key.bin", scoreKeyHost.data(), scoreKeyFileSize);
    }
    if (runProcess == RunProcess::kAll || runProcess == RunProcess::kTopK) {
        WriteFile("../output/output_idx.bin", outIdxHost.data(), outIdxFileSize);
        WriteFile("../output/output_topk_multiset.bin", topkMultisetHost.data(), topkMultisetFileSize);
    }

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(src2Device);
    aclrtFree(src3Device);
    aclrtFree(postScaleDevice);
    aclrtFree(scoreDevice);
    aclrtFree(scoreBf16Device);
    aclrtFree(outIdxDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(src2Host);
    aclrtFreeHost(src3Host);
    aclrtFreeHost(postScaleHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    if (runProcess == RunProcess::kAll) {
        VerifyMatmul<T>(cFileSize);
        VerifyScore<T>(scoreFileSize);
        VerifyTopkMultiset<T>(topkMultisetFileSize);
        VerifyTopkIndex(outIdxFileSize);
    } else if (runProcess == RunProcess::kMatmul) {
        VerifyMatmul<T>(cFileSize);
    } else if (runProcess == RunProcess::kPostProcess) {
        VerifyScore<T>(scoreFileSize);
    } else {
        VerifyTopkMultiset<T>(topkMultisetFileSize);
        VerifyTopkIndex(outIdxFileSize);
    }
}

int main(int argc, char **argv)
{
    RunProcess runProcess = ParseProcess(argc, argv);
    if (runProcess == RunProcess::kAll) {
        printf("Run process: all\n");
    } else if (runProcess == RunProcess::kMatmul) {
        printf("Run process: 1 (matmul)\n");
    } else if (runProcess == RunProcess::kPostProcess) {
        printf("Run process: 2 (postprocess)\n");
    } else {
        printf("Run process: 3 (topk)\n");
    }

    constexpr uint32_t m = 128;
    constexpr uint32_t k = 1024;
    constexpr uint32_t n = INDEXER_TEST_N;
    constexpr uint32_t topk = INDEXER_TOPK;
    static_assert(topk <= n, "INDEXER_TOPK must be <= INDEXER_TEST_N.");
    constexpr uint32_t singleCoreM = 128;
    constexpr uint32_t singleCoreK = 128;
    constexpr uint32_t singleCoreN = 131072;
    constexpr uint32_t blockDim = 1;
    constexpr uint32_t baseM = 128;
    constexpr uint32_t baseK = 128;
    constexpr uint32_t baseN = 128;
    constexpr uint32_t stepM = 1;
    constexpr uint32_t stepKa = 1;
    constexpr uint32_t stepKb = 1;
    constexpr uint32_t stepN = 1;

    MxMatmul<float, uint8_t, uint8_t, blockDim, m, k, n, singleCoreM, singleCoreK, singleCoreN, baseM, baseK, baseN,
             stepM, stepKa, stepKb, stepN>(runProcess);
}