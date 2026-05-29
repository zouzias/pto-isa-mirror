/**
 * main.cpp — host driver for sparse_attn.qk_matmul (one pipelined-block slice).
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ./input/input_q.bin         H*D       bfloat16
 *   ./input/input_kv_block.bin  BLOCK*D   bfloat16
 *   ./output/golden_acc_s.bin   H*BLOCK   float32
 *   ./output/output_acc_s.bin   H*BLOCK   float32
 *
 * softmax_scale is read from a tiny side-car file written by gen_data.py:
 *   ./input/softmax_scale.f32   1 float
 */

#include "test_common.h"
#include "acl/acl.h"
#include "../../../kernel_timing.h"
#include "generated_cases.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

extern "C" void launch_qk_matmul(uint8_t *acc_s, uint8_t *q, uint8_t *kv_block,
                                 float softmax_scale, void *stream);

int main()
{
    constexpr int kH     = kCsaH;
    constexpr int kD     = kCsaD;
    constexpr int kBlock = kCsaBlock;

    constexpr size_t bf16Bytes  = 2;
    constexpr size_t floatBytes = 4;

    size_t qBytes   = static_cast<size_t>(kH)     * kD     * bf16Bytes;
    size_t kvBytes  = static_cast<size_t>(kBlock) * kD     * bf16Bytes;
    size_t outBytes = static_cast<size_t>(kH)     * kBlock * floatBytes;

    // Read softmax_scale (tiny side-car, FP32 little-endian).
    float softmaxScale = 0.0f;
    {
        std::vector<float> tmp(1);
        ReadFile("./input/softmax_scale.f32", sizeof(float), tmp.data(), sizeof(float));
        softmaxScale = tmp[0];
    }

    printf("[qk_matmul] H=%d D=%d BLOCK=%d  softmax_scale=%g\n",
           kH, kD, kBlock, softmaxScale);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *qHost  = nullptr, *kvHost  = nullptr, *outHost = nullptr;
    uint8_t *qDev   = nullptr, *kvDev   = nullptr, *outDev  = nullptr;

    aclrtMallocHost((void **)&qHost,  qBytes);
    aclrtMallocHost((void **)&kvHost, kvBytes);
    aclrtMallocHost((void **)&outHost, outBytes);

    aclrtMalloc((void **)&qDev,  qBytes,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&kvDev, kvBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&outDev, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("./input/input_q.bin",        qBytes,  qHost,  qBytes);
    ReadFile("./input/input_kv_block.bin", kvBytes, kvHost, kvBytes);

    aclrtMemcpy(qDev,  qBytes,  qHost,  qBytes,  ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(kvDev, kvBytes, kvHost, kvBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("qk_matmul", stream, [&]() {
        launch_qk_matmul(outDev, qDev, kvDev, softmaxScale, stream);
    });

    aclrtMemcpy(outHost, outBytes, outDev, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile("./output/output_acc_s.bin", outHost, outBytes);

    aclrtFree(outDev);
    aclrtFree(kvDev);
    aclrtFree(qDev);
    aclrtFreeHost(outHost);
    aclrtFreeHost(kvHost);
    aclrtFreeHost(qHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(outBytes / sizeof(float));
    std::vector<float> dev(outBytes    / sizeof(float));
    ReadFile("./output/golden_acc_s.bin", outBytes, golden.data(), outBytes);
    ReadFile("./output/output_acc_s.bin", outBytes, dev.data(),    outBytes);

    // BF16 inputs + FP32 accumulator + a single scale multiply: drift bounded
    // by ~D ULPs.
    bool ok = ResultCmp(golden, dev, 1e-2f);
    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ok ? 0 : 1;
}
