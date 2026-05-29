/**
 * main.cpp — host driver for DeepSeek-V4 shared_expert_ffn.
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ./input/input_X.bin    T * DIM                bfloat16
 *   ./input/input_W1.bin   INTER_DIM * DIM        bfloat16
 *   ./input/input_W3.bin   INTER_DIM * DIM        bfloat16
 *   ./input/input_W2.bin   DIM * INTER_DIM        bfloat16
 *   ./output/golden_Y.bin  T * DIM                bfloat16  (numpy reference)
 *   ./output/output_Y.bin  T * DIM                bfloat16  (kernel output)
 */

#include "test_common.h"
#include "acl/acl.h"
#include "../../../kernel_timing.h"
#include "generated_cases.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

extern "C" void launch_shared_expert_ffn(uint8_t *Y, uint8_t *X,
                                         uint8_t *W1, uint8_t *W3, uint8_t *W2,
                                         void *stream);

int main()
{
    constexpr int kT     = kDsmoeT;
    constexpr int kDim   = kDsmoeDim;
    constexpr int kInter = kDsmoeInterDim;

    constexpr size_t bf16Bytes = 2;
    size_t xBytes  = static_cast<size_t>(kT)     * kDim   * bf16Bytes;
    size_t yBytes  = xBytes;
    size_t w1Bytes = static_cast<size_t>(kInter) * kDim   * bf16Bytes;
    size_t w3Bytes = w1Bytes;
    size_t w2Bytes = static_cast<size_t>(kDim)   * kInter * bf16Bytes;

    printf("[shared_expert_ffn] T=%d DIM=%d INTER_DIM=%d\n", kT, kDim, kInter);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *xHost = nullptr, *yHost = nullptr;
    uint8_t *w1Host = nullptr, *w3Host = nullptr, *w2Host = nullptr;
    uint8_t *xDev = nullptr, *yDev = nullptr;
    uint8_t *w1Dev = nullptr, *w3Dev = nullptr, *w2Dev = nullptr;

    aclrtMallocHost((void **)&xHost,  xBytes);
    aclrtMallocHost((void **)&yHost,  yBytes);
    aclrtMallocHost((void **)&w1Host, w1Bytes);
    aclrtMallocHost((void **)&w3Host, w3Bytes);
    aclrtMallocHost((void **)&w2Host, w2Bytes);

    aclrtMalloc((void **)&xDev,  xBytes,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&yDev,  yBytes,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&w1Dev, w1Bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&w3Dev, w3Bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&w2Dev, w2Bytes, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("./input/input_X.bin",  xBytes,  xHost,  xBytes);
    ReadFile("./input/input_W1.bin", w1Bytes, w1Host, w1Bytes);
    ReadFile("./input/input_W3.bin", w3Bytes, w3Host, w3Bytes);
    ReadFile("./input/input_W2.bin", w2Bytes, w2Host, w2Bytes);

    aclrtMemcpy(xDev,  xBytes,  xHost,  xBytes,  ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(w1Dev, w1Bytes, w1Host, w1Bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(w3Dev, w3Bytes, w3Host, w3Bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(w2Dev, w2Bytes, w2Host, w2Bytes, ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("shared_expert_ffn", stream, [&]() {
        launch_shared_expert_ffn(yDev, xDev, w1Dev, w3Dev, w2Dev, stream);
    });

    aclrtMemcpy(yHost, yBytes, yDev, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile("./output/output_Y.bin", yHost, yBytes);

    aclrtFree(w2Dev);
    aclrtFree(w3Dev);
    aclrtFree(w1Dev);
    aclrtFree(yDev);
    aclrtFree(xDev);
    aclrtFreeHost(w2Host);
    aclrtFreeHost(w3Host);
    aclrtFreeHost(w1Host);
    aclrtFreeHost(yHost);
    aclrtFreeHost(xHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    // Compare BF16 outputs as raw uint16 bit patterns? For SwiGLU we use a
    // tolerance compare since FP32-accum→BF16 cast introduces ULP drift.
    // Read BF16 as uint16, convert to FP32 for ResultCmp.
    std::vector<uint16_t> goldBits(yBytes / sizeof(uint16_t));
    std::vector<uint16_t> outBits (yBytes / sizeof(uint16_t));
    ReadFile("./output/golden_Y.bin", yBytes, goldBits.data(), yBytes);
    ReadFile("./output/output_Y.bin", yBytes, outBits.data(),  yBytes);

    std::vector<float> gold(goldBits.size());
    std::vector<float> dev (outBits.size());
    for (size_t i = 0; i < gold.size(); ++i) {
        uint32_t g = static_cast<uint32_t>(goldBits[i]) << 16;
        uint32_t o = static_cast<uint32_t>(outBits [i]) << 16;
        std::memcpy(&gold[i], &g, sizeof(float));
        std::memcpy(&dev [i], &o, sizeof(float));
    }

    bool ok = ResultCmp(gold, dev, 1e-2f);
    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ok ? 0 : 1;
}
