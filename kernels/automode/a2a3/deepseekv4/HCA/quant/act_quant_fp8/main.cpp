/**
 * main.cpp - host driver for act_quant_fp8 (inplace BF16 round-trip + FP32 scale).
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ./input/input_x.bin           M*N                bfloat16 (original x)
 *   ./output/golden_x.bin         M*N                bfloat16 (dequant x after fp8 round-trip)
 *   ./output/golden_scale.bin     M*ceildiv(N, BLK)  float32  (per-block scales)
 *   ./output/output_x.bin         M*N                bfloat16 (kernel write-back)
 *   ./output/output_scale.bin     M*ceildiv(N, BLK)  float32  (kernel scales)
 */

#include "test_common.h"
#include "acl/acl.h"
#include "../../../../kernel_timing.h"
#include "generated_cases.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

extern "C" void launch_act_quant_fp8(uint8_t *x, uint8_t *scale,
                                     uint64_t M, uint64_t N, uint64_t BLK,
                                     void *stream);

int main()
{
    constexpr int kM   = kQuantM;
    constexpr int kN   = kQuantN;
    constexpr int kBLK = kQuantBlockSize;
    constexpr int kNB  = (kN + kBLK - 1) / kBLK;

    constexpr size_t bf16Bytes  = 2;
    constexpr size_t floatBytes = 4;

    size_t xBytes = static_cast<size_t>(kM) * kN  * bf16Bytes;
    size_t sBytes = static_cast<size_t>(kM) * kNB * floatBytes;

    printf("[act_quant_fp8] M=%d N=%d BLK=%d nBlocks=%d inplace=%d\n",
           kM, kN, kBLK, kNB, kQuantInplace);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *xHost = nullptr, *sHost = nullptr;
    uint8_t *xDev  = nullptr, *sDev  = nullptr;

    aclrtMallocHost((void **)&xHost, xBytes);
    aclrtMallocHost((void **)&sHost, sBytes);

    aclrtMalloc((void **)&xDev, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&sDev, sBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("./input/input_x.bin", xBytes, xHost, xBytes);

    aclrtMemcpy(xDev, xBytes, xHost, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("act_quant_fp8", stream, [&]() {
        launch_act_quant_fp8(xDev, sDev,
                             static_cast<uint64_t>(kM),
                             static_cast<uint64_t>(kN),
                             static_cast<uint64_t>(kBLK),
                             stream);
    });

    aclrtMemcpy(xHost, xBytes, xDev, xBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(sHost, sBytes, sDev, sBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile("./output/output_x.bin",     xHost, xBytes);
    WriteFile("./output/output_scale.bin", sHost, sBytes);

    aclrtFree(sDev);
    aclrtFree(xDev);
    aclrtFreeHost(sHost);
    aclrtFreeHost(xHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    // Compare scale (FP32) first — tolerant since scale is rounded pow2.
    std::vector<float> golden_s(sBytes / sizeof(float));
    std::vector<float> dev_s(sBytes / sizeof(float));
    ReadFile("./output/golden_scale.bin", sBytes, golden_s.data(), sBytes);
    ReadFile("./output/output_scale.bin", sBytes, dev_s.data(),    sBytes);

    bool ok = ResultCmp(golden_s, dev_s, 5e-2f);
    // Then dequant'd BF16 round-trip; loose tolerance per README.
    // For BF16 buffer we reinterpret the raw uint16 to FP32 via a simple
    // expand for the comparator.
    std::vector<uint16_t> golden_x(xBytes / sizeof(uint16_t));
    std::vector<uint16_t> dev_x(xBytes / sizeof(uint16_t));
    ReadFile("./output/golden_x.bin", xBytes, golden_x.data(), xBytes);
    ReadFile("./output/output_x.bin", xBytes, dev_x.data(),    xBytes);

    auto bf16_to_f32 = [](uint16_t b) {
        uint32_t w = static_cast<uint32_t>(b) << 16;
        float f;
        std::memcpy(&f, &w, sizeof(f));
        return f;
    };
    std::vector<float> golden_x_f(golden_x.size()), dev_x_f(dev_x.size());
    for (size_t i = 0; i < golden_x.size(); ++i) {
        golden_x_f[i] = bf16_to_f32(golden_x[i]);
        dev_x_f[i]    = bf16_to_f32(dev_x[i]);
    }
    ok = ok && ResultCmp(golden_x_f, dev_x_f, 5e-2f);

    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ok ? 0 : 1;
}
