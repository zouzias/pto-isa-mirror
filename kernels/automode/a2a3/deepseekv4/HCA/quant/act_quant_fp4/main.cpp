/**
 * main.cpp - host driver for act_quant_fp4 (inplace BF16 round-trip + uint8 scale).
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ./input/input_x.bin           M*N                bfloat16 (original x)
 *   ./output/golden_x.bin         M*N                bfloat16 (dequant after fp4 round-trip)
 *   ./output/golden_scale.bin     M*ceildiv(N, BLK)  uint8    (E8M0 biased exponent)
 *   ./output/output_x.bin         M*N                bfloat16 (kernel write-back)
 *   ./output/output_scale.bin     M*ceildiv(N, BLK)  uint8    (kernel scales)
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

extern "C" void launch_act_quant_fp4(uint8_t *x, uint8_t *scale_e8m0,
                                     uint64_t M, uint64_t N, uint64_t BLK,
                                     void *stream);

int main()
{
    constexpr int kM   = kQuantM;
    constexpr int kN   = kQuantN;
    constexpr int kBLK = kQuantBlockSize;
    constexpr int kNB  = (kN + kBLK - 1) / kBLK;

    constexpr size_t bf16Bytes = 2;
    constexpr size_t u8Bytes   = 1;

    size_t xBytes = static_cast<size_t>(kM) * kN  * bf16Bytes;
    size_t sBytes = static_cast<size_t>(kM) * kNB * u8Bytes;

    printf("[act_quant_fp4] M=%d N=%d BLK=%d nBlocks=%d inplace=%d\n",
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

    (void)PtoTiming::TimeKernelCallUs("act_quant_fp4", stream, [&]() {
        launch_act_quant_fp4(xDev, sDev,
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

    // Scale (uint8 E8M0) bit-exact compare.
    std::vector<uint8_t> golden_s(sBytes);
    std::vector<uint8_t> dev_s(sBytes);
    ReadFile("./output/golden_scale.bin", sBytes, golden_s.data(), sBytes);
    ReadFile("./output/output_scale.bin", sBytes, dev_s.data(),    sBytes);
    bool ok = std::memcmp(golden_s.data(), dev_s.data(), sBytes) == 0;
    if (!ok) {
        printf("[act_quant_fp4] scale (E8M0) bytes differ\n");
    }

    // BF16 dequant round-trip — loose tolerance per README (FP4 has ~2-3
    // mantissa bits).
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
    ok = ok && ResultCmp(golden_x_f, dev_x_f, 2e-1f);

    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ok ? 0 : 1;
}
