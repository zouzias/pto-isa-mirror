/**
 * main.cpp — host driver for sparse_attn.pv_matmul (one pipelined-block iter).
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ./input/input_acc_o.bin      H*D       float32   (pre-rescaled by driver)
 *   ./input/input_acc_s_cast.bin H*BLOCK   bfloat16
 *   ./input/input_kv_block.bin   BLOCK*D   bfloat16
 *   ./output/golden_acc_o.bin    H*D       float32
 *   ./output/output_acc_o.bin    H*D       float32
 *
 * The driver pre-applies `acc_o *= scores_scale[h]` before this kernel; the
 * input file already encodes that rescale.
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

extern "C" void launch_pv_matmul(uint8_t *acc_o, uint8_t *acc_s_cast,
                                 uint8_t *kv_block, void *stream);

int main()
{
    constexpr int kH     = kCsaH;
    constexpr int kD     = kCsaD;
    constexpr int kBlock = kCsaBlock;

    constexpr size_t bf16Bytes  = 2;
    constexpr size_t floatBytes = 4;

    size_t accBytes  = static_cast<size_t>(kH)     * kD     * floatBytes;
    size_t scBytes   = static_cast<size_t>(kH)     * kBlock * bf16Bytes;
    size_t kvBytes   = static_cast<size_t>(kBlock) * kD     * bf16Bytes;

    printf("[pv_matmul] H=%d D=%d BLOCK=%d\n", kH, kD, kBlock);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *accHost = nullptr, *scHost = nullptr, *kvHost = nullptr;
    uint8_t *accDev  = nullptr, *scDev  = nullptr, *kvDev  = nullptr;

    aclrtMallocHost((void **)&accHost, accBytes);
    aclrtMallocHost((void **)&scHost,  scBytes);
    aclrtMallocHost((void **)&kvHost,  kvBytes);

    aclrtMalloc((void **)&accDev, accBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&scDev,  scBytes,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&kvDev,  kvBytes,  ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("./input/input_acc_o.bin",      accBytes, accHost, accBytes);
    ReadFile("./input/input_acc_s_cast.bin", scBytes,  scHost,  scBytes);
    ReadFile("./input/input_kv_block.bin",   kvBytes,  kvHost,  kvBytes);

    aclrtMemcpy(accDev, accBytes, accHost, accBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(scDev,  scBytes,  scHost,  scBytes,  ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(kvDev,  kvBytes,  kvHost,  kvBytes,  ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("pv_matmul", stream, [&]() {
        launch_pv_matmul(accDev, scDev, kvDev, stream);
    });

    aclrtMemcpy(accHost, accBytes, accDev, accBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile("./output/output_acc_o.bin", accHost, accBytes);

    aclrtFree(kvDev); aclrtFree(scDev); aclrtFree(accDev);
    aclrtFreeHost(kvHost); aclrtFreeHost(scHost); aclrtFreeHost(accHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(accBytes / sizeof(float));
    std::vector<float> dev(accBytes    / sizeof(float));
    ReadFile("./output/golden_acc_o.bin", accBytes, golden.data(), accBytes);
    ReadFile("./output/output_acc_o.bin", accBytes, dev.data(),    accBytes);

    bool ok = ResultCmp(golden, dev, 1e-2f);
    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ok ? 0 : 1;
}
