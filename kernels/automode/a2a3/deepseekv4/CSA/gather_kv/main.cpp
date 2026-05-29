/**
 * main.cpp — host driver for sparse_attn.gather_kv (one pipelined-block slice).
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ./input/input_kv.bin           kv_rows*D   bfloat16
 *   ./input/input_topk_idxs.bin    TOPK        int32
 *   ./output/golden_kv_gathered.bin BLOCK*D    bfloat16   (numpy reference)
 *   ./output/output_kv_gathered.bin BLOCK*D    bfloat16   (kernel output)
 *
 * Hardcoded compile-time block index for the prototype: t = 0 (first inner
 * block). The full FA driver iterates t over [0, kCsaNumBlocks).
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

extern "C" void launch_gather_kv(uint8_t *kv_gathered, uint8_t *kv,
                                 uint8_t *topk_idxs,
                                 uint32_t t_block, uint32_t topk_len,
                                 uint32_t kv_rows, void *stream);

int main()
{
    constexpr int kBlock     = kCsaBlock;
    constexpr int kD         = kCsaD;
    constexpr int kTopK      = kCsaS;
    // For the prototype we exercise a single (b, m, t) tuple; we size the KV
    // cache so that every valid topk index falls in [0, kKvRows).
    constexpr int kKvRows    = kCsaS;          // Assumption: kv_rows >= max(idx)+1
    constexpr uint32_t kTblock = 0;            // first pipelined block

    constexpr size_t bf16Bytes = 2;
    constexpr size_t i32Bytes  = 4;

    size_t kvBytes      = static_cast<size_t>(kKvRows) * kD * bf16Bytes;
    size_t idxBytes     = static_cast<size_t>(kTopK)         * i32Bytes;
    size_t outBytes     = static_cast<size_t>(kBlock) * kD   * bf16Bytes;

    printf("[gather_kv] H=%d D=%d S=%d BLOCK=%d  t_block=%u  kv_rows=%d\n",
           kCsaH, kD, kTopK, kBlock, kTblock, kKvRows);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *kvHost  = nullptr, *idxHost = nullptr, *outHost = nullptr;
    uint8_t *kvDev   = nullptr, *idxDev  = nullptr, *outDev  = nullptr;

    aclrtMallocHost((void **)&kvHost,  kvBytes);
    aclrtMallocHost((void **)&idxHost, idxBytes);
    aclrtMallocHost((void **)&outHost, outBytes);

    aclrtMalloc((void **)&kvDev,  kvBytes,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&idxDev, idxBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&outDev, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("./input/input_kv.bin",        kvBytes,  kvHost,  kvBytes);
    ReadFile("./input/input_topk_idxs.bin", idxBytes, idxHost, idxBytes);

    aclrtMemcpy(kvDev,  kvBytes,  kvHost,  kvBytes,  ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(idxDev, idxBytes, idxHost, idxBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("gather_kv", stream, [&]() {
        launch_gather_kv(outDev, kvDev, idxDev,
                         kTblock,
                         static_cast<uint32_t>(kTopK),
                         static_cast<uint32_t>(kKvRows),
                         stream);
    });

    aclrtMemcpy(outHost, outBytes, outDev, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile("./output/output_kv_gathered.bin", outHost, outBytes);

    aclrtFree(outDev);
    aclrtFree(idxDev);
    aclrtFree(kvDev);
    aclrtFreeHost(outHost);
    aclrtFreeHost(idxHost);
    aclrtFreeHost(kvHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    // Compare raw BF16 bytes — gather is bit-exact (no arithmetic).
    std::vector<uint16_t> golden(outBytes / sizeof(uint16_t));
    std::vector<uint16_t> dev(outBytes / sizeof(uint16_t));
    ReadFile("./output/golden_kv_gathered.bin", outBytes, golden.data(), outBytes);
    ReadFile("./output/output_kv_gathered.bin", outBytes, dev.data(),    outBytes);

    bool ok = (golden == dev);
    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ok ? 0 : 1;
}
