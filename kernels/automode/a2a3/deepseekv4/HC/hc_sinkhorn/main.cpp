/**
 * main.cpp - host driver for hc_sinkhorn (FP32 -> FP32 pre/post/comb).
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ./input/input_mixes.bin     N*MIX_HC  float32   (per-row sinkhorn input)
 *   ./input/input_hc_scale.bin  3          float32   (per-head scale)
 *   ./input/input_hc_base.bin   MIX_HC     float32   (per-feature bias)
 *   ./output/golden_pre.bin     N*HC_MULT          float32
 *   ./output/golden_post.bin    N*HC_MULT          float32
 *   ./output/golden_comb.bin    N*HC_MULT*HC_MULT  float32
 *   ./output/output_pre.bin     N*HC_MULT          float32
 *   ./output/output_post.bin    N*HC_MULT          float32
 *   ./output/output_comb.bin    N*HC_MULT*HC_MULT  float32
 *
 *   N = B*S,  MIX_HC = (2+HC_MULT)*HC_MULT  (see ../README.md).
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

extern "C" void launch_hc_sinkhorn(uint8_t *pre, uint8_t *post, uint8_t *comb,
                                   uint8_t *mixes, uint8_t *hc_scale, uint8_t *hc_base,
                                   void *stream);

int main()
{
    constexpr int kB     = kHcB;
    constexpr int kS     = kHcS;
    constexpr int kHc    = kHcMult;
    constexpr int kMixHc = kHcMixHc;

    constexpr int kN = kB * kS;

    constexpr size_t floatBytes = 4;
    size_t mixBytes   = static_cast<size_t>(kN) * kMixHc * floatBytes;
    size_t scaleBytes = static_cast<size_t>(3)  * floatBytes;
    size_t baseBytes  = static_cast<size_t>(kMixHc) * floatBytes;
    size_t preBytes   = static_cast<size_t>(kN) * kHc * floatBytes;
    size_t postBytes  = preBytes;
    size_t combBytes  = static_cast<size_t>(kN) * kHc * kHc * floatBytes;

    printf("[hc_sinkhorn] B=%d S=%d HC_MULT=%d MIX_HC=%d iters=%d  -> N=%d\n",
           kB, kS, kHc, kMixHc, kHcSinkhornIters, kN);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *mixHost = nullptr, *scaleHost = nullptr, *baseHost = nullptr;
    uint8_t *preHost = nullptr, *postHost = nullptr, *combHost = nullptr;
    uint8_t *mixDev  = nullptr, *scaleDev  = nullptr, *baseDev  = nullptr;
    uint8_t *preDev  = nullptr, *postDev  = nullptr, *combDev  = nullptr;

    aclrtMallocHost((void **)&mixHost,   mixBytes);
    aclrtMallocHost((void **)&scaleHost, scaleBytes);
    aclrtMallocHost((void **)&baseHost,  baseBytes);
    aclrtMallocHost((void **)&preHost,   preBytes);
    aclrtMallocHost((void **)&postHost,  postBytes);
    aclrtMallocHost((void **)&combHost,  combBytes);

    aclrtMalloc((void **)&mixDev,   mixBytes,   ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&scaleDev, scaleBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&baseDev,  baseBytes,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&preDev,   preBytes,   ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&postDev,  postBytes,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&combDev,  combBytes,  ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("./input/input_mixes.bin",    mixBytes,   mixHost,   mixBytes);
    ReadFile("./input/input_hc_scale.bin", scaleBytes, scaleHost, scaleBytes);
    ReadFile("./input/input_hc_base.bin",  baseBytes,  baseHost,  baseBytes);

    aclrtMemcpy(mixDev,   mixBytes,   mixHost,   mixBytes,   ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(scaleDev, scaleBytes, scaleHost, scaleBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(baseDev,  baseBytes,  baseHost,  baseBytes,  ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("hc_sinkhorn", stream, [&]() {
        launch_hc_sinkhorn(preDev, postDev, combDev,
                           mixDev, scaleDev, baseDev, stream);
    });

    aclrtMemcpy(preHost,  preBytes,  preDev,  preBytes,  ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(postHost, postBytes, postDev, postBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(combHost, combBytes, combDev, combBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("./output/output_pre.bin",  preHost,  preBytes);
    WriteFile("./output/output_post.bin", postHost, postBytes);
    WriteFile("./output/output_comb.bin", combHost, combBytes);

    aclrtFree(combDev);
    aclrtFree(postDev);
    aclrtFree(preDev);
    aclrtFree(baseDev);
    aclrtFree(scaleDev);
    aclrtFree(mixDev);
    aclrtFreeHost(combHost);
    aclrtFreeHost(postHost);
    aclrtFreeHost(preHost);
    aclrtFreeHost(baseHost);
    aclrtFreeHost(scaleHost);
    aclrtFreeHost(mixHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> goldenPre(preBytes / sizeof(float));
    std::vector<float> devPre(preBytes / sizeof(float));
    std::vector<float> goldenPost(postBytes / sizeof(float));
    std::vector<float> devPost(postBytes / sizeof(float));
    std::vector<float> goldenComb(combBytes / sizeof(float));
    std::vector<float> devComb(combBytes / sizeof(float));
    ReadFile("./output/golden_pre.bin",  preBytes,  goldenPre.data(),  preBytes);
    ReadFile("./output/output_pre.bin",  preBytes,  devPre.data(),     preBytes);
    ReadFile("./output/golden_post.bin", postBytes, goldenPost.data(), postBytes);
    ReadFile("./output/output_post.bin", postBytes, devPost.data(),    postBytes);
    ReadFile("./output/golden_comb.bin", combBytes, goldenComb.data(), combBytes);
    ReadFile("./output/output_comb.bin", combBytes, devComb.data(),    combBytes);

    // Tolerance: pre/post are single sigmoid evals; comb compounds many
    // divisions over (sinkhorn_iters) iterations -> looser tolerance.
    bool okPre  = ResultCmp(goldenPre,  devPre,  1e-4f);
    bool okPost = ResultCmp(goldenPost, devPost, 1e-4f);
    bool okComb = ResultCmp(goldenComb, devComb, 1e-3f);
    bool ok = okPre && okPost && okComb;
    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed  (pre=%d, post=%d, comb=%d)\n",
               static_cast<int>(okPre), static_cast<int>(okPost),
               static_cast<int>(okComb));
    }
    return ok ? 0 : 1;
}
