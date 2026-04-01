/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "test_common.h"

using namespace PtoTestCommon;

template <int TopK>
void LaunchRadixTopKDraft(uint16_t *src, uint32_t *outIdx, void *stream);

constexpr int kN = 2048;
constexpr int kTopK = 512;

namespace {

bool ValidateTopKMultiset(const std::vector<uint16_t> &keys, const std::vector<uint32_t> &outIdx,
                          const std::vector<uint16_t> &goldenRefMultiset)
{
    if (outIdx.size() != static_cast<size_t>(kTopK)) {
        ERROR_LOG("Expected %d indices, got %zu", kTopK, outIdx.size());
        return false;
    }
    std::vector<uint16_t> got;
    got.reserve(kTopK);
    for (uint32_t i : outIdx) {
        if (i >= static_cast<uint32_t>(kN)) {
            ERROR_LOG("Index out of range: %u", i);
            return false;
        }
        got.push_back(keys[i]);
    }
    // Device output order is arbitrary; only the multiset of values must match golden.
    std::sort(got.begin(), got.end());
    if (got != goldenRefMultiset) {
        ERROR_LOG("Top-K value multiset mismatch.");
        return false;
    }
    INFO_LOG("Top-K multiset check passed.");
    return true;
}

} // namespace

int main()
{
    constexpr size_t keysBytes = kN * sizeof(uint16_t);
    constexpr size_t outIdxBytes = kTopK * sizeof(uint32_t);
    constexpr size_t goldenBytes = kTopK * sizeof(uint16_t);

    std::vector<uint16_t> keys(kN);
    std::vector<uint16_t> goldenRefMultiset(kTopK);
    std::vector<uint32_t> outIdx(kTopK);

    size_t kfs = keysBytes;
    size_t gfs = goldenBytes;
    if (!ReadFile("../input/keys.bin", kfs, keys.data(), keysBytes) || kfs != keysBytes) {
        ERROR_LOG("Read ../input/keys.bin failed (run scripts/gen_data.py from topk directory).");
        return 1;
    }
    if (!ReadFile("../output/golden_topk_multiset.bin", gfs, goldenRefMultiset.data(), goldenBytes) ||
        gfs != goldenBytes) {
        ERROR_LOG("Read ../output/golden_topk_multiset.bin failed.");
        return 1;
    }

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint16_t *srcDevice = nullptr;
    uint32_t *idxDevice = nullptr;
    aclrtMalloc(reinterpret_cast<void **>(&srcDevice), keysBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void **>(&idxDevice), outIdxBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(idxDevice, outIdxBytes, 0, outIdxBytes);

    aclrtMemcpy(srcDevice, keysBytes, keys.data(), keysBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchRadixTopKDraft<kTopK>(srcDevice, idxDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outIdx.data(), outIdxBytes, idxDevice, outIdxBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("../output/out_idx.bin", outIdx.data(), outIdxBytes);

    aclrtFree(srcDevice);
    aclrtFree(idxDevice);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    bool ok = ValidateTopKMultiset(keys, outIdx, goldenRefMultiset);
    printf("%s\n", ok ? "RESULT: PASS" : "RESULT: FAIL");
    return ok ? 0 : 1;
}
