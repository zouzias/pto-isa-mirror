/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>
#include <cstdio>

#include <securec.h>

#include "acl/acl.h"
#include "test_common.h"

using namespace std;
using namespace PtoTestCommon;

void LaunchHif4Matmul(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* src3, void* stream);

std::vector<float> Bf16BytesToFloat(const uint8_t* raw, size_t numElements)
{
    std::vector<float> v(numElements);
    const auto* u16 = reinterpret_cast<const uint16_t*>(raw);
    for (size_t i = 0; i < numElements; i++) {
        uint32_t bits = static_cast<uint32_t>(u16[i]) << 16;
        memcpy_s(&v[i], sizeof(float), &bits, sizeof(bits));
    }
    return v;
}

void VerifyResult(size_t numElements)
{
    size_t byteSize = numElements * sizeof(uint16_t);
    std::vector<uint8_t> goldenBytes(byteSize);
    std::vector<uint8_t> outBytes(byteSize);

    size_t rd = byteSize;
    ReadFile("../output/golden.bin", rd, goldenBytes.data(), byteSize);
    rd = byteSize;
    ReadFile("../output/output_z.bin", rd, outBytes.data(), byteSize);

    auto golden = Bf16BytesToFloat(goldenBytes.data(), numElements);
    auto out = Bf16BytesToFloat(outBytes.data(), numElements);

    bool ret = ResultCmp<float>(golden, out, 0.03f);
    if (ret) {
        printf("test success\n");
    } else {
        printf("test failed\n");
    }
}

template <uint32_t m, uint32_t k, uint32_t n>
void Hif4Matmul()
{
    // HiF4 data: 4-bit packed, 2 elements per byte.
    // HiF4 scale: 4 bytes per 64-element group, pre-fractalized [16,4] cells.
    size_t aFileSize = static_cast<size_t>(m) * k / 2;
    size_t bFileSize = static_cast<size_t>(k) * n / 2;
    size_t aScaleFileSize = static_cast<size_t>(m) * k / 64 * 4;
    size_t bScaleFileSize = static_cast<size_t>(k) * n / 64 * 4;
    size_t cFileSize = static_cast<size_t>(m) * n * sizeof(uint16_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host, *src2Host, *src3Host;
    uint8_t *dstDevice, *src0Device, *src1Device, *src2Device, *src3Device;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&src0Host), aFileSize);
    aclrtMallocHost((void**)(&src1Host), bFileSize);
    aclrtMallocHost((void**)(&src2Host), aScaleFileSize);
    aclrtMallocHost((void**)(&src3Host), bScaleFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src2Device, aScaleFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src3Device, bScaleFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t rd = aFileSize;
    ReadFile("../input/x1_gm.bin", rd, src0Host, aFileSize);
    rd = bFileSize;
    ReadFile("../input/x2_gm.bin", rd, src1Host, bFileSize);
    rd = aScaleFileSize;
    ReadFile("../input/x1_scale_gm.bin", rd, src2Host, aScaleFileSize);
    rd = bScaleFileSize;
    ReadFile("../input/x2_scale_gm.bin", rd, src3Host, bScaleFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src2Device, aScaleFileSize, src2Host, aScaleFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src3Device, bScaleFileSize, src3Host, bScaleFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchHif4Matmul(dstDevice, src0Device, src1Device, src2Device, src3Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("../output/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(src2Device);
    aclrtFree(src3Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(src2Host);
    aclrtFreeHost(src3Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    VerifyResult(static_cast<size_t>(m) * n);
}

int main()
{
    constexpr uint32_t m = 2048;
    constexpr uint32_t k = 2048;
    constexpr uint32_t n = 2048;

    Hif4Matmul<m, k, n>();
}
