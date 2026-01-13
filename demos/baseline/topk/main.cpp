/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include "acl/acl.h"

using namespace std;
using namespace PtoTestCommon;

template <typename T, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
    int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4, int topk, int start>
void launchTopk(uint8_t *out, uint8_t *index, uint8_t *src, void *stream);

template <typename T, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
    int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4, int topk, int start>
void Topk()
{
    constexpr int rows = gWholeShape0 * gWholeShape1 * gWholeShape2 * gWholeShape3;
    constexpr int cols = gWholeShape4;
    constexpr int valid_row = gShape0 * gShape1 * gShape2 * gShape3;
    using indexT = uint32_t;
    size_t inFileSize = rows * cols * sizeof(T);
    constexpr int TYPE_COEF = sizeof(float)/sizeof(T);
    size_t outFileSize = valid_row * topk * sizeof(T);
    size_t indexFileSize = valid_row * topk * sizeof(indexT);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *srcHost, *indexHost;
    uint8_t *dstDevice, *srcDevice, *indexDevice;

    aclrtMallocHost((void **)(&dstHost), outFileSize);
    aclrtMallocHost((void **)(&indexHost), indexFileSize);
    aclrtMallocHost((void **)(&srcHost), inFileSize);

    aclrtMalloc((void **)&dstDevice, outFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, inFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&indexDevice, indexFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("../input/x1_gm.bin", inFileSize, srcHost, inFileSize);

    aclrtMemcpy(srcDevice, inFileSize, srcHost, inFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTopk<T, gShape0, gShape1, gShape2, gShape3, gShape4,
        gWholeShape0, gWholeShape1, gWholeShape2, gWholeShape3, gWholeShape4,
        topk, start>(dstDevice, indexDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outFileSize, dstDevice, outFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(indexHost, indexFileSize, indexDevice, indexFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("../output/output_z.bin", dstHost, outFileSize);
    WriteFile("../output/index_z.bin", indexHost, indexFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFree(indexDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(indexHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(outFileSize);
    std::vector<T> devFinal(outFileSize);
    ReadFile("../output/golden_d.bin", outFileSize, golden.data(), outFileSize);
    ReadFile("../output/output_z.bin", outFileSize, devFinal.data(), outFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    if (ret) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }

    std::vector<indexT> golden_i(indexFileSize);
    std::vector<indexT> devFinal_i(indexFileSize);
    ReadFile("../output/golden_i.bin", indexFileSize, golden_i.data(), indexFileSize);
    ReadFile("../output/index_z.bin", indexFileSize, devFinal_i.data(), indexFileSize);

    ret = ResultCmp(golden_i, devFinal_i, 0.001f);
    if (ret) {
        printf("test index success\n");
    } else {
        printf("test index failed\n");
    }
}

int main()
{
    Topk<float, 1, 1, 1, 1, 1280, 1, 1, 1, 1, 1280, 1280, k>();
}
