/*
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
#include <gtest/gtest.h>

//#define DEBUG_SOFTMAX_FA

using namespace std;
using namespace PtoTestCommon;


template <int seq, int headSize, int init, bool CAUSAL_MASK>
void launchTSOFTMAX_dn_fusion(aclFloat16 *x_exp, float *input_x, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);

template <int seq, int headSize, int init, bool CAUSAL_MASK>
void launchTSOFTMAX_dn_nofusion(aclFloat16 *x_exp, float *input_x, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);

template <int seq, int headSize, int init, bool CAUSAL_MASK>
void launchTSOFTMAX_nd_fusion(aclFloat16 *x_exp, float *input_x, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);

template <int seq, int headSize, int init, bool CAUSAL_MASK>
void launchTSOFTMAX_nd_nofusion(aclFloat16 *x_exp, float *input_x, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);


class TSOFTMAXFATest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir() {
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <int seq, int headSize, int init, bool CAUSAL_MASK>
void test_softmax_dn_fusion()
{
    size_t src0FileSize = headSize * seq * sizeof(float);  // input_x_local_UB
    size_t dst0FileSize = headSize * sizeof(float);  // local_max, local_sum, new_global_max, new_global_sum, exp_max
    size_t dst1FileSize = headSize * seq * sizeof(aclFloat16);  // x_exp

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    aclFloat16 *x_exp_Host;
    float *input_Host, *local_max_Host, *local_sum_Host, *new_global_max_Host, *new_global_sum_Host, *exp_max_Host;
    aclFloat16 *x_exp_Device;
    float *input_Device, *local_max_Device, *local_sum_Device, *new_global_max_Device, *new_global_sum_Device, *exp_max_Device;

    aclrtMallocHost((void**)(&input_Host), src0FileSize);
    aclrtMallocHost((void**)(&local_max_Host), dst0FileSize);
    aclrtMallocHost((void**)(&local_sum_Host), dst0FileSize);
    aclrtMallocHost((void**)(&new_global_max_Host), dst0FileSize);
    aclrtMallocHost((void**)(&new_global_sum_Host), dst0FileSize);
    aclrtMallocHost((void**)(&exp_max_Host), dst0FileSize);
    aclrtMallocHost((void**)(&x_exp_Host), dst1FileSize);


    aclrtMalloc((void**)&input_Device, src0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&local_max_Device, dst0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&local_sum_Device, dst0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&new_global_max_Device, dst0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&new_global_sum_Device, dst0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&exp_max_Device, dst0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&x_exp_Device, dst1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    

    ReadFile(GetGoldenDir() + "/input.bin", src0FileSize, input_Host, src0FileSize);
    ReadFile(GetGoldenDir() + "/golden_global_max_in.bin", dst0FileSize, new_global_max_Host, dst0FileSize);
    ReadFile(GetGoldenDir() + "/golden_global_sum_in.bin", dst0FileSize, new_global_sum_Host, dst0FileSize);

    #ifdef DEBUG_SOFTMAX_FA
        for(int i = 0; i < 10; i++) {
            std::cout << "input_Host[" << i << "] = " << input_Host[i] << std::endl;
        }
    #endif

    aclrtMemcpy(input_Device, src0FileSize, input_Host, src0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(new_global_max_Device, dst0FileSize, new_global_max_Host, dst0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(new_global_sum_Device, dst0FileSize, new_global_sum_Host, dst0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTSOFTMAX_dn_fusion<seq, headSize, init, CAUSAL_MASK>(x_exp_Device, input_Device, local_max_Device, local_sum_Device, new_global_max_Device, new_global_sum_Device, exp_max_Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(x_exp_Host, dst1FileSize, x_exp_Device, dst1FileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(local_max_Host, dst0FileSize, local_max_Device, dst0FileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(exp_max_Host, dst0FileSize, exp_max_Device, dst0FileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(new_global_max_Host, dst0FileSize, new_global_max_Device, dst0FileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(local_sum_Host, dst0FileSize, local_sum_Device, dst0FileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(new_global_sum_Host, dst0FileSize, new_global_sum_Device, dst0FileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_x_exp.bin", x_exp_Host, dst1FileSize);
    WriteFile(GetGoldenDir() + "/output_local_max.bin", local_max_Host, dst0FileSize);
    WriteFile(GetGoldenDir() + "/output_global_max.bin", new_global_max_Host, dst0FileSize);
    WriteFile(GetGoldenDir() + "/output_exp_max.bin", exp_max_Host, dst0FileSize);
    WriteFile(GetGoldenDir() + "/output_local_sum.bin", local_sum_Host, dst0FileSize);
    WriteFile(GetGoldenDir() + "/output_global_sum.bin", new_global_sum_Host, dst0FileSize);

    aclrtFree(input_Device);
    aclrtFree(local_max_Device);
    aclrtFree(local_sum_Device);
    aclrtFree(new_global_max_Device);
    aclrtFree(new_global_sum_Device);
    aclrtFree(exp_max_Device);
    aclrtFree(x_exp_Device);

    aclrtFreeHost(input_Host);
    aclrtFreeHost(local_max_Host);
    aclrtFreeHost(local_sum_Host);
    aclrtFreeHost(new_global_max_Host);
    aclrtFreeHost(new_global_sum_Host);
    aclrtFreeHost(exp_max_Host);
    aclrtFreeHost(x_exp_Host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<aclFloat16> golden(dst1FileSize);
    std::vector<aclFloat16> devFinal(dst1FileSize);
    ReadFile(GetGoldenDir() + "/golden_x_exp.bin", dst1FileSize, golden.data(), dst1FileSize);
    ReadFile(GetGoldenDir() + "/output_x_exp.bin", dst1FileSize, devFinal.data(), dst1FileSize);
    std::vector<float> golden1(dst0FileSize);
    std::vector<float> devFinal1(dst0FileSize);
    std::vector<float> golden2(dst0FileSize);
    std::vector<float> devFinal2(dst0FileSize);
    std::vector<float> golden3(dst0FileSize);
    std::vector<float> devFinal3(dst0FileSize);
    std::vector<float> golden4(dst0FileSize);
    std::vector<float> devFinal4(dst0FileSize);
    std::vector<float> golden5(dst0FileSize);
    std::vector<float> devFinal5(dst0FileSize);
    ReadFile(GetGoldenDir() + "/golden_local_max.bin", dst0FileSize, golden1.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/output_local_max.bin", dst0FileSize, devFinal1.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/golden_exp_max.bin", dst0FileSize, golden2.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/output_exp_max.bin", dst0FileSize, devFinal2.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/golden_local_sum.bin", dst0FileSize, golden3.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/output_local_sum.bin", dst0FileSize, devFinal3.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/golden_global_max.bin", dst0FileSize, golden4.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/output_global_max.bin", dst0FileSize, devFinal4.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/golden_global_sum.bin", dst0FileSize, golden5.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/output_global_sum.bin", dst0FileSize, devFinal5.data(), dst0FileSize);

    bool ret0 = false;
    bool ret1 = false;
    bool ret2 = false;
    bool ret3 = false;
    bool ret4 = false;
    bool ret5 = false;
    bool ret = false;
    if(init){
        ret0 = ResultCmp(golden, devFinal, 0.01f);
        ret4 = ResultCmp(golden4, devFinal4, 0.001f);
        ret5 = ResultCmp(golden5, devFinal5, 0.001f);
        ret = ret0 && ret4 && ret5;
    }
    else {
        ret0 = ResultCmp(golden, devFinal, 0.01f);
        ret1 = ResultCmp(golden1, devFinal1, 0.001f);
        ret2 = ResultCmp(golden2, devFinal2, 0.001f);
        ret3 = ResultCmp(golden3, devFinal3, 0.001f);
        ret5 = ResultCmp(golden5, devFinal5, 0.001f);
        ret = ret0 && ret1 && ret2 && ret3 && ret5;
    }
    

    EXPECT_TRUE(ret);
}

template <int seq, int headSize, int init, bool CAUSAL_MASK>
void test_softmax_dn_no_fusion()
{
    size_t src0FileSize = headSize * seq * sizeof(float);  // input_x_local_UB
    size_t dst0FileSize = headSize * sizeof(float);  // local_max, local_sum, new_global_max, new_global_sum, exp_max
    size_t dst1FileSize = headSize * seq * sizeof(aclFloat16);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    aclFloat16 *x_exp_Host;
    float *input_Host, *local_max_Host, *local_sum_Host, *new_global_max_Host, *new_global_sum_Host, *exp_max_Host;
    aclFloat16 *x_exp_Device;
    float *input_Device, *local_max_Device, *local_sum_Device, *new_global_max_Device, *new_global_sum_Device, *exp_max_Device;

    aclrtMallocHost((void**)(&input_Host), src0FileSize);
    aclrtMallocHost((void**)(&local_max_Host), dst0FileSize);
    aclrtMallocHost((void**)(&local_sum_Host), dst0FileSize);
    aclrtMallocHost((void**)(&new_global_max_Host), dst0FileSize);
    aclrtMallocHost((void**)(&new_global_sum_Host), dst0FileSize);
    aclrtMallocHost((void**)(&exp_max_Host), dst0FileSize);
    aclrtMallocHost((void**)(&x_exp_Host), dst1FileSize);


    aclrtMalloc((void**)&input_Device, src0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&local_max_Device, dst0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&local_sum_Device, dst0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&new_global_max_Device, dst0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&new_global_sum_Device, dst0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&exp_max_Device, dst0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&x_exp_Device, dst1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    

    ReadFile(GetGoldenDir() + "/input.bin", src0FileSize, input_Host, src0FileSize);
    ReadFile(GetGoldenDir() + "/golden_global_max_in.bin", dst0FileSize, new_global_max_Host, dst0FileSize);
    ReadFile(GetGoldenDir() + "/golden_global_sum_in.bin", dst0FileSize, new_global_sum_Host, dst0FileSize);

    aclrtMemcpy(input_Device, src0FileSize, input_Host, src0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(new_global_max_Device, dst0FileSize, new_global_max_Host, dst0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(new_global_max_Host, dst0FileSize, new_global_max_Device, dst0FileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(new_global_sum_Device, dst0FileSize, new_global_sum_Host, dst0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTSOFTMAX_dn_nofusion<seq, headSize, init, CAUSAL_MASK>(x_exp_Device, input_Device, local_max_Device, local_sum_Device, new_global_max_Device, new_global_sum_Device, exp_max_Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(x_exp_Host, dst1FileSize, x_exp_Device, dst1FileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(local_max_Host, dst0FileSize, local_max_Device, dst0FileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(exp_max_Host, dst0FileSize, exp_max_Device, dst0FileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(local_sum_Host, dst0FileSize, local_sum_Device, dst0FileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(new_global_sum_Host, dst0FileSize, new_global_sum_Device, dst0FileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_x_exp.bin", x_exp_Host, dst1FileSize);
    WriteFile(GetGoldenDir() + "/output_local_max.bin", local_max_Host, dst0FileSize);
    WriteFile(GetGoldenDir() + "/output_global_max.bin", new_global_max_Host, dst0FileSize);
    WriteFile(GetGoldenDir() + "/output_exp_max.bin", exp_max_Host, dst0FileSize);
    WriteFile(GetGoldenDir() + "/output_local_sum.bin", local_sum_Host, dst0FileSize);
    WriteFile(GetGoldenDir() + "/output_global_sum.bin", new_global_sum_Host, dst0FileSize);

    aclrtFree(input_Device);
    aclrtFree(local_max_Device);
    aclrtFree(local_sum_Device);
    aclrtFree(new_global_max_Device);
    aclrtFree(new_global_sum_Device);
    aclrtFree(exp_max_Device);
    aclrtFree(x_exp_Device);

    aclrtFreeHost(input_Host);
    aclrtFreeHost(local_max_Host);
    aclrtFreeHost(local_sum_Host);
    aclrtFreeHost(new_global_max_Host);
    aclrtFreeHost(new_global_sum_Host);
    aclrtFreeHost(exp_max_Host);
    aclrtFreeHost(x_exp_Host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<aclFloat16> golden(dst1FileSize);
    std::vector<aclFloat16> devFinal(dst1FileSize);
    ReadFile(GetGoldenDir() + "/golden_x_exp.bin", dst1FileSize, golden.data(), dst1FileSize);
    ReadFile(GetGoldenDir() + "/output_x_exp.bin", dst1FileSize, devFinal.data(), dst1FileSize);
    std::vector<float> golden1(dst0FileSize);
    std::vector<float> devFinal1(dst0FileSize);
    std::vector<float> golden2(dst0FileSize);
    std::vector<float> devFinal2(dst0FileSize);
    std::vector<float> golden3(dst0FileSize);
    std::vector<float> devFinal3(dst0FileSize);
    std::vector<float> golden4(dst0FileSize);
    std::vector<float> devFinal4(dst0FileSize);
    std::vector<float> golden5(dst0FileSize);
    std::vector<float> devFinal5(dst0FileSize);
    ReadFile(GetGoldenDir() + "/golden_local_max.bin", dst0FileSize, golden1.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/output_local_max.bin", dst0FileSize, devFinal1.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/golden_exp_max.bin", dst0FileSize, golden2.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/output_exp_max.bin", dst0FileSize, devFinal2.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/golden_local_sum.bin", dst0FileSize, golden3.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/output_local_sum.bin", dst0FileSize, devFinal3.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/golden_global_max.bin", dst0FileSize, golden4.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/output_global_max.bin", dst0FileSize, devFinal4.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/golden_global_sum.bin", dst0FileSize, golden5.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/output_global_sum.bin", dst0FileSize, devFinal5.data(), dst0FileSize);

    bool ret0 = false;
    bool ret1 = false;
    bool ret2 = false;
    bool ret3 = false;
    bool ret4 = false;
    bool ret5 = false;
    bool ret = false;
    if(init){
        // ret = ret0 && ret1 && ret5;
        ret0 = ResultCmp(golden, devFinal, 0.01f);
        ret5 = ResultCmp(golden5, devFinal5, 0.001f);
        ret = ret0 && ret5;    //dont need to check local max
    }
    else {
        // ret = ret0 && ret1 && ret2 && ret3 && ret5;
        ret0 = ResultCmp(golden, devFinal, 0.01f);
        ret2 = ResultCmp(golden2, devFinal2, 0.001f);
        ret3 = ResultCmp(golden3, devFinal3, 0.001f);
        ret5 = ResultCmp(golden5, devFinal5, 0.001f);
        ret = ret0 && ret2 && ret3 && ret5;    //dont need to check local max
    }

    EXPECT_TRUE(ret);
}

template <int seq, int headSize, int init, bool CAUSAL_MASK>
void test_softmax_nd_fusion()
{
    size_t src0FileSize = headSize * seq * sizeof(float);  // input_x_local_UB
    size_t dst0FileSize = headSize * sizeof(float);  // local_max, local_sum, new_global_max, new_global_sum, exp_max
    size_t dst1FileSize = headSize * seq * sizeof(aclFloat16);  // x_exp

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    aclFloat16 *x_exp_Host;
    float *input_Host, *local_max_Host, *local_sum_Host, *new_global_max_Host, *new_global_sum_Host, *exp_max_Host;
    aclFloat16 *x_exp_Device;
    float *input_Device, *local_max_Device, *local_sum_Device, *new_global_max_Device, *new_global_sum_Device, *exp_max_Device;

    aclrtMallocHost((void**)(&input_Host), src0FileSize);
    aclrtMallocHost((void**)(&local_max_Host), dst0FileSize);
    aclrtMallocHost((void**)(&local_sum_Host), dst0FileSize);
    aclrtMallocHost((void**)(&new_global_max_Host), dst0FileSize);
    aclrtMallocHost((void**)(&new_global_sum_Host), dst0FileSize);
    aclrtMallocHost((void**)(&exp_max_Host), dst0FileSize);
    aclrtMallocHost((void**)(&x_exp_Host), dst1FileSize);


    aclrtMalloc((void**)&input_Device, src0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&local_max_Device, dst0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&local_sum_Device, dst0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&new_global_max_Device, dst0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&new_global_sum_Device, dst0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&exp_max_Device, dst0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&x_exp_Device, dst1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    

    ReadFile(GetGoldenDir() + "/input.bin", src0FileSize, input_Host, src0FileSize);
    ReadFile(GetGoldenDir() + "/golden_global_max_in.bin", dst0FileSize, new_global_max_Host, dst0FileSize);
    ReadFile(GetGoldenDir() + "/golden_global_sum_in.bin", dst0FileSize, new_global_sum_Host, dst0FileSize);

    aclrtMemcpy(input_Device, src0FileSize, input_Host, src0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(new_global_max_Device, dst0FileSize, new_global_max_Host, dst0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(new_global_sum_Device, dst0FileSize, new_global_sum_Host, dst0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTSOFTMAX_nd_fusion<seq, headSize, init, CAUSAL_MASK>(x_exp_Device, input_Device, local_max_Device, local_sum_Device, new_global_max_Device, new_global_sum_Device, exp_max_Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(x_exp_Host, dst1FileSize, x_exp_Device, dst1FileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(local_max_Host, dst0FileSize, local_max_Device, dst0FileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(exp_max_Host, dst0FileSize, exp_max_Device, dst0FileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(local_sum_Host, dst0FileSize, local_sum_Device, dst0FileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(new_global_max_Host, dst0FileSize, new_global_max_Device, dst0FileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(new_global_sum_Host, dst0FileSize, new_global_sum_Device, dst0FileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_x_exp.bin", x_exp_Host, dst1FileSize);
    WriteFile(GetGoldenDir() + "/output_local_max.bin", local_max_Host, dst0FileSize);
    WriteFile(GetGoldenDir() + "/output_global_max.bin", new_global_max_Host, dst0FileSize);
    WriteFile(GetGoldenDir() + "/output_exp_max.bin", exp_max_Host, dst0FileSize);
    WriteFile(GetGoldenDir() + "/output_local_sum.bin", local_sum_Host, dst0FileSize);
    WriteFile(GetGoldenDir() + "/output_global_sum.bin", new_global_sum_Host, dst0FileSize);

    aclrtFree(input_Device);
    aclrtFree(local_max_Device);
    aclrtFree(local_sum_Device);
    aclrtFree(new_global_max_Device);
    aclrtFree(new_global_sum_Device);
    aclrtFree(exp_max_Device);
    aclrtFree(x_exp_Device);

    aclrtFreeHost(input_Host);
    aclrtFreeHost(local_max_Host);
    aclrtFreeHost(local_sum_Host);
    aclrtFreeHost(new_global_max_Host);
    aclrtFreeHost(new_global_sum_Host);
    aclrtFreeHost(exp_max_Host);
    aclrtFreeHost(x_exp_Host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<aclFloat16> golden(dst1FileSize);
    std::vector<aclFloat16> devFinal(dst1FileSize);
    ReadFile(GetGoldenDir() + "/golden_x_exp.bin", dst1FileSize, golden.data(), dst1FileSize);
    ReadFile(GetGoldenDir() + "/output_x_exp.bin", dst1FileSize, devFinal.data(), dst1FileSize);
    std::vector<float> golden1(dst0FileSize);
    std::vector<float> devFinal1(dst0FileSize);
    std::vector<float> golden2(dst0FileSize);
    std::vector<float> devFinal2(dst0FileSize);
    std::vector<float> golden3(dst0FileSize);
    std::vector<float> devFinal3(dst0FileSize);
    std::vector<float> golden4(dst0FileSize);
    std::vector<float> devFinal4(dst0FileSize);
    std::vector<float> golden5(dst0FileSize);
    std::vector<float> devFinal5(dst0FileSize);
    ReadFile(GetGoldenDir() + "/golden_local_max.bin", dst0FileSize, golden1.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/output_local_max.bin", dst0FileSize, devFinal1.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/golden_exp_max.bin", dst0FileSize, golden2.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/output_exp_max.bin", dst0FileSize, devFinal2.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/golden_local_sum.bin", dst0FileSize, golden3.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/output_local_sum.bin", dst0FileSize, devFinal3.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/golden_global_max.bin", dst0FileSize, golden4.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/output_global_max.bin", dst0FileSize, devFinal4.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/golden_global_sum.bin", dst0FileSize, golden5.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/output_global_sum.bin", dst0FileSize, devFinal5.data(), dst0FileSize);

    bool ret0 = false;
    bool ret1 = false;
    bool ret2 = false;
    bool ret3 = false;
    bool ret4 = false;
    bool ret5 = false;
    bool ret = false;
    if(init){
        ret0 = ResultCmp(golden, devFinal, 0.01f);
        ret4 = ResultCmp(golden4, devFinal4, 0.001f);
        ret5 = ResultCmp(golden5, devFinal5, 0.001f);
        ret = ret0 && ret4 && ret5;
    }
    else {
        ret0 = ResultCmp(golden, devFinal, 0.01f);
        ret1 = ResultCmp(golden1, devFinal1, 0.001f);
        ret2 = ResultCmp(golden2, devFinal2, 0.001f);
        ret3 = ResultCmp(golden3, devFinal3, 0.001f);
        ret5 = ResultCmp(golden5, devFinal5, 0.001f);
        ret = ret0 && ret1 && ret2 && ret3 && ret5;
    }

    EXPECT_TRUE(ret);
}

template <int seq, int headSize, int init, bool CAUSAL_MASK>
void test_softmax_nd_no_fusion()
{
    size_t src0FileSize = headSize * seq * sizeof(float);  // input_x_local_UB
    size_t dst0FileSize = headSize * sizeof(float);  // local_max, local_sum, new_global_max, new_global_sum, exp_max
    size_t dst1FileSize = headSize * seq * sizeof(aclFloat16);  // x_exp

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    aclFloat16 *x_exp_Host;
    float *input_Host, *local_max_Host, *local_sum_Host, *new_global_max_Host, *new_global_sum_Host, *exp_max_Host;
    aclFloat16 *x_exp_Device;
    float *input_Device, *local_max_Device, *local_sum_Device, *new_global_max_Device, *new_global_sum_Device, *exp_max_Device;

    aclrtMallocHost((void**)(&input_Host), src0FileSize);
    aclrtMallocHost((void**)(&local_max_Host), dst0FileSize);
    aclrtMallocHost((void**)(&local_sum_Host), dst0FileSize);
    aclrtMallocHost((void**)(&new_global_max_Host), dst0FileSize);
    aclrtMallocHost((void**)(&new_global_sum_Host), dst0FileSize);
    aclrtMallocHost((void**)(&exp_max_Host), dst0FileSize);
    aclrtMallocHost((void**)(&x_exp_Host), dst1FileSize);


    aclrtMalloc((void**)&input_Device, src0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&local_max_Device, dst0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&local_sum_Device, dst0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&new_global_max_Device, dst0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&new_global_sum_Device, dst0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&exp_max_Device, dst0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&x_exp_Device, dst1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    

    ReadFile(GetGoldenDir() + "/input.bin", src0FileSize, input_Host, src0FileSize);
    ReadFile(GetGoldenDir() + "/golden_global_max_in.bin", dst0FileSize, new_global_max_Host, dst0FileSize);
    ReadFile(GetGoldenDir() + "/golden_global_sum_in.bin", dst0FileSize, new_global_sum_Host, dst0FileSize);

    aclrtMemcpy(input_Device, src0FileSize, input_Host, src0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(new_global_max_Device, dst0FileSize, new_global_max_Host, dst0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(new_global_sum_Device, dst0FileSize, new_global_sum_Host, dst0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTSOFTMAX_nd_nofusion<seq, headSize, init, CAUSAL_MASK>(x_exp_Device, input_Device, local_max_Device, local_sum_Device, new_global_max_Device, new_global_sum_Device, exp_max_Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(x_exp_Host, dst1FileSize, x_exp_Device, dst1FileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(local_max_Host, dst0FileSize, local_max_Device, dst0FileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(exp_max_Host, dst0FileSize, exp_max_Device, dst0FileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(local_sum_Host, dst0FileSize, local_sum_Device, dst0FileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(new_global_max_Host, dst0FileSize, new_global_max_Device, dst0FileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(new_global_sum_Host, dst0FileSize, new_global_sum_Device, dst0FileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_x_exp.bin", x_exp_Host, dst1FileSize);
    WriteFile(GetGoldenDir() + "/output_local_max.bin", local_max_Host, dst0FileSize);
    WriteFile(GetGoldenDir() + "/output_global_max.bin", new_global_max_Host, dst0FileSize);
    WriteFile(GetGoldenDir() + "/output_exp_max.bin", exp_max_Host, dst0FileSize);
    WriteFile(GetGoldenDir() + "/output_local_sum.bin", local_sum_Host, dst0FileSize);
    WriteFile(GetGoldenDir() + "/output_global_sum.bin", new_global_sum_Host, dst0FileSize);

    aclrtFree(input_Device);
    aclrtFree(local_max_Device);
    aclrtFree(local_sum_Device);
    aclrtFree(new_global_max_Device);
    aclrtFree(new_global_sum_Device);
    aclrtFree(exp_max_Device);
    aclrtFree(x_exp_Device);

    aclrtFreeHost(input_Host);
    aclrtFreeHost(local_max_Host);
    aclrtFreeHost(local_sum_Host);
    aclrtFreeHost(new_global_max_Host);
    aclrtFreeHost(new_global_sum_Host);
    aclrtFreeHost(exp_max_Host);
    aclrtFreeHost(x_exp_Host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<aclFloat16> golden(dst1FileSize);
    std::vector<aclFloat16> devFinal(dst1FileSize);
    ReadFile(GetGoldenDir() + "/golden_x_exp.bin", dst1FileSize, golden.data(), dst1FileSize);
    ReadFile(GetGoldenDir() + "/output_x_exp.bin", dst1FileSize, devFinal.data(), dst1FileSize);
    std::vector<float> golden1(dst0FileSize);
    std::vector<float> devFinal1(dst0FileSize);
    std::vector<float> golden2(dst0FileSize);
    std::vector<float> devFinal2(dst0FileSize);
    std::vector<float> golden3(dst0FileSize);
    std::vector<float> devFinal3(dst0FileSize);
    std::vector<float> golden4(dst0FileSize);
    std::vector<float> devFinal4(dst0FileSize);
    std::vector<float> golden5(dst0FileSize);
    std::vector<float> devFinal5(dst0FileSize);
    ReadFile(GetGoldenDir() + "/golden_local_max.bin", dst0FileSize, golden1.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/output_local_max.bin", dst0FileSize, devFinal1.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/golden_exp_max.bin", dst0FileSize, golden2.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/output_exp_max.bin", dst0FileSize, devFinal2.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/golden_local_sum.bin", dst0FileSize, golden3.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/output_local_sum.bin", dst0FileSize, devFinal3.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/golden_global_max.bin", dst0FileSize, golden4.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/output_global_max.bin", dst0FileSize, devFinal4.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/golden_global_sum.bin", dst0FileSize, golden5.data(), dst0FileSize);
    ReadFile(GetGoldenDir() + "/output_global_sum.bin", dst0FileSize, devFinal5.data(), dst0FileSize);

    bool ret0 = false;
    bool ret1 = false;
    bool ret2 = false;
    bool ret3 = false;
    bool ret4 = false;
    bool ret5 = false;
    bool ret = false;
    if(init){
        ret0 = ResultCmp(golden, devFinal, 0.01f);
        ret4 = ResultCmp(golden4, devFinal4, 0.001f);
        ret5 = ResultCmp(golden5, devFinal5, 0.001f);
        ret = ret0 && ret4 && ret5;
    }
    else {
        ret0 = ResultCmp(golden, devFinal, 0.01f);
        ret1 = ResultCmp(golden1, devFinal1, 0.001f);
        ret2 = ResultCmp(golden2, devFinal2, 0.001f);
        ret3 = ResultCmp(golden3, devFinal3, 0.001f);
        ret5 = ResultCmp(golden5, devFinal5, 0.001f);
        ret = ret0 && ret1 && ret2 && ret3 && ret5;
    }

    EXPECT_TRUE(ret);
}

TEST_F(TSOFTMAXFATest, case1_B1_N1_S128_H64_DN_fusion_init)
{
    test_softmax_dn_fusion<128, 64, 1, 0>();
}

TEST_F(TSOFTMAXFATest, case2_B1_N1_S256_H64_DN_fusion_init)
{
    test_softmax_dn_fusion<256, 64, 1, 0>();
}

TEST_F(TSOFTMAXFATest, case3_B1_N1_S256_H64_DN_no_fusion_init)
{
    test_softmax_dn_no_fusion<256, 64, 1, 0>();
}

TEST_F(TSOFTMAXFATest, case4_B1_N1_S128_H64_DN_no_fusion_init)
{
    test_softmax_dn_no_fusion<128, 64, 1, 0>();
}

TEST_F(TSOFTMAXFATest, case5_B1_N1_S256_H64_ND_no_fusion_init)
{
    test_softmax_nd_no_fusion<256, 64, 1, 0>();
}

TEST_F(TSOFTMAXFATest, case6_B1_N1_S128_H64_ND_no_fusion_init)
{
    test_softmax_nd_no_fusion<128, 64, 1, 0>();
}

TEST_F(TSOFTMAXFATest, case7_B1_N1_S128_H64_DN_fusion_no_init)
{
    test_softmax_dn_fusion<128, 64, 0, 0>();
}

TEST_F(TSOFTMAXFATest, case8_B1_N1_S128_H64_DN_no_fusion_no_init)
{
    test_softmax_dn_no_fusion<128, 64, 0, 0>();
}

TEST_F(TSOFTMAXFATest, case9_B1_N1_S128_H64_ND_no_fusion_no_init)
{
    test_softmax_nd_no_fusion<128, 64, 0, 0>();
}

TEST_F(TSOFTMAXFATest, case10_B1_N1_S256_H64_ND_no_fusion_no_init)
{
    test_softmax_nd_no_fusion<256, 64, 0, 0>();
}

TEST_F(TSOFTMAXFATest, case11_B1_N1_S256_H64_DN_fusion_no_init)
{
    test_softmax_dn_fusion<256, 64, 0, 0>();
}

TEST_F(TSOFTMAXFATest, case12_B1_N1_S256_H64_DN_no_fusion_no_init)
{
    test_softmax_dn_no_fusion<256, 64, 0, 0>();
}

TEST_F(TSOFTMAXFATest, case13_B1_N1_S128_H64_ND_fusion_init)
{
    test_softmax_nd_fusion<128, 64, 1, 0>();
}

TEST_F(TSOFTMAXFATest, case14_B1_N1_S128_H64_ND_fusion_no_init)
{
    test_softmax_nd_fusion<128, 64, 0, 0>();
}

TEST_F(TSOFTMAXFATest, case15_B1_N1_S256_H64_ND_fusion_init)
{
    test_softmax_nd_fusion<256, 64, 1, 0>();
}

TEST_F(TSOFTMAXFATest, case16_B1_N1_S256_H64_ND_fusion_no_init)
{
    test_softmax_nd_fusion<256, 64, 0, 0>();
}

TEST_F(TSOFTMAXFATest, case17_B1_N1_S64_H64_DN_fusion_init)
{
    test_softmax_dn_fusion<64, 64, 1, 0>();
}

TEST_F(TSOFTMAXFATest, case18_B1_N1_S64_H64_DN_fusion_no_init)
{
    test_softmax_dn_fusion<64, 64, 0, 0>();
}

TEST_F(TSOFTMAXFATest, case19_B1_N1_S64_H64_ND_fusion_init)
{
    test_softmax_nd_fusion<64, 64, 1, 0>();
}

TEST_F(TSOFTMAXFATest, case20_B1_N1_S64_H64_ND_fusion_no_init)
{
    test_softmax_nd_fusion<64, 64, 0, 0>();
}

TEST_F(TSOFTMAXFATest, case21_B1_N1_S64_H128_ND_no_fusion_no_init)
{
    test_softmax_nd_no_fusion<64, 128, 0, 0>();
}

TEST_F(TSOFTMAXFATest, case22_B1_N1_S64_H128_ND_no_fusion_init)
{
    test_softmax_nd_no_fusion<64, 128, 1, 0>();
}

TEST_F(TSOFTMAXFATest, case23_B1_N1_S64_H128_ND_fusion_no_init)
{
    test_softmax_nd_fusion<64, 128, 0, 0>();
}

TEST_F(TSOFTMAXFATest, case24_B1_N1_S64_H128_ND_fusion_init)
{
    test_softmax_nd_fusion<64, 128, 1, 0>();
}

TEST_F(TSOFTMAXFATest, case25_B1_N1_S128_H128_ND_no_fusion_no_init)
{
    test_softmax_nd_no_fusion<128, 128, 0, 0>();
}

TEST_F(TSOFTMAXFATest, case26_B1_N1_S128_H128_ND_no_fusion_init)
{
    test_softmax_nd_no_fusion<128, 128, 1, 0>();
}

TEST_F(TSOFTMAXFATest, case27_B1_N1_S128_H128_ND_fusion_no_init)
{
    test_softmax_nd_fusion<128, 128, 0, 0>();
}

TEST_F(TSOFTMAXFATest, case28_B1_N1_S128_H128_ND_fusion_init)
{
    test_softmax_nd_fusion<128, 128, 1, 0>();
}

TEST_F(TSOFTMAXFATest, case29_B1_N1_S64_H128_ND_fusion_no_init)
{
    test_softmax_nd_fusion<64, 128, 0, 0>();
}

TEST_F(TSOFTMAXFATest, case30_B1_N1_S128_H128_DN_no_fusion_init)
{
    test_softmax_dn_no_fusion<128, 128, 1, 0>();
}

TEST_F(TSOFTMAXFATest, case31_B1_N1_S128_H128_DN_fusion_init)
{
    test_softmax_dn_fusion<128, 128, 1, 0>();
}

TEST_F(TSOFTMAXFATest, case32_B1_N1_S128_H128_DN_no_fusion_no_init)
{
    test_softmax_dn_no_fusion<128, 128, 0, 0>();
}

TEST_F(TSOFTMAXFATest, case33_B1_N1_S128_H128_DN_fusion_no_init)
{
    test_softmax_dn_fusion<128, 128, 0, 0>();
}

TEST_F(TSOFTMAXFATest, case34_B1_N1_S128_H64_ND_no_fusion_init)
{
    test_softmax_nd_no_fusion<128, 64, 1, 1>();
}

TEST_F(TSOFTMAXFATest, case35_B1_N1_S512_H32_ND_no_fusion_init)
{
    test_softmax_nd_no_fusion<512, 32, 1, 0>();
}

TEST_F(TSOFTMAXFATest, case36_B1_N1_S1024_H16_ND_no_fusion_init)
{
    test_softmax_nd_no_fusion<1024, 16, 1, 0>();
}

TEST_F(TSOFTMAXFATest, case37_B1_N1_S512_H32_ND_fusion_init)
{
    test_softmax_nd_fusion<512, 32, 1, 0>();
}

TEST_F(TSOFTMAXFATest, case38_B1_N1_S1024_H16_ND_fusion_init)
{
    test_softmax_nd_fusion<1024, 16, 1, 0>();
}

TEST_F(TSOFTMAXFATest, case39_B1_N1_S512_H32_DN_fusion_init)
{
    test_softmax_dn_fusion<512, 32, 1, 0>();
}