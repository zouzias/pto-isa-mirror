#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <typename T, int Rows, int Cols>
void LaunchTgu(T *out, T *prev, T *est, T *exp_max, T *pv_pend, void *stream);

template <typename T, int Rows, int Cols>
void LaunchTguLast(T *out, T *prev, T *est, T *exp_max, T *global_sum, T *pv_pend, void *stream);

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    const std::string suiteName = testInfo->test_suite_name();
    return "../" + suiteName + "." + caseName;
}

template <typename T, int Rows, int Cols>
void test_tgu(bool last_tile)
{
    size_t num_elems = static_cast<size_t>(Rows) * static_cast<size_t>(Cols);
    size_t bytes = num_elems * sizeof(T);
    size_t reduce_bytes = static_cast<size_t>(Rows) * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *prevHost = nullptr;
    T *estHost = nullptr;
    T *pvHost = nullptr;
    T *outHost = nullptr;
    T *expMaxHost = nullptr;
    T *globalSumHost = nullptr;

    aclrtMallocHost(reinterpret_cast<void **>(&prevHost), bytes);
    aclrtMallocHost(reinterpret_cast<void **>(&estHost), bytes);
    aclrtMallocHost(reinterpret_cast<void **>(&pvHost), bytes);
    aclrtMallocHost(reinterpret_cast<void **>(&outHost), bytes);
    aclrtMallocHost(reinterpret_cast<void **>(&expMaxHost), reduce_bytes);
    aclrtMallocHost(reinterpret_cast<void **>(&globalSumHost), reduce_bytes);

    ReadFile(GetGoldenDir() + "/input0.bin", bytes, prevHost, bytes);
    ReadFile(GetGoldenDir() + "/input1.bin", bytes, estHost, bytes);
    ReadFile(GetGoldenDir() + "/input2.bin", reduce_bytes, expMaxHost, reduce_bytes);
    ReadFile(GetGoldenDir() + "/input3.bin", bytes, pvHost, bytes);
    if (last_tile) {
        ReadFile(GetGoldenDir() + "/global_sum.bin", reduce_bytes, globalSumHost, reduce_bytes);
    }

    T *prevDevice = nullptr;
    T *estDevice = nullptr;
    T *pvDevice = nullptr;
    T *outDevice = nullptr;
    T *expMaxDevice = nullptr;
    T *globalSumDevice = nullptr;

    aclrtMalloc(reinterpret_cast<void **>(&prevDevice), bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void **>(&estDevice), bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void **>(&pvDevice), bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void **>(&outDevice), bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void **>(&expMaxDevice), reduce_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void **>(&globalSumDevice), reduce_bytes, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(prevDevice, bytes, prevHost, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(estDevice, bytes, estHost, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(pvDevice, bytes, pvHost, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(expMaxDevice, reduce_bytes, expMaxHost, reduce_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (last_tile) {
        aclrtMemcpy(globalSumDevice, reduce_bytes, globalSumHost, reduce_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }

    if (last_tile) {
        LaunchTguLast<T, Rows, Cols>(outDevice, prevDevice, estDevice, expMaxDevice, globalSumDevice, pvDevice, stream);
    } else {
        LaunchTgu<T, Rows, Cols>(outDevice, prevDevice, estDevice, expMaxDevice, pvDevice, stream);
    }

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outHost, bytes, outDevice, bytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", outHost, bytes);

    std::vector<T> golden(num_elems);
    std::vector<T> devFinal(num_elems);
    ReadFile(GetGoldenDir() + "/golden.bin", bytes, golden.data(), bytes);
    ReadFile(GetGoldenDir() + "/output.bin", bytes, devFinal.data(), bytes);
    bool cmp = ResultCmp(golden, devFinal, 1e-3f);

    aclrtFree(prevDevice);
    aclrtFree(estDevice);
    aclrtFree(pvDevice);
    aclrtFree(outDevice);
    aclrtFree(expMaxDevice);
    aclrtFree(globalSumDevice);

    aclrtFreeHost(prevHost);
    aclrtFreeHost(estHost);
    aclrtFreeHost(pvHost);
    aclrtFreeHost(outHost);
    aclrtFreeHost(expMaxHost);
    aclrtFreeHost(globalSumHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    EXPECT_TRUE(cmp);
}

TEST(TGUTest, case1_regular)
{
    test_tgu<float, 32, 128>(false);
}

TEST(TGUTest, case2_last_tile)
{
    test_tgu<float, 32, 128>(true);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
