/**
 * main.cpp - host driver for add_tile_array (auto-mode A3 prototype).
 *
 * Pattern source: kernels/manual/a2a3/topk/main.cpp (uses test_common.h
 * ReadFile / WriteFile / ResultCmp; reads ../input/*.bin and writes
 * ../output/*.bin relative to the build/ directory).
 *
 * I/O contract (all little-endian float32, contiguous, no header):
 *   ../input/input_a.bin    (TOTAL_ELEMENTS * sizeof(float) bytes)
 *   ../input/input_b.bin    (same)
 *   ../output/golden_c.bin  (same; written by scripts/gen_data.py)
 *   ../output/output_c.bin  (same; written by this driver)
 *
 * Inputs are integer-valued floats in [1, 10] (see scripts/gen_data.py),
 * so element-wise sum is exact in IEEE-754 float32; ResultCmp tolerance
 * 0.001f is comfortably wide.
 */

#include "test_common.h"
#include "acl/acl.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

template <typename T>
void launchAddTileArray(T *c, T *a, T *b, void *stream);

template <typename T, int NUM_TILES_, int TILE_ROWS_, int TILE_COLS_>
inline bool ValidateDataResults(size_t outFileSize)
{
    std::vector<T> golden(outFileSize / sizeof(T));
    std::vector<T> devFinal(outFileSize / sizeof(T));

    ReadFile("../output/golden_c.bin", outFileSize, golden.data(), outFileSize);
    ReadFile("../output/output_c.bin", outFileSize, devFinal.data(), outFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    if (ret) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ret;
}

template <typename T, int NUM_TILES_, int TILE_ROWS_, int TILE_COLS_>
void AddTileArray()
{
    constexpr int totalRows = NUM_TILES_ * TILE_ROWS_;
    constexpr int cols      = TILE_COLS_;
    constexpr size_t totalElements = static_cast<size_t>(totalRows) * cols;
    const size_t fileSize = totalElements * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *aHost = nullptr, *bHost = nullptr, *cHost = nullptr;
    T *aDevice = nullptr, *bDevice = nullptr, *cDevice = nullptr;

    aclrtMallocHost((void **)(&aHost), fileSize);
    aclrtMallocHost((void **)(&bHost), fileSize);
    aclrtMallocHost((void **)(&cHost), fileSize);

    aclrtMalloc((void **)&aDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&bDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&cDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("../input/input_a.bin", fileSize, aHost, fileSize);
    ReadFile("../input/input_b.bin", fileSize, bHost, fileSize);

    aclrtMemcpy(aDevice, fileSize, aHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(bDevice, fileSize, bHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launchAddTileArray<T>(cDevice, aDevice, bDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(cHost, fileSize, cDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("../output/output_c.bin", cHost, fileSize);

    aclrtFree(cDevice);
    aclrtFree(bDevice);
    aclrtFree(aDevice);
    aclrtFreeHost(cHost);
    aclrtFreeHost(bHost);
    aclrtFreeHost(aHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    bool dataSuccess = ValidateDataResults<T, NUM_TILES_, TILE_ROWS_, TILE_COLS_>(fileSize);
    if (dataSuccess) {
        printf("test success\n");
    } else {
        printf("test failed\n");
    }
}

int main()
{
    constexpr int NUM_TILES = 4;
    constexpr int TILE_ROWS = 64;
    constexpr int TILE_COLS = 64;
    AddTileArray<float, NUM_TILES, TILE_ROWS, TILE_COLS>();
    return 0;
}
