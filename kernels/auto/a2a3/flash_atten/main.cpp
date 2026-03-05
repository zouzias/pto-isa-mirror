#include "runtime/rt.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// Matrix Dimensions
#define ROWS 2048
#define COLS 128
#define MATRIX_SIZE (ROWS * COLS)

#define RT_CHECK(e)                                                                \
    {                                                                              \
        if (e != RT_ERROR_NONE) {                                                  \
            fprintf(stderr, "RT Error %d at %s:%d\n", (int)e, __FILE__, __LINE__); \
            exit(-1);                                                              \
        }                                                                          \
    }

template <typename T>
static vector<T> readFile(const string &fileName)
{
    ifstream ifs(fileName.c_str(), ios::in | ios::binary | ios::ate);
    if (!ifs) {
        fprintf(stderr, "Failed to open: %s\n", fileName.c_str());
        abort();
    }
    size_t sz = ifs.tellg();
    ifs.seekg(0, ios::beg);
    vector<T> buf(sz / sizeof(T));
    ifs.read(reinterpret_cast<char *>(buf.data()), sz);
    return buf;
}

template <typename T>
static void writeFile(const string &path, const vector<T> &data)
{
    ofstream f(path, ios::binary);
    if (data.empty())
        return;
    f.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(T));
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <kernel_elf> <kernel_name>\n", argv[0]);
        return 1;
    }

    // 1. Setup Device
    RT_CHECK(rtSetDevice(0));
    rtStream_t stream = nullptr;
    RT_CHECK(rtStreamCreate(&stream, 0));

    // Number of AICORE blocks to launch
    uint32_t blockDim = 16;

    // 2. Load Kernel
    vector<char> elf = readFile<char>(argv[1]);
    rtDevBinary_t binary{RT_DEV_BINARY_MAGIC_ELF, 0, elf.data(), elf.size()};
    void *binHandle = nullptr;
    RT_CHECK(rtDevBinaryRegister(&binary, &binHandle));
    uint32_t funcStub = 0;
    RT_CHECK(rtFunctionRegister(binHandle, &funcStub, argv[2], argv[2], 3));

    // 3. Prepare Data
    string inPath = "./input_data/in.data";
    vector<uint16_t> host_data = readFile<uint16_t>(inPath);

    if (host_data.size() != 3 * MATRIX_SIZE) {
        fprintf(stderr, "Input size mismatch! Expected %d, got %zu\n", 3 * MATRIX_SIZE, host_data.size());
        return -1;
    }

    uint16_t *host_q = host_data.data();
    uint16_t *host_k = host_data.data() + MATRIX_SIZE;
    uint16_t *host_v = host_data.data() + 2 * MATRIX_SIZE;

    // Output Float32
    vector<float> host_out(MATRIX_SIZE, 0.0f);

    // 4. Device Allocations
    void *d_out = nullptr, *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_ws = nullptr;

    size_t size_inputs = MATRIX_SIZE * sizeof(uint16_t); // FP16 size
    size_t size_output = MATRIX_SIZE * sizeof(float);    // FP32 size

    // Workspace for scratch buffers (per AICORE block)
    constexpr size_t BLOCK_M = 32;
    constexpr size_t BLOCK_N = 32;
    const size_t score_tile_bytes = BLOCK_M * BLOCK_N * sizeof(float);
    const size_t prob_tile_bytes = BLOCK_M * BLOCK_N * sizeof(uint16_t); // fp16
    const size_t pv_tile_bytes = BLOCK_M * COLS * sizeof(float);
    const size_t ws_per_block = 2 * score_tile_bytes + 2 * prob_tile_bytes + pv_tile_bytes;

    RT_CHECK(rtMalloc(&d_out, size_output, RT_MEMORY_HBM, 0));
    RT_CHECK(rtMalloc(&d_q, size_inputs, RT_MEMORY_HBM, 0));
    RT_CHECK(rtMalloc(&d_k, size_inputs, RT_MEMORY_HBM, 0));
    RT_CHECK(rtMalloc(&d_v, size_inputs, RT_MEMORY_HBM, 0));
    RT_CHECK(rtMalloc(&d_ws, ws_per_block * blockDim, RT_MEMORY_HBM, 0));
    RT_CHECK(rtMemset(d_ws, ws_per_block * blockDim, 0, ws_per_block * blockDim));

    // 5. Host to Device Copy
    RT_CHECK(rtMemcpy(d_q, size_inputs, host_q, size_inputs, RT_MEMCPY_HOST_TO_DEVICE));
    RT_CHECK(rtMemcpy(d_k, size_inputs, host_k, size_inputs, RT_MEMCPY_HOST_TO_DEVICE));
    RT_CHECK(rtMemcpy(d_v, size_inputs, host_v, size_inputs, RT_MEMCPY_HOST_TO_DEVICE));
    RT_CHECK(rtMemset(d_out, size_output, 0, size_output));

    // 6. Launch Kernel
    float scale = 1.0f / sqrtf(float(COLS));
    uint32_t scale_bits;
    memcpy(&scale_bits, &scale, sizeof(scale));
    void *scale_arg = (void *)(uintptr_t)scale_bits;

    void *ffts_addr;
    uint32_t ffts_len;
    RT_CHECK(rtGetC2cCtrlAddr((uint64_t *)&ffts_addr, &ffts_len));

    void *args[] = {d_out, d_q, d_k, d_v, d_ws, ffts_addr, scale_arg};

    printf("Launching kernel with scale %f...\n", scale);
    auto start = chrono::high_resolution_clock::now();

    RT_CHECK(rtKernelLaunch(&funcStub, blockDim, args, sizeof(args), nullptr, stream));
    RT_CHECK(rtStreamSynchronize(stream));

    auto end = chrono::high_resolution_clock::now();
    printf("Kernel finished in %ld us\n", chrono::duration_cast<chrono::microseconds>(end - start).count());

    // 7. Device to Host Copy
    RT_CHECK(rtMemcpy(host_out.data(), size_output, d_out, size_output, RT_MEMCPY_DEVICE_TO_HOST));

    // 8. Save Output
    std::filesystem::create_directories("./output_data");
    writeFile("./output_data/out.data", host_out);
    printf("Saved output at out.data\n");

    // Cleanup
    rtFree(d_out);
    rtFree(d_q);
    rtFree(d_k);
    rtFree(d_v);
    rtStreamDestroy(stream);
    rtDeviceReset(0);

    // 9. Result Check (Golden vs Device)
    const std::string goldenPath = "./golden_data/out.data";
    const std::string devicePath = "./output_data/out.data";

    if (!std::filesystem::exists(goldenPath)) {
        fprintf(stderr, "Golden file missing: %s\n", goldenPath.c_str());
        return 1;
    }
    if (!std::filesystem::exists(devicePath)) {
        fprintf(stderr, "Device output missing: %s\n", devicePath.c_str());
        return 1;
    }

    std::vector<float> golden = readFile<float>(goldenPath);
    std::vector<float> device = readFile<float>(devicePath);

    if (golden.size() != device.size()) {
        fprintf(stderr, "Size Mismatch! Golden: %zu, Device: %zu\n", golden.size(), device.size());
        return 1;
    }
    if (golden.size() != MATRIX_SIZE) {
        fprintf(stderr, "Unexpected output size! Expected %d floats, got %zu\n", MATRIX_SIZE, golden.size());
        return 1;
    }

    printf("Golden Sample (Row 0, first 8): ");
    for (int i = 0; i < 8; ++i)
        printf("%s%.6f", (i ? ", " : ""), golden[i]);
    printf("\n");

    printf("Device Sample (Row 0, first 8): ");
    for (int i = 0; i < 8; ++i)
        printf("%s%.6f", (i ? ", " : ""), device[i]);
    printf("\n");

    const float atol = 5e-2f;
    const float rtol = 5e-2f;

    size_t num_mismatches = 0;
    size_t max_idx = 0;
    float max_diff = 0.0f;

    for (size_t i = 0; i < golden.size(); ++i) {
        const float g = golden[i];
        const float d = device[i];
        const float diff = std::fabs(g - d);
        const float tol = atol + rtol * std::fabs(g);
        const bool match = (diff <= tol);

        if (!match)
            ++num_mismatches;
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
    }

    if (num_mismatches == 0) {
        printf("\n[PASSED] Device output matches Golden reference.\n");
    } else {
        const int r = int(max_idx / COLS);
        const int c = int(max_idx % COLS);
        printf("\n[FAILED] Output mismatch!\n");
        printf("Total Mismatches: %zu / %zu\n", num_mismatches, golden.size());
        printf("Max Diff: %.6f at index (%d, %d)\n", max_diff, r, c);
        printf("Golden: %.6f, Device: %.6f\n", golden[max_idx], device[max_idx]);
        return 1;
    }

    return 0;
}