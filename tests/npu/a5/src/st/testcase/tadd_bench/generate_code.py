#!/usr/bin/env python3
"""
Generate main.cpp and tadd_bench_kernel.cpp from input.csv
Usage: python3 generate_code.py
"""
import os
from pathlib import Path

def read_input_csv(csv_path):
    cases = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) >= 6:
                op, dtype, tile_h, tile_w, valid_h, valid_w = parts[:6]
                cases.append({
                    'op': op.strip(),
                    'dtype': dtype.strip(),
                    'tile_h': int(tile_h),
                    'tile_w': int(tile_w),
                    'valid_h': int(valid_h),
                    'valid_w': int(valid_w)
                })
    return cases

def dtype_to_cpp(dtype):
    mapping = {
        'float32': 'float',
        'float16': 'half',
        'bfloat16': 'bfloat16_t',
        'int32': 'int32_t',
        'int16': 'int16_t',
        'int8': 'int8_t',
        'uint8': 'uint8_t'
    }
    return mapping.get(dtype, 'float')

def dtype_to_gtest(dtype):
    mapping = {
        'float32': 'float',
        'float16': 'half',
        'bfloat16': 'bfloat16',
        'int32': 'int32',
        'int16': 'int16',
        'int8': 'int8',
        'uint8': 'uint8'
    }
    return mapping.get(dtype, dtype)

def op_to_macro(op):
    """Convert op name to PTO macro"""
    return op.upper()

def generate_main_cpp(cases, output_path):
    # Group by op
    ops = set(c['op'] for c in cases)
    
    header = '''/**
 * TADD Benchmark Test Suite - Auto-generated from input.csv
 * Run: python3 generate_code.py
 */
#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

'''
    
    # Generate test class for each op
    classes = ""
    for op in sorted(ops):
        class_name = f"{op.upper()}BenchTest"
        classes += f'''class {class_name} : public testing::Test {{
protected:
    void SetUp() override {{}}
    void TearDown() override {{}}
}};

'''
    
    golden_dir = '''std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    return "../" + std::string(testInfo->test_suite_name()) + "." + testInfo->name();
}

'''
    
    # Template declaration
    template_decl = '''template <typename T, int tileH, int tileW, int vRows, int vCols>
void LaunchTAddBench(T *out, T *src0, T *src1, void *stream);

'''
    
    # Test function template
    test_fn = '''template <typename T, int tileH, int tileW, int vRows, int vCols>
void test_tadd_bench()
{
    size_t fileSize = tileH * tileW * sizeof(T);
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;
    aclrtMallocHost((void **)(&dstHost), fileSize);
    aclrtMallocHost((void **)(&src0Host), fileSize);
    aclrtMallocHost((void **)(&src1Host), fileSize);
    aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", fileSize, src0Host, fileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", fileSize, src1Host, fileSize);
    aclrtMemcpy(src0Device, fileSize, src0Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSize, src1Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTAddBench<T, tileH, tileW, vRows, vCols>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    
    aclrtFree(dstDevice); aclrtFree(src0Device); aclrtFree(src1Device);
    aclrtFreeHost(dstHost); aclrtFreeHost(src0Host); aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

'''
    
    # Generate test cases
    tests = "// Test cases (auto-generated from input.csv)\n"
    for c in cases:
        op = c['op']
        dtype = c['dtype']
        h, w = c['valid_h'], c['valid_w']
        th, tw = c['tile_h'], c['tile_w']
        
        class_name = f"{op.upper()}BenchTest"
        dtype_gtest = dtype_to_gtest(dtype)
        dtype_cpp = dtype_to_cpp(dtype)
        case_name = f"case_{dtype_gtest}_{h}x{w}"
        
        tests += f"TEST_F({class_name}, {case_name}) {{ test_tadd_bench<{dtype_cpp}, {th}, {tw}, {h}, {w}>(); }}\n"
    
    with open(output_path, 'w') as f:
        f.write(header + classes + golden_dir + template_decl + test_fn + tests)
    
    print(f"Generated: {output_path}")

def generate_kernel_cpp(cases, output_path):
    header = '''/**
 * TADD Benchmark Kernel - Auto-generated from input.csv
 * 10x chained TADD with pipe_barrier for isolated VF measurement
 */
#include "kernel_operator.h"

using namespace AscendC;

'''
    
    # Collect unique (tileH, tileW, vRows, vCols) combinations
    templates = set()
    for c in cases:
        templates.add((c['tile_h'], c['tile_w'], c['valid_h'], c['valid_w']))
    
    kernels = ""
    for th, tw, vh, vw in sorted(templates):
        kernels += f'''template <>
__aicore__ inline void LaunchTAddBench<float, {th}, {tw}, {vh}, {vw}>(float *dst, float *src0, float *src1, void *stream)
{{
    constexpr int tileH = {th};
    constexpr int tileW = {tw};
    constexpr int vRows = {vh};
    constexpr int vCols = {vw};
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc0(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc1(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc0 = (GM_ADDR)src0;
    GM_ADDR gmSrc1 = (GM_ADDR)src1;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    // 10x TADD with barriers for isolated VF measurement
    #pragma unroll
    for (int i = 0; i < 10; i++) {{
        TLOAD(tileSrc0, gmSrc0, tileW * sizeof(float));
        TLOAD(tileSrc1, gmSrc1, tileW * sizeof(float));
        TADD(tileDst, tileSrc0, tileSrc1);
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }}
}}

'''
    
    with open(output_path, 'w') as f:
        f.write(header + kernels)
    
    print(f"Generated: {output_path}")

def main():
    script_dir = Path(__file__).parent
    csv_path = script_dir / "input.csv"
    main_path = script_dir / "main.cpp"
    kernel_path = script_dir / "tadd_bench_kernel.cpp"
    
    cases = read_input_csv(csv_path)
    if not cases:
        print(f"No cases found in {csv_path}")
        return
    
    print(f"Found {len(cases)} test cases in input.csv")
    generate_main_cpp(cases, main_path)
    generate_kernel_cpp(cases, kernel_path)
    print(f"\nDone! Rebuild with: python3 tests/script/run_st.py -r sim -v a5 -t tadd_bench --rebuild")

if __name__ == "__main__":
    main()
