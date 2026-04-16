#!/usr/bin/env python3
"""
Generate main.cpp and kernel.cpp from input.csv
Supports: binary ops (tadd), unary ops (texp), scalar ops (tadds)
Usage: python3 generate_code.py
"""
import os
from pathlib import Path

UNARY_OPS = {'texp', 'tlog', 'tsqrt', 'tabs', 'tneg', 'trcp', 'trsqrt'}
SCALAR_OPS = {'tadds', 'tsubs', 'tmuls', 'tdivs', 'tmaxs', 'tmins'}

def get_op_type(op):
    """Determine op type: binary, unary, or scalar"""
    op_lower = op.lower()
    if op_lower in UNARY_OPS:
        return 'unary'
    elif op_lower in SCALAR_OPS:
        return 'scalar'
    else:
        return 'binary'

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
                scalar = float(parts[6]) if len(parts) > 6 else 1.0
                cases.append({
                    'op': op.strip(),
                    'dtype': dtype.strip(),
                    'tile_h': int(tile_h),
                    'tile_w': int(tile_w),
                    'valid_h': int(valid_h),
                    'valid_w': int(valid_w),
                    'scalar': scalar,
                    'op_type': get_op_type(op.strip())
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
    """Convert op name to PTO macro (uppercase)"""
    return op.upper()

def generate_main_cpp(cases, output_path):
    ops = set(c['op'] for c in cases)
    
    header = '''/**
 * Tile Op Benchmark Test Suite - Auto-generated from input.csv
 * Supports: binary (tadd), unary (texp), scalar (tadds) ops
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
    
    # Template declarations for all op types
    template_decls = '''// Binary op launcher (2 tiles -> 1 tile)
template <typename T, int tileH, int tileW, int vRows, int vCols>
void LaunchBinaryBench(const char* op, T *out, T *src0, T *src1, void *stream);

// Unary op launcher (1 tile -> 1 tile)
template <typename T, int tileH, int tileW, int vRows, int vCols>
void LaunchUnaryBench(const char* op, T *out, T *src, void *stream);

// Scalar op launcher (1 tile + scalar -> 1 tile)
template <typename T, int tileH, int tileW, int vRows, int vCols>
void LaunchScalarBench(const char* op, T *out, T *src, T scalar, void *stream);

'''
    
    # Test functions for each op type
    test_fns = '''// Binary test (tadd, tsub, tmul, etc.)
template <typename T, int tileH, int tileW, int vRows, int vCols>
void test_binary_bench(const char* op)
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

    LaunchBinaryBench<T, tileH, tileW, vRows, vCols>(op, dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtFree(dstDevice); aclrtFree(src0Device); aclrtFree(src1Device);
    aclrtFreeHost(dstHost); aclrtFreeHost(src0Host); aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

// Unary test (texp, tlog, tsqrt, etc.)
template <typename T, int tileH, int tileW, int vRows, int vCols>
void test_unary_bench(const char* op)
{
    size_t fileSize = tileH * tileW * sizeof(T);
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;
    aclrtMallocHost((void **)(&dstHost), fileSize);
    aclrtMallocHost((void **)(&srcHost), fileSize);
    aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", fileSize, srcHost, fileSize);
    aclrtMemcpy(srcDevice, fileSize, srcHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchUnaryBench<T, tileH, tileW, vRows, vCols>(op, dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtFree(dstDevice); aclrtFree(srcDevice);
    aclrtFreeHost(dstHost); aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

// Scalar test (tadds, tmuls, etc.)
template <typename T, int tileH, int tileW, int vRows, int vCols>
void test_scalar_bench(const char* op, T scalar)
{
    size_t fileSize = tileH * tileW * sizeof(T);
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;
    aclrtMallocHost((void **)(&dstHost), fileSize);
    aclrtMallocHost((void **)(&srcHost), fileSize);
    aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", fileSize, srcHost, fileSize);
    aclrtMemcpy(srcDevice, fileSize, srcHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchScalarBench<T, tileH, tileW, vRows, vCols>(op, dstDevice, srcDevice, scalar, stream);

    aclrtSynchronizeStream(stream);
    aclrtFree(dstDevice); aclrtFree(srcDevice);
    aclrtFreeHost(dstHost); aclrtFreeHost(srcHost);
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
        op_type = c['op_type']
        scalar = c['scalar']
        
        class_name = f"{op.upper()}BenchTest"
        dtype_gtest = dtype_to_gtest(dtype)
        dtype_cpp = dtype_to_cpp(dtype)
        case_name = f"case_{dtype_gtest}_{h}x{w}"
        
        if op_type == 'binary':
            tests += f'TEST_F({class_name}, {case_name}) {{ test_binary_bench<{dtype_cpp}, {th}, {tw}, {h}, {w}>("{op}"); }}\n'
        elif op_type == 'unary':
            tests += f'TEST_F({class_name}, {case_name}) {{ test_unary_bench<{dtype_cpp}, {th}, {tw}, {h}, {w}>("{op}"); }}\n'
        elif op_type == 'scalar':
            tests += f'TEST_F({class_name}, {case_name}) {{ test_scalar_bench<{dtype_cpp}, {th}, {tw}, {h}, {w}>("{op}", {scalar}f); }}\n'
    
    with open(output_path, 'w') as f:
        f.write(header + classes + golden_dir + template_decls + test_fns + tests)
    
    print(f"Generated: {output_path}")

def generate_kernel_cpp(cases, output_path):
    header = '''/**
 * Tile Op Benchmark Kernel - Auto-generated from input.csv
 * Supports: binary (tadd), unary (texp), scalar (tadds) ops
 * 10x ops with pipe_barrier for isolated VF measurement
 */
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

using namespace pto;

'''
    
    # Collect unique (op_type, tileH, tileW, vRows, vCols) combinations
    binary_templates = set()
    unary_templates = set()
    scalar_templates = set()
    
    for c in cases:
        key = (c['tile_h'], c['tile_w'], c['valid_h'], c['valid_w'])
        if c['op_type'] == 'binary':
            binary_templates.add(key)
        elif c['op_type'] == 'unary':
            unary_templates.add(key)
        elif c['op_type'] == 'scalar':
            scalar_templates.add(key)
    
    kernels = ""
    
    # Binary op kernels
    for th, tw, vh, vw in sorted(binary_templates):
        kernels += f'''// Binary op kernel: 2 tiles -> 1 tile
template <>
__aicore__ inline void LaunchBinaryBench<float, {th}, {tw}, {vh}, {vw}>(const char* op, float *dst, float *src0, float *src1, void *stream)
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
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {{
        TLOAD(tileSrc0, gmSrc0, tileW * sizeof(float));
        TLOAD(tileSrc1, gmSrc1, tileW * sizeof(float));
        TADD(tileDst, tileSrc0, tileSrc1);  // Binary op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }}
}}

'''
    
    # Unary op kernels (TEXP)
    for th, tw, vh, vw in sorted(unary_templates):
        kernels += f'''// Unary op kernel: 1 tile -> 1 tile
template <>
__aicore__ inline void LaunchUnaryBench<float, {th}, {tw}, {vh}, {vw}>(const char* op, float *dst, float *src, void *stream)
{{
    constexpr int tileH = {th};
    constexpr int tileW = {tw};
    constexpr int vRows = {vh};
    constexpr int vCols = {vw};
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc = (GM_ADDR)src;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {{
        TLOAD(tileSrc, gmSrc, tileW * sizeof(float));
        TEXP(tileDst, tileSrc);  // Unary op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }}
}}

'''
    
    # Scalar op kernels (TADDS)
    for th, tw, vh, vw in sorted(scalar_templates):
        kernels += f'''// Scalar op kernel: 1 tile + scalar -> 1 tile
template <>
__aicore__ inline void LaunchScalarBench<float, {th}, {tw}, {vh}, {vw}>(const char* op, float *dst, float *src, float scalar, void *stream)
{{
    constexpr int tileH = {th};
    constexpr int tileW = {tw};
    constexpr int vRows = {vh};
    constexpr int vCols = {vw};
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc = (GM_ADDR)src;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {{
        TLOAD(tileSrc, gmSrc, tileW * sizeof(float));
        TADDS(tileDst, tileSrc, scalar);  // Scalar op
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
    
    # Count by op type
    binary_cnt = sum(1 for c in cases if c['op_type'] == 'binary')
    unary_cnt = sum(1 for c in cases if c['op_type'] == 'unary')
    scalar_cnt = sum(1 for c in cases if c['op_type'] == 'scalar')
    print(f"  Binary: {binary_cnt}, Unary: {unary_cnt}, Scalar: {scalar_cnt}")
    
    generate_main_cpp(cases, main_path)
    generate_kernel_cpp(cases, kernel_path)
    print(f"\nDone! Rebuild with: python3 tests/script/run_st.py -r sim -v a5 -t tadd_bench")

if __name__ == "__main__":
    main()
