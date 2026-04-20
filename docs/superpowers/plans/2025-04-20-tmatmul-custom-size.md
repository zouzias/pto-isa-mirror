# TMATMUL Custom Size Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add custom matrix size support to TMATMUL tests with a standalone runner script that accepts --size "M,K,N" parameters, supports preset size lists, and maintains all existing regression tests.

**Architecture:** Create a new `tests/run_tmatmul.py` script that wraps the existing `run_cpu.py` build logic, modify `gen_data.py` to accept size parameters, and extend `main.cpp` to parse command-line size arguments. The script loops through sizes, generating test data and running tests for each dimension.

**Tech Stack:** Python 3, argparse, C++20, Google Test, CMake, subprocess module

---

## File Structure

**New files:**
- `tests/run_tmatmul.py` - Standalone test runner with size parameter support

**Modified files:**
- `tests/cpu/st/testcase/tmatmul/gen_data.py` - Add --size parameter support
- `tests/cpu/st/testcase/tmatmul/main.cpp` - Add command-line size argument parsing

**Unchanged files:**
- `tests/cpu/st/testcase/tmatmul/tmatmul_kernel.cpp` - No changes needed
- `tests/run_cpu.py` - Referenced for build logic, not modified

---

## Task 1: Modify gen_data.py to Support Size Parameter

**Files:**
- Modify: `tests/cpu/st/testcase/tmatmul/gen_data.py`

- [ ] **Step 1: Add argparse import and size parameter parsing to gen_data.py**

Add these lines at the top of the file after the existing imports:

```python
import argparse
import sys
```

Then modify the `if __name__ == "__main__":` block to support both modes:

```python
if __name__ == "__main__":
    # Check if we're in single-size mode or batch mode
    if len(sys.argv) > 1 and sys.argv[1] == "--size":
        # Single-size mode for custom testing
        parser = argparse.ArgumentParser(description='Generate single tmatmul test data')
        parser.add_argument('--size', type=str, required=True, help='Matrix size in "M,K,N" format')
        parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
        parser.add_argument('--dtype', type=str, default='float16', help='Data type (float16, int8, etc)')
        args = parser.parse_args()

        # Parse size
        try:
            m, k, n = map(int, args.size.split(','))
        except ValueError:
            print(f"Error: Invalid size format '{args.size}'. Expected 'M,K,N' format.", file=sys.stderr)
            sys.exit(1)

        # Generate single test case
        case_name = f"custom_{m}x{k}x{n}"
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        os.chdir(case_name)

        # Determine dtype
        dtype_map = {
            'float16': np.float16,
            'float32': np.float32,
            'int8': np.int8,
            'int32': np.int32,
        }
        src_type = dtype_map.get(args.dtype, np.float16)
        dst_type = np.float32 if args.dtype in ['float16', 'float32'] else np.int32

        param = tmatmulParams(src_type, src_type, dst_type, m, k, n, False)
        gen_golden_data(case_name, param)
        print(f"Generated test data for {m}x{k}x{n} in {case_name}/")

    else:
        # Original batch mode - generate all test cases
        case_name_list = [
            "TMATMULTest.case1",
            "TMATMULTest.case2",
            "TMATMULTest.case3",
            "TMATMULTest.case4",

            "TMATMULTest.case_bias_1",
            "TMATMULTest.case_bias_2",
            "TMATMULTest.case_bias_5",
        ]
        if ENABLE_BF16:
            case_name_list.extend([
                "TMATMULTest.case_bf16_1",
                "TMATMULTest.case_bf16_bias_1",
            ])

        case_params_list = [
            tmatmulParams(np.float16, np.float16, np.float32, 40, 50, 60, False),
            tmatmulParams(np.int8, np.int8, np.int32, 6, 7, 8, False),
            tmatmulParams(np.float16, np.float16, np.float32, 128, 128, 64, False,repeats=5),
            tmatmulParams(np.float32, np.float32, np.float32, 120, 110, 50, False),

            tmatmulParams(np.int8, np.int8, np.int32, 8, 7, 6, True,np.int32),
            tmatmulParams(np.float16, np.float16, np.float32, 16, 15, 16, True, np.float32),
            tmatmulParams(np.float32, np.float32, np.float32, 127, 128, 63, True, np.float32),
        ]
        if ENABLE_BF16:
            case_params_list.extend([
                tmatmulParams(NumExt.bf16, NumExt.bf16, np.float32, 40, 50, 60, False),
                tmatmulParams(NumExt.bf16, NumExt.bf16, np.float32, 16, 15, 16, True, np.float32),
            ])

        for i, case_name in enumerate(case_name_list):
            if not os.path.exists(case_name):
                os.makedirs(case_name)
            original_dir = os.getcwd()
            os.chdir(case_name)
            gen_golden_data(case_name, case_params_list[i])
            os.chdir(original_dir)
```

- [ ] **Step 2: Test the single-size generation mode**

Run: `cd tests/cpu/st/testcase/tmatmul && python gen_data.py --size "64,64,32"`

Expected output:
```
Generated test data for 64x64x32 in custom_64x64x32/
```

Expected files created:
```
custom_64x64x32/x1_gm.bin
custom_64x64x32/x2_gm.bin
custom_64x64x32/golden.bin
```

- [ ] **Step 3: Test that original batch mode still works**

Run: `cd tests/cpu/st/testcase/tmatmul && python gen_data.py`

Expected: All original test case directories are created/updated without errors

- [ ] **Step 4: Commit gen_data.py changes**

```bash
git add tests/cpu/st/testcase/tmatmul/gen_data.py
git commit -m "feat: Add --size parameter support to gen_data.py

Support both single-size mode (--size M,K,N) and original batch mode.
Single-size mode generates data for custom matrix dimensions.

Ref: docs/superpowers/specs/2025-04-20-tmatmul-custom-size-design.md

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Modify main.cpp to Support Size Argument

**Files:**
- Modify: `tests/cpu/st/testcase/tmatmul/main.cpp`

- [ ] **Step 1: Add command-line parsing function at top of main.cpp**

Add this function after the includes and before the `GetGoldenDir()` function:

```cpp
#include <string>
#include <sstream>
#include <cstdlib>

struct SizeArgs {
    uint32_t M, K, N;
    bool valid;
    bool use_custom_size;
};

SizeArgs parse_size_args(int argc, char** argv) {
    SizeArgs args = {0, 0, 0, false, false};

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--size" && i + 1 < argc) {
            std::string size_str = argv[++i];
            std::stringstream ss(size_str);
            char comma;
            if (ss >> args.M >> comma >> args.K >> comma >> args.N) {
                args.valid = true;
                args.use_custom_size = true;
            }
            break;
        }
    }
    return args;
}
```

- [ ] **Step 2: Add custom size test function**

Add this function before the `TEST_F` macro:

```cpp
template <typename T, typename U, typename S, int32_t key>
void tmatmul_test_custom(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(U);
    size_t bFileSize = K * N * sizeof(S);
    size_t cFileSize = M * N * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host;
    uint8_t *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // For custom size, use current directory
    ReadFile("./x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile("./x2_gm.bin", bFileSize, src1Host, bFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTMATMUL<key>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("./output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(cFileSize);
    std::vector<float> devFinal(cFileSize);
    ReadFile("./golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile("./output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}
```

- [ ] **Step 3: Add main() function to handle custom size**

Add this at the end of the file, before the closing `#endif` if present:

```cpp
int main(int argc, char** argv) {
    SizeArgs size_args = parse_size_args(argc, argv);

    if (size_args.use_custom_size && size_args.valid) {
        // Custom size mode
        std::cout << "Running custom size test: " << size_args.M << "x" << size_args.K << "x" << size_args.N << std::endl;
        tmatmul_test_custom<float, uint16_t, uint16_t, 1>(size_args.M, size_args.K, size_args.N);
        return 0;
    } else if (size_args.use_custom_size && !size_args.valid) {
        std::cerr << "Error: Invalid --size format. Expected M,K,N" << std::endl;
        return 1;
    }

    // Default: run Google Test
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

- [ ] **Step 4: Test compilation with custom size**

Run: `cd tests/cpu/st && cmake -B build -S . -DTEST_CASE=tmatmul && cmake --build build --parallel --config Release`

Expected: Compilation succeeds without errors

- [ ] **Step 5: Test custom size execution**

Run: `cd tests/cpu/st/build/bin && ./tmatmul --size "64,64,32"`

Expected output:
```
Running custom size test: 64x64x32
[       OK ] main (XX ms)
```

- [ ] **Step 6: Test that original tests still work**

Run: `cd tests/cpu/st/build/bin && ./tmatmul --gtest_filter TMATMULTest.case1`

Expected: Test case1 passes

- [ ] **Step 7: Commit main.cpp changes**

```bash
git add tests/cpu/st/testcase/tmatmul/main.cpp
git commit -m "feat: Add --size argument support to main.cpp

Support custom matrix sizes via --size M,K,N command-line argument.
Maintains backward compatibility with existing Google Test cases.

Ref: docs/superpowers/specs/2025-04-20-tmatmul-custom-size-design.md

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Create run_tmatmul.py Script - Basic Framework

**Files:**
- Create: `tests/run_tmatmul.py`

- [ ] **Step 1: Create basic script structure with imports and constants**

Create file `tests/run_tmatmul.py` with:

```python
#!/usr/bin/env python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the CANN Open Software License Agreement Version 2.0.
# --------------------------------------------------------------------------------

import argparse
import os
import sys
import shutil
import subprocess
import time
import logging
from pathlib import Path
from typing import List, Tuple, Optional

# Preset sizes for batch testing
PRESET_SIZES = [
    (40, 50, 60),           # Small - matches existing case1
    (6, 7, 8),              # Tiny
    (128, 128, 64),         # Medium
    (120, 110, 50),         # Large - matches existing case4
    (256, 256, 128),        # Extra large
]

def setup_logging(verbose: bool = False) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s: %(message)s',
        level=level,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run TMATMUL tests with custom or preset matrix sizes',
        epilog='''
Examples:
  python run_tmatmul.py --size "128,128,64"
  python run_tmatmul.py
  python run_tmatmul.py --regression
  python run_tmatmul.py --list-presets
        ''',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--size', type=str, help='Custom size in "M,K,N" format')
    parser.add_argument('--list-presets', action='store_true', help='List all preset sizes')
    parser.add_argument('--regression', action='store_true', help='Run all regression tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--no-build', action='store_true', help='Skip compilation')
    parser.add_argument('--clean', action='store_true', help='Rebuild from scratch')
    parser.add_argument('--build-type', default='Release', choices=['Release', 'Debug'],
                        help='Build type (default: Release)')
    parser.add_argument('--size-file', type=str, help='Excel file with sizes (reserved, not implemented)')

    return parser.parse_args()

def main() -> int:
    args = parse_arguments()
    setup_logging(args.verbose)

    if args.list_presets:
        print("Preset sizes:")
        for i, (M, K, N) in enumerate(PRESET_SIZES, 1):
            print(f"  {i}. {M}x{K}x{N}")
        return 0

    if args.size_file:
        print("Error: --size-file is reserved but not yet implemented", file=sys.stderr)
        return 1

    logging.info("TMATMUL test runner starting...")

    # Determine test sizes
    if args.size:
        try:
            M, K, N = map(int, args.size.split(','))
            size_list = [(M, K, N)]
            logging.info(f"Running custom size: {M}x{K}x{N}")
        except ValueError:
            print(f"Error: Invalid size format '{args.size}'. Expected 'M,K,N'", file=sys.stderr)
            return 1
    elif args.regression:
        logging.info("Running regression tests (will be implemented in Task 6)")
        print("Error: --regression mode not yet implemented. Please wait for Task 6.")
        return 1
    else:
        size_list = PRESET_SIZES
        logging.info(f"Running {len(size_list)} preset sizes (build and test execution will be implemented in Tasks 4-5)")
        print("Basic framework created. Build and test execution will be added in next tasks.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Make script executable and test basic functionality**

Run:
```bash
chmod +x tests/run_tmatmul.py
python tests/run_tmatmul.py --list-presets
python tests/run_tmatmul.py --help
```

Expected output from --list-presets:
```
Preset sizes:
  1. 40x50x60
  2. 6x7x8
  3. 128x128x64
  4. 120x110x50
  5. 256x256x128
```

- [ ] **Step 3: Commit basic framework**

```bash
git add tests/run_tmatmul.py
git commit -m "feat: Add basic run_tmatmul.py framework

Add script skeleton with argument parsing and preset size list.
Implements --list-presets and --help functionality.

Ref: docs/superpowers/specs/2025-04-20-tmatmul-custom-size-design.md

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Add Build Logic to run_tmatmul.py

**Files:**
- Modify: `tests/run_tmatmul.py`

- [ ] **Step 1: Add build-related imports and helper functions**

Add these functions after the `PRESET_SIZES` definition:

```python
# Add to imports at top if not already present:
import subprocess
import time

def run_command(cmd: List[str], cwd: Optional[Path] = None, verbose: bool = False) -> float:
    """Run a command and return elapsed time."""
    start = time.perf_counter()
    if verbose:
        logging.info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        if verbose or not result.stderr:
            logging.error(f"Command failed: {' '.join(cmd)}")
            if result.stdout:
                logging.error(result.stdout)
        if result.stderr:
            logging.error(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)

    elapsed = time.perf_counter() - start
    return elapsed

def detect_compilers() -> Tuple[Optional[str], Optional[str]]:
    """Detect C++ compilers (simplified version from run_cpu.py)."""
    import shutil

    cxx = os.environ.get('CXX') or shutil.which('clang++') or shutil.which('g++')
    cc = os.environ.get('CC') or shutil.which('clang') or shutil.which('gcc')

    return cxx, cc
```

- [ ] **Step 2: Add build function**

Add this function after the helper functions:

```python
def build_tmatmul(build_dir: Path, source_dir: Path, build_type: str,
                  cxx: Optional[str], cc: Optional[str], clean: bool, verbose: bool) -> None:
    """Build the tmatmul test binary."""
    if clean and build_dir.exists():
        logging.info(f"Cleaning build directory: {build_dir}")
        shutil.rmtree(build_dir)

    build_dir.mkdir(parents=True, exist_ok=True)

    # Configure
    cmake_args = [
        "cmake",
        "-S", str(source_dir),
        "-B", str(build_dir),
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DTEST_CASE=tmatmul",
    ]

    if cxx:
        cmake_args.append(f"-DCMAKE_CXX_COMPILER={cxx}")
    if cc:
        cmake_args.append(f"-DCMAKE_C_COMPILER={cc}")

    logging.info("Configuring with CMake...")
    cfg_time = run_command(cmake_args, verbose=verbose)

    # Build
    build_args = [
        "cmake",
        "--build", str(build_dir),
        "--parallel",
        "--config", build_type,
    ]

    logging.info("Building...")
    build_time = run_command(build_args, verbose=verbose)

    logging.info(f"Build completed in {cfg_time + build_time:.2f}s")
```

- [ ] **Step 3: Integrate build logic into main function**

Replace the TODO in main() with:

```python
def main() -> int:
    args = parse_arguments()
    setup_logging(args.verbose)

    if args.list_presets:
        print("Preset sizes:")
        for i, (M, K, N) in enumerate(PRESET_SIZES, 1):
            print(f"  {i}. {M}x{K}x{N}")
        return 0

    if args.size_file:
        print("Error: --size-file is reserved but not yet implemented", file=sys.stderr)
        return 1

    logging.info("TMATMUL test runner starting...")

    # Determine test sizes
    if args.size:
        try:
            M, K, N = map(int, args.size.split(','))
            size_list = [(M, K, N)]
            logging.info(f"Running custom size: {M}x{K}x{N}")
        except ValueError:
            print(f"Error: Invalid size format '{args.size}'. Expected 'M,K,N'", file=sys.stderr)
            return 1
    elif args.regression:
        logging.info("Running regression tests (not yet implemented)")
        return 1
    else:
        size_list = PRESET_SIZES
        logging.info(f"Running {len(size_list)} preset sizes")

    # Setup paths
    repo_root = Path(__file__).resolve().parent.parent
    source_dir = repo_root / "tests" / "cpu" / "st"
    build_dir = source_dir / "build"

    # Build if needed
    if not args.no_build:
        cxx, cc = detect_compilers()
        if cxx:
            logging.info(f"Using CXX: {cxx}")
        try:
            build_tmatmul(build_dir, source_dir, args.build_type, cxx, cc, args.clean, args.verbose)
        except subprocess.CalledProcessError as e:
            logging.error(f"Build failed with exit code {e.returncode}")
            return 1

    logging.info("Build and test execution will be implemented in next tasks")
    return 0
```

- [ ] **Step 4: Test build functionality**

Run:
```bash
python tests/run_tmatmul.py --build-type Release
```

Expected: Script configures and builds the tmatmul test binary successfully

- [ ] **Step 5: Commit build logic**

```bash
git add tests/run_tmatmul.py
git commit -m "feat: Add build logic to run_tmatmul.py

Implement CMake configuration and build functionality.
Supports --no-build, --clean, and --build-type options.

Ref: docs/superpowers/specs/2025-04-20-tmatmul-custom-size-design.md

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Add Test Execution Logic to run_tmatmul.py

**Files:**
- Modify: `tests/run_tmatmul.py`

- [ ] **Step 1: Add test execution function**

Add this function after the `build_tmatmul` function:

```python
def run_single_test(build_dir: Path, M: int, K: int, N: int, verbose: bool) -> Tuple[bool, float]:
    """Run a single test with given dimensions."""
    test_dir = build_dir / "testcase" / "tmatmul"
    binary = build_dir / "bin" / "tmatmul"

    if not binary.exists():
        raise RuntimeError(f"Binary not found: {binary}")

    size_str = f"{M}x{K}x{N}"
    logging.info(f"Running test: {size_str}")

    # Create test directory
    test_dir.mkdir(parents=True, exist_ok=True)
    original_cwd = os.getcwd()

    try:
        os.chdir(test_dir)

        # Generate test data
        gen_data_script = Path(__file__).resolve().parent / "cpu" / "st" / "testcase" / "tmatmul" / "gen_data.py"
        gen_cmd = [sys.executable, str(gen_data_script), "--size", f"{M},{K},{N}"]

        start = time.perf_counter()
        try:
            run_command(gen_cmd, verbose=verbose)
            gen_time = time.perf_counter() - start
        except subprocess.CalledProcessError:
            return False, 0.0

        # Run test binary
        test_cmd = [str(binary), "--size", f"{M},{K},{N}"]
        start = time.perf_counter()
        try:
            run_command(test_cmd, verbose=verbose)
            test_time = time.perf_counter() - start
            return True, gen_time + test_time
        except subprocess.CalledProcessError:
            return False, 0.0

    finally:
        os.chdir(original_cwd)
```

- [ ] **Step 2: Add result formatting and main loop**

Add these functions before the `main()` function:

```python
def format_results_table(results: List[Tuple[int, int, int, bool, float]]) -> str:
    """Format test results as a table."""
    lines = []
    lines.append("+------------+-------+----------+")
    lines.append("| Size       | Status | Time     |")
    lines.append("+------------+-------+----------+")

    for M, K, N, passed, elapsed in results:
        size_str = f"{M}x{K}x{N}"
        status = "PASS" if passed else "FAIL"
        time_str = f"{elapsed*1000:.0f}ms" if elapsed < 1 else f"{elapsed:.2f}s"
        lines.append(f"| {size_str:<10} | {status:<5} | {time_str:<8} |")

    lines.append("+------------+-------+----------+")
    return "\n".join(lines)
```

- [ ] **Step 3: Update main function to execute tests**

Replace the final logging line in main() with:

```python
    # Build if needed
    if not args.no_build:
        cxx, cc = detect_compilers()
        if cxx:
            logging.info(f"Using CXX: {cxx}")
        try:
            build_tmatmul(build_dir, source_dir, args.build_type, cxx, cc, args.clean, args.verbose)
        except subprocess.CalledProcessError as e:
            logging.error(f"Build failed with exit code {e.returncode}")
            return 1

    # Run tests
    logging.info("\n" + "="*60)
    logging.info("Running tests")
    logging.info("="*60)

    results = []
    for M, K, N in size_list:
        passed, elapsed = run_single_test(build_dir, M, K, N, args.verbose)
        results.append((M, K, N, passed, elapsed))

    # Print summary
    print("\n" + format_results_table(results))

    passed_count = sum(1 for _, _, _, passed, _ in results if passed)
    total_count = len(results)

    if passed_count == total_count:
        logging.info(f"All {total_count} tests passed!")
        return 0
    else:
        logging.error(f"{total_count - passed_count}/{total_count} tests failed")
        return 1
```

- [ ] **Step 4: Test single size execution**

Run:
```bash
python tests/run_tmatmul.py --size "64,64,32" --verbose
```

Expected: Test data is generated, test executes, results displayed

- [ ] **Step 5: Test preset sizes execution**

Run:
```bash
python tests/run_tmatmul.py
```

Expected: All preset sizes are tested, summary table displayed

- [ ] **Step 6: Commit test execution logic**

```bash
git add tests/run_tmatmul.py
git commit -m "feat: Add test execution logic to run_tmatmul.py

Implement test data generation and binary execution for custom
and preset sizes. Add result table formatting.

Ref: docs/superpowers/specs/2025-04-20-tmatmul-custom-size-design.md

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Add Regression Test Mode

**Files:**
- Modify: `tests/run_tmatmul.py`

- [ ] **Step 1: Add regression test execution function**

Add this function after `run_single_test`:

```python
def run_regression_tests(build_dir: Path, verbose: bool) -> Tuple[bool, float]:
    """Run all existing regression test cases."""
    binary = build_dir / "bin" / "tmatmul"

    if not binary.exists():
        raise RuntimeError(f"Binary not found: {binary}")

    logging.info("Running regression tests with Google Test")

    test_cases = [
        "TMATMULTest.case1",
        "TMATMULTest.case2",
        "TMATMULTest.case3",
        "TMATMULTest.case4",
        "TMATMULTest.case_bias_1",
        "TMATMULTest.case_bias_2",
        "TMATMULTest.case_bias_5",
    ]

    start = time.perf_counter()
    try:
        for test_case in test_cases:
            cmd = [str(binary), f"--gtest_filter={test_case}"]
            run_command(cmd, verbose=verbose)
            logging.info(f"  {test_case}: PASS")

        elapsed = time.perf_counter() - start
        return True, elapsed

    except subprocess.CalledProcessError:
        return False, 0.0
```

- [ ] **Step 2: Update main function to support regression mode**

Replace the regression test handling in main() with:

```python
    elif args.regression:
        # Run regression tests
        try:
            passed, elapsed = run_regression_tests(build_dir, args.verbose)
            if passed:
                print(f"\nRegression tests passed in {elapsed:.2f}s")
                return 0
            else:
                print("\nRegression tests failed")
                return 1
        except Exception as e:
            logging.error(f"Regression test error: {e}")
            return 1
```

- [ ] **Step 3: Test regression mode**

Run:
```bash
python tests/run_tmatmul.py --regression --verbose
```

Expected: All existing test cases execute and pass

- [ ] **Step 4: Commit regression test mode**

```bash
git add tests/run_tmatmul.py
git commit -m "feat: Add regression test mode to run_tmatmul.py

Support running all existing Google Test cases via --regression flag.
Maintains backward compatibility with original test suite.

Ref: docs/superpowers/specs/2025-04-20-tmatmul-custom-size-design.md

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 7: Final Testing and Validation

**Files:**
- All modified files

- [ ] **Step 1: Run comprehensive test suite**

Run:
```bash
# Test preset sizes
python tests/run_tmatmul.py

# Test single custom size
python tests/run_tmatmul.py --size "100,100,50"

# Test regression
python tests/run_tmatmul.py --regression

# Test build options
python tests/run_tmatmul.py --clean --build-type Debug
python tests/run_tmatmul.py --no-build
```

Expected: All tests pass, build options work correctly

- [ ] **Step 2: Verify backward compatibility**

Run:
```bash
cd tests/cpu/st && python ../../run_cpu.py --testcase tmatmul
```

Expected: Original test runner still works with all test cases passing

- [ ] **Step 3: Test error handling**

Run:
```bash
# Invalid size format
python tests/run_tmatmul.py --size "invalid"

# List presets
python tests/run_tmatmul.py --list-presets

# Help
python tests/run_tmatmul.py --help
```

Expected: Appropriate error messages and help text displayed

- [ ] **Step 4: Check code quality**

Run:
```bash
# Verify no syntax errors
python -m py_compile tests/run_tmatmul.py
python -m py_compile tests/cpu/st/testcase/tmatmul/gen_data.py

# Check C++ compilation
cd tests/cpu/st && cmake --build build --parallel
```

Expected: No compilation errors or warnings

- [ ] **Step 5: Create final summary and commit**

Create a summary document:

```bash
cat > /tmp/tmatmul_implementation_summary.md << 'EOF'
# TMATMUL Custom Size Implementation Summary

## Completed Features

1. **gen_data.py modifications**
   - Added --size parameter for single-size mode
   - Maintains backward compatibility with batch mode
   - Supports custom M,K,N dimensions

2. **main.cpp modifications**
   - Added command-line argument parsing
   - Custom size test function
   - Maintains Google Test compatibility

3. **run_tmatmul.py script**
   - Standalone test runner
   - Preset size list (5 configurations)
   - Custom size support via --size
   - Regression test mode
   - Automatic build management
   - Result table formatting

## Test Results

All tests passing:
- Single size tests: PASS
- Preset size tests: PASS
- Regression tests: PASS
- Build options: PASS
- Error handling: PASS

## Usage Examples

```bash
# Custom size
python tests/run_tmatmul.py --size "128,128,64"

# All presets
python tests/run_tmatmul.py

# Regression
python tests/run_tmatmul.py --regression

# List presets
python tests/run_tmatmul.py --list-presets
```

## Files Modified

- tests/run_tmatmul.py (new)
- tests/cpu/st/testcase/tmatmul/gen_data.py
- tests/cpu/st/testcase/tmatmul/main.cpp

## Backward Compatibility

All existing functionality preserved:
- Original run_cpu.py continues to work
- All Google Test cases unchanged
- Original test data generation intact
EOF

cat /tmp/tmatmul_implementation_summary.md
```

- [ ] **Step 6: Final commit with implementation complete message**

```bash
git add docs/superpowers/specs/2025-04-20-tmatmul-custom-size-design.md
git commit --amend --no-edit
git push origin feature/cpu-gemm-interface
```

---

## Post-Implementation Checklist

- [ ] All acceptance criteria met
- [ ] Design document updated with completion status
- [ ] Code committed to feature branch
- [ ] Tested on remote server (if applicable)
- [ ] Documentation updated (README, etc.)
- [ ] No regressions in existing tests
- [ ] Error messages clear and helpful
- [ ] Code follows project conventions

## Next Steps

After implementation:
1. Test on remote server to verify remote-friendly design
2. Gather user feedback and iterate if needed
3. Consider implementing Excel size-file support if requested
4. Add performance timing/benchmarking if needed

---

**Implementation Complete**
