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

# Test data script path (relative to this script)
TEST_DATA_SCRIPT_PATH = Path("cpu/st/testcase/tmatmul/gen_data.py")

def run_command(cmd: List[str], cwd: Optional[Path] = None, verbose: bool = False) -> float:
    """Run a command and return elapsed time."""
    start = time.perf_counter()
    if verbose:
        logging.info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if result.returncode != 0:
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
    cxx = os.environ.get('CXX') or shutil.which('clang++') or shutil.which('g++')
    cc = os.environ.get('CC') or shutil.which('clang') or shutil.which('gcc')

    return cxx, cc

def build_tmatmul(build_dir: Path, source_dir: Path, build_type: str,
                  cxx: Optional[str], cc: Optional[str], clean: bool, verbose: bool) -> None:
    """Build the tmatmul test binary."""

    # Check if cmake is available
    if not shutil.which("cmake"):
        logging.error("cmake is not installed or not in PATH")
        raise RuntimeError("cmake is required to build the project")

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

    logging.info(f"Configure completed in {cfg_time:.2f}s")
    logging.info(f"Build completed in {build_time:.2f}s")
    logging.info(f"Total build time: {cfg_time + build_time:.2f}s")

def run_single_test(build_dir: Path, M: int, K: int, N: int, verbose: bool) -> Tuple[bool, float]:
    """Run a single test with given dimensions."""
    test_dir = build_dir / "testcase" / "tmatmul"
    binary = build_dir / "bin" / "tmatmul"

    if not binary.exists():
        raise RuntimeError(f"Binary not found: {binary}")

    size_str = f"{M}x{K}x{N}"
    logging.info(f"Running test: {size_str}")

    # Create test directory (cleanup first if exists)
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    original_cwd = os.getcwd()

    try:
        os.chdir(test_dir)

        # Generate test data
        gen_data_script = Path(__file__).resolve().parent / TEST_DATA_SCRIPT_PATH
        gen_cmd = [sys.executable, str(gen_data_script), "--size", f"{M},{K},{N}"]

        start = time.perf_counter()
        try:
            run_command(gen_cmd, verbose=verbose)
            gen_time = time.perf_counter() - start
        except subprocess.CalledProcessError as e:
            elapsed = time.perf_counter() - start
            logging.error(f"Data generation failed for {size_str} after {elapsed:.2f}s (exit code {e.returncode})")
            return False, elapsed

        # Run test binary
        test_cmd = [str(binary), "--size", f"{M},{K},{N}"]
        start = time.perf_counter()
        try:
            run_command(test_cmd, verbose=verbose)
            test_time = time.perf_counter() - start
            return True, gen_time + test_time
        except subprocess.CalledProcessError as e:
            elapsed = gen_time + (time.perf_counter() - start)
            logging.error(f"Test execution failed for {size_str} after {elapsed:.2f}s (exit code {e.returncode})")
            return False, elapsed

    finally:
        os.chdir(original_cwd)

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

    # Run regression tests if requested
    if args.regression:
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

    # Determine test sizes
    if args.size:
        try:
            M, K, N = map(int, args.size.split(','))
            size_list = [(M, K, N)]
            logging.info(f"Running custom size: {M}x{K}x{N}")
        except ValueError:
            print(f"Error: Invalid size format '{args.size}'. Expected 'M,K,N'", file=sys.stderr)
            return 1
    else:
        size_list = PRESET_SIZES
        logging.info(f"Running {len(size_list)} preset sizes")

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

if __name__ == "__main__":
    sys.exit(main())
