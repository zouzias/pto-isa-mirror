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
