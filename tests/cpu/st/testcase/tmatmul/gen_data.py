#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os
import struct
import math
import numpy as np
import argparse
import sys
from utils import NumExt

np.random.seed(19)
ENABLE_BF16 = os.environ.get("PTO_CPU_SIM_ENABLE_BF16") == "1"

def matmul_reference(a, b, out_dtype, trans='NN'):
    """
    Reference matmul that avoids BLAS calls (some macOS Python distributions may
    ship a broken/unsupported BLAS backend that returns incorrect results).

    a: (m, k) or (k, m) depending on trans
    b: (k, n) or (n, k) depending on trans
    trans: 'NN', 'NT', 'TN', 'TT'
    returns: (m, n)
    """
    a = a.astype(out_dtype, copy=False)
    b = b.astype(out_dtype, copy=False)

    return (a[:, :, None] * b[None, :, :]).sum(axis=1, dtype=out_dtype)

def gen_golden_data(case_name, param):
    src_type = param.atype
    dst_type = param.ctype

    m, k, n, is_bias, trans = param.m, param.k, param.n, param.is_bias, param.trans 
    repeats = param.repeats

    x1_gm = NumExt.astype(np.random.randint(1, 5, [repeats, m, k]), src_type)
    x2_gm = NumExt.astype(np.random.randint(1, 5, [repeats, k, n]), src_type)
    golden = np.zeros([m, n], dst_type)

    x1_gm_s = NumExt.astype(np.random.randint(1, 5, [repeats, m if trans in ['NN', 'NT'] else k, k if trans in ['NN', 'NT'] else m]), src_type)
    x2_gm_s = NumExt.astype(np.random.randint(1, 5, [repeats, k if trans in ['NN', 'TN'] else n, n if trans in ['NN', 'TN'] else k]), src_type)
    

    bias_gm = np.random.randint(1, 10, [n, ]).astype(param.bias_type)
    golden = np.zeros([m, n], dst_type)

    for i in range(repeats):
        golden = golden + matmul_reference(x1_gm[i], x2_gm[i], dst_type, trans).astype(dst_type)

        if trans in ['NT', 'TT']:
            x1_gm_s[i] = x1_gm.transpose()
        
        if trans in ['TN', 'TT']:
            x2_gm_s[i] = x2_gm.transpose()

    if is_bias:
        golden += bias_gm

    NumExt.write_array("./x1_gm.bin", x1_gm_s, src_type)
    NumExt.write_array("./x2_gm.bin", x2_gm_s, src_type)
    bias_gm.tofile("./bias_gm.bin")
    golden.tofile("./golden.bin")


class tmatmulParams:
    def __init__(self, atype, btype, ctype, m, k, n, is_bias, bias_type=None, repeats=1, trans='NN'):
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.m = m
        self.k = k
        self.n = n
        self.repeats = repeats
        self.is_bias = is_bias
        self.trans = trans
        if (bias_type):
            self.bias_type = bias_type
        else:
            self.bias_type = ctype


if __name__ == "__main__":
    # Check if we're in single-size mode or batch mode
    if len(sys.argv) > 1 and sys.argv[1] == "--size":
        # Single-size mode for custom testing
        parser = argparse.ArgumentParser(description='Generate single tmatmul test data')
        parser.add_argument('--size', type=str, required=True, help='Matrix size in "M,K,N" format (comma-separated without spaces)')
        parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
        parser.add_argument('--dtype', type=str, default='float16', help='Data type (float16, int8, bf16, etc)')
        parser.add_argument('--bias', action='store_true', help='Include bias in test')
        parser.add_argument('--repeats', type=int, default=1, help='Number of repeats (default: 1)')
        parser.add_argument('--trans', type=str, default='NN', choices=['NN', 'NT', 'TN', 'TT'],
                           help='Transpose: NN (default), NT, TN, TT')
        args = parser.parse_args()

        # Parse size
        try:
            m, k, n = map(int, args.size.split(','))
        except ValueError:
            print(f"Error: Invalid size format '{args.size}'. Expected 'M,K,N' format (comma-separated without spaces).", file=sys.stderr)
            sys.exit(1)

        # Validate size values
        if m <= 0 or k <= 0 or n <= 0:
            print(f"Error: Size values must be positive integers, got M={m}, K={k}, N={n}", file=sys.stderr)
            sys.exit(1)

        # Generate single test case
        case_name = f"custom_{m}x{k}x{n}_{args.trans}"
        if args.output_dir != '.':
            # Use custom output directory
            output_path = args.output_dir
        else:
            # Use default case_name subdirectory
            output_path = case_name

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        original_dir = os.getcwd()
        os.chdir(output_path)

        # Determine dtype
        dtype_map = {
            'float16': np.float16,
            'float32': np.float32,
            'int8': np.int8,
            'int32': np.int32,
        }
        if ENABLE_BF16:
            dtype_map['bf16'] = NumExt.bf16

        src_type = dtype_map.get(args.dtype, np.float16)
        dst_type = np.float32 if args.dtype in ['float16', 'float32', 'bf16'] else np.int32

        param = tmatmulParams(src_type, src_type, dst_type, m, k, n, args.bias, repeats=args.repeats, trans=args.trans)
        gen_golden_data(case_name, param)
        os.chdir(original_dir)
        print(f"Generated test data for {m}x{k}x{n} with trans={args.trans} in {output_path}/")

    else:
        # Original batch mode - generate all test cases
        # 用例名称
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
