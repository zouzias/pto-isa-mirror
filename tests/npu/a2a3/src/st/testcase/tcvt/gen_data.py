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
import numpy as np
try:
    import torch
    HAS_TORCH = True
except ImportError:
    print("Warning: PyTorch not available, using NumPy for saturation tests")
    HAS_TORCH = False

np.random.seed(19)

def gen_golden(case_name, param):
    srctype = param.srctype
    dsttype = param.dsttype
    m, n = param.m, param.n
    is_saturation_test = "saturation_" in case_name

    # Generate input data with reasonable ranges
    if is_saturation_test:
        # For saturation tests, use only special values: inf, -inf, nan, and 2 overflow values
        # Pad to 32 elements to meet minimum shape requirement
        if srctype == np.float32 or srctype == np.float16:
            if dsttype == np.int8:
                # Special values: -inf, inf, nan, and 2 overflow values
                special_values = [
                    -np.inf,  # -infinity
                    np.inf,   # +infinity
                    np.nan,   # NaN
                    -200.0,   # Overflow below min (-128)
                    200.0,    # Overflow above max (127)
                ]
                # Pad with zeros to reach m*n elements
                x1_gm = np.array(special_values + [0.0] * (m*n - len(special_values))).astype(srctype).reshape([m, n])
            elif dsttype == np.uint8:
                special_values = [
                    -np.inf,  # -infinity
                    np.inf,   # +infinity
                    np.nan,   # NaN
                    -100.0,   # Overflow below min (0)
                    300.0,    # Overflow above max (255)
                ]
                x1_gm = np.array(special_values + [0.0] * (m*n - len(special_values))).astype(srctype).reshape([m, n])
            elif dsttype == np.int16:
                special_values = [
                    -np.inf,    # -infinity
                    np.inf,     # +infinity
                    np.nan,     # NaN
                    -40000.0,   # Overflow below min (-32768)
                    40000.0,    # Overflow above max (32767)
                ]
                x1_gm = np.array(special_values + [0.0] * (m*n - len(special_values))).astype(srctype).reshape([m, n])
            elif dsttype == np.int32:
                special_values = [
                    -np.inf,  # -infinity
                    np.inf,   # +infinity
                    np.nan,   # NaN
                    -3e9,     # Overflow below min
                    3e9,      # Overflow above max
                ]
                x1_gm = np.array(special_values + [0.0] * (m*n - len(special_values))).astype(srctype).reshape([m, n])
            else:
                x1_gm = (np.random.random([m, n]) * 200 - 100).astype(srctype)
        elif srctype == np.int64:
            # int64 to int32 saturation test - only overflow values (no inf/nan for integers)
            if dsttype == np.int32:
                special_values = [
                    -3000000000,  # Overflow below min
                    3000000000,   # Overflow above max
                    -2147483648,  # At min boundary
                    2147483647,   # At max boundary
                    0,            # Zero
                ]
                x1_gm = np.array(special_values + [0] * (m*n - len(special_values))).astype(srctype).reshape([m, n])
            else:
                x1_gm = np.random.randint(-10000, 10000, [m, n]).astype(srctype)
        elif srctype == np.int32:
            # int32 to int16 saturation test - only overflow values
            if dsttype == np.int16:
                special_values = [
                    -40000,   # Overflow below min
                    40000,    # Overflow above max
                    -32768,   # At min boundary
                    32767,    # At max boundary
                    0,        # Zero
                ]
                x1_gm = np.array(special_values + [0] * (m*n - len(special_values))).astype(srctype).reshape([m, n])
            else:
                x1_gm = np.random.randint(-10000, 10000, [m, n]).astype(srctype)
        else:
            x1_gm = (np.random.random([m, n]) * 200 - 100).astype(srctype)
    elif srctype == np.float32 or srctype == np.float16:
        # Floating point: range [-100, 100]
        x1_gm = (np.random.random([m, n]) * 200 - 100).astype(srctype)
    elif srctype == np.int8:
        # int8: full range [-128, 127]
        x1_gm = np.random.randint(-128, 128, [m, n]).astype(srctype)
    elif srctype == np.uint8:
        # uint8: full range [0, 255]
        x1_gm = np.random.randint(0, 256, [m, n]).astype(srctype)
    elif srctype == np.int16:
        # int16: reasonable range [-1000, 1000]
        x1_gm = np.random.randint(-1000, 1000, [m, n]).astype(srctype)
    elif srctype == np.uint16:
        # uint16: reasonable range [0, 10000]
        x1_gm = np.random.randint(0, 10000, [m, n]).astype(srctype)
    elif srctype == np.int32:
        # int32: reasonable range [-10000, 10000]
        x1_gm = np.random.randint(-10000, 10000, [m, n]).astype(srctype)
    elif srctype == np.uint32:
        # uint32: reasonable range [0, 10000]
        x1_gm = np.random.randint(0, 10000, [m, n]).astype(srctype)
    elif srctype == np.int64:
        # int64: reasonable range [-10000, 10000]
        x1_gm = np.random.randint(-10000, 10000, [m, n]).astype(srctype)
    else:
        # Default: signed int range
        x1_gm = np.random.randint(-10000, 10000, [m, n]).astype(srctype)

    # Apply rounding mode for conversions
    mode = param.mode
    
    # Perform conversion first
    if np.issubdtype(srctype, np.floating):
        if np.issubdtype(dsttype, np.integer):
            # Floating point to integer conversion
            if mode == "RoundMode::CAST_RINT":
                converted_golden = np.rint(x1_gm)
            elif mode == "RoundMode::CAST_ROUND":
                converted_golden = np.round(x1_gm)
            elif mode == "RoundMode::CAST_FLOOR":
                converted_golden = np.floor(x1_gm)
            elif mode == "RoundMode::CAST_CEIL":
                converted_golden = np.ceil(x1_gm)
            elif mode == "RoundMode::CAST_TRUNC":
                converted_golden = np.trunc(x1_gm)
            else:
                converted_golden = x1_gm
        elif srctype == np.float32 and dsttype == np.float32:
            # FP32 to FP32 conversion - apply rounding to integer values but keep as float
            if mode == "RoundMode::CAST_RINT":
                converted_golden = np.rint(x1_gm)
            elif mode == "RoundMode::CAST_ROUND":
                converted_golden = np.round(x1_gm)
            elif mode == "RoundMode::CAST_FLOOR":
                converted_golden = np.floor(x1_gm)
            elif mode == "RoundMode::CAST_CEIL":
                converted_golden = np.ceil(x1_gm)
            elif mode == "RoundMode::CAST_TRUNC":
                converted_golden = np.trunc(x1_gm)
            else:
                converted_golden = x1_gm
        else:
            # Other float to float conversions - no rounding applied
            converted_golden = x1_gm
    else:
        # Integer to any type conversion
        converted_golden = x1_gm

    # Clamp the result to the destination type's representable range.
    # NOTE: np.clip casts a_min/a_max to the input array dtype, so for integer->integer
    # widening (e.g. int32 -> int64), clip() must run on a widened dtype first.
    if np.issubdtype(dsttype, np.integer):
        info = np.iinfo(dsttype)
        tmp = converted_golden
        if np.issubdtype(tmp.dtype, np.integer):
            if np.issubdtype(dsttype, np.signedinteger):
                tmp = tmp.astype(np.int64, copy=False)
            else:
                tmp = tmp.astype(np.uint64, copy=False)
        else:
            tmp = tmp.astype(np.float64, copy=False)
        golden = np.clip(tmp, info.min, info.max).astype(dsttype)
    elif np.issubdtype(dsttype, np.floating):
        info = np.finfo(dsttype)
        golden = np.clip(converted_golden.astype(np.float64, copy=False), info.min, info.max).astype(dsttype)
    else:
        golden = converted_golden.astype(dsttype)
            
    x1_gm.tofile("./x1_gm.bin")
    golden.tofile("./golden.bin")
    
    # For saturation tests, generate golden data using PyTorch behavior
    if is_saturation_test:
        if np.issubdtype(dsttype, np.integer):
            info = np.iinfo(dsttype)
            
            # Use PyTorch for golden data generation if available
            # For saturation tests, we need two different outputs:
            # 1. Saturated: clamp to valid range
            # 2. Truncated: bit extraction (modulo behavior)
            
            # Saturated mode: clamp to datatype range
            saturated = np.clip(converted_golden.astype(np.int64), info.min, info.max).astype(dsttype)
            
            # Truncated mode: bit extraction (matches PyTorch for integer conversions)
            as_int64 = converted_golden.astype(np.int64)
            if dsttype == np.int8:
                truncated = np.array([np.int8(val & 0xFF) for val in as_int64.flat], dtype=np.int8).reshape([m, n])
            elif dsttype == np.uint8:
                truncated = np.array([np.uint8(val & 0xFF) for val in as_int64.flat], dtype=np.uint8).reshape([m, n])
            elif dsttype == np.int16:
                truncated = np.array([np.int16(val & 0xFFFF) for val in as_int64.flat], dtype=np.int16).reshape([m, n])
            elif dsttype == np.int32:
                truncated = np.array([np.int32(val & 0xFFFFFFFF) for val in as_int64.flat], dtype=np.int32).reshape([m, n])
            else:
                truncated = saturated
            
            # Verify with PyTorch if available (truncated should match PyTorch)
            if HAS_TORCH:
                np_to_torch = {
                    np.float32: torch.float32,
                    np.float16: torch.float16,
                    np.int64: torch.int64,
                    np.int32: torch.int32,
                    np.int16: torch.int16,
                    np.int8: torch.int8,
                    np.uint8: torch.uint8,
                }
                
                if srctype in np_to_torch and dsttype in np_to_torch:
                    # Convert input to torch tensor
                    if np.issubdtype(srctype, np.floating):
                        torch_input = torch.from_numpy(x1_gm.astype(np.float32))
                        torch_input = torch_input.to(np_to_torch[srctype])
                    else:
                        torch_input = torch.from_numpy(x1_gm)
                        if srctype in np_to_torch:
                            torch_input = torch_input.to(np_to_torch[srctype])
                    
                    # PyTorch conversion - this should match truncated mode
                    torch_output = torch_input.to(np_to_torch[dsttype])
                    torch_result = torch_output.numpy().astype(dsttype)
                    
                    # Verify truncated matches PyTorch
                    if not np.array_equal(truncated, torch_result):
                        print(f"Warning: Truncated mode doesn't match PyTorch for {srctype.__name__} → {dsttype.__name__}")
                        mismatches = np.sum(truncated != torch_result)
                        print(f"  Mismatches: {mismatches}/{truncated.size}")
            
            saturated.tofile("./golden_saturated.bin")
            truncated.tofile("./golden_truncated.bin")
                
class tcvtParams:
    def __init__(self, srctype, dsttype, m, n, mode):
        self.srctype = srctype
        self.dsttype = dsttype
        self.m = m
        self.n = n
        self.mode = mode

if __name__ == "__main__":
    # Type conversion pairs: (name_suffix, source_type, destination_type)
    type_pairs = [
        # FP32 Source
        ("fp32_fp32", np.float32, np.float32),
        ("fp32_fp16", np.float32, np.float16),
        ("fp32_int32", np.float32, np.int32),
        ("fp32_int16", np.float32, np.int16),
        ("fp32_int64", np.float32, np.int64),
        
        # FP16 Source
        ("fp16_fp32", np.float16, np.float32),
        ("fp16_int32", np.float16, np.int32),
        ("fp16_int16", np.float16, np.int16),
        ("fp16_int8", np.float16, np.int8),
        ("fp16_uint8", np.float16, np.uint8),

        # INT32 Source
        ("int32_fp32", np.int32, np.float32),
        ("int32_int16", np.int32, np.int16),
        ("int32_int64", np.int32, np.int64),

        # INT16 Source
        ("int16_fp16", np.int16, np.float16),
        ("int16_fp32", np.int16, np.float32),

        # INT8 Source
        ("int8_fp16", np.int8, np.float16),

        # UINT8 Source
        ("uint8_fp16", np.uint8, np.float16),

        # INT64 Source
        ("int64_fp32", np.int64, np.float32),
        ("int64_int32", np.int64, np.int32),
    ]

    # Different shape configurations (m, n)
    shapes = [
        (2, 128),
        (2, 32),
        (1, 64),
        (4, 64),
    ]

    case_name_list = []
    case_params_list = []

    # Generate test cases for each type pair and shape combination
    for type_name, src, dst in type_pairs:
        for m, n in shapes:
            case_name = f"case_{type_name}_{m}x{n}"
            case_name_list.append(f"TCVTTest.{case_name}")
            case_params_list.append(tcvtParams(src, dst, m, n, "RoundMode::CAST_RINT"))

    # Add saturation mode test cases (only for supported conversions on A2A3)
    # Note: fp32→int8 is NOT supported on A2A3 hardware
    # Using 1x5 shape: inf, -inf, nan, and 2 overflow values
    saturation_tests = [
        ("saturation_fp16_int8_1x32", np.float16, np.int8, 1, 32),
        ("saturation_fp32_int16_1x32", np.float32, np.int16, 1, 32),
        ("saturation_fp32_int32_1x32", np.float32, np.int32, 1, 32),
        ("saturation_fp16_uint8_1x32", np.float16, np.uint8, 1, 32),
        ("saturation_fp16_int32_1x32", np.float16, np.int32, 1, 32),
        ("saturation_int64_int32_1x32", np.int64, np.int32, 1, 32),
        ("saturation_int32_int16_1x32", np.int32, np.int16, 1, 32),
    ]
    
    for test_name, src, dst, m, n in saturation_tests:
        case_name_list.append(f"TCVTTest.{test_name}")
        case_params_list.append(tcvtParams(src, dst, m, n, "RoundMode::CAST_RINT"))

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden(case_name, case_params_list[i])

        os.chdir(original_dir)
