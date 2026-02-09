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

def default_saturation_off(srctype, dsttype):
    """Check if this conversion's default saturation mode is OFF.
    
    Default OFF conversions (truncation/bit-extraction behavior):
    - fp16 → uint8
    - fp16 → int8
    - fp32 → int16
    - fp16 → int16
    - int64 → int32
    - int32 → int16
    
    All other conversions default to ON (clamping).
    """
    return (
        (srctype == np.float16 and dsttype == np.uint8) or
        (srctype == np.float16 and dsttype == np.int8) or
        (srctype == np.float32 and dsttype == np.int16) or
        (srctype == np.float16 and dsttype == np.int16) or
        (srctype == np.int64 and dsttype == np.int32) or
        (srctype == np.int32 and dsttype == np.int16)
    )

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

    # Generate golden data based on default saturation mode for this conversion
    if np.issubdtype(dsttype, np.integer):
        info = np.iinfo(dsttype)
        
        # Determine if this conversion has default saturation OFF (truncation) or ON (clamping)
        sat_off = default_saturation_off(srctype, dsttype)
        
        if sat_off:
            # OFF (truncation): bit extraction - wrap around using modulo
            golden_list = []
            for val in converted_golden.flat:
                if np.isnan(val) or np.isinf(val):
                    int_val = 0
                else:
                    int_val = int(np.int64(val))
                
                # Extract lower N bits and interpret as signed/unsigned
                if dsttype == np.int8:
                    byte_val = int_val & 0xFF
                    truncated_val = byte_val if byte_val < 128 else byte_val - 256
                elif dsttype == np.uint8:
                    truncated_val = int_val & 0xFF
                elif dsttype == np.int16:
                    word_val = int_val & 0xFFFF
                    truncated_val = word_val if word_val < 32768 else word_val - 65536
                elif dsttype == np.int32:
                    dword_val = int_val & 0xFFFFFFFF
                    truncated_val = dword_val if dword_val < 2147483648 else dword_val - 4294967296
                else:
                    truncated_val = int_val
                
                golden_list.append(truncated_val)
            golden = np.array(golden_list, dtype=dsttype).reshape(converted_golden.shape)
        else:
            # ON (saturation): clamp to datatype range
            # NOTE: np.clip casts a_min/a_max to the input array dtype, so for integer->integer
            # widening (e.g. int32 -> int64), clip() must run on a widened dtype first.
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
            
            # Use PyTorch for golden data generation (preferred method)
            use_torch = HAS_TORCH
            if use_torch:
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
                    
                    # Generate truncated mode using PyTorch (default PyTorch behavior)
                    torch_output = torch_input.to(np_to_torch[dsttype])
                    truncated = torch_output.numpy().astype(dsttype)
                    
                    # Generate saturated mode by handling special values
                    if np.issubdtype(srctype, np.floating):
                        # Create masks for special values BEFORE conversion
                        is_nan = torch.isnan(torch_input)
                        is_pos_inf = torch.isinf(torch_input) & (torch_input > 0)
                        is_neg_inf = torch.isinf(torch_input) & (torch_input < 0)
                        
                        # Convert to integer (PyTorch clamps finite values)
                        saturated_output = torch_input.to(np_to_torch[dsttype])
                        
                        # Replace special values with appropriate boundary values after conversion
                        saturated_output = torch.where(is_nan,
                                                      torch.tensor(info.min if np.issubdtype(dsttype, np.signedinteger) else 0, dtype=np_to_torch[dsttype]),
                                                      saturated_output)
                        saturated_output = torch.where(is_pos_inf,
                                                      torch.tensor(info.max, dtype=np_to_torch[dsttype]),
                                                      saturated_output)
                        saturated_output = torch.where(is_neg_inf,
                                                      torch.tensor(info.min, dtype=np_to_torch[dsttype]),
                                                      saturated_output)
                        saturated = saturated_output.numpy().astype(dsttype)
                    else:
                        # For integer to integer, clamp the input
                        clamped_input = torch.clamp(torch_input, info.min, info.max)
                        saturated_output = clamped_input.to(np_to_torch[dsttype])
                        saturated = saturated_output.numpy().astype(dsttype)
                    
                    print(f"Generated saturation golden data using PyTorch for {srctype.__name__} → {dsttype.__name__}")
                else:
                    print(f"Warning: PyTorch conversion not supported for {srctype.__name__} → {dsttype.__name__}, using NumPy fallback")
                    use_torch = False
            
            # NumPy fallback when PyTorch is not available or conversion not supported
            if not use_torch:
                # Saturated mode: clamp to datatype range, handling special FP values
                saturated_list = []
                for val in converted_golden.flat:
                    if np.isnan(val):
                        saturated_val = info.min if np.issubdtype(dsttype, np.signedinteger) else 0
                    elif np.isinf(val):
                        saturated_val = info.max if val > 0 else info.min
                    else:
                        int_val = int(np.int64(val))
                        saturated_val = max(info.min, min(info.max, int_val))
                    saturated_list.append(saturated_val)
                saturated = np.array(saturated_list, dtype=dsttype).reshape([m, n])
                
                # Truncated mode: bit extraction (modulo behavior)
                truncated_list = []
                for val in converted_golden.flat:
                    if np.isnan(val) or np.isinf(val):
                        int_val = 0
                    else:
                        int_val = int(np.int64(val))
                    
                    # Extract lower N bits and interpret as signed/unsigned
                    if dsttype == np.int8:
                        byte_val = int_val & 0xFF
                        truncated_val = byte_val if byte_val < 128 else byte_val - 256
                    elif dsttype == np.uint8:
                        truncated_val = int_val & 0xFF
                    elif dsttype == np.int16:
                        word_val = int_val & 0xFFFF
                        truncated_val = word_val if word_val < 32768 else word_val - 65536
                    elif dsttype == np.int32:
                        dword_val = int_val & 0xFFFFFFFF
                        truncated_val = dword_val if dword_val < 2147483648 else dword_val - 4294967296
                    else:
                        truncated_val = int_val
                    
                    truncated_list.append(truncated_val)
                truncated = np.array(truncated_list, dtype=dsttype).reshape([m, n])
                
                print(f"Generated saturation golden data using NumPy fallback for {srctype.__name__} → {dsttype.__name__}")
            
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
