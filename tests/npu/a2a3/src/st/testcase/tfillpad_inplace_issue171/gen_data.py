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
"""
Issue #171: TFILLPAD_INPLACE produces incorrect padding on simulator (all N) and hardware (N=16)

Generate test data for the bug scenario:
- N=16, valid_len in [1,8] (triggers Path B which is buggy)
- Pad with -inf (PadValue::Min for float)

Cases:
  1: M=16, N=16, validLen=1  (pads 15 columns) -> SHOULD FAIL on SIM
  2: M=16, N=16, validLen=8  (pads 8 columns)  -> SHOULD FAIL on SIM
  3: M=16, N=16, validLen=9  (pads 7 columns)  -> SHOULD PASS (Path B NO-OP)
  4: M=16, N=16, validLen=15 (pads 1 column)   -> SHOULD PASS (Path B NO-OP)
"""

import os
import numpy as np
np.random.seed(42)


def gen_golden_data(case_name, M, N, valid_len):
    """Generate input and golden data for TFILLPAD_INPLACE test."""
    
    # Sequential values for easy debugging
    input_data = np.arange(M * N, dtype=np.float32).reshape(M, N)
    
    # Golden: same as input for valid columns, -inf for pad columns
    golden = input_data.copy()
    golden[:, valid_len:] = -np.inf
    
    # Save the input and golden data to binary files
    input_data.tofile("input.bin")   # Source data (will be loaded by TLOAD)
    golden.tofile("golden.bin")       # Expected output after TFILLPAD_INPLACE
    
    print(f"Generated {case_name}:")
    print(f"  M={M}, N={N}, validLen={valid_len}")
    print(f"  Pad columns: {N - valid_len}")
    print(f"  Input shape: {input_data.shape}")
    
    return input_data, golden


class TFillPadParams:
    def __init__(self, case_id, M, N, valid_len, should_fail):
        self.case_id = case_id
        self.M = M
        self.N = N
        self.valid_len = valid_len
        self.should_fail = should_fail
    
    def case_name(self):
        suffix = ("SHOULD_FAIL" if self.should_fail else "SHOULD_PASS") if self.should_fail is not None else ""
        suffix_part = f"_{suffix}" if suffix else ""
        return f"TFILLPADIssue171Test.case_float_{self.M}x{self.N}_validlen_{self.valid_len}{suffix_part}"


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Test cases based on Issue #171 analysis
    case_params_list = [
        TFillPadParams(1, 16, 16, 1, should_fail=True),   # Path B runs -> BUG
        TFillPadParams(2, 16, 16, 8, should_fail=True),   # Path B runs -> BUG
        TFillPadParams(3, 16, 16, 9, should_fail=False),  # Path B NO-OP
        TFillPadParams(4, 16, 16, 15, should_fail=False), # Path B NO-OP
        # Issue #171 exact cases: N=32/64/128 with validLen=1
        TFillPadParams(5, 16, 32, 1, should_fail=None),   # N=32, pads 31 columns
        TFillPadParams(6, 16, 64, 1, should_fail=None),   # N=64, pads 63 columns
        TFillPadParams(7, 16, 128, 1, should_fail=None),  # N=128, pads 127 columns
    ]
    
    for param in case_params_list:
        case_name = param.case_name()
        case_dir = os.path.join(script_dir, case_name)
        
        if not os.path.exists(case_dir):
            os.makedirs(case_dir)
        
        original_dir = os.getcwd()
        os.chdir(case_dir)
        gen_golden_data(case_name, param.M, param.N, param.valid_len)
        os.chdir(original_dir)
    
    print(f"\nGenerated {len(case_params_list)} test cases in {script_dir}")
