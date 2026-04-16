#!/usr/bin/env python3
"""Generate test data for TADD benchmark - reads from input.csv"""
import numpy as np
import os
import csv
from pathlib import Path

def dtype_to_np(dtype):
    """Convert dtype string to numpy dtype"""
    mapping = {
        'float32': np.float32,
        'float16': np.float16,
        'bfloat16': np.float16,  # numpy doesn't have bfloat16, use float16 for data gen
        'int32': np.int32,
        'int16': np.int16,
        'int8': np.int8,
        'uint8': np.uint8
    }
    return mapping.get(dtype, np.float32)

def dtype_to_gtest(dtype):
    """Convert dtype to gtest case naming"""
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

def read_input_csv(csv_path):
    """Read test cases from input.csv"""
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

def main():
    script_dir = Path(__file__).parent
    csv_path = script_dir / "input.csv"
    
    cases = read_input_csv(csv_path)
    if not cases:
        print(f"No cases found in {csv_path}")
        return
    
    np.random.seed(42)
    
    for c in cases:
        op = c['op']
        dtype = c['dtype']
        h, w = c['valid_h'], c['valid_w']
        elems = h * w
        
        dtype_gtest = dtype_to_gtest(dtype)
        np_dtype = dtype_to_np(dtype)
        
        # Test name follows gtest convention
        test_name = f"{op.upper()}BenchTest.case_{dtype_gtest}_{h}x{w}"
        
        os.makedirs(test_name, exist_ok=True)
        
        # Generate random input data
        if np_dtype in [np.float32, np.float16]:
            np.random.rand(elems).astype(np_dtype).tofile(f"{test_name}/input1.bin")
            np.random.rand(elems).astype(np_dtype).tofile(f"{test_name}/input2.bin")
            np.zeros(elems, dtype=np_dtype).tofile(f"{test_name}/golden.bin")
        else:
            np.random.randint(0, 100, elems, dtype=np_dtype).tofile(f"{test_name}/input1.bin")
            np.random.randint(0, 100, elems, dtype=np_dtype).tofile(f"{test_name}/input2.bin")
            np.zeros(elems, dtype=np_dtype).tofile(f"{test_name}/golden.bin")
        
        print(f"Generated: {test_name}")
    
    print(f"\nGenerated data for {len(cases)} test cases")

if __name__ == "__main__":
    main()
