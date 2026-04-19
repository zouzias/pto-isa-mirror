#!/usr/bin/env python3
"""Generate test data for tile op benchmark - reads from input.csv"""
import numpy as np
import os
import sys
from pathlib import Path

UNARY_OPS = {'texp', 'tlog', 'tsqrt', 'tabs', 'tneg', 'trcp', 'trsqrt'}
SCALAR_OPS = {'tadds', 'tsubs', 'tmuls', 'tdivs', 'tmaxs', 'tmins'}
REDUCE_ROW_OPS = {'trowsum'}
REDUCE_COL_OPS = {'tcolsum'}
BROADCAST_SCALAR_OPS = {'texpands'}
BROADCAST_ROW_OPS = {'trowexpand'}
BROADCAST_COL_OPS = {'tcolexpand'}

def get_op_type(op):
    op_lower = op.lower()
    if op_lower in UNARY_OPS:
        return 'unary'
    elif op_lower in SCALAR_OPS:
        return 'scalar'
    elif op_lower in REDUCE_ROW_OPS:
        return 'reduce_row'
    elif op_lower in REDUCE_COL_OPS:
        return 'reduce_col'
    elif op_lower in BROADCAST_SCALAR_OPS:
        return 'broadcast_scalar'
    elif op_lower in BROADCAST_ROW_OPS:
        return 'broadcast_row'
    elif op_lower in BROADCAST_COL_OPS:
        return 'broadcast_col'
    return 'binary'

def dtype_to_np(dtype):
    mapping = {
        'float32': np.float32,
        'float16': np.float16,
        'bfloat16': np.float16,
        'int32': np.int32,
        'int16': np.int16,
        'int8': np.int8,
        'uint8': np.uint8
    }
    return mapping.get(dtype, np.float32)

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

def main():
    # Try multiple locations for input.csv
    script_dir = Path(__file__).parent
    possible_paths = [
        script_dir / "input.csv",  # Same dir as script
        script_dir.parent / "testcase" / "tile_perf" / "input.csv",  # build/../testcase/tile_perf/
        Path("../testcase/tile_perf/input.csv"),  # Relative from build/
    ]
    
    csv_path = None
    for p in possible_paths:
        if p.exists():
            csv_path = p
            break
    
    if csv_path is None:
        print(f"ERROR: input.csv not found in any of: {[str(p) for p in possible_paths]}")
        sys.exit(1)
    
    print(f"Using: {csv_path}")
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
        op_type = c['op_type']
        
        dtype_gtest = dtype_to_gtest(dtype)
        np_dtype = dtype_to_np(dtype)
        
        test_name = f"{op.upper()}BenchTest.case_{dtype_gtest}_{h}x{w}"
        os.makedirs(test_name, exist_ok=True)
        
        # Generate input1 (always needed except broadcast_scalar)
        if op_type == 'broadcast_scalar':
            # TEXPANDS: no input tile, just scalar → output
            np.zeros(elems, dtype=np_dtype).tofile(f"{test_name}/golden.bin")
            print(f"Generated: {test_name} ({op_type}, scalar only)")
            continue

        # Determine input shape for broadcast ops
        if op_type == 'broadcast_row':
            # TROWEXPAND: src is (H, 1) column vector
            input1_elems = h
        elif op_type == 'broadcast_col':
            # TCOLEXPAND: src is (1, W) row vector
            input1_elems = w
        else:
            input1_elems = elems

        if np_dtype in [np.float32, np.float16]:
            # For exp, use small values to avoid overflow
            if op.lower() in ['texp']:
                data1 = (np.random.rand(input1_elems) * 2 - 1).astype(np_dtype)  # [-1, 1]
            else:
                data1 = np.random.rand(input1_elems).astype(np_dtype)
            data1.tofile(f"{test_name}/input1.bin")
        else:
            np.random.randint(0, 100, input1_elems, dtype=np_dtype).tofile(f"{test_name}/input1.bin")
        
        # Generate input2 (only for binary ops)
        if op_type == 'binary':
            if np_dtype in [np.float32, np.float16]:
                np.random.rand(elems).astype(np_dtype).tofile(f"{test_name}/input2.bin")
            else:
                np.random.randint(0, 100, elems, dtype=np_dtype).tofile(f"{test_name}/input2.bin")
        
        # Generate golden placeholder
        np.zeros(elems, dtype=np_dtype).tofile(f"{test_name}/golden.bin")
        
        print(f"Generated: {test_name} ({op_type})")
    
    print(f"\nGenerated data for {len(cases)} test cases")

if __name__ == "__main__":
    main()
