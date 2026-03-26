#!/usr/bin/env python3
"""
Generate golden data for TADD tutorial test cases.
"""
import os
import numpy as np

case_name_list = [
    "TADDTutorialTest.case_float_16x64_16x64",
]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(script_dir, "..", "..", "build")
    
    for case_name in case_name_list:
        case_dir = os.path.join(build_dir, case_name)
        os.makedirs(case_dir, exist_ok=True)
        
        # Parse dimensions from case name
        # case_float_16x64_16x64 -> float, 16, 64
        parts = case_name.split(".")[-1].split("_")
        dtype_str = parts[1]  # float
        rows = int(parts[2].split("x")[0])  # 16
        cols = int(parts[2].split("x")[1])  # 64
        
        dtype = np.float32 if dtype_str == "float" else np.float16
        
        # Generate random input data
        np.random.seed(42)
        input1 = np.random.randn(rows, cols).astype(dtype)
        input2 = np.random.randn(rows, cols).astype(dtype)
        output = input1 + input2
        
        # Save to binary files
        input1.tofile(os.path.join(case_dir, "input1.bin"))
        input2.tofile(os.path.join(case_dir, "input2.bin"))
        output.tofile(os.path.join(case_dir, "output.bin"))
        
        print(f"Generated: {case_dir}")

if __name__ == "__main__":
    main()
