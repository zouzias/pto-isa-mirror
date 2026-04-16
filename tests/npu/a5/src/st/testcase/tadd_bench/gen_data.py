#!/usr/bin/env python3
"""Generate test data for TADD benchmark"""
import numpy as np
import os

# All test cases
CASES = [
    ("TADDBenchTest.case_float_64x64", 64, 64),
    ("TADDBenchTest.case_float_8x512", 8, 512),
    ("TADDBenchTest.case_float_1x4096", 1, 4096),
    ("TADDBenchTest.case_float_64x128", 64, 128),
    ("TADDBenchTest.case_float_16x512", 16, 512),
    ("TADDBenchTest.case_float_1x8192", 1, 8192),
    ("TADDBenchTest.case_float_128x128", 128, 128),
    ("TADDBenchTest.case_float_32x512", 32, 512),
    ("TADDBenchTest.case_float_1x16384", 1, 16384),
]

np.random.seed(42)
for name, h, w in CASES:
    os.makedirs(name, exist_ok=True)
    elems = h * w
    np.random.rand(elems).astype(np.float32).tofile(f"{name}/input1.bin")
    np.random.rand(elems).astype(np.float32).tofile(f"{name}/input2.bin")
    np.zeros(elems, dtype=np.float32).tofile(f"{name}/golden.bin")

print(f"Generated data for {len(CASES)} test cases")
