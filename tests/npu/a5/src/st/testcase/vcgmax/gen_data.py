#!/usr/bin/python3
import os
import numpy as np

np.random.seed(1)

def make_fp16_bits(x):
    value = np.float16(np.nan if x == 'nan' else x)
    return np.frombuffer(value.tobytes(), dtype=np.uint16)[0]

if __name__ == '__main__':
    case_dir = 'VCGMAXTest.case1_nan_fp16'
    if not os.path.exists(case_dir):
        os.makedirs(case_dir)
    os.chdir(case_dir)

    vals = [1.0, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 'nan', 15.0, 14.0, -8.0, 0.5, -3.0, 2.0, 9.0, 10.0]
    bits = np.array([make_fp16_bits(v) for v in vals], dtype=np.uint16)
    bits.tofile('input.bin')

    gold = np.array([make_fp16_bits('nan')], dtype=np.uint16)
    gold.tofile('golden.bin')
