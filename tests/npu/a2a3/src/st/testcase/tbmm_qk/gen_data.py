#!/usr/bin/env python3
"""
Generate Q/K input and golden output for TBMM_QK 128x128x128 (float32)
Writes: q.bin, k.bin, golden.bin
"""
import os
import numpy as np

np.random.seed(7)

def gen_case(path, M, K, N):
    # generate inputs in FP16, compute golden in FP32
    q_fp32 = (np.random.randn(M, K).astype(np.float16) * 1.5).astype(np.float32)
    k_fp32 = (np.random.randn(K, N).astype(np.float16) * 1.5).astype(np.float32)
    q = q_fp32.astype(np.float16)
    k = k_fp32.astype(np.float16)
    golden = (q_fp32.dot(k_fp32)).astype(np.float32)

    kt = k.T.astype(np.float16)

    # write FP16 inputs and FP32 golden
    q.tofile(os.path.join(path, 'q.bin'))
    k.tofile(os.path.join(path, 'k.bin'))
    kt.tofile(os.path.join(path, 'kt.bin'))
    golden.tofile(os.path.join(path, 'golden.bin'))


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cases = [
        ('TBMMQKTest.case_float_128x128_128x128_NN', (128, 128, 128)),
        ('TBMMQKTest.case_float_256x128_64x64_NN', (256, 128, 64)),
        ('TBMMQKTest.case_float_64x256_64x64_NN', (64, 256, 64)),
        ('TBMMQKTest.case_float_128x256_256x128_NN', (128, 256, 128)),
        ('TBMMQKTest.case_float_256x128_128x128_NN', (256, 128, 128)),
        ('TBMMQKTest.case_float_128x128_128x128_NT', (128, 128, 128)),
        ('TBMMQKTest.case_float_256x128_64x64_NT', (256, 128, 64)),
        ('TBMMQKTest.case_float_64x256_64x64_NT', (64, 256, 64)),
        ('TBMMQKTest.case_float_128x256_256x128_NT', (128, 256, 128)),
        ('TBMMQKTest.case_float_256x128_128x128_NT', (256, 128, 128)),
    ]
    for name, (M, K, N) in cases:
        case_dir = os.path.join(script_dir, name)
        os.makedirs(case_dir, exist_ok=True)
        gen_case(case_dir, M, K, N)
