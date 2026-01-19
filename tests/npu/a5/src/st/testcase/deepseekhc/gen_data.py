#!/usr/bin/python3
# coding=utf-8

import os
import numpy as np
np.random.seed(19)

def gen_golden_data_deepseek(case_name, param):
    dtype = param.dtype

    S0, S1, S2, S3 = [param.shape0, param.shape1, param.shape2, param.shape3]
    h_valid, w_valid = [param.valid_row, param.valid_col]

    # Generate random input arrays
    input1 = np.random.randint(1, 10, size=[S0, S1, S2, S3]).astype(dtype)
    input2 = np.random.randint(1, 10, size=[S0, S1, S2, S3]).astype(dtype)
    input3 = np.array([S0, S1, h_valid, w_valid]).astype(np.uint32)

    # Perform the addbtraction
    golden = input1 + input2

    # Apply valid region constraints
    output = np.zeros([S0, S1, S2, S3]).astype(dtype)
    for i in range(S0):
        for j in range(S1):
            for h in range(S2):
                for w in range(S3):
                    if h >= h_valid or w >= w_valid:
                        golden[i][j][h][w] = output[i][j][h][w]

    # Save the input and golden data to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    input3.tofile("input3.bin")
    golden.tofile("golden.bin")

    return output, input1, input2, golden

class deepseekParams:
    def __init__(self, dtype, shape0, shape1, shape2, shape3, vRows, vCols):
        self.dtype = dtype
        self.shape0 = shape0
        self.shape1 = shape1
        self.shape2 = shape2
        self.shape3 = shape3
        self.valid_row = vRows
        self.valid_col = vCols 

def generate_case_name(param):
    dtype_str = {
        np.float32: 'float',
        np.float16: 'half',
        np.int8: 'int8',
        np.int32: 'int32',
        np.int16: 'int16'
    }[param.dtype]
    return f"DsHcPostTest.case_{dtype_str}_{param.shape0}_{param.shape1}_{param.shape2}_{param.shape3}"

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        # deepseekParams(np.float32, 1, 1, 107, 152, 107, 152),
        # deepseekParams(np.float32, 1, 8, 1, 2048, 1, 2048),
        # deepseekParams(np.float32, 1, 8, 1, 32, 1, 32),
        # deepseekParams(np.float32, 1, 1, 256, 16, 256, 16),
        # deepseekParams(np.float32, 1, 8, 8, 128, 8, 128),
        # deepseekParams(np.float32, 16, 16, 1, 64, 1, 64),
        # deepseekParams(np.float32, 16, 8, 2, 64, 2, 64),
        # deepseekParams(np.float32, 1, 1, 1, 16384, 1, 16384),
        deepseekParams(np.float32, 3, 33, 1, 8, 1, 8),
        # deepseekParams(np.float32, 2, 2, 2, 1024, 2, 1024),
    ]

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_deepseek(case_name, param)
        os.chdir(original_dir)
