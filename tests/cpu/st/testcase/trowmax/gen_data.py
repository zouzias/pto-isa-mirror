# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import numpy as np
import sys

def get_test_params():
    params = []
    # FP32 tile shapes
    params.append({'type': np.float32, 'global_row': 64, 'global_col': 64, 'tile_row': 64, 'tile_col': 64, 'valid_row': 64, 'valid_col': 64})
    params.append({'type': np.float32, 'global_row': 77, 'global_col': 81, 'tile_row': 32, 'tile_col': 16, 'valid_row': 32, 'valid_col': 16})
    params.append({'type': np.float32, 'global_row': 32, 'global_col': 32, 'tile_row': 32, 'tile_col': 16, 'valid_row': 32, 'valid_col': 16})
    # FP16 tile shapes
    params.append({'type': np.float16, 'global_row': 64, 'global_col': 64, 'tile_row': 64, 'tile_col': 64, 'valid_row': 64, 'valid_col': 64})
    params.append({'type': np.float16, 'global_row': 161, 'global_col': 161, 'tile_row': 32, 'tile_col': 32, 'valid_row': 32, 'valid_col': 32})
    # BF16 tile shapes
    params.append({'type': np.bfloat16, 'global_row': 64, 'global_col': 64, 'tile_row': 64, 'tile_col': 64, 'valid_row': 64, 'valid_col': 64})
    # INT8 tile shapes
    params.append({'type': np.int8, 'global_row': 64, 'global_col': 64, 'tile_row': 64, 'tile_col': 64, 'valid_row': 64, 'valid_col': 64})
    params.append({'type': np.int8, 'global_row': 128, 'global_col': 128, 'tile_row': 32, 'tile_col': 32, 'valid_row': 32, 'valid_col': 32})
    params.append({'type': np.int8, 'global_row': 64, 'global_col': 32, 'tile_row': 32, 'tile_col': 16, 'valid_row': 32, 'valid_col': 16})
    params.append({'type': np.int8, 'global_row': 96, 'global_col': 96, 'tile_row': 48, 'tile_col': 48, 'valid_row': 48, 'valid_col': 48})
    # UINT8 tile shapes
    params.append({'type': np.uint8, 'global_row': 64, 'global_col': 64, 'tile_row': 64, 'tile_col': 64, 'valid_row': 64, 'valid_col': 64})
    params.append({'type': np.uint8, 'global_row': 128, 'global_col': 128, 'tile_row': 32, 'tile_col': 32, 'valid_row': 32, 'valid_col': 32})
    params.append({'type': np.uint8, 'global_row': 64, 'global_col': 32, 'tile_row': 32, 'tile_col': 16, 'valid_row': 32, 'valid_col': 16})
    params.append({'type': np.uint8, 'global_row': 96, 'global_col': 96, 'tile_row': 48, 'tile_col': 48, 'valid_row': 48, 'valid_col': 48})
    return params

def gen_golden_data_float(input_data, global_row, global_col, tile_row, tile_col, valid_row, valid_col):
    dst = np.zeros((global_row, global_col), dtype=np.float32)
    golden = np.zeros(global_row, dtype=np.float32)
    for i in range(global_row):
        golden[i] = input_data[i * global_col]
        for j in range(1, global_col):
            golden[i] = max(golden[i], input_data[i * global_col + j])
    return golden

def gen_golden_data_float16(input_data, global_row, global_col, tile_row, tile_col, valid_row, valid_col):
    return gen_golden_data_float(input_data, global_row, global_col, tile_row, tile_col, valid_row, valid_col)

def gen_golden_data_bfloat16(input_data, global_row, global_col, tile_row, tile_col, valid_row, valid_col):
    return gen_golden_data_float(input_data, global_row, global_col, tile_row, tile_col, valid_row, valid_col)

def gen_golden_data_int8(input_data, global_row, global_col, tile_row, tile_col, valid_row, valid_col):
    golden = np.zeros(global_row, dtype=np.int8)
    for i in range(global_row):
        golden[i] = input_data[i * global_col]
        for j in range(1, global_col):
            golden[i] = max(golden[i], input_data[i * global_col + j])
    return golden

def gen_golden_data_uint8(input_data, global_row, global_col, tile_row, tile_col, valid_row, valid_col):
    golden = np.zeros(global_row, dtype=np.uint8)
    for i in range(global_row):
        golden[i] = input_data[i * global_col]
        for j in range(1, global_col):
            golden[i] = max(golden[i], input_data[i * global_col + j])
    return golden

def main():
    params = get_test_params()
    for param in params:
        print("Generate test data for type:", param["type"])
        if not np.issubdtype(param["type"], np.integer):
            input_data = np.random.rand(param["global_row"], param["global_col"]).astype(param["type"])
        else:
            iinfo = np.iinfo(param["type"])
            input_data = np.random.randint(iinfo.min, iinfo.max + 1, (param["global_row"], param["global_col"]), dtype=param["type"])

        if param["type"] == np.float32:
            golden = gen_golden_data_float(input_data, param["global_row"], param["global_col"], param["tile_row"], param["tile_col"], param["valid_row"], param["valid_col"])
        elif param["type"] == np.float16:
            golden = gen_golden_data_float16(input_data, param["global_row"], param["global_col"], param["tile_row"], param["tile_col"], param["valid_row"], param["valid_col"])
        elif param["type"] == np.bfloat16:
            golden = gen_golden_data_bfloat16(input_data, param["global_row"], param["global_col"], param["tile_row"], param["tile_col"], param["valid_row"], param["valid_col"])
        elif param["type"] == np.int8:
            golden = gen_golden_data_int8(input_data, param["global_row"], param["global_col"], param["tile_row"], param["tile_col"], param["valid_row"], param["valid_col"])
        elif param["type"] == np.uint8:
            golden = gen_golden_data_uint8(input_data, param["global_row"], param["global_col"], param["tile_row"], param["tile_col"], param["valid_row"], param["valid_col"])
        else:
            print("Unsupported data type")
            sys.exit(1)

        input_path = "data/input1.bin"
        golden_path = "data/golden.bin"

        np.savetxt(input_path, input_data.reshape(-1), fmt='%g')
        np.savetxt(golden_path, golden, fmt='%g')
        print("Test data generated successfully")

if __name__ == "__main__":
    main()
