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
np.random.seed(19)


def gen_golden_data_tdequant(case_name, param):
    dstDType = param.dstDType
    srcDType = param.srcDType

    m, n = [param.dstValidRows, param.dstValidCols]
    dst_tile_shape = (param.dstRows, param.dstCols)
    src_tile_shape = (param.srcRows, param.srcCols)
    para_tile_shape = (param.paraRows, param.paraCols)

    # src [m, n] -> padding [srcRows, srcCols]
    # int8: [-128, 127], int16: [-32768, 32767]
    if srcDType == np.int8:
        src_valid = np.random.randint(-128, 128, size=(m, n), dtype=np.int8)
    elif srcDType == np.int16:
        src_valid = np.random.randint(-32768, 32768, size=(m, n), dtype=np.int16)
    else:
        raise ValueError(f"Unsupported src dtype: {srcDType}")
    src = np.zeros(src_tile_shape, dtype=srcDType)
    src[:m, :n] = src_valid
    
    scale_valid = np.random.uniform(0.001, 1.0, size=(m, 1)).astype(np.float32)
    scale = np.zeros(para_tile_shape, dtype=np.float32)
    scale[:m, :1] = scale_valid

    if srcDType == np.int8:
        offset_valid = np.random.uniform(-128, 127, size=(m, 1)).astype(np.float32)
    else:  # int16
        offset_valid = np.random.uniform(-32768, 32767, size=(m, 1)).astype(np.float32)
    offset = np.zeros(para_tile_shape, dtype=np.float32)
    offset[:m, :1] = offset_valid

    src_float = src_valid.astype(np.float32)
    offset_broadcast = np.broadcast_to(offset_valid[:, :1], (m, n))
    scale_broadcast = np.broadcast_to(scale_valid[:, :1], (m, n))

    dst_valid = (src_float - offset_broadcast) * scale_broadcast
    dst_valid = dst_valid.astype(np.float32)
    dst = np.zeros(dst_tile_shape, dtype=np.float32)
    dst[:m, :n] = dst_valid

    dst.tofile("golden.bin")
    src.tofile("srcInput.bin")
    scale.tofile("scaleInput.bin")
    offset.tofile("offsetInput.bin")

    return dst, src, scale, offset


class TDequantParams:
    def __init__(self, name, dstDType, srcDType, dstRows, dstCols, srcRows, srcCols, 
                 dstValidRows, dstValidCols, paraRows, paraCols):
        self.name = name
        self.dstDType = dstDType
        self.srcDType = srcDType
        self.dstRows = dstRows
        self.dstCols = dstCols
        self.srcRows = srcRows
        self.srcCols = srcCols
        self.dstValidRows = dstValidRows
        self.dstValidCols = dstValidCols
        self.paraRows = paraRows
        self.paraCols = paraCols

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        # TDequantParams("TDEQUANTTest.case1", np.float32, np.int8, 32, 32, 32, 32, 32, 32, 32, 32),
        # TDequantParams("TDEQUANTTest.case2", np.float32, np.int16, 32, 32, 32, 32, 32, 32, 32, 32),
        TDequantParams("TDEQUANTTest.case3", np.float32, np.float32, 32, 32, 32, 32, 32, 32, 32, 32),
    ]

    for param in case_params_list:
        case_name = param.name
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tdequant(case_name, param)
        os.chdir(original_dir)