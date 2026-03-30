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
import struct
import ctypes
import numpy as np
np.random.seed(23)


def gen_golden_data(param):
    data_type = param.data_type
    validRow = param.validRow
    validCol = param.validCol

    input_arr = np.random.uniform(low=-8, high=8, size=(param.iRow, param.iCol)).astype(data_type)
    divider = np.random.uniform(low=-8, high=8, size=(1, 1)).astype(data_type)
    output_arr = np.zeros((param.oRow, param.oCol), dtype=data_type)
    for i in range(validRow):
        for j in range(validCol):
            output_arr[i, j] = input_arr[i, j] + divider[0, 0]
    input_arr.tofile('input.bin')
    with open("divider.bin", 'wb') as f:
        f.write(struct.pack('f', np.float32(divider[0, 0])))
    output_arr.tofile('golden.bin')


class TAddsParams:
    def __init__(self, name, data_type, validRow, validCol, iRow=None, iCol=None, oRow=None, oCol=None):
        self.name = name
        self.data_type = data_type
        self.validRow = validRow
        self.validCol = validCol
        self.iRow = validRow if iRow is None else iRow
        self.iCol = validCol if iCol is None else iCol
        self.oRow = validRow if oRow is None else oRow
        self.oCol = validCol if oCol is None else oCol


if __name__ == "__main__":
    case_params_list = [
        TAddsParams("TADDSTest.case1", np.float32, 32, 64),
        TAddsParams("TADDSTest.case2", np.float16, 63, 64),
        TAddsParams("TADDSTest.case3", np.int32, 31, 128),
        TAddsParams("TADDSTest.case4", np.int16, 15, 64 * 3),
        TAddsParams("TADDSTest.case5", np.float32, 7, 64 * 7),
        TAddsParams("TADDSTest.case6", np.float32, 256, 16),
        TAddsParams("TADDSTest.case7", np.float32, 16, 16, 32, 32, 30, 30)
    ]

    for _, case in enumerate(case_params_list):
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)
