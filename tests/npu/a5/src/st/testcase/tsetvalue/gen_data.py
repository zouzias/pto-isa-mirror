#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os
import struct
import numpy as np
import ml_dtypes
from typing import Tuple, Optional

bfloat16 = ml_dtypes.bfloat16
np.random.seed(19)


def gen_golden_data(case_name, param):
    src_type = param.type
    if param.is_conv_tile:
        # 对于conv tile模式，使用5维shape
        c1, h, w, c0 = param.shape_nc1hwc0[1], param.shape_nc1hwc0[2], param.shape_nc1hwc0[3], param.shape_nc1hwc0[4]
        n = param.shape_nc1hwc0[0]  # N维度
        input = np.random.randint(-10, 10, [n, c1, h, w, c0]).astype(src_type)
        golden = np.full([n, c1, h, w, c0], param.value, dtype=src_type)
    else:
        # 对于2维矩阵模式，使用m, n
        input = np.random.randint(-10, 10, [param.m, param.n]).astype(src_type)
        golden = np.full([param.m, param.n], param.value, dtype=src_type)
    input.tofile("./input.bin")
    golden.tofile("./golden.bin")


class TsetvalueParams:
    def __init__(
        self,
        type,
        value,
        m : Optional[int] = None,
        n : Optional[int] = None,
        shape_nc1hwc0: Optional[Tuple[int, int, int, int, int]] = None,
        is_conv_tile = False,
    ):
        self.type = type
        self.value = value
        self.m = m
        self.n = n 
        self.shape_nc1hwc0 = shape_nc1hwc0
        self.is_conv_tile = is_conv_tile


if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TSETVALUETest.case1",
        "TSETVALUETest.case2",
        "TSETVALUETest.case3",
        "TSETVALUETest.case4",
        "TSETVALUETest.case5",
        "TSETVALUETest.case6",
        "TSETVALUETest.case7",
    ]

    case_params_list = [
        # 2维矩阵测试用例
        TsetvalueParams(np.float16, value=2, m=128, n=128, is_conv_tile=False),
        TsetvalueParams(np.int16, value=5, m=32, n=64, is_conv_tile=False),
        TsetvalueParams(np.float32, value=3, m=32, n=32, is_conv_tile=False),
        TsetvalueParams(np.int8, value=1, m=32, n=32, is_conv_tile=False),
        TsetvalueParams(bfloat16, value=7, m=256, n=256, is_conv_tile=False),
        
        # conv tile测试用例 (N, C1, H, W, C0)
        TsetvalueParams(np.float16, value=3, shape_nc1hwc0=(1, 16, 7, 7, 16), is_conv_tile=True),
        TsetvalueParams(np.float32, value=4, shape_nc1hwc0=(2, 32, 14, 14, 8), is_conv_tile=True),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)