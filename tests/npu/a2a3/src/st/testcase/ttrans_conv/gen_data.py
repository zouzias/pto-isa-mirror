#!/user/bin/python3
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

def gen_golden_data(g_info):
    data_type = g_info.data_type
    g_shape0 = g_info.g_shape0
    g_shape1 = g_info.g_shape1
    g_shape2 = g_info.g_shape2
    g_shape3 = g_info.g_shape3
    g_shape4 = g_info.g_shape4
    g_whole_shape0 = g_info.g_whole_shape0
    g_whole_shape1 = g_info.g_whole_shape1
    g_whole_shape2 = g_info.g_whole_shape2
    g_whole_shape3 = g_info.g_whole_shape3
    g_whole_shape4 = g_info.g_whole_shape4

    if g_info.format == "NCHW2NC1HWC0":
        # NCHW -> NC1HWC0
        rows = g_whole_shape1 * g_whole_shape2 * g_whole_shape3
        cols = g_whole_shape4
        input_arr = np.zeros(shape=(rows, cols), dtype=data_type)
        for i in range (rows):
            input_arr[i:] = i
        output_arr = np.zeros(shape=(g_shape0, g_shape1, g_shape2, g_shape3, g_shape4), dtype=data_type)
        output_arr = input_arr.reshape([g_shape0, g_shape1, g_shape4, g_shape2, g_shape3]).transpose([0, 1, 3, 4, 2])
    elif g_info.format == "NC1HWC02C1HWNC0":
        num = g_whole_shape0
        rows = g_whole_shape1 * g_whole_shape2 * g_whole_shape3
        cols = g_whole_shape4
        input_arr = np.arange(rows,)[np.newaxis, :, np.newaxis].astype(data_type)
        input_arr = np.tile(input_arr, (num, 1, cols))
        output_arr = input_arr.reshape([num, rows, cols]).transpose([1, 0, 2])

    input_arr.tofile("./input.bin")
    output_arr.tofile("./golden.bin")


class TTRANSParams:
    def __init__(self, case_name, data_type, format, g_shape0, g_shape1, g_shape2, g_shape3, g_shape4,
                 g_whole_shape0, g_whole_shape1, g_whole_shape2, g_whole_shape3, g_whole_shape4):
        self.case_name = case_name 
        self.data_type = data_type
        self.format = format
        self.g_shape0 = g_shape0
        self.g_shape1 = g_shape1
        self.g_shape2 = g_shape2
        self.g_shape3 = g_shape3
        self.g_shape4 = g_shape4
        self.g_whole_shape0 = g_whole_shape0
        self.g_whole_shape1 = g_whole_shape1
        self.g_whole_shape2 = g_whole_shape2
        self.g_whole_shape3 = g_whole_shape3
        self.g_whole_shape4 = g_whole_shape4


if __name__ == "__main__":

    case_params_list = [
        # N, C1, H, W, C0 <- 1, N, C, H, W
        TTRANSParams("TTRANSConvTest.float32_1_32_6_56", np.float32, "NCHW2NC1HWC0",
                     1, 4, 6, 56, 8, 1, 1, 32, 6, 56),
        TTRANSParams("TTRANSConvTest.float32_1_32_2_16", np.float32, "NCHW2NC1HWC0",
                     1, 2, 2, 16, 16, 1, 1, 32, 2, 16),
        # C1, H, W, N, C0 <- N, C1, H, W, C0
        TTRANSParams("TTRANSConvTest.float32_2_2_2_16_16", np.float32, "NC1HWC02C1HWNC0",
                     2, 2, 16, 2, 16, 2, 2, 2, 16, 16),
    ]

    for case_params in case_params_list:
        case_name = case_params.case_name
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden_data(case_params)
        os.chdir(original_dir)