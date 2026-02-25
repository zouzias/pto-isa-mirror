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
from enum import Enum
np.random.seed(19)

class DataFormat(Enum):
    NCHW2NC1HWC0 = 1
    NC1HWC02C1HWN1N0C0 = 2


def nchw_to_nc1hwc0(nchw_tensor: np.ndarray, c0: int) -> np.ndarray:
    if nchw_tensor.ndim != 4:
        raise ValueError(f"The input must be a 4-dimensional NCHW tensor, current dim : {nchw_tensor.ndim}")
    if (c0 & (c0 - 1)) != 0 and c0 != 1:
        raise ValueError(f"C0 should be 8/16/32, now is : {c0}")

    n, c, h, w = nchw_tensor.shape
    c1 = (c + c0 - 1) // c0
    pad_c = c1 * c0 - c
    if pad_c > 0:
        pad_width = ((0, 0), (0, pad_c), (0, 0), (0, 0))
        nchw_padded = np.pad(nchw_tensor, pad_width, mode="constant", constant_values=0)
    else:
        nchw_padded = nchw_tensor

    nc1c0hw_tensor = nchw_padded.reshape(n, c1, c0, h, w)

    # NC1C0HW → NC1HWC0
    # origin index：0(n),1(c1),2(C0),3(h),4(w) → new index ：0,1,3,4,2
    nc1hwc0_tensor = np.transpose(nc1c0hw_tensor, axes=(0, 1, 3, 4, 2))

    return nc1hwc0_tensor


def gen_golden_data(g_info):
    data_type = g_info.data_type
    g_shape0 = g_info.g_shape0
    g_shape1 = g_info.g_shape1
    g_shape2 = g_info.g_shape2
    g_shape3 = g_info.g_shape3
    g_shape4 = g_info.g_shape4
    g_shape5 = g_info.g_shape5
    g_whole_shape0 = g_info.g_whole_shape0
    g_whole_shape1 = g_info.g_whole_shape1
    g_whole_shape2 = g_info.g_whole_shape2
    g_whole_shape3 = g_info.g_whole_shape3
    g_whole_shape4 = g_info.g_whole_shape4

    input_arr = np.random.randint(1, 5, [g_shape3, g_shape4]).astype(data_type)
    output_arr = np.zeros([g_shape3, g_shape4]).astype(data_type)
    if g_info.format == DataFormat["NCHW2NC1HWC0"].value:
        # NCHW -> NC1HWC0
        input_arr = np.random.randint(1, 5, size=(g_whole_shape1,
                                  g_whole_shape2, g_whole_shape3, g_whole_shape4)).astype(data_type)
        golden_nchw = np.zeros(shape=(g_shape0, g_shape1 * g_shape4, g_shape2, g_shape3), dtype=data_type)
        golden_nchw = input_arr[0:g_shape0, 0:g_shape1 * g_shape4, 0:g_shape2, 0:g_shape3]
        pad_c = g_shape1 * g_shape4 - g_whole_shape2
        if pad_c > 0:
            pad_width = ((0, 0), (0, pad_c), (0, 0), (0, 0))
            input_arr = np.pad(input_arr, pad_width, mode="constant", constant_values=0)
        output_arr = nchw_to_nc1hwc0(golden_nchw, g_shape4)
    elif g_info.format == DataFormat["NC1HWC02C1HWN1N0C0"].value:
        src_n = g_whole_shape0
        src_c1 = g_whole_shape1
        src_h = g_whole_shape2
        src_w = g_whole_shape3
        src_c0 = g_whole_shape4
        dst_n1 = g_shape3
        dst_n0 = g_shape4
        input_ori = np.random.randint(1, 5, size=(src_n, src_c1, src_h, src_w, src_c0)).astype(data_type)
        pad_n = dst_n1 * dst_n0 - g_whole_shape0
        if pad_n > 0:
            pad_width = ((0, pad_n), (0, 0), (0, 0), (0, 0), (0, 0))
            input_arr = np.pad(input_ori, pad_width, mode="constant", constant_values=0)
        else:
            input_arr = input_ori
        # [N, C1, H, W, C0] -> [C1, H, W, N1, N0, C0]
        output_arr = input_arr.transpose([1, 2, 3, 0, 4])
    input_arr.tofile("./input.bin")
    output_arr.tofile("./golden.bin")


class TTRANSParams:
    def __init__(self, case_name, data_type, format, g_shape0, g_shape1, g_shape2, g_shape3, g_shape4, g_shape5,
                 g_whole_shape0, g_whole_shape1, g_whole_shape2, g_whole_shape3, g_whole_shape4):
        self.case_name = case_name 
        self.data_type = data_type
        self.format = format
        self.g_shape0 = g_shape0
        self.g_shape1 = g_shape1
        self.g_shape2 = g_shape2
        self.g_shape3 = g_shape3
        self.g_shape4 = g_shape4
        self.g_shape5 = g_shape5
        self.g_whole_shape0 = g_whole_shape0
        self.g_whole_shape1 = g_whole_shape1
        self.g_whole_shape2 = g_whole_shape2
        self.g_whole_shape3 = g_whole_shape3
        self.g_whole_shape4 = g_whole_shape4


if __name__ == "__main__":

    case_params_list = [
        # N, C1, H, W, C0, 1 <- 1, N, C, H, W
        TTRANSParams("TTRANSConvTest.float32_1_32_6_56", np.float32, DataFormat["NCHW2NC1HWC0"].value,
                     1, 4, 6, 56, 8, 1, 1, 1, 32, 6, 56),
        TTRANSParams("TTRANSConvTest.float32_5_57_4_16", np.float32, DataFormat["NCHW2NC1HWC0"].value,
                     5, 4, 4, 16, 16, 1, 1, 5, 57, 4, 16),
        TTRANSParams("TTRANSConvTest.half_1_30_2_16", np.float16, DataFormat["NCHW2NC1HWC0"].value,
                     1, 2, 2, 16, 16, 1, 1, 1, 30, 2, 16), 
        TTRANSParams("TTRANSConvTest.int8_1_63_2_56", np.int8, DataFormat["NCHW2NC1HWC0"].value,
                     3, 2, 2, 56, 32, 1, 1, 3, 64, 2, 56),
        TTRANSParams("TTRANSConvTest.int8_3_63_2_56", np.int8, DataFormat["NCHW2NC1HWC0"].value,
                     1, 2, 2, 56, 32, 1, 1, 1, 63, 2, 56),
        TTRANSParams("TTRANSConvTest.int8_5_58_2_16", np.int8, DataFormat["NCHW2NC1HWC0"].value,
                     5, 2, 2, 16, 32, 1, 1, 5, 58, 2, 16),
        TTRANSParams("TTRANSConvTest.int8_2_63_2_16", np.int8, DataFormat["NCHW2NC1HWC0"].value,
                     2, 2, 2, 16, 32, 1, 1, 2, 63, 2, 16),
        # C1, H, W, N1, N0, C0 <- N, C1, H, W, C0
        TTRANSParams("TTRANSConvTest.float32_3_2_2_16_16", np.float32, DataFormat["NC1HWC02C1HWN1N0C0"].value,
                     2, 2, 16, 2, 2, 16, 3, 2, 2, 16, 16),
    ]

    for case_params in case_params_list:
        case_name = case_params.case_name
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden_data(case_params)
        os.chdir(original_dir)