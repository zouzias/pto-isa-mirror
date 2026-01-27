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
import numpy as np
import ml_dtypes

bfloat16 = ml_dtypes.bfloat16
np.random.seed(19)

def float32_to_hf32(x, round_mode="roundTiesToEven"):
    """
    将float32转换为HF32格式(E8M11)
    
    HF32格式: 1位符号位 + 8位指数位 + 11位尾数位（共20位有效位）
    """
    # 将float32转换为二进制表示
    packed = struct.pack('f', x)
    bits = struct.unpack('I', packed)[0]
    
    # 提取符号、指数和尾数
    sign = (bits >> 31) & 0x1
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF  # 23位尾数
    
    # 处理特殊值（NaN/Inf）
    if exponent == 0xFF:
        if mantissa != 0:
            return float('nan')
        return float('inf') * (-1 if sign else 1)
    
    # 处理次正规数和零
    if exponent == 0:
        if mantissa == 0:  # 零值
            return 0.0 if sign == 0 else -0.0
        # 对于次正规数，我们将其视为0（HF32不支持次正规数）
        return 0.0 if sign == 0 else -0.0
    
    # 将23位尾数转换为11位尾数
    mantissa_23bit = mantissa
    mantissa_11bit = mantissa_23bit >> 12  # 丢弃低12位
    
    # 应用舍入模式
    lost_bits = mantissa_23bit & 0xFFF  # 被丢弃的低12位
    
    if round_mode == "CAST_RINT":
        # roundTiesToEven: 四舍六入五成双
        if lost_bits > 0x800:  # 大于一半 (0x800 = 2048)
            mantissa_11bit += 1
        elif lost_bits == 0x800:  # 正好一半
            # 检查最低有效位
            if mantissa_11bit & 0x1:
                mantissa_11bit += 1  # 奇数则入
            # 偶数则保持
    elif round_mode == "CAST_ROUND":
        # roundTiesAway: 四舍五入，0.5时远离零
        if lost_bits >= 0x800:  # 大于等于一半
            mantissa_11bit += 1
    else:
        raise ValueError(f"Unsupported round mode: {round_mode}")
    
    # 检查尾数是否溢出（进位到指数）
    # 11位尾数的最大值是0x7FF (2047)
    if mantissa_11bit >= 0x800:  # 0x800 = 2048，需要进位
        mantissa_11bit = mantissa_11bit >> 1
        exponent += 1
    
    # 检查指数溢出
    if exponent >= 0xFF:
        return float('inf') if sign == 0 else -float('inf')
    
    # 重建HF32浮点数（实际上存储为float32，但精度为HF32）
    # 注意：HF32尾数只有11位有效，所以低12位尾数为0
    hf32_mantissa = mantissa_11bit << 12
    hf32_bits = (sign << 31) | (exponent << 23) | hf32_mantissa
    
    # 转换回float32
    return struct.unpack('f', struct.pack('I', hf32_bits))[0]

def gen_golden_data(case_name, param):
    src_type = param.atype
    dst_type = param.ctype
    bias_type = param.bias_type

    m, k, n, is_bias, is_atrans, is_btrans = param.m, param.k, param.n, param.is_bias, False, False

    x1_gm = np.random.uniform(1, 5, [m, k]).astype(src_type)
    x2_gm = np.random.uniform(1, 5, [k, n]).astype(src_type)
    bias_gm = np.random.uniform(1, 10, [n, ]).astype(bias_type)

    if is_atrans:
        x1_gm = x1_gm.transpose()
    if is_btrans:
        x2_gm = x2_gm.transpose()

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")

    if param.is_hf32:
        # 确定舍入模式
        round_mode =param.hf32_trans
        
        hf32_func = np.vectorize(lambda x: float32_to_hf32(x, round_mode))
        
        # 应用HF32转换
        x1_gm = hf32_func(x1_gm.astype(np.float32))
        x2_gm = hf32_func(x2_gm.astype(np.float32))

    if is_bias:
        golden = np.matmul(x1_gm.astype(dst_type), x2_gm.astype(dst_type)).astype(dst_type) + bias_gm.astype(dst_type)
    else:
        golden = np.matmul(x1_gm.astype(dst_type), x2_gm.astype(dst_type)).astype(dst_type)

    bias_gm.tofile("./bias_gm.bin")
    golden.tofile("./golden.bin")


class tmatmulParams:
    def __init__(self, atype, btype, ctype, m, k, n, is_bias, bias_type=None, is_hf32 = False, hf32_trans = "CAST_RINT"):
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.m = m
        self.k = k
        self.n = n 
        self.is_bias = is_bias
        if (bias_type):
            self.bias_type = bias_type
        else:
            self.bias_type = ctype
        self.is_hf32 = is_hf32
        self.hf32_trans = hf32_trans


if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TMATMULTest.case1", # 此名称要和TEST_F(TMATMULTest, case1)定义的名称一致
        "TMATMULTest.case2",
        "TMATMULTest.case3",
        "TMATMULTest.case4",
        "TMATMULTest.case5",
        "TMATMULTest.case6",
        "TMATMULTest.case7",
        "TMATMULBIASTest.case1",
        "TMATMULBIASTest.case2",
        "TMATMULBIASTest.case3",
        "TMATMULBIASTest.case4",
        "TMATMULBIASTest.case5",
        "TMATMULBIASTest.case6",
        "TMATMULBIASTest.case7",
    ]

    case_params_list = [
        tmatmulParams(np.float16, np.float16, np.float32, 31, 120, 58, False),
        tmatmulParams(np.int8, np.int8, np.int32, 65, 90, 89, False),
        tmatmulParams(np.float16, np.float16, np.float32, 5, 75, 11, False),
        tmatmulParams(np.float16, np.float16, np.float32, 1, 256, 64, False),
        tmatmulParams(np.float16, np.float16, np.float32, 1, 16, 32, False),
        tmatmulParams(np.float16, np.float16, np.float32, 1, 200, 32, False),
        tmatmulParams(np.float32, np.float32, np.float32, 16, 32, 64, False, np.float32, True, "CAST_RINT"),
        # bias test
        tmatmulParams(np.float16, np.float16, np.float32, 26, 100, 94, True),
        tmatmulParams(np.float16, np.float16, np.float32, 101, 288, 67, True),
        tmatmulParams(np.float32, np.float32, np.float32, 15, 16, 15, True),
        tmatmulParams(np.int8, np.int8, np.int32, 55, 127, 29, True),
        tmatmulParams(bfloat16, bfloat16, np.float32, 11, 402, 30, True),
        tmatmulParams(np.int8, np.int8, np.int32, 150, 89, 50, True),
        # bias + acc test
        tmatmulParams(np.int8, np.int8, np.int32, 135, 64, 88, True),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)