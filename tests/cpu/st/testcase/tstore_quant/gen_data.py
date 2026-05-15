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

np.random.seed(19)

from utils import NumExt

PRINT_C_CASE = True

bfloat16 = NumExt.bf16

class QuantMode:
    F32_TO_B8 = 0
    F32_TO_F16 = 1
    F32_TO_BF16 = 2
    I32_TO_F16 = 3
    I32_TO_I16 = 4
    I32_TO_B8 = 5
    BYPASS = 6


def get_quant_mode(src_dtype, dst_dtype):
    if src_dtype in [np.float32, np.int32] and dst_dtype in [np.int8, np.uint8]:
        if src_dtype == np.int32:
            return QuantMode.I32_TO_B8
        return QuantMode.F32_TO_B8
    elif dst_dtype == np.float16:
        return QuantMode.F32_TO_F16 if src_dtype == np.float32 else QuantMode.I32_TO_F16
    elif dst_dtype == bfloat16:
        return QuantMode.F32_TO_BF16
    elif dst_dtype == np.int16:
        return QuantMode.I32_TO_I16
    return QuantMode.BYPASS


def get_quant_vector(dst_dtype, n):
    result = []

    for _ in range(n):
        f_val = np.random.uniform(0.0, 5.0)
        f_bits = np.float32(f_val).view(np.uint32)
        offset_val = np.random.randint(0, 512)
        shift_bits = np.random.randint(0, 16)

        sign_bit = 1 if (dst_dtype == np.int8) else 0

        packed = (int(shift_bits & 0xF) << 32) | \
                    (int(sign_bit) << 46) | \
                    (int(offset_val & 0x1FF) << 37) | \
                    (int(f_bits))
        result.append(packed)

    return np.array(result, dtype=np.uint64)


def extract_quant_params(quant_gm):
    """
    Extract the parameters M1, offset, and sign from the quant_gm of type uint64.
    Args:
        quant_g: An integer of type uint64
    Return:
        m1: A floating-point number in custom format (1,8,10)
        offset: A 9-bit integer
        sign: A 1-bit boolean value (0 or 1)
    """
    quant_gm = int(quant_gm)
    m1_bits = (quant_gm >> 13) & 0x7FFFF
    offset = (quant_gm >> 37) & 0x1FF
    sign = (quant_gm >> 46) & 0x1

    # Parse M1 into a floating-point number in (1,8,10) format.
    sign_bit = (m1_bits >> 18) & 0x1
    exponent = (m1_bits >> 10) & 0xFF
    mantissa = m1_bits & 0x3FF
    exponent_bias = 127  # Assuming the exponent bias is 127, which aligns with float32.
    m1 = (-1) ** sign_bit * (1 + mantissa / 1024) * (2 ** (exponent - exponent_bias))

    return m1, offset, sign


def apply_quant_element(src_val, quant_gm, mode, dst_dtype, use_relu=False, saturate_inf=False):
    quant_gm = int(quant_gm)
    m1, offset, sign = extract_quant_params(quant_gm)
    res = src_val.astype(np.float32) * m1

    if mode in [QuantMode.F32_TO_B8, QuantMode.I32_TO_B8]:
        res = res + offset
        res = np.round(res)
        min_v = -128 if dst_dtype == np.int8 else 0
        max_v = 127 if dst_dtype == np.int8 else 255
        res = np.clip(res, min_v, max_v)
    elif mode in [QuantMode.F32_TO_F16]:
        f16_lim = np.finfo(np.float16)
        if np.isnan(res) and saturate_inf:
            res = 0
        elif np.isfinite(res) or saturate_inf:
            res = np.clip(res, f16_lim.min, f16_lim.max)
    elif mode == QuantMode.I32_TO_F16:
        f16_lim = np.finfo(np.float16)
        res = np.clip(res, f16_lim.min, f16_lim.max)
    elif mode == QuantMode.I32_TO_I16:
        src_val = int(src_val)
        shift_bits = ((quant_gm >> 32) & 0xF) + 1
        src_val = src_val >> shift_bits
        int16_lim = np.iinfo(np.int16)
        res = np.clip(src_val, int16_lim.min, int16_lim.max)

    if use_relu:
        res = np.maximum(res, 0)

    return NumExt.astype(np.array([res]), dst_dtype)[0]


def process_quant(data_array, quant_array, src_dtype, dst_dtype, is_vector, use_relu, saturate_inf):
    mode = get_quant_mode(src_dtype, dst_dtype)
    rows, cols = data_array.shape
    if NumExt.is_bf16(dst_dtype):
        out = np.zeros((rows, cols), dtype=np.float32)
    else:
        out = np.zeros_like(data_array, dtype=dst_dtype)

    for j in range(cols):
        q_param = quant_array[j] if is_vector else quant_array[0]
        for i in range(rows):
            out[i, j] = apply_quant_element(data_array[i, j], q_param, mode, dst_dtype, use_relu, saturate_inf)
    
    return out

def gen_golden_data(case_name, gInfo):
    src_type = gInfo.src_type
    dst_type = gInfo.dst_type
    gShape0 = gInfo.gShape0
    gShape1 = gInfo.gShape1
    gShape2 = gInfo.gShape2
    gShape3 = gInfo.gShape3
    gShape4 = gInfo.gShape4
    gWholeShape0 = gInfo.gWholeShape0
    gWholeShape1 = gInfo.gWholeShape1
    gWholeShape2 = gInfo.gWholeShape2
    gWholeShape3 = gInfo.gWholeShape3
    gWholeShape4 = gInfo.gWholeShape4

    # 1. Generate Raw Input Data
    if gInfo.format in ["ND", "NZ"]:
        input_shape = (gWholeShape0, gWholeShape1, gWholeShape2, gWholeShape3, gWholeShape4)
        active_slice = (slice(0, gShape0), slice(0, gShape1), slice(0, gShape2), slice(0, gShape3), slice(0, gShape4))
    elif gInfo.format == "DN":
        # Note the swap of 3 and 4 for DN layout
        input_shape = (gWholeShape0, gWholeShape1, gWholeShape2, gWholeShape4, gWholeShape3)
        active_slice = (slice(0, gShape0), slice(0, gShape1), slice(0, gShape2), slice(0, gShape4), slice(0, gShape3))

    input_arr = np.random.randint(-5, 5, size=input_shape).astype(src_type)
    
    # 2. Prepare for Quantization
    # We collapse everything except the last physical dimension into "Rows"
    # This matches how vector quantization usually applies per-channel (last dim)
    original_shape = input_arr.shape
    rows = np.prod(original_shape[:-1])
    cols = original_shape[-1]
    reshaped_input = input_arr.reshape((rows, cols))

    # 3. Generate Quantization Parameters
    # If is_v_quant (vector quant), we need 'cols' parameters. Else, just 1.
    quant_num = cols if gInfo.is_v_quant else 1
    quant_array = get_quant_vector(dst_type, quant_num)

    # 4. Apply Quantization
    # We pass the reshaped 2D data into your existing process_quant
    quantized_2d = process_quant(
        reshaped_input, 
        quant_array, 
        src_dtype=src_type, 
        dst_dtype=dst_type, # Or your target dst_dtype
        is_vector=gInfo.is_v_quant,
        use_relu=gInfo.use_relu,
        saturate_inf=gInfo.saturate_inf
    )

    # 5. Restore Shape and handle Padding
    output_arr = quantized_2d.reshape(original_shape)
    
    # Masking: Ensure only the "active" shape contains quantized data, 
    # and the padding (WholeShape - Shape) is zeroed out as per your requirement.
    final_output = np.zeros_like(output_arr)
    final_output[active_slice] = output_arr[active_slice]

    # 6. Save Files
    input_arr.tofile("./input.bin")
    final_output.tofile("./golden.bin")
    quant_array.tofile("./quant.bin")

class GlobalTensorInfo:
    def __init__(self, src_type, dst_type, format,
                is_v_quant: bool, 
                saturate_inf: bool,
                use_relu: bool,
                gShape0, gShape1, gShape2, gShape3, gShape4,
                gWholeShape0, gWholeShape1, gWholeShape2, gWholeShape3, gWholeShape4):
        self.src_type = src_type
        self.dst_type = dst_type
        self.format = format
        self.is_v_quant = is_v_quant
        self.saturate_inf = saturate_inf
        self.use_relu = use_relu
        self.gShape0 = gShape0
        self.gShape1 = gShape1
        self.gShape2 = gShape2
        self.gShape3 = gShape3
        self.gShape4 = gShape4
        self.gWholeShape0 = gWholeShape0
        self.gWholeShape1 = gWholeShape1
        self.gWholeShape2 = gWholeShape2
        self.gWholeShape3 = gWholeShape3
        self.gWholeShape4 = gWholeShape4

if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TStoreTest.ND_1",
        "TStoreTest.ND_2",
        "TStoreTest.ND_3",
        "TStoreTest.DN_4",
        "TStoreTest.DN_5",
        "TStoreTest.DN_6",
        "TStoreTest.NZ_7",
        "TStoreTest.NZ_8",
        "TStoreTest.NZ_9",
    ]

    case_params_list = [
        GlobalTensorInfo(np.float32, np.int8, "ND", True, True, True, 1, 1, 1, 2, 128, 1, 1, 1, 2, 128),
        GlobalTensorInfo(np.int32, np.int16, "ND", True, True, False, 1, 2, 1, 23, 121, 3, 2, 2, 35, 125),
        GlobalTensorInfo(np.int32, np.int8, "ND", True, False, True, 2, 2, 3, 23, 47, 3, 3, 4, 32, 50),
        GlobalTensorInfo(np.float32, np.float16, "DN",False, True, True, 1, 1, 1, 4, 21, 1, 1, 1, 8, 32),
        GlobalTensorInfo(np.float32, np.int16, "DN", True, False, False, 3, 1, 1, 1, 124, 5, 1, 1, 2, 128),
        GlobalTensorInfo(np.int32, np.int8, "DN", False, True, False, 2, 1, 2, 32, 32, 3, 4, 3, 64, 35),
        GlobalTensorInfo(np.float32, np.float16, "NZ", False, False, True, 1, 1, 1, 16, 8, 1, 1, 2, 16, 8),
        GlobalTensorInfo(np.int32, np.int16, "NZ", False, False, False, 2, 2, 2, 16, 16, 5, 3, 3, 16, 16),
        GlobalTensorInfo(np.int32, np.int8, "NZ", True, True, True, 1, 2, 1, 16, 32, 2, 4, 2, 16, 32),
    ]

    for i, case_name  in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)