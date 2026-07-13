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
import numpy as np


np.random.seed(19)


class TMatmulParams:
    def __init__(self, a_type, b_type, out_type, m, k, n, layout="dn", b_int4=False):
        self.a_type = a_type
        self.b_type = b_type
        self.out_type = out_type
        self.m = m
        self.k = k
        self.n = n
        self.layout = layout
        self.b_int4 = b_int4


def pack_int4_rows_with_stride(values_2d: np.ndarray, row_stride_bytes: int) -> np.ndarray:
    """Pack signed int4 matrix rows into bytes and place each packed row at row_stride_bytes stride.

    Packed byte format follows [low nibble, high nibble] for consecutive logical elements.
    """
    rows, cols = values_2d.shape
    packed_cols = (cols + 1) // 2
    out = np.zeros((rows, row_stride_bytes), dtype=np.uint8)

    for r in range(rows):
        row = values_2d[r].astype(np.int16)
        if row.size % 2 != 0:
            row = np.append(row, 0)
        lo = (row[0::2] & 0x0F).astype(np.uint8)
        hi = ((row[1::2] & 0x0F) << 4).astype(np.uint8)
        packed = (lo | hi).astype(np.uint8)
        out[r, :packed_cols] = packed[:packed_cols]

    return out.reshape(-1)


def gen_golden_data(param):
    x1_gm = np.random.uniform(-5, 5, [param.m, param.k]).astype(param.a_type)
    x2_gm = np.random.uniform(-5, 5, [param.k, param.n]).astype(param.b_type)

    # Use integer-friendly generation for int8 cases to keep deterministic results stable.
    if param.a_type == np.int8 and param.b_type == np.int8:
        x1_gm = np.random.randint(-8, 8, [param.m, param.k], dtype=np.int8)
        x2_gm = np.random.randint(-8, 8, [param.k, param.n], dtype=np.int8)

    if param.b_int4:
        x1_gm = np.random.randint(-8, 8, [param.m, param.k], dtype=np.int8)
        x2_gm = np.random.randint(-8, 8, [param.k, param.n], dtype=np.int8)

    golden = np.matmul(x1_gm.astype(param.out_type), x2_gm.astype(param.out_type)).astype(param.out_type)

    if param.layout == "dn":
        # DN layout: store transposed bytes to preserve the same logical matrix values.
        x1_gm.T.tofile("x1_gm.bin")
        if param.b_int4:
            # Keep one-byte dtype row-stride in GM while storing packed s4 payload at row head.
            x2_dn = x2_gm.T
            x2_store = pack_int4_rows_with_stride(x2_dn, row_stride_bytes=param.k)
            x2_store.tofile("x2_gm.bin")
        else:
            x2_gm.T.tofile("x2_gm.bin")
    else:
        # ND layout: write plain row-major tensors.
        x1_gm.tofile("x1_gm.bin")
        if param.b_int4:
            x2_store = pack_int4_rows_with_stride(x2_gm, row_stride_bytes=param.n)
            x2_store.tofile("x2_gm.bin")
        else:
            x2_gm.tofile("x2_gm.bin")
    golden.tofile("golden.bin")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    case_name_list = [
        "TMATMULTest.case_fp16_fp16_to_fp32_31x96x47",
        "TMATMULTest.case_int8_int8_to_int32_65x90x89",
        "TMATMULTest.case_fp32_fp32_to_fp32_16x32x64",
        "TMATMULTest.case_fp16_fp16_to_fp32_1x256x64",
        "TMATMULTest.case_nd_fp16_fp16_to_fp32_64x64x64",
        "TMATMULTest.case_nd_int8_int8_to_int32_96x128x65",
        "TMATMULTest.case_nd_fp32_fp32_to_fp32_33x63x31",
        "TMATMULTest.case_nd_fp16_fp16_to_fp32_2x80x48",
        "TMATMULTest.case_fp16_fp16_to_fp32_127x33x95",
        "TMATMULTest.case_int8_int8_to_int32_17x33x31",
        "TMATMULTest.case_fp32_fp32_to_fp32_63x31x15",
        "TMATMULTest.case_nd_fp16_fp16_to_fp32_95x33x79",
        "TMATMULTest.case_nd_int8_int8_to_int32_129x95x33",
        "TMATMULTest.case_nd_fp32_fp32_to_fp32_47x29x25",
        "TMATMULTest.case_nd_int8_int4_to_int32_64x64x64",
        "TMATMULTest.case_nd_int8_int4_to_int32_96x128x65",
        "TMATMULTest.case_nd_int8_int4_to_int32_129x95x33",
        "TMATMULTest.case_nd_int8_int4_to_int32_17x33x31",
        "TMATMULTest.case_nd_int8_int4_to_int32_2x80x48",
        "TMATMULTest.case_nd_fp16_int8_to_fp32_64x64x64",
    ]

    case_params_list = [
        TMatmulParams(np.float16, np.float16, np.float32, 31, 96, 47, "dn"),
        TMatmulParams(np.int8, np.int8, np.int32, 65, 90, 89, "dn"),
        TMatmulParams(np.float32, np.float32, np.float32, 16, 32, 64, "dn"),
        TMatmulParams(np.float16, np.float16, np.float32, 1, 256, 64, "dn"),
        TMatmulParams(np.float16, np.float16, np.float32, 64, 64, 64, "nd"),
        TMatmulParams(np.int8, np.int8, np.int32, 96, 128, 65, "nd"),
        TMatmulParams(np.float32, np.float32, np.float32, 33, 63, 31, "nd"),
        TMatmulParams(np.float16, np.float16, np.float32, 2, 80, 48, "nd"),
        TMatmulParams(np.float16, np.float16, np.float32, 127, 33, 95, "dn"),
        TMatmulParams(np.int8, np.int8, np.int32, 17, 33, 31, "dn"),
        TMatmulParams(np.float32, np.float32, np.float32, 63, 31, 15, "dn"),
        TMatmulParams(np.float16, np.float16, np.float32, 95, 33, 79, "nd"),
        TMatmulParams(np.int8, np.int8, np.int32, 129, 95, 33, "nd"),
        TMatmulParams(np.float32, np.float32, np.float32, 47, 29, 25, "nd"),
        TMatmulParams(np.int8, np.int8, np.int32, 64, 64, 64, "nd", b_int4=True),
        TMatmulParams(np.int8, np.int8, np.int32, 96, 128, 65, "nd", b_int4=True),
        TMatmulParams(np.int8, np.int8, np.int32, 129, 95, 33, "nd", b_int4=True),
        TMatmulParams(np.int8, np.int8, np.int32, 17, 33, 31, "nd", b_int4=True),
        TMatmulParams(np.int8, np.int8, np.int32, 2, 80, 48, "nd", b_int4=True),
        TMatmulParams(np.float16, np.int8, np.float32, 64, 64, 64, "nd"),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_params_list[i])
        os.chdir(original_dir)
