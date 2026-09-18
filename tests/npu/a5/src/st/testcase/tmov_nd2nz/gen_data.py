#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------
import os
import numpy as np

np.random.seed(42)


def nd_to_nz(data, rows, cols, c0=32, n0=16):
    """Convert ND (row-major) layout to NZ fractal layout.

    NZ layout: [c1, n1, n0, c0] where c1 = cols/c0, n1 = rows/n0.
    Requires rows % n0 == 0 and cols % c0 == 0.
    """
    c1 = cols // c0
    n1 = rows // n0
    # Reshape to [n1, n0, c1, c0] then transpose to [c1, n1, n0, c0]
    nz = data.reshape(n1, n0, c1, c0).transpose(2, 0, 1, 3).reshape(-1)
    return nz


def nd_to_nz_f4(data, rows, cols_elems, c0_bytes=32, n0=16):
    """Convert packed f4 (2 nibbles/byte) ND data to NZ fractal layout.

    data is the byte-packed ND matrix [rows, cols_elems/2] (DENSE, unpadded).
    Each row is zero-padded up to the full NZ panel width (c0_bytes per panel
    row = 64 nibbles), matching the kernel: TLOAD zero-fills the per-row gap
    in MTE and the NZ output always contains full [n0, c0_bytes] panels.
    """
    byte_cols = data.shape[1]
    aligned_byte_cols = (byte_cols + c0_bytes - 1) // c0_bytes * c0_bytes
    pad = aligned_byte_cols - byte_cols
    if pad > 0:
        data = np.concatenate([data, np.zeros((rows, pad), dtype=np.uint8)], axis=1)
    c1 = aligned_byte_cols // c0_bytes
    n1 = rows // n0
    # Reshape to [n1, n0, c1, c0_bytes] then transpose to [c1, n1, n0, c0_bytes]
    nz = data.reshape(n1, n0, c1, c0_bytes).transpose(2, 0, 1, 3).reshape(-1)
    return nz


def gen_golden(case_name, rows, cols):
    input_arr = np.random.randint(0, 256, size=(rows, cols), dtype=np.uint8)
    input_arr.tofile("input_arr.bin")
    golden = nd_to_nz(input_arr, rows, cols)
    golden.tofile("golden.bin")


def gen_golden_f4(case_name, rows, cols_elems):
    # Input: DENSE byte-packed ND matrix [rows, cols_elems/2] — NOT pre-padded.
    # The kernel TLOADs it with MTE zero-padding to the NZ panel width, so the
    # golden is the NZ fractal over the zero-padded width.
    byte_cols = cols_elems // 2
    input_arr = np.random.randint(0, 256, size=(rows, byte_cols), dtype=np.uint8)
    input_arr.tofile("input_arr.bin")
    golden = nd_to_nz_f4(input_arr, rows, cols_elems)
    golden.tofile("golden.bin")


class CaseParams:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols


if __name__ == "__main__":
    case_name_list = [
        "TMovNd2NzTest.case_hif8_32x32",
        "TMovNd2NzTest.case_hif8_32x64",
        "TMovNd2NzTest.case_hif8_64x64",
        "TMovNd2NzTest.case_f4e1m2_16x32",
        "TMovNd2NzTest.case_f4e2m1_16x32",
        "TMovNd2NzTest.case_f4e1m2_16x8160",
        "TMovNd2NzTest.case_f4e2m1_16x8160",
        "TMovNd2NzTest.case_f4e1m2_4080x32",
        "TMovNd2NzTest.case_f4e2m1_4080x32",
        "TMovNd2NzTest.case_f4e1m2_32x32",
        "TMovNd2NzTest.case_f4e2m1_32x32",
        "TMovNd2NzTest.case_f4e1m2_32x64",
        "TMovNd2NzTest.case_f4e2m1_32x64",
        "TMovNd2NzTest.case_f4e1m2_64x64",
        "TMovNd2NzTest.case_f4e2m1_64x64",
    ]

    case_params_list = [
        CaseParams(32, 32), CaseParams(32, 64), CaseParams(64, 64),
        CaseParams(16, 32), CaseParams(16, 32),
        CaseParams(16, 8160), CaseParams(16, 8160),
        CaseParams(4080, 32), CaseParams(4080, 32),
        CaseParams(32, 32), CaseParams(32, 32),
        CaseParams(32, 64), CaseParams(32, 64),
        CaseParams(64, 64), CaseParams(64, 64),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)

        original_dir = os.getcwd()
        os.chdir(case_name)

        if case_name.startswith("TMovNd2NzTest.case_f4"):
            gen_golden_f4(case_name, case_params_list[i].rows, case_params_list[i].cols)
        else:
            gen_golden(case_name, case_params_list[i].rows, case_params_list[i].cols)

        os.chdir(original_dir)
