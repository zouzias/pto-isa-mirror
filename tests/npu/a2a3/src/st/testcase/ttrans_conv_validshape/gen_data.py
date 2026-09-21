#!/user/bin/python3
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

"""Golden data for TTRANSConvValidShapeTest (NPU a2a3).

NPU semantics modeled here (see TTransImplConvTile in include/pto/npu/a2a3/TTrans.hpp):
  * TTRANS derives every axis of the transform from GetShape() (the physical
    shape), so the ValidShape configured on any axis does not change the result:
    the dst holds the full physical transform.
  * The src channel dim is read as the padded C1 * C0, so input.bin carries
    C1 * C0 channels with zeros in the padding.
"""

import os
from enum import Enum

import numpy as np

np.random.seed(19)

SUITE_NAME = "TTRANSConvValidShapeTest"
BLOCK_BYTE_SIZE = 32


class DataFormat(Enum):
    NCHW2NC1HWC0 = 0
    NC1HWC02NCHW = 1
    GNCHW2GNC1HWC0 = 2


def block_elem(data_type) -> int:
    return BLOCK_BYTE_SIZE // np.dtype(data_type).itemsize


def golden_NCHW2NC1HWC0(case):
    n, c, h, w = case.phys
    vn = case.valid[0]
    c0 = block_elem(case.data_type)
    c1 = (c + c0 - 1) // c0

    data = np.random.randint(1, 5, size=(n, c, h, w)).astype(case.data_type)
    input_arr = np.pad(data, ((0, 0), (0, c1 * c0 - c), (0, 0), (0, 0)), mode="constant")
    input_arr.tofile("./input.bin")

    golden = input_arr.reshape(n, c1, c0, h, w).transpose(0, 1, 3, 4, 2)
    golden = np.ascontiguousarray(golden)
    golden.tofile("./golden.bin")
    print(f"  -> Input ({n},{c1 * c0},{h},{w}) | Golden {golden.shape} | C0={c0} C1={c1} VN={vn}")


def golden_NC1HWC02NCHW(case):
    n, c1, h, w, c0 = case.phys
    vn = case.valid[0]
    assert c0 == block_elem(case.data_type), f"C0 must be {block_elem(case.data_type)} for this dtype"

    input_arr = np.random.randint(1, 5, size=(n, c1, h, w, c0)).astype(case.data_type)
    input_arr.tofile("./input.bin")

    golden = input_arr.transpose(0, 1, 4, 2, 3).reshape(n, c1 * c0, h, w)
    golden = np.ascontiguousarray(golden)
    golden.tofile("./golden.bin")
    print(f"  -> Input ({n},{c1},{h},{w},{c0}) | Golden {golden.shape} | VN={vn}")


def golden_GNCHW2GNC1HWC0(case):
    g, n, c, h, w = case.phys
    c0 = block_elem(case.data_type)
    c1 = (c + c0 - 1) // c0

    data = np.random.randint(1, 5, size=(g, n, c, h, w)).astype(case.data_type)
    input_arr = np.pad(data, ((0, 0), (0, 0), (0, c1 * c0 - c), (0, 0), (0, 0)), mode="constant")
    input_arr.tofile("./input.bin")

    golden = input_arr.reshape(g, n, c1, c0, h, w).transpose(0, 1, 2, 4, 5, 3)
    golden = np.ascontiguousarray(golden)
    golden.tofile("./golden.bin")
    print(f"  -> Input ({g},{n},{c1 * c0},{h},{w}) | Golden {golden.shape} | C0={c0} C1={c1}")


def gen_golden_data(case):
    if case.fmt == DataFormat.NCHW2NC1HWC0.value:
        golden_NCHW2NC1HWC0(case)
    elif case.fmt == DataFormat.NC1HWC02NCHW.value:
        golden_NC1HWC02NCHW(case)
    elif case.fmt == DataFormat.GNCHW2GNC1HWC0.value:
        golden_GNCHW2GNC1HWC0(case)
    else:
        raise ValueError(f"unsupported format {case.fmt}")


class TTRANSValidParams:
    def __init__(self, case_name, data_type, fmt, phys, valid):
        self.case_name = case_name
        self.data_type = data_type
        self.fmt = fmt
        self.phys = phys
        self.valid = valid


FWD = DataFormat.NCHW2NC1HWC0.value
REV = DataFormat.NC1HWC02NCHW.value
GRP = DataFormat.GNCHW2GNC1HWC0.value

test_cases_registry = [
    # ---------------- NCHW -> NC1HWC0: phys (N, C, H, W), valid (VN, VC, VH, VW) ----------------
    TTRANSValidParams("NCHW2NC1HWC0_valid_n_half", np.float16, FWD, (4, 20, 4, 16), (2, 20, 4, 16)),
    TTRANSValidParams("NCHW2NC1HWC0_valid_c_half", np.float16, FWD, (4, 20, 4, 16), (4, 7, 4, 16)),
    TTRANSValidParams("NCHW2NC1HWC0_valid_h_half", np.float16, FWD, (4, 16, 8, 16), (4, 16, 5, 16)),
    TTRANSValidParams("NCHW2NC1HWC0_valid_w_half", np.float16, FWD, (4, 16, 4, 16), (4, 16, 4, 4)),
    TTRANSValidParams("NCHW2NC1HWC0_valid_all_half", np.float16, FWD, (4, 20, 8, 16), (2, 7, 5, 4)),
    TTRANSValidParams("NCHW2NC1HWC0_valid_all_float", np.float32, FWD, (4, 20, 4, 16), (2, 7, 3, 4)),
    TTRANSValidParams("NCHW2NC1HWC0_valid_eq_phys_half", np.float16, FWD, (4, 20, 4, 16), (4, 20, 4, 16)),
    TTRANSValidParams("NCHW2NC1HWC0_valid_min_half", np.float16, FWD, (4, 20, 4, 16), (1, 1, 1, 1)),

    # ------------- NC1HWC0 -> NCHW: phys (N, C1, H, W, C0), valid (VN, VC1, VH, VW, VC0) -------------
    TTRANSValidParams("NC1HWC02NCHW_valid_n_half", np.float16, REV, (4, 2, 4, 16, 16), (3, 2, 4, 16, 16)),
    TTRANSValidParams("NC1HWC02NCHW_valid_c1_half", np.float16, REV, (4, 2, 4, 16, 16), (4, 1, 4, 16, 16)),
    TTRANSValidParams("NC1HWC02NCHW_valid_c0_half", np.float16, REV, (4, 2, 4, 16, 16), (4, 2, 4, 16, 7)),
    TTRANSValidParams("NC1HWC02NCHW_valid_h_half", np.float16, REV, (4, 2, 8, 16, 16), (4, 2, 5, 16, 16)),
    TTRANSValidParams("NC1HWC02NCHW_valid_w_half", np.float16, REV, (4, 2, 4, 16, 16), (4, 2, 4, 4, 16)),
    TTRANSValidParams("NC1HWC02NCHW_valid_all_half", np.float16, REV, (4, 2, 8, 16, 16), (2, 1, 5, 4, 7)),
    TTRANSValidParams("NC1HWC02NCHW_valid_all_int32", np.int32, REV, (4, 2, 4, 8, 8), (2, 1, 3, 4, 5)),
    TTRANSValidParams("NC1HWC02NCHW_valid_eq_phys_half", np.float16, REV, (4, 2, 4, 16, 16), (4, 2, 4, 16, 16)),

    # ------------- GNCHW -> GNC1HWC0: phys (G, N, C, H, W), valid (VG, VN, VC, VH, VW) -------------
    TTRANSValidParams("GNCHW2GNC1HWC0_valid_g_half", np.float16, GRP, (2, 2, 20, 4, 16), (1, 2, 20, 4, 16)),
    TTRANSValidParams("GNCHW2GNC1HWC0_valid_all_half", np.float16, GRP, (2, 2, 20, 4, 16), (1, 1, 7, 3, 4)),
]

if __name__ == "__main__":
    print(f"Generating golden data for {len(test_cases_registry)} cases of {SUITE_NAME}...\n")

    for case in test_cases_registry:
        dirname = f"{SUITE_NAME}.{case.case_name}"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        original_dir = os.getcwd()
        os.chdir(dirname)
        print(f"Case: {case.case_name} | dtype: {case.data_type.__name__} | valid: {case.valid}")
        gen_golden_data(case)
        os.chdir(original_dir)

    print("\nAll binary test files (input.bin, golden.bin) have been generated successfully.")
