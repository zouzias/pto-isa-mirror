#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os
import numpy as np


# MMAD.f16f32 golden generator: A[half] x B[half] -> C[float].
# Inputs are fp16; the reference matmul upcasts to float32 and accumulates in
# float32, matching the cube L0C accumulator dtype.
np.random.seed(19)


def gen_golden_data(m, k, n, layout):
    """Generate A[m,k], B[k,n] in fp16 and the fp32 golden = A @ B.

    layout="nd": write row-major tensors (stride last-dim = 1).
    layout="dn": write transposed tensors (DN layout: stride last-dim = the
        other extent). The device kernel's GlobalTensor DN stride reads these
        back as the logical [m,k]/[k,n] matrix.
    """
    x1_gm = np.random.uniform(-5, 5, [m, k]).astype(np.float16)
    x2_gm = np.random.uniform(-5, 5, [k, n]).astype(np.float16)

    # Reference: fp16 operands upcast to fp32, accumulate in fp32 (L0C dtype).
    golden = np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32)).astype(np.float32)

    if layout == "dn":
        x1_gm.T.tofile("x1_gm.bin")
        x2_gm.T.tofile("x2_gm.bin")
    else:  # nd
        x1_gm.tofile("x1_gm.bin")
        x2_gm.tofile("x2_gm.bin")
    golden.tofile("golden.bin")


# (case_name, m, k, n, layout). Keys match tmatmul_f16_kernel.cpp /
# main.cpp. ND first (keys 1-10), then DN (keys 11-15).
CASES = [
    # --- ND (row-major) ---
    ("TMATMUL_F16_TEST.case_mmad_f16f32_nd_16x16x16", 16, 16, 16, "nd"),
    ("TMATMUL_F16_TEST.case_mmad_f16f32_nd_64x64x64", 64, 64, 64, "nd"),
    ("TMATMUL_F16_TEST.case_mmad_f16f32_nd_128x128x128", 128, 128, 128, "nd"),
    ("TMATMUL_F16_TEST.case_mmad_f16f32_nd_17x33x31", 17, 33, 31, "nd"),
    ("TMATMUL_F16_TEST.case_mmad_f16f32_nd_95x33x79", 95, 33, 79, "nd"),
    ("TMATMUL_F16_TEST.case_mmad_f16f32_nd_127x96x95", 127, 96, 95, "nd"),
    ("TMATMUL_F16_TEST.case_mmad_f16f32_nd_1x256x64", 1, 256, 64, "nd"),
    ("TMATMUL_F16_TEST.case_mmad_f16f32_nd_2x80x48", 2, 80, 48, "nd"),
    ("TMATMUL_F16_TEST.case_mmad_f16f32_nd_128x128x256", 128, 128, 256, "nd"),
    ("TMATMUL_F16_TEST.case_mmad_f16f32_nd_129x95x33", 129, 95, 33, "nd"),
    # --- DN (transposed) ---
    ("TMATMUL_F16_TEST.case_mmad_f16f32_dn_31x96x47", 31, 96, 47, "dn"),
    ("TMATMUL_F16_TEST.case_mmad_f16f32_dn_127x33x95", 127, 33, 95, "dn"),
    ("TMATMUL_F16_TEST.case_mmad_f16f32_dn_64x64x64", 64, 64, 64, "dn"),
    ("TMATMUL_F16_TEST.case_mmad_f16f32_dn_1x256x64", 1, 256, 64, "dn"),
    ("TMATMUL_F16_TEST.case_mmad_f16f32_dn_65x90x89", 65, 90, 89, "dn"),
]

# The first case is also emitted at the script dir for run_st.py's in-tree
# direct-execution fallback (main.cpp GetGoldenDir resolves to ../suite.case).
DEFAULT_CASE = CASES[0]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=-1)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.case < 0:
        for case_name, m, k, n, layout in CASES:
            os.makedirs(os.path.join(script_dir, case_name), exist_ok=True)
            original_dir = os.getcwd()
            os.chdir(os.path.join(script_dir, case_name))
            gen_golden_data(m, k, n, layout)
            os.chdir(original_dir)
        # Backward-compat: also emit the first case at the script dir.
        name0, m0, k0, n0, layout0 = DEFAULT_CASE
        original_dir = os.getcwd()
        os.chdir(script_dir)
        gen_golden_data(m0, k0, n0, layout0)
        os.chdir(original_dir)
    else:
        case_name, m, k, n, layout = CASES[args.case]
        os.makedirs(os.path.join(script_dir, case_name), exist_ok=True)
        original_dir = os.getcwd()
        os.chdir(os.path.join(script_dir, case_name))
        gen_golden_data(m, k, n, layout)
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
