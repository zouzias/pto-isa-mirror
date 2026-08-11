#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import argparse
import importlib.util
import os

import ml_dtypes
import numpy as np
from ml_dtypes import bfloat16


E8M0_BIAS = 127
MX_SCALE_GROUP = 32


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_sibling_gen_data(sibling):
    """Locate a sibling testcase's gen_data.py.

    run_st.py copies this script to build/gen_data.py and runs it from build/,
    so __file__-relative '..' points to tests/npu/a6/src/st (one level above
    build), NOT to testcase/. The sibling scripts live under testcase/<sibling>/.
    Fall back to '../<sibling>' for direct in-source execution.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "..", "testcase", sibling, "gen_data.py"),
        os.path.join(script_dir, "..", sibling, "gen_data.py"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"could not find sibling gen_data.py for '{sibling}'")


def _left_data_and_dequant(rng, left_kind, m, k):
    src = rng.uniform(-8.0, 8.0, (m, k)).astype(np.float32)
    if left_kind == "e4m3":
        left = src.astype(ml_dtypes.float8_e4m3fn)
        left_bytes = left.view(np.uint8).tobytes()
        left_float = left.astype(np.float32)
    elif left_kind == "fp16":
        left = src.astype(np.float16)
        left_bytes = left.tobytes()
        left_float = left.astype(np.float32)
    else:
        left = src.astype(bfloat16)
        left_bytes = left.tobytes()
        left_float = left.astype(np.float32)

    # fp8_e4m3 is a microscaled type (OCP MX) -> it takes a real e8m0 A-scale.
    # half/bf16 are full-precision 16-bit types, NOT microscaled -> the MMAD_MX
    # instruction does NOT apply an A-side e8m0 scale for them. Feed neutral
    # (all-127, scale=1.0) A-scale bytes so the tile is harmless if the hardware
    # reads it, and the golden must NOT apply an A-scale (see _gen_case).
    if left_kind == "e4m3":
        a_scale = rng.integers(126, 130, size=(m, k // MX_SCALE_GROUP), dtype=np.uint8)
    else:  # fp16 / bf16: neutral A-scale
        a_scale = np.full((m, k // MX_SCALE_GROUP), E8M0_BIAS, dtype=np.uint8)
    return left_bytes, left_float, a_scale


def _gen_case(case, e2m1_mod, hif4_mod, out_dir):
    case_id, case_name, left_kind, right_kind, m, k, n = case
    rng = np.random.default_rng(100 + case_id)

    left_bytes, left_float, a_scale = _left_data_and_dequant(rng, left_kind, m, k)
    a_scale_zz = e2m1_mod.convert_x1_scale_format(a_scale, 16, 2)
    # fp8_e4m3 A is microscaled -> apply e8m0 A-scale. half/bf16 A are
    # full-precision -> NOT microscaled, no A-scale applied (neutral bytes fed).
    if left_kind == "e4m3":
        a_scale_factor = np.power(2.0, a_scale.astype(np.int16) - E8M0_BIAS).astype(np.float32)
        left_deq = left_float * np.repeat(a_scale_factor, MX_SCALE_GROUP, axis=1)
    else:  # fp16 / bf16: no A-scale
        left_deq = left_float

    if right_kind == "e2m1":
        b_src = hif4_mod.make_bf16_matrix(k, n, group_axis="col").astype(np.float32)
        b_codes, b_scale = e2m1_mod.e2m1_mx_quantize(b_src, group_axis="col")
        b_data = e2m1_mod.pack_fp4_nd(b_codes.ravel())
        b_scale_nn = e2m1_mod.convert_x2_scale_format(b_scale, 16, 2)
        b_scale_bytes = b_scale_nn.tobytes()

        b_scale_factor = np.power(2.0, b_scale.astype(np.int16) - E8M0_BIAS).astype(np.float32)
        b_deq = e2m1_mod.decode_e2m1(b_codes) * np.repeat(b_scale_factor, MX_SCALE_GROUP, axis=0)
    else:
        b_bf16 = hif4_mod.make_bf16_matrix(k, n, group_axis="col")
        b_data, b_scale_bytes = hif4_mod.quantize_to_hif4_b(b_bf16)
        b_deq = hif4_mod.dequantize_for_matmul(b_bf16.T.copy()).T.astype(np.float32)

    golden = (left_deq.astype(np.float32) @ b_deq.astype(np.float32)).astype(np.float32).astype(bfloat16)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "a_data.bin"), "wb") as f:
        f.write(left_bytes)
    with open(os.path.join(out_dir, "a_scale.bin"), "wb") as f:
        f.write(a_scale_zz.tobytes())
    with open(os.path.join(out_dir, "b_data.bin"), "wb") as f:
        f.write(b_data)
    with open(os.path.join(out_dir, "b_scale.bin"), "wb") as f:
        f.write(b_scale_bytes)
    with open(os.path.join(out_dir, "golden_out.bin"), "wb") as f:
        f.write(golden.tobytes())
    with open(os.path.join(out_dir, "golden.bin"), "wb") as f:
        f.write(golden.tobytes())

    print(
        f"[{case_name}] M={m} K={k} N={n}: "
        f"a_data={len(left_bytes)}B a_scale={a_scale_zz.nbytes}B "
        f"b_data={len(b_data)}B b_scale={len(b_scale_bytes)}B golden={golden.nbytes}B"
    )


CASES = [
    (1, "TMATMUL_MX_MIXED_A6_TEST.case_mmad_mx_e4m3hi4_128x128x128_nd", "e4m3", "hif4", 128, 128, 128),
    (2, "TMATMUL_MX_MIXED_A6_TEST.case_mmad_mx_e4m3hi4_64x128x64_nd", "e4m3", "hif4", 64, 128, 64),
    (3, "TMATMUL_MX_MIXED_A6_TEST.case_mmad_mx_fp16e2m1_128x128x128_nd", "fp16", "e2m1", 128, 128, 128),
    (4, "TMATMUL_MX_MIXED_A6_TEST.case_mmad_mx_fp16e2m1_64x128x64_nd", "fp16", "e2m1", 64, 128, 64),
    (5, "TMATMUL_MX_MIXED_A6_TEST.case_mmad_mx_bf16e2m1_128x128x128_nd", "bf16", "e2m1", 128, 128, 128),
    (6, "TMATMUL_MX_MIXED_A6_TEST.case_mmad_mx_bf16e2m1_64x128x64_nd", "bf16", "e2m1", 64, 128, 64),
    (7, "TMATMUL_MX_MIXED_A6_TEST.case_mmad_mx_fp16hi4_128x128x128_nd", "fp16", "hif4", 128, 128, 128),
    (8, "TMATMUL_MX_MIXED_A6_TEST.case_mmad_mx_fp16hi4_64x128x64_nd", "fp16", "hif4", 64, 128, 64),
    (9, "TMATMUL_MX_MIXED_A6_TEST.case_mmad_mx_bf16hif4_128x128x128_nd", "bf16", "hif4", 128, 128, 128),
    (10, "TMATMUL_MX_MIXED_A6_TEST.case_mmad_mx_bf16hif4_64x128x64_nd", "bf16", "hif4", 64, 128, 64),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=-1)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    e2m1_mod = _load_module("mx_e2m1_mod", _resolve_sibling_gen_data("tmatmul_mx_e4m3e2m1"))
    hif4_mod = _load_module("mx_hif4_mod", _resolve_sibling_gen_data("tmatmul_mx_hif4"))

    if args.case < 0:
        for case in CASES:
            _, name, _, _, _, _, _ = case
            _gen_case(case, e2m1_mod, hif4_mod, os.path.join(script_dir, name))
            if case[0] == 1:
                _gen_case(case, e2m1_mod, hif4_mod, script_dir)
    else:
        case = CASES[args.case]
        _gen_case(case, e2m1_mod, hif4_mod, os.path.join(script_dir, case[1]))


if __name__ == "__main__":
    main()
