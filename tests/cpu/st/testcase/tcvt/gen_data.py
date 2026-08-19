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
np.random.seed(19)


_FP4_E1M2_POS = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75], dtype=np.float32)
_FP4_E2M1_POS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

class Int4:
    pass


class Float4E2M1:
    @classmethod
    def quantize(cls, value):
        return mx_from_double(value, 2, 1, 1)

    @classmethod
    def decode(cls, raw):
        return mx_to_double(raw, 2, 1, 1)


class Float4E1M2:
    @classmethod
    def quantize(cls, value):
        return mx_from_double(value, 1, 2, 1)

    @classmethod
    def decode(cls, raw):
        return mx_to_double(raw, 1, 2, 1)

MAN_DBL = 52
EXP_DBL = 11
EXP_DBL_BIAS = 1023
RESERVED_EXPONENT_COUNT = 2

FP4_LIMITS = {
    Int4: (-8, 7),
    Float4E2M1: (-6.0, 6.0),
    Float4E1M2: (-4.0, 3.5)
}

def _quantize_to_fp4_nibble(data, fp4_type, rmode: str = "RoundMode::CAST_RINT"):
    """Return the 4-bit nibble code for a scalar float value.

    Nibble layout: bit3 = sign, bits[2:0] = magnitude code (index into pos_grid).
    Saturation ON: values beyond max are clamped to max.
    rmode controls rounding mode:
        - CAST_RINT / CAST_ROUND: round-to-nearest-even (RNE)
        - CAST_FLOOR: round toward -inf
        - CAST_CEIL: round toward +inf
        - CAST_TRUNC: round toward zero
        - CAST_ODD: round-to-nearest-odd (pick odd magnitude code for ties)
        - CAST_NONE: same as RNE (no rounding needed for exact quantization)

    For directional rounding (FLOOR/CEIL/TRUNC), the rounding direction applies
    to ALL non-exact values, not just midpoint ties. For example:
        FLOOR(0.9) → 0.75 (not 1.0, even though 1.0 is closer)
        FLOOR(-0.1) → -0.25 (more negative, not -0.0 even though -0.0 is closer)
    """
    pos_grid = _FP4_E1M2_POS if fp4_type == Float4E1M2 else _FP4_E2M1_POS
    data_flat = data.flatten()
    result = np.zeros(len(data_flat), dtype=np.uint8)
    for i, val in enumerate(data_flat):
        sign = 0
        if val < 0:
            sign = 1
            val = -val
        # Saturate values beyond the representable range.
        max_code = len(pos_grid) - 1
        if val >= float(pos_grid[max_code]):
            mag_code = max_code
            result[i] = (sign << 3) | mag_code
            continue

        # Find the two bracketing grid values: lower_idx <= val <= upper_idx
        # pos_grid is sorted ascending (e.g., [0.0, 0.25, 0.5, ..., 1.75])
        upper_idx = int(np.searchsorted(pos_grid, val, side='right'))
        if upper_idx == 0:
            # val < pos_grid[0] — underflow to zero
            mag_code = 0
            result[i] = (sign << 3) | mag_code
            continue
        lower_idx = upper_idx - 1

        # Check for exact match with lower bound
        if np.isclose(val, float(pos_grid[lower_idx]), rtol=0, atol=1e-8):
            mag_code = lower_idx
            result[i] = (sign << 3) | mag_code
            continue

        # If beyond last grid value (should be caught by saturation, but be safe)
        if upper_idx >= len(pos_grid):
            mag_code = max_code
            result[i] = (sign << 3) | mag_code
            continue

        # Check for exact match with upper bound
        if np.isclose(val, float(pos_grid[upper_idx]), rtol=0, atol=1e-8):
            mag_code = upper_idx
            result[i] = (sign << 3) | mag_code
            continue

        # val is strictly between pos_grid[lower_idx] and pos_grid[upper_idx]
        lower_val = float(pos_grid[lower_idx])
        upper_val = float(pos_grid[upper_idx])
        midpoint = (lower_val + upper_val) / 2.0
        is_tie = np.isclose(val, midpoint, rtol=0, atol=1e-8)

        # Apply rounding mode based on the actual mathematical direction
        if rmode in ("RoundMode::CAST_FLOOR",):
            # FLOOR: round toward -inf
            # positive (sign=0): pick lower (smaller value → toward -inf)
            # negative (sign=1): abs picks upper (larger magnitude → more negative → toward -inf)
            if sign == 0:
                mag_code = lower_idx
            else:
                mag_code = upper_idx
        elif rmode in ("RoundMode::CAST_CEIL",):
            # CEIL: round toward +inf
            # positive (sign=0): pick upper (larger value → toward +inf)
            # negative (sign=1): abs picks lower (smaller magnitude → less negative → toward +inf)
            if sign == 0:
                mag_code = upper_idx
            else:
                mag_code = lower_idx
        elif rmode in ("RoundMode::CAST_TRUNC",):
            # TRUNC: round toward zero
            # Both positive and negative: abs picks lower (smaller magnitude → toward zero)
            mag_code = lower_idx
        elif rmode in ("RoundMode::CAST_RINT", "RoundMode::CAST_ROUND"):
            # RNE: round to nearest, ties to even
            if is_tie:
                even_candidates = [i for i in [lower_idx, upper_idx] if i % 2 == 0]
                mag_code = even_candidates[0] if even_candidates else lower_idx
            elif val < midpoint:
                mag_code = lower_idx
            else:
                mag_code = upper_idx
        elif rmode in ("RoundMode::CAST_ODD",):
            # Round-to-nearest-odd: ties to odd
            if is_tie:
                odd_candidates = [i for i in [lower_idx, upper_idx] if i % 2 == 1]
                mag_code = odd_candidates[0] if odd_candidates else lower_idx
            elif val < midpoint:
                mag_code = lower_idx
            else:
                mag_code = upper_idx
        elif rmode in ("RoundMode::CAST_NONE",):
            # CAST_NONE: round to nearest, ties to even (same as RNE)
            if is_tie:
                even_candidates = [i for i in [lower_idx, upper_idx] if i % 2 == 0]
                mag_code = even_candidates[0] if even_candidates else lower_idx
            elif val < midpoint:
                mag_code = lower_idx
            else:
                mag_code = upper_idx
        else:
            # Default: RNE
            if is_tie:
                even_candidates = [i for i in [lower_idx, upper_idx] if i % 2 == 0]
                mag_code = even_candidates[0] if even_candidates else lower_idx
            elif val < midpoint:
                mag_code = lower_idx
            else:
                mag_code = upper_idx

        result[i] = (sign << 3) | mag_code
    return result

def double_to_bits(value):
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def bits_to_double(bits):
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def mx_from_double(value, exp_bits, man_bits, bias):
    if value == 0.0:
        return 0

    bits = double_to_bits(value)
    sign = (bits >> (MAN_DBL + EXP_DBL)) & 1
    exponent = (bits >> MAN_DBL) & ((1 << EXP_DBL) - 1)
    mantissa = bits & ((1 << MAN_DBL) - 1)
    out_exp = 0
    out_man = 0

    if exponent - EXP_DBL_BIAS > ((1 << exp_bits) - RESERVED_EXPONENT_COUNT):
        if exponent == (1 << EXP_DBL) - 1:
            out_exp = (1 << exp_bits) - 1
            out_man = mantissa & 1

    if exponent > 0:
        out_man = mantissa >> (MAN_DBL - man_bits)
        out_exp = exponent + bias - EXP_DBL_BIAS
        if out_exp < 0:
            out_man = (out_man | (1 << man_bits)) >> (1 - out_exp)
            out_exp = 0
    else:
        return 0

    return (
        (sign << (exp_bits + man_bits)) |
        ((out_exp & ((1 << exp_bits) - 1)) << man_bits) |
        (out_man & ((1 << man_bits) - 1))
    ) & 0xF


def mx_to_double(raw, exp_bits, man_bits, bias):
    mantissa = raw & ((1 << man_bits) - 1)
    exponent = (raw >> man_bits) & ((1 << exp_bits) - 1)
    sign = (raw >> (exp_bits + man_bits)) & 1

    if exponent > 0 or man_bits == 0:
        bits = (
            (sign << (MAN_DBL + EXP_DBL)) |
            ((exponent + EXP_DBL_BIAS - bias) << MAN_DBL) |
            (mantissa << (MAN_DBL - man_bits))
        )
        return bits_to_double(bits)

    if raw == 0:
        return 0.0
    if raw == (1 << (exp_bits + man_bits)):
        return -0.0
    i = man_bits - 1
    while i >= 0 and ((mantissa >> i) & 1) == 0:
        i -= 1

    bits = (
        (sign << (MAN_DBL + EXP_DBL)) |
        ((EXP_DBL_BIAS - bias + 1 + (i - man_bits)) << MAN_DBL) |
        ((mantissa & ((1 << i) - 1)) << (MAN_DBL - i))
    )
    return bits_to_double(bits)


def get_limits(t):
    if t in FP4_LIMITS:
        return FP4_LIMITS[t]
    try:
        info = np.iinfo(t)
        return info.min, info.max
    except ValueError:
        info = np.finfo(t)
        return info.min, info.max


def quantize_to_fp4(data, fp4_type):
    data_flat = data.flatten()
    result = np.zeros(len(data_flat), dtype=np.uint8)
    for i, val in enumerate(data_flat):
        result[i] = fp4_type.quantize(val)
    return result.reshape(data.shape)


def pack_fp4(codes):
    codes_flat = codes.flatten()
    if len(codes_flat) % 2 != 0:
        codes_flat = np.append(codes_flat, 0)
    packed = np.zeros(len(codes_flat) // 2, dtype=np.uint8)
    for i in range(0, len(codes_flat), 2):
        packed[i // 2] = (codes_flat[i + 1] << 4) | (codes_flat[i] & 0x0F)
    return packed


def write_int4(filename, data):
    data = np.asarray(data, dtype=np.int8)
    encoded = np.bitwise_and(data.astype(np.int16), 0x0F).astype(np.uint8)
    encoded.tofile(filename)


def generate_input_data(param):
    m, n = param.m, param.n
    s_min, s_max = get_limits(param.srctype)

    if param.srctype is Int4:
        return np.random.randint(s_min, s_max + 1, size=[m, n], dtype=np.int8)
    elif param.srctype in (Float4E2M1, Float4E1M2):
        float_data = np.random.uniform(s_min + 0.5, s_max - 0.5, size=[m, n]).astype(np.float32)
        return quantize_to_fp4(float_data, param.srctype)
    else:
        return np.random.uniform(s_min + 5, s_max - 5, size=[m, n]).astype(param.srctype)


def decode_input_data(data, srctype):
    if srctype in (Float4E2M1, Float4E1M2):
        return decode_fp4(data, srctype)
    elif srctype is Int4:
        return data.astype(np.float32)
    else:
        return data.astype(np.float32)


def decode_fp4(data, fp4_type):
    out = np.empty(data.shape, dtype=np.float32)
    flat_in = data.ravel()
    flat_out = out.ravel()
    for i, raw in enumerate(flat_in):
        flat_out[i] = fp4_type.decode(int(raw))
    return out


def apply_saturation(data, dsttype):
    if dsttype is Int4:
        return np.clip(data, -8, 7)
    elif dsttype is Float4E2M1:
        return np.clip(data, -6.0, 6.0)
    elif dsttype is Float4E1M2:
        return np.clip(data, -4.0, 3.5)
    else:
        d_min, d_max = get_limits(dsttype)
        return np.clip(data, d_min, d_max)


def apply_rounding(data, mode):
    if mode == "RoundMode::CAST_RINT":
        return np.rint(data)
    elif mode == "RoundMode::CAST_ROUND":
        return np.round(data)
    elif mode == "RoundMode::CAST_FLOOR":
        return np.floor(data)
    elif mode == "RoundMode::CAST_CEIL":
        return np.ceil(data)
    elif mode == "RoundMode::CAST_TRUNC":
        return np.trunc(data)
    elif mode == "RoundMode::CAST_ODD":
        result = np.empty_like(data)
        for i, val in enumerate(data.flatten()):
            f = np.floor(val)
            frac = val - f
            if frac > 0.5:
                result.flat[i] = f + 1
            elif frac < 0.5:
                result.flat[i] = f
            else:
                result.flat[i] = f if (int(f) & 1) else f + 1
        return result.reshape(data.shape)
    else:
        return data


def convert_to_dsttype(data, dsttype):
    if dsttype is Int4:
        data = np.clip(data, -8, 7)
        return data.astype(np.int8)
    elif dsttype in (Float4E2M1, Float4E1M2):
        return quantize_to_fp4(data, dsttype)
    else:
        return data.astype(dsttype)


def write_output_data(data, dtype, filename):
    if dtype is Int4:
        write_int4(filename, data)
    elif dtype in (Float4E2M1, Float4E1M2):
        pack_fp4(data).tofile(filename)
    else:
        data.tofile(filename)

def gen_golden(param):
    m, n = param.m, param.n

    x1_gm = generate_input_data(param)
    input_values = decode_input_data(x1_gm, param.srctype)

    if param.dsttype in [Float4E1M2, Float4E2M1]:
        golden = _quantize_to_fp4_nibble(input_values, param.dsttype, param.mode)
    else:
        if param.saturation_mode == "SatMode::ON":
            data_to_cast = apply_saturation(input_values, param.dsttype)
        else:
            data_to_cast = input_values

        rounded_data = apply_rounding(data_to_cast, param.mode)
        golden = convert_to_dsttype(rounded_data, param.dsttype)
    write_output_data(x1_gm, param.srctype, "./x1_gm.bin")
    write_output_data(golden, param.dsttype, "./golden.bin")


class TCvtParams:
    def __init__(self, srctype, dsttype, m, n, mode, saturation_mode="SatMode::OFF"):
        self.srctype = srctype
        self.dsttype = dsttype
        self.m = m
        self.n = n
        self.mode = mode
        self.saturation_mode = saturation_mode

if __name__ == "__main__":
    case_name_list = [
        "TCVTTest.case1",
        "TCVTTest.case2",
        "TCVTTest.case3",
        "TCVTTest.case4",
        "TCVTTest.case5",
        "TCVTTest.case6",
        "TCVTTest.case7",
        "TCVTTest.case8",
        "TCVTTest.case9",

        "TCVTTest.case10",
        "TCVTTest.case11",
        "TCVTTest.case12",
        "TCVTTest.case13",
        "TCVTTest.case14",
        "TCVTTest.case15",

        "TCVTTest.case16",
        "TCVTTest.case17",
        "TCVTTest.case18",
        "TCVTTest.case19",

        "TCVTTest.case20",
        "TCVTTest.case21",
        "TCVTTest.case22",
        "TCVTTest.case23",

        "TCVTTest.case24",
        "TCVTTest.case25",
        "TCVTTest.case26",
        "TCVTTest.case27",
        "TCVTTest.case28",
        "TCVTTest.case29",
        "TCVTTest.case30",
        "TCVTTest.case31"
    ]
   
    case_params_list = [
        TCvtParams(np.float32, np.int32, 128, 128, "RoundMode::CAST_RINT"),
        TCvtParams(np.int32, np.float32, 256, 64, "RoundMode::CAST_RINT"),
        TCvtParams(np.float32, np.int16, 16, 32, "RoundMode::CAST_RINT"),
        TCvtParams(np.float32, np.int32, 32, 512, "RoundMode::CAST_RINT"),
        TCvtParams(np.int16, np.int32, 2, 512, "RoundMode::CAST_RINT"),
        TCvtParams(np.float32, np.int32, 4, 4096, "RoundMode::CAST_RINT"),
        TCvtParams(np.int16, np.float32, 64, 64, "RoundMode::CAST_RINT"),
        TCvtParams(np.float32, np.float16, 64, 64, "RoundMode::CAST_RINT"),
        TCvtParams(np.float16, np.uint8, 64, 64, "RoundMode::CAST_RINT"),

        TCvtParams(np.int32, np.float32, 64, 64, "RoundMode::CAST_RINT", "SatMode::ON"),
        TCvtParams(np.int8, np.float32, 128, 128, "RoundMode::CAST_RINT", "SatMode::ON"),
        TCvtParams(np.float32, np.uint8, 64, 64, "RoundMode::CAST_RINT", "SatMode::ON"),
        TCvtParams(np.int32, np.int16, 64, 64, "RoundMode::CAST_RINT", "SatMode::ON"),
        TCvtParams(np.float16, np.int8, 32, 32, "RoundMode::CAST_RINT", "SatMode::ON"),
        TCvtParams(np.float16, np.uint8, 64, 64, "RoundMode::CAST_RINT", "SatMode::ON"),

        TCvtParams(np.float32, np.uint16, 64, 64, "RoundMode::CAST_RINT"),
        TCvtParams(np.uint16, np.float32, 64, 64, "RoundMode::CAST_RINT"),
        TCvtParams(np.int32, np.uint16, 64, 64, "RoundMode::CAST_RINT", "SatMode::ON"),
        TCvtParams(np.uint16, np.int32, 64, 64, "RoundMode::CAST_RINT", "SatMode::ON"),

        TCvtParams(np.float32, Int4, 64, 64, "RoundMode::CAST_RINT"),
        TCvtParams(Int4, np.float32, 64, 64, "RoundMode::CAST_RINT"),
        TCvtParams(np.float32, Int4, 64, 64, "RoundMode::CAST_RINT", "SatMode::ON"),
        TCvtParams(Int4, np.float32, 64, 64, "RoundMode::CAST_RINT", "SatMode::ON"),

        TCvtParams(np.float32, Float4E2M1, 64, 64, "RoundMode::CAST_RINT"),
        TCvtParams(Float4E2M1, np.float32, 64, 64, "RoundMode::CAST_RINT"),
        TCvtParams(np.float32, Float4E2M1, 64, 64, "RoundMode::CAST_RINT"),
        TCvtParams(Float4E2M1, np.float32, 64, 64, "RoundMode::CAST_RINT", "SatMode::ON"),

        TCvtParams(np.float32, Float4E1M2, 64, 64, "RoundMode::CAST_RINT"),
        TCvtParams(Float4E1M2, np.float32, 64, 64, "RoundMode::CAST_RINT"),
        TCvtParams(np.float32, Float4E1M2, 64, 64, "RoundMode::CAST_RINT"),
        TCvtParams(Float4E1M2, np.float32, 64, 64, "RoundMode::CAST_RINT", "SatMode::ON")
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden(case_params_list[i])

        os.chdir(original_dir)
