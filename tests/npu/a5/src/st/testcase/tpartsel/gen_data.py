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

from pathlib import Path

import numpy as np


def gen_golden_data(case_name, dtype, rows, cols):
    rng = np.random.default_rng(19)
    input0 = rng.integers(1, 64, size=(rows, cols)).astype(dtype)
    input1 = rng.integers(65, 128, size=(rows, cols)).astype(dtype)
    if np.dtype(dtype).itemsize == 8:
        input0 += np.array(1 << 40, dtype=dtype)
        input1 += np.array(1 << 48, dtype=dtype)
    mask = rng.integers(0, 256, size=(rows, (cols + 7) // 8), dtype=np.uint8)
    mask[0, :] = 0
    mask[1, :] = 255
    mask[2, :] = 0x55
    bits = np.unpackbits(mask, axis=1, bitorder="little")[:, :cols]
    golden = np.where(bits, input0, input1)
    case_dir = Path(case_name)
    case_dir.mkdir(exist_ok=True)
    for name, data in [("input0", input0), ("input1", input1), ("mask", mask), ("golden", golden)]:
        data.tofile(case_dir / f"{name}.bin")


if __name__ == "__main__":
    cases = [
        ("TPARTSELTest.float_full", np.float32, 4, 128),
        ("TPARTSELTest.float_tail", np.float32, 3, 131),
        ("TPARTSELTest.half_full", np.float16, 4, 128),
        ("TPARTSELTest.half_tail", np.float16, 3, 131),
        ("TPARTSELTest.int32_tail", np.int32, 3, 37),
        ("TPARTSELTest.int16_tail", np.int16, 3, 37),
        ("TPARTSELTest.int8_tail", np.int8, 3, 259),
        ("TPARTSELTest.uint8_tail", np.uint8, 3, 259),
        ("TPARTSELTest.int64_tail", np.int64, 3, 67),
        ("TPARTSELTest.uint64_tail", np.uint64, 3, 67),
    ]
    for case in cases:
        gen_golden_data(*case)
