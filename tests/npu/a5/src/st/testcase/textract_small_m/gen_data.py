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

from pathlib import Path

import ml_dtypes
import numpy as np

CASES = [
    (1, 0, 0, 0, 16, 64, 4, 64, 0, 0, 4),
    (2, 1, 0, 0, 16, 64, 4, 64, 0, 0, 4),
    (3, 0, 1, 1, 32, 128, 4, 63, 15, 16, 1),
    (4, 1, 1, 1, 32, 128, 4, 63, 15, 16, 1),
    (5, 0, 0, 0, 32, 64, 15, 64, 17, 16, 1),
    (6, 1, 1, 0, 32, 64, 8, 17, 24, 16, 1),
    (7, 0, 0, 0, 16, 64, 1, 1, 15, 0, 1),
    (8, 1, 0, 0, 16, 64, 2, 64, 13, 0, 1),
    (9, 0, 0, 1, 32, 64, 16, 64, 16, 16, 1),
    (10, 1, 0, 0, 16, 512, 4, 512, 12, 0, 1),
    (11, 0, 1, 0, 16, 256, 4, 256, 12, 0, 1),
    (12, 0, 0, 1, 32, 128, 4, 63, 28, 16, 1),
    (13, 0, 1, 1, 32, 128, 16, 63, 16, 16, 1),
]

for key, bf16, compact, dynamic, rows, cols, m, k, row, col, groups in CASES:
    dtype = ml_dtypes.bfloat16 if bf16 else np.float16
    rng = np.random.default_rng(444 + key)
    a = rng.integers(-8, 9, (rows, cols + col)).astype(dtype)
    b = rng.integers(-4, 5, (cols, 32)).astype(dtype)
    b[k:, :] = 0
    golden = np.concatenate(
        [
            a[row + group * 4 : row + group * 4 + m, col : col + k].astype(np.float32) @ b[:k].astype(np.float32)
            for group in range(groups)
        ]
    )
    out = Path(f"TEXTRACTSmallMTest.case{key}")
    out.mkdir(exist_ok=True)
    a.tofile(out / "a.bin")
    b.tofile(out / "b.bin")
    golden.tofile(out / "golden.bin")
