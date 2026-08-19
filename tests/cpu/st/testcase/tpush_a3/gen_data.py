#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np
import os

a = np.random.randn(32, 64).astype(np.float32)
b = np.random.randn(64, 512).astype(np.float32)
c_prev = np.random.randn(32, 512).astype(np.float32)

c_golden = c_prev + np.matmul(a + 1.0, b)

CASE_NAME = "TPUSH_A3Test.case_1"
if not os.path.exists(CASE_NAME):
    os.makedirs(CASE_NAME)
ORIGINAL_DIR = os.getcwd()
os.chdir(CASE_NAME)

a.tofile("a.bin")
b.tofile("b.bin")
c_prev.tofile("c.bin")
c_golden.tofile("golden.bin")

os.chdir(ORIGINAL_DIR)
