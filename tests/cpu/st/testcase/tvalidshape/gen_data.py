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
import numpy as np
from utils import NumExt


def generate_cpu_test_suite(prefix, cases):
    """
    Generates test data for ConvTile validShape verification.

    For each case:
      - input.bin:  Full physical data in NC1HWC0 layout.
      - golden.bin: Tile data after TLOAD with valid < physical.
                    Valid positions contain source data; invalid positions are zero.
    """
    type_map = {
        'float16': (np.float16, 2),
        'half': (np.float16, 2),
    }

    for suffix, phys_shape, valid_dims, dtype_str in cases:
        folder_name = f"{prefix}.{suffix}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        np_type, type_size = type_map.get(dtype_str.lower(), (np.float16, 2))

        total_elements = int(np.prod(phys_shape))
        data = NumExt.astype((np.arange(total_elements) % 256).reshape(phys_shape), np_type)

        golden = np.zeros(phys_shape, dtype=data.dtype)
        slices = tuple(slice(0, v) for v in valid_dims) + (slice(None),)
        golden[slices] = data[slices]

        input_path = os.path.join(folder_name, "input.bin")
        golden_path = os.path.join(folder_name, "golden.bin")

        NumExt.write_array(input_path, data, np_type)
        NumExt.write_array(golden_path, golden, np_type)

        print(f"Generated: {folder_name}")
        print(f"  -> Physical: {phys_shape} | Valid: {valid_dims} | DType: {dtype_str}")
        print(f"  -> Input size: {os.path.getsize(input_path)}B | Golden size: {os.path.getsize(golden_path)}B")


test_cases = [
    # (suffix, physical_shape (N,C1,H,W,C0), valid_dims (VN,VC1,VH,VW), dtype)
    # Physical: N=4, C1=2, H=4, W=4, C0=16 (half: 32/2=16)
    # Valid:    N=2, C1=1, H=2, W=4 (W and C0 fully valid)
    ("case_tload_nc1hwc0_valid", (4, 2, 4, 4, 16), (2, 1, 2, 4), "float16"),
]

if __name__ == "__main__":
    generate_cpu_test_suite("TValidShapeTest", test_cases)
