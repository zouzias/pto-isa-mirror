#!/usr/bin/python3
# coding=utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import sys
import numpy as np


def main():
    if len(sys.argv) < 5:
        print("Usage: verify_result.py <output_file> <golden_file> <padded_M> <padded_N> [orig_M] [orig_N]")
        sys.exit(1)

    output_file = sys.argv[1]
    golden_file = sys.argv[2]
    padded_m = int(sys.argv[3])
    padded_n = int(sys.argv[4])
    orig_m = int(sys.argv[5]) if len(sys.argv) > 5 else padded_m
    orig_n = int(sys.argv[6]) if len(sys.argv) > 6 else padded_n

    output = np.fromfile(output_file, dtype=np.float32).reshape(padded_m, padded_n)
    golden = np.fromfile(golden_file, dtype=np.float32).reshape(orig_m, orig_n)

    if orig_m != padded_m or orig_n != padded_n:
        output = output[:orig_m, :orig_n]

    rtol = 0.001
    atol = 0.001
    close = np.allclose(output, golden, rtol=rtol, atol=atol)

    if close:
        print(f"[PASS] Output matches golden (rtol={rtol}, atol={atol})")
        sys.exit(0)
    else:
        diff = np.abs(output - golden)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        mismatch_count = np.sum(~np.isclose(output, golden, rtol=rtol, atol=atol))
        total = output.size
        print(f"[FAIL] Output does NOT match golden!")
        print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
        print(f"  Mismatched elements: {mismatch_count}/{total} ({mismatch_count/total*100:.2f}%)")
        sys.exit(1)


if __name__ == "__main__":
    main()
