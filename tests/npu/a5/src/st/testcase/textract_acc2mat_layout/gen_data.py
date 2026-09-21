# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import re
import struct
from pathlib import Path


def encode(value, kind):
    if kind == 0:
        return struct.pack("<e", value)
    if kind == 1:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        return struct.pack("<H", (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16)
    if kind == 2:
        return struct.pack("<I", int(value) & 0xFFFFFFFF)
    return struct.pack("<f", value)


def generate(case, root):
    key, kind, fs, m, k, n, rows, cols, vr, vc, ir, ic, phase, split, bias, relu, plain = case
    rng = 42

    def sample():
        nonlocal rng
        rng = (1664525 * rng + 1013904223) & 0xFFFFFFFF
        return 0 if bias else (1 + (rng >> 16) % 4 if key < 8 else (rng >> 16) % 15 - 7)

    a = [[sample() for _ in range(k)] for _ in range(m)]
    b = [[sample() for _ in range(k)] for _ in range(n)]
    # Include values whose float bit interpretation is subnormal, infinity or NaN,
    # and integers outside float's exact-integer range.
    patterns = [
        0,
        1,
        -1,
        -2147483648,
        2147483647,
        16777217,
        -16777217,
        0x7F800000,
        0x7FC00001,
        0x7FA12345,
        -8388608,
        -4194303,
        0x00800000,
        0x007FFFFF,
    ]
    biases = [patterns[c % len(patterns)] if bias else 0 for c in range(n)]
    count = 2 if phase == 2 else 1
    size = 4 if kind in (2, 3) else 2
    c0 = fs // 16 // size
    output = bytearray(b"\xa5" * (256 + count * rows * cols * size))
    for block in range(count):
        for r in range(vr):
            for c in range(vc):
                sr, sc = (ir + r, ic + c) if block == 0 else (r, c)
                value = biases[sc] + sum(a[sr][j] * b[sc][j] for j in range(k))
                if relu:
                    value = max(value, 0)
                index = 128 + size * (block * rows * cols + (c // c0) * rows * c0 + r * c0 + c % c0)
                output[index : index + size] = encode(value, kind)
    path = root / f"TEXTRACTAcc2MatLayoutTest.case{key}"
    path.mkdir(parents=True, exist_ok=True)
    input_kind = 2 if kind == 2 else 0

    def pack_input(values):
        return b"".join(struct.pack("<b", v) if input_kind == 2 else struct.pack("<e", v) for v in values)

    (path / "a.bin").write_bytes(pack_input(v for row in a for v in row))
    (path / "b.bin").write_bytes(pack_input(v for row in b for v in row))
    (path / "bias.bin").write_bytes(b"".join(struct.pack("<i", v) for v in biases))
    (path / "golden.bin").write_bytes(output)


if __name__ == "__main__":
    source = Path(__file__).with_name("textract_acc2mat_layout_cases.h").read_text()
    for values in re.findall(r"^LAYOUT_CASE\(([\d, ]+)\)", source, re.MULTILINE):
        generate(tuple(map(int, values.split(","))), Path.cwd())
