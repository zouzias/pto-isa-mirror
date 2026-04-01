#!/usr/bin/env python3
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

"""Generate [1, 2048] uint16 keys and golden multiset for Top-512 (largest keys)."""

import argparse
import os

import numpy as np

N = 2048
TOPK = 512


def main():
    parser = argparse.ArgumentParser(description="Generate keys.bin and golden_topk_multiset.bin")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (default: random each run)",
    )
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    np.random.seed(seed)
    keys = np.random.randint(0, 65536, size=N, dtype=np.uint16)
    # Reference multiset of the K largest keys (written ascending only for host-side multiset compare)
    topk_vals = np.sort(keys)[-TOPK:]

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    keys.tofile(os.path.join("input", "keys.bin"))
    topk_vals.tofile(os.path.join("output", "golden_topk_multiset.bin"))
    print(f"Wrote input/keys.bin ({N} x uint16), output/golden_topk_multiset.bin ({TOPK} x uint16).")
    print(f"seed = {seed}  (re-run with: python3 scripts/gen_data.py --seed {seed})")


if __name__ == "__main__":
    main()
