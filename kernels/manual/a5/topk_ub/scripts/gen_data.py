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

"""Generate uint16 keys, golden Top-K indices, and optional value multiset (Top-512 largest)."""

import argparse
import os

import numpy as np

N = 65536
TOPK = 512


def main():
    parser = argparse.ArgumentParser(description="Generate keys.bin, golden_topk_idx.bin, golden_topk_multiset.bin")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (default: random each run)",
    )
    parser.add_argument(
        "--min-key",
        type=int,
        default=0,
        help="Inclusive lower bound of generated uint16 keys (default: 0)",
    )
    parser.add_argument(
        "--max-key",
        type=int,
        default=65535,
        help="Inclusive upper bound of generated uint16 keys (default: 65535)",
    )
    parser.add_argument(
        "--constant-key",
        type=int,
        default=None,
        help="If set, all N keys are this uint16 value (uniform input)",
    )
    args = parser.parse_args()
    if not (0 <= args.min_key <= 65535 and 0 <= args.max_key <= 65535):
        raise ValueError("min-key/max-key must be in [0, 65535]")
    if args.min_key > args.max_key:
        raise ValueError("min-key must be <= max-key")
    if args.constant_key is not None and not (0 <= args.constant_key <= 65535):
        raise ValueError("constant-key must be in [0, 65535]")
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    np.random.seed(seed)
    if args.constant_key is not None:
        keys = np.full(N, args.constant_key, dtype=np.uint16)
    else:
        keys = np.random.randint(args.min_key, args.max_key + 1, size=N, dtype=np.uint16)
    # Canonical Top-K indices: sort by key descending, stable tie-break (smaller index first among equals).
    order = np.argsort(-keys.astype(np.int64), kind="stable")
    topk_idx = order[:TOPK].astype(np.uint32)
    # Reference multiset of the K largest key values (ascending; optional tooling / sanity).
    topk_vals = np.sort(keys)[-TOPK:]

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    keys.tofile(os.path.join("input", "keys.bin"))
    topk_idx.tofile(os.path.join("output", "golden_topk_idx.bin"))
    topk_vals.tofile(os.path.join("output", "golden_topk_multiset.bin"))
    print(
        f"Wrote input/keys.bin ({N} x uint16), output/golden_topk_idx.bin ({TOPK} x uint32), "
        f"golden_topk_multiset.bin ({TOPK} x uint16)."
    )
    if args.constant_key is not None:
        print(f"constant_key = {args.constant_key} (all {N} inputs identical)")
        print(f"re-run: python3 scripts/gen_data.py --constant-key {args.constant_key} --seed {seed}")
    else:
        print(f"key_range = [{args.min_key}, {args.max_key}]")
        print(
            "seed = "
            f"{seed}  (re-run with: python3 scripts/gen_data.py --seed {seed} --min-key {args.min_key} --max-key {args.max_key})"
        )


if __name__ == "__main__":
    main()
