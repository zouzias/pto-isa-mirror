#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Count keys in keys.bin whose high byte (MSB) equals 0xBE (190).

Also prints per-256-chunk: local MSB==0xBE count and the first two keys in the chunk as hex.
"""

import argparse
import os
import sys

import numpy as np

TARGET_MSB = 0xBE
CHUNK = 256


def main() -> int:
    parser = argparse.ArgumentParser(description="Count keys with (uint16 >> 8) == 0xBE")
    parser.add_argument(
        "keys_bin",
        nargs="?",
        default=None,
        help="Path to keys.bin (default: ../input/keys.bin relative to script)",
    )
    args = parser.parse_args()

    if args.keys_bin is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        keys_path = os.path.join(root, "input", "keys.bin")
    else:
        keys_path = os.path.abspath(args.keys_bin)

    if not os.path.isfile(keys_path):
        print(f"Missing file: {keys_path}", file=sys.stderr)
        return 1

    keys = np.fromfile(keys_path, dtype=np.dtype("<u2"))
    msb = (keys.astype(np.uint32) >> 8) & 0xFF
    n_total = int(np.sum(msb == TARGET_MSB))

    print(f"file: {keys_path}")
    print(f"total keys: {keys.size}")
    print(f"total count (key >> 8) == 0x{TARGET_MSB:02x} ({TARGET_MSB}): {n_total}")
    print()
    print(f"per-{CHUNK} keys: chunk_idx  index_range  msb==0x{TARGET_MSB:02x}_count  key0_hex  key1_hex")

    for chunk_idx, start in enumerate(range(0, keys.size, CHUNK)):
        end = min(start + CHUNK, keys.size)
        sl = slice(start, end)
        c = int(np.sum(msb[sl] == TARGET_MSB))
        k0 = int(keys[start])
        if end - start >= 2:
            k1 = int(keys[start + 1])
            h1 = f"0x{k1:04x}"
        else:
            h1 = "n/a"
        print(f"  chunk {chunk_idx:3d}  [{start:5d}, {end:5d})  {c:3d}  0x{k0:04x}  {h1}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
