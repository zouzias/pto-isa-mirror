#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse kernels/manual/a5/topk/build/core0.veccore0.ub.wr_log.dump:
  - chistMSB lives at UB 0x14000 .. 0x143ff (256 x uint32 LE).
  - For each 4B-aligned address, keep the last write in file order (sim time order).

Prints:
  - Last-write tick histogram (see THISTOGRAM: 4 x 64 words batches).
  - Compare linear C_sim[b] to CPU ascending cumulative C_cpu[b] from input/keys.bin.
  - Note: THISTOGRAM stores bins in hardware layout; linear b is not necessarily MSB value b.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np

ADDR_PAT = re.compile(r"Address\s+([0-9a-fA-F]+)\s*=\s*\[([0-9a-fA-F]+)\]")
TICK_PAT = re.compile(r"\[info\]\s+\[(\d+)\]\s+")


def parse_ub_wr_log(path: Path, region_base: int, region_size: int) -> tuple[dict[int, int], dict[int, int]]:
    """Returns (last_val, last_tick) per address in [base, base+size)."""
    last_val: dict[int, int] = {}
    last_tick: dict[int, int] = {}
    cur_tick: int | None = None
    hi = region_base + region_size
    with path.open(errors="replace") as f:
        for line in f:
            tm = TICK_PAT.search(line)
            if tm:
                cur_tick = int(tm.group(1))
            for m in ADDR_PAT.finditer(line):
                a = int(m.group(1), 16)
                if region_base <= a < hi:
                    v = int(m.group(2), 16)
                    last_val[a] = v
                    if cur_tick is not None:
                        last_tick[a] = cur_tick
    return last_val, last_tick


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "build" / "core0.veccore0.ub.wr_log.dump",
    )
    ap.add_argument("--base", type=lambda x: int(x, 0), default=0x14000)
    ap.add_argument("--keys", type=Path, default=Path(__file__).resolve().parent.parent / "input" / "keys.bin")
    args = ap.parse_args()

    base = args.base
    last_val, last_tick = parse_ub_wr_log(args.log, base, 256 * 4)

    C_sim = np.array([last_val[base + 4 * b] for b in range(256)], dtype=np.uint64)
    ticks = [last_tick.get(base + 4 * b) for b in range(256)]
    print("=== Last-write tick per word index (expect 4 batches x 64 for THISTOGRAM tail stores) ===")
    print(Counter(t for t in ticks if t is not None))

    if args.keys.is_file():
        keys = np.fromfile(args.keys, dtype=np.dtype("<u2"))
        msb = (keys.astype(np.uint32) >> 8) & 0xFF
        raw_cpu = np.bincount(msb, minlength=256).astype(np.uint64)
        C_cpu = np.cumsum(raw_cpu).astype(np.uint64)
        print()
        print("=== vs CPU (keys.bin): ascending cumulative by MSB byte value 0..255 ===")
        print("C_cpu[255] =", int(C_cpu[255]), "  C_sim[255] =", int(C_sim[255]))
        mism = np.flatnonzero(C_sim != C_cpu)
        print("uint32 mismatches (linear index b vs logical MSB b):", mism.size)
        raw_phys = np.zeros(256, dtype=np.int64)
        raw_phys[0] = int(C_sim[0])
        raw_phys[1:] = (C_sim[1:].astype(np.int64) - C_sim[:-1].astype(np.int64))
        print("multiset(raw_phys diffs) == multiset(raw_cpu)?", np.array_equal(np.sort(raw_phys), np.sort(raw_cpu.astype(np.int64))))
        print("(If layout is permuted, linear diffs are not logical per-bin counts.)")
    else:
        print("No keys.bin; skip CPU compare.")


if __name__ == "__main__":
    main()
