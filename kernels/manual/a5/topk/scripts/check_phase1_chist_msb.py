#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare sim chistMSB @ UB 0x14000 after Phase1 vs CPU reference from keys.bin."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ADDR_PAT = re.compile(r"Address\s+([0-9a-fA-F]+)\s*=\s*\[([0-9a-fA-F]+)\]")
TICK_PAT = re.compile(r"\[info\]\s+\[(\d+)\]\s+")


def parse_ub_region(path: Path, base: int, n_words: int) -> np.ndarray:
    last_val: dict[int, int] = {}
    cur_tick: int | None = None
    hi = base + n_words * 4
    with path.open(errors="replace") as f:
        for line in f:
            tm = TICK_PAT.search(line)
            if tm:
                cur_tick = int(tm.group(1))
            for m in ADDR_PAT.finditer(line):
                a = int(m.group(1), 16)
                if base <= a < hi:
                    last_val[a] = int(m.group(2), 16)
    return np.array([last_val.get(base + 4 * b, 0) for b in range(n_words)], dtype=np.uint64)


def cpu_msb_chist(keys: np.ndarray) -> np.ndarray:
    msb = (keys.astype(np.uint32) >> 8) & 0xFF
    counts = np.bincount(msb.astype(np.int64), minlength=256).astype(np.uint64)
    return np.cumsum(counts)


def main() -> int:
    topk_dir = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keys", type=Path, default=topk_dir / "input" / "keys.bin")
    ap.add_argument(
        "--ub-log",
        type=Path,
        default=topk_dir / "build" / "core0.veccore0.ub.wr_log.dump",
    )
    ap.add_argument("--base", type=lambda x: int(x, 0), default=0x14000, help="chistMSB UB base")
    args = ap.parse_args()

    if not args.keys.is_file():
        print(f"error: missing {args.keys}", file=sys.stderr)
        return 1
    if not args.ub_log.is_file():
        print(f"error: missing {args.ub_log} (run ./topk in build/ first)", file=sys.stderr)
        return 1

    keys = np.fromfile(args.keys, dtype=np.dtype("<u2"))
    C_cpu = cpu_msb_chist(keys)
    C_sim = parse_ub_region(args.ub_log, args.base, 256)

    print(f"keys: {args.keys}  N={keys.size}")
    print(f"ub:   {args.ub_log}  chistMSB @ {args.base:#x}")
    print(f"C[255] cpu={int(C_cpu[255])}  sim={int(C_sim[255])}")

    mism = np.flatnonzero(C_sim != C_cpu)
    print(f"linear uint32 mismatches: {mism.size}")
    if mism.size:
        for b in mism[:16]:
            print(f"  b={b}: cpu=0x{int(C_cpu[b]):08x} sim=0x{int(C_sim[b]):08x}")
        return 2

    raw_cpu = np.diff(np.concatenate(([0], C_cpu)))
    raw_sim = np.diff(np.concatenate(([0], C_sim)))
    multiset_ok = np.array_equal(np.sort(raw_sim), np.sort(raw_cpu))
    print(f"multiset(per-bin counts) match: {multiset_ok}")

    thr = keys.size - 512
    w_cpu = int(np.argmax(C_cpu >= thr))
    w_sim = int(np.argmax(C_sim >= thr))
    print(f"msb_winner (min b: C[b]>=N-TopK): cpu={w_cpu} sim={w_sim}")
    return 0 if w_cpu == w_sim else 3


if __name__ == "__main__":
    raise SystemExit(main())
