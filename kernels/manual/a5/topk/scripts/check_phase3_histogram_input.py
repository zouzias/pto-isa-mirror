#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check Phase3 THISTOGRAM<BYTE_0> inputs vs draft.cpp tiling.

Inputs per slice:
  - idxFilter (uint8): MSB byte == msbWinnerSaved (raw find bin), NOT msbWinnerBin (find-1).
  - inTile: TLOAD of keys[base+sub : base+sub+subValid] as uint16 row.

Verifies:
  1) Sim idxFilter @ UB 0x1C000
  2) CPU replay of slice-wise filtered LSB hist + TADD matches sim chistLSB @ 0x18000
  3) Per-slice filtered key counts (sanity)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ADDR_PAT = re.compile(r"Address\s+([0-9a-fA-F]+)\s*=\s*\[([0-9a-fA-F]+)\]")
TICK_PAT = re.compile(r"\[info\]\s+\[(\d+)\]\s+")

UB_CHIST_LSB = 0x18000
UB_IDX = 0x1C000
UB_MSB_SAVED = 0x24D80

N_DEFAULT = 8192
TILE_COLS_DEFAULT = 2048


def parse_last_byte(path: Path, addr: int) -> int | None:
    last: dict[int, int] = {}
    with path.open(errors="replace") as f:
        for line in f:
            for m in ADDR_PAT.finditer(line):
                a = int(m.group(1), 16)
                if addr <= a < addr + 64:
                    last[a] = int(m.group(2), 16)
    v = last.get(addr)
    return (v & 0xFF) if v is not None else None


def parse_chist_linear(path: Path, base: int = UB_CHIST_LSB) -> np.ndarray:
    last: dict[int, int] = {}
    hi = base + 256 * 4
    with path.open(errors="replace") as f:
        for line in f:
            for m in ADDR_PAT.finditer(line):
                a = int(m.group(1), 16)
                if base <= a < hi:
                    last[a] = int(m.group(2), 16)
    return np.array([last.get(base + 4 * b, 0) for b in range(256)], dtype=np.uint64)


def cpu_msb_find(keys: np.ndarray, topk: int) -> int:
    msb = ((keys.astype(np.uint32) >> 8) & 0xFF).astype(np.int64)
    c = np.cumsum(np.bincount(msb, minlength=256))
    thr = keys.size - topk
    for b in range(256):
        if c[b] >= thr:
            return b
    return 0


def slice_hist_lsb(keys_slice: np.ndarray, msb_filter: int) -> np.ndarray:
    """Ascending cumulative LSB hist for keys with MSB == msb_filter in this slice."""
    msb = ((keys_slice.astype(np.uint32) >> 8) & 0xFF).astype(np.int64)
    lsb = (keys_slice.astype(np.uint32) & 0xFF).astype(np.int64)
    sel = lsb[msb == msb_filter]
    counts = np.bincount(sel, minlength=256).astype(np.uint64)
    return np.cumsum(counts)


def replay_phase3_inputs(
    keys: np.ndarray,
    msb_filter: int,
    tile_cols: int,
    hist_chunk_cols: int,
) -> tuple[np.ndarray, list[dict]]:
    n = keys.size
    chist = np.zeros(256, dtype=np.uint64)
    meta: list[dict] = []
    k_loop = (n + tile_cols - 1) // tile_cols
    for i in range(k_loop):
        base = i * tile_cols
        valid = min(tile_cols, n - base)
        for sub in range(0, valid, hist_chunk_cols):
            sv = min(hist_chunk_cols, valid - sub)
            chunk = keys[base + sub : base + sub + sv]
            msb = ((chunk.astype(np.uint32) >> 8) & 0xFF).astype(np.int64)
            n_filt = int(np.sum(msb == msb_filter))
            partial = slice_hist_lsb(chunk, msb_filter)
            chist += partial
            meta.append(
                {
                    "tile": i,
                    "sub": sub,
                    "base": base,
                    "subValid": sv,
                    "n_filtered": n_filt,
                    "partial_c255": int(partial[255]),
                }
            )
    return chist, meta


def main() -> int:
    topk_dir = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keys", type=Path, default=topk_dir / "input" / "keys.bin")
    ap.add_argument(
        "--ub-log",
        type=Path,
        default=topk_dir / "build" / "core0.veccore0.ub.wr_log.dump",
    )
    ap.add_argument("--topk", type=int, default=512)
    ap.add_argument("--tile-cols", type=int, default=TILE_COLS_DEFAULT)
    ap.add_argument("--hist-chunk-cols", type=int, default=256)
    args = ap.parse_args()

    if not args.keys.is_file():
        print(f"error: missing {args.keys}", file=sys.stderr)
        return 1

    keys = np.fromfile(args.keys, dtype=np.dtype("<u2"))
    msb_filter = cpu_msb_find(keys, args.topk)

    print(f"keys: {args.keys}  N={keys.size}  TopK={args.topk}")
    print(f"tiling: tile_cols={args.tile_cols}  hist_chunk_cols={args.hist_chunk_cols}")
    print(f"CPU msbWinnerSaved / idxFilter byte = {msb_filter} (0x{msb_filter:02x})")
    print()

    chist_cpu, meta = replay_phase3_inputs(keys, msb_filter, args.tile_cols, args.hist_chunk_cols)
    total_filt = sum(m["n_filtered"] for m in meta)
    print(f"CPU replay: {len(meta)} THISTOGRAM slices, total keys with MSB==filter: {total_filt}")
    print(f"CPU replay chistLSB[255] = {int(chist_cpu[255])}")
    print()

    ok = True
    if args.ub_log.is_file():
        sim_idx = parse_last_byte(args.ub_log, UB_IDX)
        sim_msb_saved = None
        last: dict[int, int] = {}
        with args.ub_log.open(errors="replace") as f:
            for line in f:
                for m in ADDR_PAT.finditer(line):
                    a = int(m.group(1), 16)
                    if UB_MSB_SAVED <= a < UB_MSB_SAVED + 4:
                        last[a] = int(m.group(2), 16)
        if last:
            sim_msb_saved = last.get(UB_MSB_SAVED)

        chist_sim = parse_chist_linear(args.ub_log)
        print("=== Sim UB ===")
        print(f"  idxFilter @ {UB_IDX:#x}       = {sim_idx}")
        print(f"  msbWinnerSaved[0] @ {UB_MSB_SAVED:#x} = {sim_msb_saved}")
        print(f"  chistLSB[255] @ {UB_CHIST_LSB:#x} = {int(chist_sim[255])}")
        print()

        if sim_idx is None or int(sim_idx) != msb_filter:
            print(f"FAIL idxFilter: sim={sim_idx} cpu={msb_filter}")
            ok = False
        if sim_msb_saved is not None and int(sim_msb_saved) != msb_filter:
            print(f"FAIL msbWinnerSaved before TCVT: sim={sim_msb_saved} cpu={msb_filter}")
            ok = False
        mism = np.flatnonzero(chist_sim != chist_cpu)
        if mism.size:
            print(f"FAIL chistLSB: {mism.size} linear bin mismatches vs CPU replay from Phase3 inputs")
            for b in mism[:8]:
                print(f"  b={b}: cpu=0x{int(chist_cpu[b]):08x} sim=0x{int(chist_sim[b]):08x}")
            ok = False
        if int(chist_sim[255]) != total_filt:
            print(f"WARN: chist[255]={int(chist_sim[255])} vs filtered key count={total_filt}")
    else:
        print(f"(no ub log; CPU-only replay chist[255]={int(chist_cpu[255])})")

    # Per-tile summary
    print()
    print("=== Per-tile filtered key counts (CPU input semantics) ===")
    for i in range((keys.size + args.tile_cols - 1) // args.tile_cols):
        tile_meta = [m for m in meta if m["tile"] == i]
        nf = sum(m["n_filtered"] for m in tile_meta)
        print(f"  tile {i}: {len(tile_meta)} slice(s), MSB=={msb_filter} keys in tile = {nf}")

    if ok:
        print()
        print("PASS: Phase3 THISTOGRAM inputs (idxFilter + per-slice key replay) consistent with sim chistLSB.")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
