#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cumulative histogram on the high 8 bits (MSB) of uint16 keys from keys.bin.
C[b] = number of keys with (key >> 8) <= b, for b in 0..255.
Same convention as tests/npu/a5/.../thistogram/gen_data.py (np.cumsum of bincount).

Optional --split-tiles: same tiling as draft.cpp (kTileCols=2048, kLoop = ceil(N/2048)),
run bincount on MSB for each tile only and print each tile's 256-bin counts.

With --split-tiles --cumhist: print each tile's own ascending cumulative histogram
C_tile[b] = #(MSB <= b) within that tile only (still length 256; C_tile[255] = tile len).

With --split-tiles --hex-pair-only --cumhist: one line per 256-key tile, two hex u32:
C_tile[0] and C_tile[255] (full tile: second value is 0x00000100 == 256).

--input-head-hex N: print first N raw keys as uint16 hex (e.g. N=2 for first two keys).

--tile-input-head-hex N: with --split-tiles, for each tile print first N keys of that tile
as uint16 hex (e.g. N=2: keys at global indices [lo, lo+N) within the tile).

--chist-hex: final MSB cumulative chist C[0..255] over all keys, u32 hex per bin (draft chistMSB).
"""

import argparse
import os

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Ascending cumulative MSB histogram for keys.bin")
    ap.add_argument(
        "keys_bin",
        nargs="?",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input", "keys.bin"),
        help="Path to keys.bin (uint16 LE)",
    )
    ap.add_argument("--all", action="store_true", help="Print all 256 C[b] values (one per line)")
    ap.add_argument(
        "--split-tiles",
        action="store_true",
        help="Split keys into tiles of 256 (like draft kLoop), print per-tile MSB bincount",
    )
    ap.add_argument(
        "--tile-cols",
        type=int,
        default=256,
        help="Keys per tile (default 256, matches draft kTileCols)",
    )
    ap.add_argument(
        "--per-tile-full",
        action="store_true",
        help="With --split-tiles, print all 256 values per tile (counts or C_tile if --cumhist)",
    )
    ap.add_argument(
        "--cumhist",
        action="store_true",
        help="With --split-tiles, print per-tile cumsum(bincount) instead of raw counts",
    )
    ap.add_argument(
        "--hex-pair",
        action="store_true",
        help="With --split-tiles: per tile, also print 2 hex u32 (arr[0] and arr[255]; use with --cumhist for C_tile)",
    )
    ap.add_argument(
        "--hex-pair-only",
        action="store_true",
        help="With --split-tiles: only print one line per tile: tile_i 0x........ 0x........ (arr[0], arr[255])",
    )
    ap.add_argument(
        "--input-head-hex",
        type=int,
        metavar="N",
        default=0,
        help="Print first N keys from keys.bin as uint16 hex (0x + 4 hex digits each). Example: --input-head-hex 2",
    )
    ap.add_argument(
        "--tile-input-head-hex",
        type=int,
        metavar="N",
        default=0,
        help="With --split-tiles: each tile, print first N keys in that tile as uint16 hex (e.g. 2 = first two inputs per tile)",
    )
    ap.add_argument(
        "--chist-hex",
        action="store_true",
        help="Print final MSB cumulative C[0..255] as u32 hex (one line per bin: b 0x........); not for --split-tiles",
    )
    args = ap.parse_args()

    keys = np.fromfile(args.keys_bin, dtype=np.dtype("<u2"))
    if args.input_head_hex > 0:
        n = min(args.input_head_hex, keys.size)
        parts = " ".join(f"0x{int(keys[i]):04x}" for i in range(n))
        print(f"input keys[0:{n}] (uint16 LE): {parts}")
        print()
    msb = (keys.astype(np.uint32) >> 8) & 0xFF
    counts = np.bincount(msb, minlength=256).astype(np.uint64)
    C = np.cumsum(counts)

    if args.chist_hex:
        if args.split_tiles:
            raise SystemExit("error: --chist-hex is for full-input MSB chist only; omit --split-tiles")
        print(f"# MSB final chist (keys.bin={args.keys_bin}, N={keys.size})  C[b]=#(MSB<=b)  u32 hex")
        for b in range(256):
            print(f"{b:3d} 0x{int(C[b]):08x}")
        return

    if args.split_tiles:
        tc = args.tile_cols
        n = keys.size
        k_loop = (n + tc - 1) // tc
        hex_only = args.hex_pair_only
        if not hex_only:
            print("keys.bin:", args.keys_bin)
            print("N =", n, " tile_cols =", tc, " kLoop =", k_loop)
            if args.cumhist:
                print("Per tile: C_tile[b] = cumsum(bincount(MSB)), i.e. #(MSB<=b) within this tile only")
            else:
                print("Per tile: bincount(MSB byte in 0..255), length 256")
            if args.hex_pair:
                print("Hex pair per tile: arr[0] arr[255] (8 hex digits each, u32 semantics)")
            if args.tile_input_head_hex > 0:
                print(
                    f"Per tile: first {args.tile_input_head_hex} input keys (uint16 LE) "
                    f"as hex (global indices [lo, lo+N) within tile)"
                )
            print()
        for i in range(k_loop):
            lo = i * tc
            hi = min(lo + tc, n)
            chunk = keys[lo:hi]
            m = ((chunk.astype(np.uint32) >> 8) & 0xFF).astype(np.int64)
            tcount = np.bincount(m, minlength=256).astype(np.uint64)
            C_tile = np.cumsum(tcount)
            arr = C_tile if args.cumhist else tcount
            a0, a255 = int(arr[0]), int(arr[255])
            tin = 0
            if args.tile_input_head_hex > 0:
                tin = min(args.tile_input_head_hex, chunk.size)
            if hex_only or args.hex_pair:
                line = f"tile {i} 0x{a0:08x} 0x{a255:08x}"
                if tin > 0:
                    inh = " ".join(f"0x{int(chunk[j]):04x}" for j in range(tin))
                    line += f"  in[{lo}:{lo + tin}] {inh}"
                print(line)
            if hex_only:
                continue
            print(f"--- tile i={i}  key indices [{lo}, {hi})  len={hi - lo} ---")
            if tin > 0:
                inh = " ".join(f"0x{int(chunk[j]):04x}" for j in range(tin))
                print(f"  input head ({tin}): {inh}")
            print("  sum(tcount) =", int(tcount.sum()), "(should equal tile len)")
            label = "C_tile" if args.cumhist else "count"
            if args.per_tile_full:
                for b in range(256):
                    print(f"  {b} {int(arr[b])}")
            else:
                print(f"  {label}[0..15]:  ", " ".join(str(int(arr[j])) for j in range(16)))
                print(f"  {label}[240..255]:", " ".join(str(int(arr[j])) for j in range(240, 256)))
                if args.cumhist:
                    print("  C_tile[255] =", int(C_tile[255]))
            print()
        return

    print("keys.bin:", args.keys_bin)
    print("N =", keys.size)
    print("Per-bin count[i] = keys with MSB byte == i")
    print("C[i] = cumulative: count(MSB <= i) = sum_{j=0..i} count[j]")
    print()
    print("C[0] =", int(C[0]), "  (MSB==0 only:", int(counts[0]), ")")
    print("C[1] =", int(C[1]), "  (through MSB<=1)")
    print("C[255] =", int(C[255]))
    print()
    print("count[0..15]:", " ".join(str(int(counts[i])) for i in range(16)))
    print("C[0..15]:    ", " ".join(str(int(C[i])) for i in range(16)))
    print()
    print("count[240..255]:", " ".join(str(int(counts[i])) for i in range(240, 256)))
    print("C[240..255]:    ", " ".join(str(int(C[i])) for i in range(240, 256)))

    if args.all:
        print()
        print("=== all C[0..255] ===")
        for i in range(256):
            print(i, int(C[i]))


if __name__ == "__main__":
    main()
