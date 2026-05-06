#!/usr/bin/env python3
# coding=utf-8
"""Validate VOR threshold using VNOT input from rvec_pv dump.

Flow:
1) Extract BF16 raw inputs from RV_VNOT `Rd Vn` blocks in rvec_pv dump.
2) Recompute ordered-key radix threshold (MSB/LSB winner).
3) Extract RV_VOR output threshold from the same dump.
4) Compare per batch and report pass/fail.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


VNOT_MARKER = "RV_VNOT"
VOR_MARKER = "RV_VOR"
RD_VN_MARKER = "Rd Vn:"
WR_VD_MARKER = "Wr Vd:"
ADDR_WORD_RE = re.compile(r"\[([0-9a-fA-F]{8})\]")


def winner_bin_min_ge(cum_hist: np.ndarray, thr: int) -> int:
    for b in range(256):
        if int(cum_hist[b]) >= thr:
            return b
    return 255


def winner_bin_min_gt(cum_hist: np.ndarray, thr: int) -> int:
    for b in range(256):
        if int(cum_hist[b]) > thr:
            return b
    return 255


def ordered_bf16_key(bits_u16: np.ndarray) -> np.ndarray:
    bits = bits_u16.astype(np.uint16)
    sign = (bits & np.uint16(0x8000)) != 0
    out = np.empty_like(bits, dtype=np.uint16)
    out[sign] = np.uint16((~bits[sign]) & np.uint16(0xFFFF))
    out[~sign] = np.uint16(bits[~sign] ^ np.uint16(0x8000))
    return out


def words_to_u16_list(words: list[str]) -> list[int]:
    out: list[int] = []
    for w in words:
        v = int(w, 16)
        # log words are displayed as 32-bit; keep little-endian half-word order
        out.append(v & 0xFFFF)
        out.append((v >> 16) & 0xFFFF)
    return out


def extract_u16_blocks(lines: list[str], marker: str, value_header: str) -> list[np.ndarray]:
    blocks: list[np.ndarray] = []
    i = 0
    n = len(lines)
    while i < n:
        if marker in lines[i]:
            j = i + 1
            while j < n and value_header not in lines[j]:
                j += 1
            if j >= n:
                break
            j += 1
            words: list[str] = []
            while j < n:
                if lines[j].startswith("[info] [") and words:
                    break
                words.extend(ADDR_WORD_RE.findall(lines[j]))
                j += 1
            if words:
                blocks.append(np.array(words_to_u16_list(words), dtype=np.uint16))
            i = j
            continue
        i += 1
    return blocks


def compute_thresholds_from_vnot(vnot_u16: np.ndarray, n: int, topk: int) -> list[int]:
    if vnot_u16.size % n != 0:
        raise ValueError(f"VNOT input size {vnot_u16.size} is not divisible by N={n}")
    bs = vnot_u16.size // n
    out: list[int] = []
    for b in range(bs):
        bits = vnot_u16[b * n : (b + 1) * n]
        keys = ordered_bf16_key(bits)
        thr_msb = n - topk
        msb = (keys.astype(np.uint32) >> 8) & 0xFF
        lsb = keys.astype(np.uint32) & 0xFF
        counts_msb = np.bincount(msb, minlength=256).astype(np.uint64)
        c_msb = np.cumsum(counts_msb).astype(np.uint64)
        msb_winner = winner_bin_min_ge(c_msb, thr_msb)
        c_below = int(c_msb[msb_winner - 1]) if msb_winner > 0 else 0
        remain_k = int(np.uint32(thr_msb - c_below))
        lsb_keys = lsb[msb == msb_winner]
        counts_lsb = np.bincount(lsb_keys.astype(np.int64), minlength=256).astype(np.uint64)
        c_lsb = np.cumsum(counts_lsb).astype(np.uint64)
        lsb_winner = winner_bin_min_gt(c_lsb, remain_k)
        out.append(int((msb_winner << 8) | lsb_winner))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Check VNOT radix threshold against VOR output.")
    parser.add_argument(
        "--dump",
        default="build/core0.veccore0.rvec_pv.dump",
        help="Path to rvec_pv dump file",
    )
    parser.add_argument("--n", type=int, default=2048, help="TopK input length N")
    parser.add_argument("--topk", type=int, default=512, help="TopK K")
    args = parser.parse_args()

    dump_path = Path(args.dump)
    if not dump_path.is_file():
        print(f"Dump not found: {dump_path}", file=sys.stderr)
        return 1
    if args.topk <= 0 or args.topk > args.n:
        print(f"Invalid K={args.topk}, must be in [1, N]", file=sys.stderr)
        return 1

    lines = dump_path.read_text(errors="ignore").splitlines()
    vnot_blocks = extract_u16_blocks(lines, VNOT_MARKER, RD_VN_MARKER)
    vor_blocks = extract_u16_blocks(lines, VOR_MARKER, WR_VD_MARKER)

    if not vnot_blocks:
        print("No VNOT blocks found.", file=sys.stderr)
        return 1
    if not vor_blocks:
        print("No VOR blocks found.", file=sys.stderr)
        return 1

    # One VNOT block contains 128 BF16 values (256 bytes). N=2048 => 16 blocks per batch.
    vals_per_vnot_block = int(vnot_blocks[0].size)
    if vals_per_vnot_block == 0:
        print("Empty VNOT blocks.", file=sys.stderr)
        return 1
    blocks_per_batch = args.n // vals_per_vnot_block
    if blocks_per_batch * vals_per_vnot_block != args.n:
        print(
            f"N={args.n} is not aligned to VNOT block size={vals_per_vnot_block}.",
            file=sys.stderr,
        )
        return 1
    if len(vnot_blocks) % blocks_per_batch != 0:
        print(
            f"VNOT blocks={len(vnot_blocks)} not divisible by blocks_per_batch={blocks_per_batch}.",
            file=sys.stderr,
        )
        return 1

    bs_vnot = len(vnot_blocks) // blocks_per_batch
    vnot_all = np.concatenate(vnot_blocks, axis=0)
    calc_thr = compute_thresholds_from_vnot(vnot_all, args.n, args.topk)

    # Each VOR `Wr Vd` holds repeated threshold in lane0 low16.
    vor_thr: list[int] = [int(block[0] & 0xFFFF) for block in vor_blocks]
    bs = min(bs_vnot, len(vor_thr))

    print("=== VNOT -> RadixTopK -> VOR Check ===")
    print(f"dump={dump_path}")
    print(f"N={args.n}, K={args.topk}, vnot_blocks={len(vnot_blocks)}, vor_blocks={len(vor_blocks)}")
    print(f"batch_from_vnot={bs_vnot}, compare_batches={bs}")

    all_ok = True
    for b in range(bs):
        ok = calc_thr[b] == vor_thr[b]
        all_ok = all_ok and ok
        tag = "OK" if ok else "MISMATCH"
        print(
            f"[b={b}] calc_threshold=0x{calc_thr[b]:04X}, vor_threshold=0x{vor_thr[b]:04X} -> {tag}"
        )

    if bs_vnot != len(vor_thr):
        print(
            f"Warning: batch count differs (from VNOT={bs_vnot}, from VOR={len(vor_thr)}).",
            file=sys.stderr,
        )
        all_ok = False

    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

