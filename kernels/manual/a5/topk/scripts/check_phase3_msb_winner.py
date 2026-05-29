#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify Phase3 MSB filter winner: msbWinnerSaved @ 0x24D80 and idxFilter @ 0x1C000.

draft.cpp Phase3: TCVT(idxFilter, msbWinnerSaved) — raw MSB find bin (TROWMIN, no -1).
Phase2 msbWinnerBin @ 0x23C00 is (find-1) for TGATHER(chistMSB); not used in Phase3.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ADDR_PAT = re.compile(r"Address\s+([0-9a-fA-F]+)\s*=\s*\[([0-9a-fA-F]+)\]")
TICK_PAT = re.compile(r"\[info\]\s+\[(\d+)\]\s+")

N_DEFAULT = 8192
TOPK_DEFAULT = 512

# UB addresses from draft.cpp
UB_MSB_WINNER_SAVED = 0x24D80  # WinnerBinTile 1x32 u32
UB_MSB_WINNER_BIN = 0x23C00  # msbWinnerBin broadcast (find-1 path)
UB_IDX_FILTER = 0x1C000  # IdxFilterTile ColMajor; valid 1x1 u8
UB_REMAIN_K = 0x25000  # remainKTile lane0


def parse_ub_writes(path: Path, base: int, size: int) -> tuple[dict[int, int], dict[int, list[tuple[int, int]]]]:
    """Last value per address + full (tick, value) history for lane0 (base)."""
    last_val: dict[int, int] = {}
    history: dict[int, list[tuple[int, int]]] = {}
    cur_tick: int | None = None
    hi = base + size
    with path.open(errors="replace") as f:
        for line in f:
            tm = TICK_PAT.search(line)
            if tm:
                cur_tick = int(tm.group(1))
            for m in ADDR_PAT.finditer(line):
                a = int(m.group(1), 16)
                if base <= a < hi:
                    v = int(m.group(2), 16)
                    last_val[a] = v
                    if a == base and cur_tick is not None:
                        history.setdefault(a, []).append((cur_tick, v))
    return last_val, history


def read_u32_lane(last_val: dict[int, int], base: int, lane: int = 0) -> int | None:
    a = base + lane * 4
    return last_val.get(a)


def cpu_msb_winners(keys: np.ndarray, topk: int) -> dict:
    n = keys.size
    msb = ((keys.astype(np.uint32) >> 8) & 0xFF).astype(np.int64)
    counts = np.bincount(msb, minlength=256).astype(np.uint64)
    c_msb = np.cumsum(counts)
    thr = n - topk

    msb_find = 0
    for b in range(256):
        if c_msb[b] >= thr:
            msb_find = b
            break

    # msbWinnerBin path: TROWMIN find, TSUB(1), wrap if find==0
    if msb_find == 0:
        msb_bin = 0
    else:
        msb_bin = msb_find - 1

    cw = int(c_msb[msb_bin])
    remain_k = thr - cw

    # Phase3 filter count sanity
    n_lsb = int(np.sum(msb == msb_find))

    return {
        "thr_msb": thr,
        "msb_find_raw": msb_find,
        "msb_winner_bin_idx": msb_bin,
        "C_at_bin_idx": cw,
        "remain_k": remain_k,
        "n_keys_msb_eq_find": n_lsb,
    }


def main() -> int:
    topk_dir = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keys", type=Path, default=topk_dir / "input" / "keys.bin")
    ap.add_argument(
        "--ub-log",
        type=Path,
        default=topk_dir / "build" / "core0.veccore0.ub.wr_log.dump",
    )
    ap.add_argument("--topk", type=int, default=TOPK_DEFAULT)
    args = ap.parse_args()

    if not args.keys.is_file():
        print(f"error: missing {args.keys}", file=sys.stderr)
        return 1
    if not args.ub_log.is_file():
        print(f"error: missing {args.ub_log}", file=sys.stderr)
        return 1

    keys = np.fromfile(args.keys, dtype=np.dtype("<u2"))
    cpu = cpu_msb_winners(keys, args.topk)

    lv_saved, hist_saved = parse_ub_writes(args.ub_log, UB_MSB_WINNER_SAVED, 32 * 4)
    lv_bin, hist_bin = parse_ub_writes(args.ub_log, UB_MSB_WINNER_BIN, 32 * 4)
    lv_idx, hist_idx = parse_ub_writes(args.ub_log, UB_IDX_FILTER, 64)
    lv_rk, hist_rk = parse_ub_writes(args.ub_log, UB_REMAIN_K, 32 * 4)

    sim_saved = read_u32_lane(lv_saved, UB_MSB_WINNER_SAVED, 0)
    bin_hist = hist_bin.get(UB_MSB_WINNER_BIN, [])
    sim_bin = next((v for _t, v in bin_hist if v == cpu["msb_winner_bin_idx"]), None)

    rk_hist = hist_rk.get(UB_REMAIN_K, [])
    sim_remain = next((v for _t, v in rk_hist if v == cpu["remain_k"]), None)

    # idxFilter: ColMajor; lane0 byte at 0x1C000
    sim_idx_byte = lv_idx.get(UB_IDX_FILTER)
    if sim_idx_byte is not None:
        sim_idx_byte = sim_idx_byte & 0xFF

    print(f"keys: {args.keys}  N={keys.size}  TopK={args.topk}")
    print(f"ub:   {args.ub_log}")
    print()
    print("=== CPU (draft semantics) ===")
    print(f"  msb_find_raw (Phase3 idxFilter / msbWinnerSaved) = {cpu['msb_find_raw']} (0x{cpu['msb_find_raw']:02x})")
    print(f"  msb_winner_bin_idx (TGATHER chistMSB, find-1)     = {cpu['msb_winner_bin_idx']} (0x{cpu['msb_winner_bin_idx']:02x})")
    print(f"  C_msb[winner_bin_idx]                           = {cpu['C_at_bin_idx']}")
    print(f"  remain_k = thr - C[winner_bin]                    = {cpu['remain_k']}")
    print(f"  keys with MSB == msb_find_raw                     = {cpu['n_keys_msb_eq_find']}")
    print()
    print("=== Sim UB (last write) ===")
    print(f"  msbWinnerSaved[0] @ {UB_MSB_WINNER_SAVED:#x}  = {sim_saved}")
    print(f"  msbWinnerBin[0]   @ {UB_MSB_WINNER_BIN:#x}  = {sim_bin}  (Phase2; final UB may be stale)")
    print(f"  idxFilter byte    @ {UB_IDX_FILTER:#x}  = {sim_idx_byte}  (Phase3 TCVT)")
    print(f"  remainKTile[0]    @ {UB_REMAIN_K:#x}  = {sim_remain}  (Phase2; 0x25000 reused for packedThr)")
    if hist_idx.get(UB_IDX_FILTER):
        t_idx = hist_idx[UB_IDX_FILTER][-1][0]
        print(f"  idxFilter write tick = {t_idx}")

    ok = True
    if sim_saved is None or int(sim_saved) != cpu["msb_find_raw"]:
        print(f"FAIL: msbWinnerSaved sim={sim_saved} cpu={cpu['msb_find_raw']}")
        ok = False
    phase3_ok = sim_saved is not None and int(sim_saved) == cpu["msb_find_raw"]
    phase3_ok = phase3_ok and sim_idx_byte is not None and int(sim_idx_byte) == cpu["msb_find_raw"]
    if not phase3_ok:
        print(f"FAIL Phase3: msbWinnerSaved={sim_saved} idxFilter={sim_idx_byte} cpu={cpu['msb_find_raw']}")
        ok = False
    if sim_bin is None or int(sim_bin) != cpu["msb_winner_bin_idx"]:
        print(f"WARN: msbWinnerBin Phase2 sim={sim_bin} cpu={cpu['msb_winner_bin_idx']} (check hist if stale)")
    if sim_remain is None or int(sim_remain) != cpu["remain_k"]:
        print(f"WARN: remainK Phase2 sim={sim_remain} cpu={cpu['remain_k']} (0x25000 overwritten in Phase4)")

    if ok and phase3_ok:
        print()
        print("PASS: Phase3 MSB winner (msbWinnerSaved / idxFilter) matches CPU.")
    return 0 if (ok and phase3_ok) else 2


if __name__ == "__main__":
    raise SystemExit(main())
