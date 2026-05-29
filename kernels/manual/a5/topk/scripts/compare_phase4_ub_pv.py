#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-check Phase4 TCMPS/TSELS/TROWMIN: ub.wr_log vs rvec_pv.dump."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ADDR = re.compile(r"Address\s+([0-9a-fA-F]+)\s*=\s*\[([0-9a-fA-F]+)\]")
TICK_BRACKET = re.compile(r"\[info\]\s+\[(\d+)\]\s+")
TICK_PLAIN = re.compile(r"\[info\]\s+(\d+):")
WR_S = re.compile(r"Wr S(\d+):\s+(0x[0-9a-fA-F]+)")
VCMPS_GT = re.compile(r"\[info\]\s+\[(\d+)\].*RV_VCMPS_GT")
VEC_LINE = re.compile(r"^(Rd|Wr)\s+(V\d+|Pg|P\d+):")

UB = {
    "chistLSB": 0x18000,
    "remainK": 0x25000,
    "mask": 0x23000,
    "lsbWinnerLanes": 0x23000,  # same tile as mask in draft
    "rowMinDst": 0x24000,
    "lsbWinnerBin": 0x23C00,
}


def parse_ub_at_tick(path: Path, tick: int, window: int = 50) -> dict[int, int]:
    last: dict[int, int] = {}
    for line in path.open(errors="replace"):
        t = TICK_BRACKET.search(line)
        if not t:
            continue
        cur = int(t.group(1))
        if cur < tick - window or cur > tick + window:
            continue
        for m in ADDR.finditer(line):
            last[int(m.group(1), 16)] = int(m.group(2), 16)
    return last


def read_u32_vec(last: dict[int, int], ub_base: int, n: int = 64) -> list[int]:
    out = []
    for i in range(n):
        a = ub_base + i * 4
        out.append(last.get(a, 0))
    return out


def parse_vec_block(lines: list[str]) -> dict[str, list[int]]:
    vecs: dict[str, list[int]] = {}
    cur: str | None = None
    for line in lines:
        m = VEC_LINE.match(line.strip())
        if m:
            cur = m.group(2)
            vecs[cur] = []
            continue
        if cur and "Address" in line:
            for m2 in ADDR.finditer(line):
                vecs[cur].append(int(m2.group(2), 16))
    return vecs


def extract_vcmps_gt_events(pv_path: Path, tick_lo: int, tick_hi: int) -> list[dict]:
    events = []
    buf: list[str] = []
    cur_tick: int | None = None
    cur_s: dict[str, int] = {}
    for line in pv_path.open(errors="replace"):
        tb = TICK_BRACKET.search(line)
        if tb:
            cur_tick = int(tb.group(1))
        else:
            tp = TICK_PLAIN.search(line)
            if tp:
                cur_tick = int(tp.group(1))
        ws = WR_S.search(line)
        if ws and cur_tick is not None:
            cur_s[f"S{ws.group(1)}"] = int(ws.group(2), 16)
        if VCMPS_GT.search(line) and cur_tick is not None and tick_lo <= cur_tick <= tick_hi:
            events.append({"tick": cur_tick, "s": dict(cur_s), "lines": list(buf)})
            buf = []
        if cur_tick is not None and tick_lo <= cur_tick <= tick_hi:
            buf.append(line)
    return events


def cpu_lsb_gt_mask(remain_k: int = 12) -> tuple[np.ndarray, int]:
    keys = np.fromfile(Path(__file__).resolve().parent.parent / "input" / "keys.bin", dtype="<u2")
    msb = ((keys >> 8) & 0xFF) == 240
    lsb = keys[msb] & 0xFF
    c = np.cumsum(np.bincount(lsb.astype(np.int64), minlength=256))
    winner = next(b for b in range(256) if c[b] > remain_k)
    return c > remain_k, winner


def main() -> int:
    topk = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, default=topk / "build")
    ap.add_argument("--tick-lo", type=int, default=22000)
    ap.add_argument("--tick-hi", type=int, default=24000)
    args = ap.parse_args()

    ub = args.build / "core0.veccore0.ub.wr_log.dump"
    pv = args.build / "core0.veccore0.rvec_pv.dump"
    if not ub.is_file() or not pv.is_file():
        print("missing ub or pv dump", file=sys.stderr)
        return 1

    gt_cpu, winner_cpu = cpu_lsb_gt_mask()
    print(f"CPU: remain_k=12, lsb_winner={winner_cpu}, bins with C>12: {int(gt_cpu.sum())}")
    print()

    events = extract_vcmps_gt_events(pv, args.tick_lo, args.tick_hi)
    print(f"RV_VCMPS_GT in tick [{args.tick_lo},{args.tick_hi}]: {len(events)} events")
    for ev in events[:6]:
        t = ev["tick"]
        print(f"\n=== tick {t} RV_VCMPS_GT ===")
        print("  scalar regs:", {k: hex(v) for k, v in sorted(ev["s"].items()) if k.startswith("S")})
        vecs = parse_vec_block(ev["lines"])
        v0 = vecs.get("V0", [])[:8]
        v1 = vecs.get("V1", [])[:8]
        print(f"  pv V0[:8] (Rd chist?) = {[hex(x) for x in v0]}")
        print(f"  pv V1[:8] (Rd remainK?) = {[hex(x) for x in v1]}")
        ubmap = parse_ub_at_tick(ub, t)
        chist = read_u32_vec(ubmap, UB["chistLSB"], 8)
        rk = read_u32_vec(ubmap, UB["remainK"], 4)
        print(f"  ub chistLSB[:8] @ {UB['chistLSB']:#x} = {chist}")
        print(f"  ub remainK[:4] @ {UB['remainK']:#x} = {rk}")
        if v0 and chist[0] != v0[0]:
            print(f"  MISMATCH: pv V0[0]={v0[0]} vs ub chist[0]={chist[0]}")
        if rk and rk[0] != 12:
            print(f"  NOTE: ub remainK[0]={rk[0]} (expect 12)")

    # Phase4 TROWMIN gather pollution check
    print("\n=== Phase4 scratch (ub, tick window) ===")
    for name, base in [("rowMinDst", UB["rowMinDst"]), ("lsbWinnerBin", UB["lsbWinnerBin"])]:
        for t in [22718, 22761, 7753, 7963]:
            ubmap = parse_ub_at_tick(ub, t, 5)
            v = ubmap.get(base, None)
            if v is not None:
                print(f"  tick {t} {name} @ {base:#x} = {v} (0x{v:08x})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
