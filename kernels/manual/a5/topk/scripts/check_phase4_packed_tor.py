#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify Phase4 packed threshold: TOR(outU, hiU, lsbU) @ kRemainUbOut (0x25000).

draft.cpp: TCVT(msbU<-msbWinnerSaved); TSHLS(hiU, msbU, 8); TCVT(lsbU<-lsbWinnerBin); TOR(outU, hiU, lsbU).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ADDR_PAT = re.compile(r"Address\s+([0-9a-fA-F]+)\s*=\s*\[([0-9a-fA-F]+)\]")
TICK_PAT = re.compile(r"\[info\]\s+\[(\d+)\]\s+")

UB_OUT = 0x25000
UB_MSB_U = 0x24E00
UB_HI_U = 0x24F80
UB_LSB_U = 0x24F00
UB_LSB_WIN = 0x23C00


def lane0_history(path: Path, addr: int) -> list[tuple[int, int]]:
    hist: list[tuple[int, int]] = []
    cur: int | None = None
    with path.open(errors="replace") as f:
        for line in f:
            tm = TICK_PAT.search(line)
            if tm:
                cur = int(tm.group(1))
            for m in ADDR_PAT.finditer(line):
                if int(m.group(1), 16) == addr and cur is not None:
                    hist.append((cur, int(m.group(2), 16)))
    return hist


def cpu_packed(keys: np.ndarray, topk: int) -> dict:
    n = keys.size
    msb = ((keys.astype(np.uint32) >> 8) & 0xFF).astype(np.int64)
    lsb = (keys.astype(np.uint32) & 0xFF).astype(np.int64)
    c_msb = np.cumsum(np.bincount(msb, minlength=256))
    thr = n - topk
    msb_w = next(b for b in range(256) if c_msb[b] >= thr)
    remain_k = thr - int(c_msb[msb_w - 1] if msb_w > 0 else 0)
    mask = msb == msb_w
    c_lsb = np.cumsum(np.bincount(lsb[mask], minlength=256))
    lsb_w = next(b for b in range(256) if c_lsb[b] > remain_k)
    packed = ((msb_w & 0xFF) << 8) | (lsb_w & 0xFF)
    return {
        "msb_w": msb_w,
        "remain_k": remain_k,
        "lsb_w": lsb_w,
        "packed": int(packed),
        "hi_u": int((msb_w & 0xFF) << 8),
        "lsb_u": int(lsb_w & 0xFF),
    }


def pick_phase4(hist: list[tuple[int, int]], after_tick: int = 5000) -> int | None:
    vals = [v for t, v in hist if t >= after_tick]
    return vals[-1] if vals else None


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
    args = ap.parse_args()

    if not args.keys.is_file() or not args.ub_log.is_file():
        print("error: missing keys.bin or ub dump", file=sys.stderr)
        return 1

    keys = np.fromfile(args.keys, dtype=np.dtype("<u2"))
    cpu = cpu_packed(keys, args.topk)

    h_out = lane0_history(args.ub_log, UB_OUT)
    h_msb = lane0_history(args.ub_log, UB_MSB_U)
    h_hi = lane0_history(args.ub_log, UB_HI_U)
    h_lsb = lane0_history(args.ub_log, UB_LSB_U)
    h_lwin = lane0_history(args.ub_log, UB_LSB_WIN)

    sim_msb = pick_phase4(h_msb)
    sim_hi = pick_phase4(h_hi)
    sim_lsb = pick_phase4(h_lsb)
    sim_lwin = pick_phase4(h_lwin)
    sim_packed = pick_phase4(h_out)
    if sim_packed is not None and sim_packed > 0xFFFF:
        sim_packed &= 0xFFFF

    tor_from_parts = None
    if sim_hi is not None and sim_lsb is not None:
        tor_from_parts = (sim_hi & 0xFFFF) | (sim_lsb & 0xFFFF)

    print(f"keys: {args.keys}  ub: {args.ub_log}")
    print()
    print("=== CPU (Phase4 semantics: MSB raw find, LSB min b with C_lsb[b] > remain_k) ===")
    print(f"  msb byte = {cpu['msb_w']} (0x{cpu['msb_w']:02x})")
    print(f"  remain_k = {cpu['remain_k']}")
    print(f"  lsb byte = {cpu['lsb_w']} (0x{cpu['lsb_w']:02x})")
    print(f"  hi_u = msb<<8 = 0x{cpu['hi_u']:04x}")
    print(f"  packed = TOR(hi|lsb) = 0x{cpu['packed']:04x} ({cpu['packed']})")
    print()
    print("=== Sim UB (Phase4 window, tick >= 5000) ===")
    print(f"  msbU @ {UB_MSB_U:#x}     = {sim_msb}")
    print(f"  hiU @ {UB_HI_U:#x}     = {sim_hi} (0x{sim_hi:04x})" if sim_hi is not None else "  hiU missing")
    print(f"  lsbU @ {UB_LSB_U:#x}     = {sim_lsb}")
    print(f"  lsbWinnerBin @ {UB_LSB_WIN:#x} = {sim_lwin}")
    print(f"  outU/packed @ {UB_OUT:#x} = {sim_packed} (0x{sim_packed:04x})" if sim_packed else "  packed missing")
    if tor_from_parts is not None:
        print(f"  hi|lsb recomputed     = 0x{tor_from_parts:04x}")

    ok = True
    if sim_packed is None or int(sim_packed) != cpu["packed"]:
        sp = f"0x{int(sim_packed):04x}" if sim_packed is not None else "None"
        print(f"FAIL packed: sim={sp} cpu=0x{cpu['packed']:04x}")
        ok = False
    if sim_lsb is not None and int(sim_lsb & 0xFF) != cpu["lsb_u"]:
        print(f"FAIL lsb byte before TOR: sim={sim_lsb & 0xFF} cpu={cpu['lsb_u']}")
        ok = False
    if sim_hi is not None and int(sim_hi & 0xFFFF) != cpu["hi_u"]:
        print(f"FAIL hi byte before TOR: sim=0x{sim_hi:04x} cpu=0x{cpu['hi_u']:04x}")
        ok = False
    if tor_from_parts is not None and sim_packed is not None and int(tor_from_parts) != int(sim_packed):
        print("FAIL TOR: outU != (hiU | lsbU) — TOR opcode/layout issue")
        ok = False
    elif tor_from_parts is not None and sim_packed is not None:
        print("TOR opcode: outU == (hiU | lsbU)  OK")

    if ok:
        print()
        print("PASS: Phase4 packed threshold (TOR) matches CPU.")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
