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

"""Parse A5 sim dumps under build/ and emit phase-level VF / PMU summary JSON."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

CHUNK_COLS = 256


def parse_vf(instr_log: Path) -> list[dict]:
    pat = re.compile(
        r"\[(\d+)\].*?\(PC: (0x[0-9a-f]+)\) PUSHQ.*?ID: (\d+)\) VF\s*, "
        r"vf_execute_time: (\d+), vf_real_execute_time: (\d+)"
    )
    return [
        {
            "retire_cycle": int(m[1]),
            "push_pc": m[2],
            "id": int(m[3]),
            "vf_execute_time": int(m[4]),
            "vf_real_execute_time": int(m[5]),
        }
        for m in pat.finditer(instr_log.read_text(errors="replace"))
    ]


def parse_summary(summary_log: Path) -> dict:
    text = summary_log.read_text(errors="replace")
    out: dict[str, int] = {}
    for key, pat in [
        ("kernel_ticks", r"kernal total ticks : (\d+)"),
        ("mte2_busy", r"mte2_veccore0_su_busy_cycle\s+\|\s+(\d+)"),
        ("rvec_busy", r"rvec_veccore0_busy_cycle\s+\|\s+(\d+)"),
        ("scalar_busy", r"CCU.scalar_veccore0_su_busy_cycle\s+\|\s+(\d+)"),
    ]:
        m = re.search(pat, text)
        if m:
            out[key] = int(m.group(1))
    return out


def classify_pcs(vfs: list[dict], n_gather_slices: int, n_hist_slices: int) -> dict[str, str]:
    """Map push_pc -> phase label using per-PC VF counts (relink-stable heuristics)."""
    counts: dict[str, int] = defaultdict(int)
    real_sum: dict[str, int] = defaultdict(int)
    min_retire: dict[str, int] = {}
    for v in vfs:
        pc = v["push_pc"]
        counts[pc] += 1
        real_sum[pc] += v["vf_real_execute_time"]
        min_retire[pc] = min(min_retire.get(pc, v["retire_cycle"]), v["retire_cycle"])

    init_pc = min(min_retire, key=min_retire.get)
    labels: dict[str, str] = {}
    for pc, cnt in counts.items():
        if pc == init_pc and cnt == 1:
            labels[pc] = "phase1_init"
        elif cnt == n_gather_slices:
            labels[pc] = "phase5_tile"
        else:
            per_tile = real_sum[pc] / cnt if cnt else 0
            # THISTOGRAM: ~45–97 vf_real per 256-col chunk; wide 2048-col hist is higher but still < Phase5.
            labels[pc] = "phase1_3_hist_tile" if per_tile < 120 else "phase2_4_ctrl"
    return labels


def build_report(
    build_dir: Path,
    seed: int | None,
    n: int,
    topk: int,
    tile_cols: int = 2048,
    chunk_cols: int = CHUNK_COLS,
    hist_chunk_cols: int = 2048,
) -> dict:
    instr = build_dir / "core0.veccore0.instr_log.dump"
    summary = build_dir / "core0_summary_log"
    if not instr.is_file():
        raise FileNotFoundError(instr)
    vfs = parse_vf(instr)
    pmu = parse_summary(summary) if summary.is_file() else {}
    n_tiles = (n + tile_cols - 1) // tile_cols
    chunks_gather = (tile_cols + chunk_cols - 1) // chunk_cols
    chunks_hist = (tile_cols + hist_chunk_cols - 1) // hist_chunk_cols
    n_gather_slices = n_tiles * chunks_gather
    n_hist_slices = n_tiles * chunks_hist
    pc_labels = classify_pcs(vfs, n_gather_slices, n_hist_slices)

    by_phase: dict[str, dict] = defaultdict(
        lambda: {"vf_count": 0, "vf_execute_time": 0, "vf_real_execute_time": 0}
    )
    for v in vfs:
        ph = pc_labels[v["push_pc"]]
        by_phase[ph]["vf_count"] += 1
        by_phase[ph]["vf_execute_time"] += v["vf_execute_time"]
        by_phase[ph]["vf_real_execute_time"] += v["vf_real_execute_time"]

    p5_detail = []
    for pc in sorted(pc for pc, lab in pc_labels.items() if lab == "phase5_tile"):
        lst = [v for v in vfs if v["push_pc"] == pc]
        n_tile = len(lst)
        se = sum(v["vf_execute_time"] for v in lst)
        sr = sum(v["vf_real_execute_time"] for v in lst)
        p5_detail.append(
            {
                "push_pc": pc,
                "vf_count": n_tile,
                "vf_execute_time_total": se,
                "vf_real_execute_time_total": sr,
                "vf_real_per_tile": round(sr / n_tile, 1) if n_tile else 0,
            }
        )
    p5_detail.sort(key=lambda x: -x["vf_real_execute_time_total"])

    rvec = pmu.get("rvec_busy") or sum(v["vf_real_execute_time"] for v in vfs)
    kt = pmu.get("kernel_ticks", 0)

    phases = []
    order = ["phase1_init", "phase1_3_hist_tile", "phase2_4_ctrl", "phase5_tile"]
    report_p5_gt = sum(
        x["vf_real_execute_time_total"]
        for x in p5_detail[:4]
    )
    report_p5_eq = sum(
        x["vf_real_execute_time_total"]
        for x in p5_detail[4:]
    )
    for ph in order:
        if ph not in by_phase:
            continue
        d = by_phase[ph]
        sr = d["vf_real_execute_time"]
        phases.append(
            {
                "phase": ph,
                "vf_count": d["vf_count"],
                "vf_execute_time": d["vf_execute_time"],
                "vf_real_execute_time": sr,
                "pct_of_rvec_real": round(100.0 * sr / rvec, 2) if rvec else 0,
                "pct_of_kernel_ticks": round(100.0 * sr / kt, 2) if kt else 0,
            }
        )

    return {
        "case": {
            "n": n,
            "topk": topk,
            "tile_cols": tile_cols,
            "chunk_cols": chunk_cols,
            "hist_chunk_cols": hist_chunk_cols,
            "num_tiles": n_tiles,
            "num_gather_slices_per_pass": n_gather_slices,
            "num_hist_slices_per_pass": n_hist_slices,
            "seed": seed,
            "soc": "Ascend950PR_9599",
            "core": "core0.veccore0",
        },
        "pmu": pmu,
        "vf_total": len(vfs),
        "vf_execute_time_sum": sum(v["vf_execute_time"] for v in vfs),
        "vf_real_execute_time_sum": sum(v["vf_real_execute_time"] for v in vfs),
        "icache_overhead_cycles": sum(v["vf_execute_time"] - v["vf_real_execute_time"] for v in vfs),
        "phases": phases,
        "phase5_vf_templates": p5_detail,
        "phase5_gt_cluster_vf_real": report_p5_gt,
        "phase5_eq_cluster_vf_real": report_p5_eq,
        "opprof_dir": None,
        "notes": [
            "vf_real_execute_time matches rvec_veccore0_busy_cycle in core0_summary_log",
            "mte2_busy dominates wall time; phase vf_real is SIMD-only",
            "draft.cpp: explicit PIPE_V/PIPE_S fences removed after Phase2 TSUB and Phase4 TOR",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--seed", type=int, default=1241200609)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--tile-cols", type=int, default=2048)
    parser.add_argument("--chunk-cols", type=int, default=CHUNK_COLS)
    parser.add_argument("--hist-chunk-cols", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--opprof-dir", type=Path, default=None)
    args = parser.parse_args()

    report = build_report(
        args.build_dir.resolve(),
        args.seed,
        args.n,
        args.topk,
        tile_cols=args.tile_cols,
        chunk_cols=args.chunk_cols,
        hist_chunk_cols=args.hist_chunk_cols,
    )
    if args.opprof_dir is not None:
        report["opprof_dir"] = str(args.opprof_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
