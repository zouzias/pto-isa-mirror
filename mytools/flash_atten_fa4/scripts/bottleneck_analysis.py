#!/usr/bin/env python3
# --------------------------------------------------------------------------------
# Author: ywangmu from HKUST
# Bottleneck analysis script for Flash Attention pipeline simulation logs.
#
# Parses the timeline CSV produced by pipeline_log_analysis.py and computes:
#   1. Overall timing and pipeline utilization breakdown
#   2. Per-MMAD instruction timing and inter-MMAD gaps
#   3. Per-tile steady-state breakdown (QK MMAD, UB stall, QK TMOV, SM stall, PV MMAD, PV TMOV)
#   4. Cross-core sync stall analysis (INTRA_BLOCK vs SET/WAIT_FLAG)
#   5. Vec compute density per time window
#   6. Optimization suggestions
#
# Usage:
#   python3 bottleneck_analysis.py --csv timeline.csv [--window 500]
#
# Typical invocation (after running run_timeline.sh):
#   cd scripts && python3 bottleneck_analysis.py --csv timeline.csv
# --------------------------------------------------------------------------------

import argparse
import csv
import sys
from collections import Counter, defaultdict


def load_timeline(csv_path: str) -> list:
    """Load timeline CSV into a list of dicts with parsed integer timestamps."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["ts_start"] = int(r["ts_start"])
            r["ts_end"] = int(r["ts_end"])
            r["duration"] = r["ts_end"] - r["ts_start"]
            rows.append(r)
    return rows


def print_section(title: str, width: int = 70):
    print()
    print("=" * width)
    print(title)
    print("=" * width)


def overall_timing(rows: list):
    """Print overall kernel timing."""
    cube_rows = [r for r in rows if r["core"] == "cube"]
    vec_rows = [r for r in rows if r["core"] == "vector"]

    cube_start = min(r["ts_start"] for r in cube_rows)
    cube_end = max(r["ts_end"] for r in cube_rows)
    vec_start = min(r["ts_start"] for r in vec_rows)
    vec_end = max(r["ts_end"] for r in vec_rows)
    overall_start = min(cube_start, vec_start)
    overall_end = max(cube_end, vec_end)
    total_span = overall_end - overall_start

    print_section("OVERALL TIMING")
    print(f"Cube:   [{cube_start}, {cube_end}], span = {cube_end - cube_start} cycles")
    print(f"Vector: [{vec_start}, {vec_end}], span = {vec_end - vec_start} cycles")
    print(f"Total kernel: [{overall_start}, {overall_end}], span = {total_span} cycles")
    return total_span


def utilization_breakdown(rows: list, total_span: int):
    """Print per-core op_class utilization breakdown."""
    print_section("PIPELINE UTILIZATION BREAKDOWN")
    print(f"Total wall time: {total_span} cycles")

    for core_name in ("cube", "vector"):
        core_rows = [r for r in rows if r["core"] == core_name]
        by_class = defaultdict(lambda: {"count": 0, "total_dur": 0, "max_dur": 0})
        for r in core_rows:
            cls = r["op_class"]
            by_class[cls]["count"] += 1
            by_class[cls]["total_dur"] += r["duration"]
            by_class[cls]["max_dur"] = max(by_class[cls]["max_dur"], r["duration"])

        total_instr = sum(d["total_dur"] for d in by_class.values())
        print(f"\n--- {core_name} ({total_instr} total instruction-cycles) ---")
        for cls in sorted(by_class, key=lambda c: -by_class[c]["total_dur"]):
            d = by_class[cls]
            pct = 100 * d["total_dur"] / total_instr if total_instr else 0
            avg = d["total_dur"] / d["count"] if d["count"] else 0
            print(
                f"  {cls:12s}: count={d['count']:5d}, total={d['total_dur']:8d} ({pct:5.1f}%), "
                f"avg={avg:.1f}, max={d['max_dur']}"
            )


def cross_core_sync_analysis(rows: list):
    """Separate INTRA_BLOCK (cross-core) from SET/WAIT_FLAG (pipeline) stalls."""
    print_section("CROSS-CORE SYNC vs PIPELINE SYNC")
    for core_name in ("cube", "vector"):
        core_rows = [r for r in rows if r["core"] == core_name]
        intra_block = [r for r in core_rows if "INTRA_BLOCK" in r["opcode"]]
        set_wait_flag = [r for r in core_rows if r["opcode"] in ("SET_FLAG", "WAIT_FLAG")]

        intra_total = sum(r["duration"] for r in intra_block)
        flag_total = sum(r["duration"] for r in set_wait_flag)
        intra_stalls = [r for r in intra_block if r["duration"] > 100]
        flag_stalls = [r for r in set_wait_flag if r["duration"] > 100]

        print(f"\n--- {core_name} ---")
        print(
            f"  INTRA_BLOCK:  count={len(intra_block):4d}, total_dur={intra_total:8d}, "
            f"stalls(>100)={len(intra_stalls)}, stall_dur={sum(r['duration'] for r in intra_stalls)}"
        )
        print(
            f"  SET/WAIT_FLAG: count={len(set_wait_flag):4d}, total_dur={flag_total:8d}, "
            f"stalls(>100)={len(flag_stalls)}, stall_dur={sum(r['duration'] for r in flag_stalls)}"
        )


def mmad_analysis(rows: list):
    """Analyze MMAD instructions and gaps between them."""
    cube_rows = [r for r in rows if r["core"] == "cube"]
    mmads = sorted(
        [r for r in cube_rows if "MMAD" in r["opcode"]], key=lambda r: r["ts_start"]
    )
    stores = sorted(
        [r for r in cube_rows if "FIX_L0C_TO_DST" in r["opcode"]],
        key=lambda r: r["ts_start"],
    )

    n_mmads = len(mmads)
    if n_mmads == 0:
        print("\n[WARN] No MMAD instructions found.")
        return mmads, stores

    total_mmad = sum(r["duration"] for r in mmads)
    total_span_approx = mmads[-1]["ts_end"] - mmads[0]["ts_start"]

    print_section("MMAD (CUBE COMPUTE) ANALYSIS")
    print(f"Total MMADs: {n_mmads}, total compute cycles: {total_mmad}")
    print(f"Average MMAD duration: {total_mmad / n_mmads:.1f} cycles")
    print(f"MMAD span: [{mmads[0]['ts_start']}, {mmads[-1]['ts_end']}] = {total_span_approx} cycles")
    print()

    print(f"{'Idx':>4s} {'MMAD span':>18s} {'dur':>5s} {'TMOV span':>18s} {'TMOV dur':>8s} {'MMAD→TMOV gap':>14s} {'Gap to next MMAD':>17s}")
    print("-" * 95)
    for i, m in enumerate(mmads):
        s = stores[i] if i < len(stores) else None
        gap_to_next = mmads[i + 1]["ts_start"] - m["ts_end"] if i + 1 < n_mmads else 0
        mmad_str = f"[{m['ts_start']:6d}-{m['ts_end']:6d}]"
        if s:
            tmov_str = f"[{s['ts_start']:6d}-{s['ts_end']:6d}]"
            tmov_dur = s["duration"]
            mmad_to_tmov = s["ts_start"] - m["ts_end"]
        else:
            tmov_str = "N/A"
            tmov_dur = 0
            mmad_to_tmov = 0
        print(
            f"  {i:2d}  {mmad_str:>18s} {m['duration']:5d} "
            f"{tmov_str:>18s} {tmov_dur:8d} {mmad_to_tmov:14d} {gap_to_next:17d}"
        )

    return mmads, stores


def per_tile_breakdown(mmads: list, stores: list, total_span: int):
    """Reconstruct per-tile QK/PV pairing and compute stall breakdown.

    Assumes the standard pipeline structure:
      Preload: QK[0], QK[1], ...
      Main loop tile_id=0: QK[preload+0], PV[0]
      Main loop tile_id=1: QK[preload+1], PV[1]
      ...
    """
    n_mmads = len(mmads)
    n_stores = len(stores)
    if n_mmads < 4:
        print("\n[INFO] Too few MMADs for per-tile breakdown.")
        return

    # Heuristic: detect preload count.
    # During preload, QK MMAD is followed immediately by TMOV (gap < 100 cycles).
    # In the main loop, QK MMAD->TMOV has a large UB stall (> 500 cycles)
    # because Cube waits for Vec to free the UB buffer.
    # So preload_count = number of leading MMADs with small MMAD→TMOV gap.
    preload_count = 0
    for i in range(min(n_mmads, n_stores)):
        mmad_to_tmov = stores[i]["ts_start"] - mmads[i]["ts_end"]
        if mmad_to_tmov > 100:
            break
        preload_count += 1
    if preload_count == 0:
        preload_count = 2  # fallback

    n_pv = (n_mmads - preload_count) // 2 + (n_mmads - preload_count) % 2
    # Tiles with both QK and PV in main loop
    n_paired = (n_mmads - preload_count) // 2

    print_section(f"PER-TILE STEADY-STATE BREAKDOWN (detected preload={preload_count})")

    # Build QK/PV index mapping:
    # QK indices: 0..preload-1 (preload), then preload, preload+2, preload+4, ...
    # PV indices: preload+1, preload+3, preload+5, ...
    qk_indices = list(range(preload_count))
    pv_indices = []
    idx = preload_count
    while idx < n_mmads:
        if idx + 1 < n_mmads:
            # paired: idx=QK, idx+1=PV
            qk_indices.append(idx)
            pv_indices.append(idx + 1)
            idx += 2
        else:
            # unpaired PV-only
            pv_indices.append(idx)
            idx += 1

    # Per-tile breakdown table
    tile_data = []
    header = f"{'Tile':>6s} {'QK MMAD':>8s} {'UB stall':>9s} {'QK TMOV':>8s} {'SM stall':>9s} {'PV MMAD':>8s} {'PV TMOV':>8s} {'Total':>7s}"
    print(header)
    print("-" * len(header))

    for tile_id in range(len(pv_indices)):
        pi = pv_indices[tile_id]
        # Find the corresponding QK index (preload_count + tile_id for paired tiles)
        qi = qk_indices[preload_count + tile_id] if (preload_count + tile_id) < len(qk_indices) else None

        if qi is not None and qi < len(mmads) and pi < len(mmads) and qi < len(stores) and pi < len(stores):
            qk_mmad = mmads[qi]["duration"]
            ub_stall = stores[qi]["ts_start"] - mmads[qi]["ts_end"]
            qk_tmov = stores[qi]["duration"]
            sm_stall = mmads[pi]["ts_start"] - stores[qi]["ts_end"]
            pv_mmad = mmads[pi]["duration"]
            pv_tmov = stores[pi]["duration"] if pi < len(stores) else 0
            total = qk_mmad + ub_stall + qk_tmov + sm_stall + pv_mmad + pv_tmov
            tile_data.append(
                {
                    "tile": tile_id,
                    "qk_mmad": qk_mmad,
                    "ub_stall": ub_stall,
                    "qk_tmov": qk_tmov,
                    "sm_stall": sm_stall,
                    "pv_mmad": pv_mmad,
                    "pv_tmov": pv_tmov,
                    "total": total,
                }
            )
            print(
                f"  t{tile_id:<4d} {qk_mmad:8d} {ub_stall:9d} {qk_tmov:8d} "
                f"{sm_stall:9d} {pv_mmad:8d} {pv_tmov:8d} {total:7d}"
            )
        else:
            # PV-only tail tile
            pv_mmad = mmads[pi]["duration"]
            pv_tmov = stores[pi]["duration"] if pi < len(stores) else 0
            prev_end = mmads[pi - 1]["ts_end"] if pi > 0 else mmads[pi]["ts_start"]
            gap = mmads[pi]["ts_start"] - prev_end
            print(
                f"  t{tile_id:<4d} {'--':>8s} {'--':>9s} {'--':>8s} "
                f"{'--':>9s} {pv_mmad:8d} {pv_tmov:8d}   (PV-only, wait={gap})"
            )

    if tile_data:
        n = len(tile_data)
        avg_ub = sum(t["ub_stall"] for t in tile_data) / n
        avg_sm = sum(t["sm_stall"] for t in tile_data) / n
        avg_total = sum(t["total"] for t in tile_data) / n
        ideal = tile_data[0]["qk_mmad"] + tile_data[0]["pv_mmad"]  # 2x MMAD

        print()
        print(f"  Avg UB buffer stall:  {avg_ub:.0f} cycles/tile")
        print(f"  Avg SM sync stall:    {avg_sm:.0f} cycles/tile")
        print(f"  Avg tile iteration:   {avg_total:.0f} cycles (ideal {ideal} for 2x MMAD)")
        print(f"  Efficiency:           {100 * ideal / avg_total:.1f}%")
        print(f"  Stall ratio:          {100 * (avg_ub + avg_sm) / avg_total:.0f}%")

    # PV-to-PV iteration time
    if len(pv_indices) >= 2:
        print()
        print("PV-to-PV iteration time:")
        for i in range(1, len(pv_indices)):
            pi_curr = pv_indices[i]
            pi_prev = pv_indices[i - 1]
            dt = mmads[pi_curr]["ts_start"] - mmads[pi_prev]["ts_start"]
            print(f"  PV[{i-1}] -> PV[{i}]: {dt} cycles")


def vec_compute_density(rows: list, window: int = 500):
    """Show Vec compute density per time window."""
    vec_rows = [r for r in rows if r["core"] == "vector"]
    if not vec_rows:
        return

    overall_start = min(r["ts_start"] for r in rows)
    overall_end = max(r["ts_end"] for r in rows)

    print_section(f"VEC COMPUTE DENSITY (per {window}-cycle window)")
    print(f"{'Window':>14s} {'Compute ops':>12s} {'Compute cyc':>12s} {'Store ops':>10s} {'Sync ops':>9s}")

    for w_start in range(
        (overall_start // window) * window, overall_end + window, window
    ):
        w_end = w_start + window
        comp = [
            r
            for r in vec_rows
            if r["ts_start"] >= w_start
            and r["ts_start"] < w_end
            and r["op_class"] == "compute"
        ]
        store = [
            r
            for r in vec_rows
            if r["ts_start"] >= w_start
            and r["ts_start"] < w_end
            and r["op_class"] == "store"
        ]
        sync = [
            r
            for r in vec_rows
            if r["ts_start"] >= w_start
            and r["ts_start"] < w_end
            and r["op_class"] in ("sync", "other")
        ]
        comp_dur = sum(r["duration"] for r in comp)
        if len(comp) == 0 and len(store) == 0 and len(sync) == 0:
            continue
        print(
            f"  [{w_start:5d}-{w_end:5d}] {len(comp):12d} {comp_dur:12d} "
            f"{len(store):10d} {len(sync):9d}"
        )


def vec_softmax_ops_breakdown(rows: list, mmads: list):
    """Show Vec opcode breakdown during a representative Cube QK->PV wait window."""
    if len(mmads) < 4:
        return

    vec_rows = [r for r in rows if r["core"] == "vector"]
    # Use the gap between MMAD[2] end and MMAD[3] start as a representative window
    # (first main-loop QK->PV gap)
    # Find first gap > 500 cycles after initial MMADs
    window_start = None
    window_end = None
    for i in range(len(mmads) - 1):
        gap = mmads[i + 1]["ts_start"] - mmads[i]["ts_end"]
        if gap > 500:
            window_start = mmads[i]["ts_end"]
            window_end = mmads[i + 1]["ts_start"]
            break

    if window_start is None:
        return

    print_section(
        f"VEC ACTIVITY DURING CUBE QK→PV WAIT [{window_start}-{window_end}] ({window_end - window_start} cycles)"
    )

    vec_in_window = [
        r
        for r in vec_rows
        if r["ts_start"] >= window_start - 500
        and r["ts_end"] <= window_end + 500
        and r["op_class"] == "compute"
    ]
    print(f"Vec compute ops overlapping window: {len(vec_in_window)}")

    op_counts = Counter()
    op_durations = Counter()
    for r in vec_in_window:
        op_counts[r["opcode"]] += 1
        op_durations[r["opcode"]] += r["duration"]

    print(f"  {'Opcode':25s} {'Count':>6s} {'Total dur':>10s}")
    for op in sorted(op_durations, key=lambda x: -op_durations[x]):
        print(f"  {op:25s} {op_counts[op]:6d} {op_durations[op]:10d}")


def buffer_breakdown(rows: list):
    """Show per-core buffer hit breakdown."""
    print_section("PER-CORE BUFFER BREAKDOWN")
    for core_name in ("cube", "vector"):
        core_rows = [r for r in rows if r["core"] == core_name]
        by_buf = defaultdict(lambda: {"count": 0, "total_dur": 0, "max_dur": 0})
        for r in core_rows:
            buf = r.get("buffer", "") or r.get("global_buffer", "") or "unknown"
            by_buf[buf]["count"] += 1
            by_buf[buf]["total_dur"] += r["duration"]
            by_buf[buf]["max_dur"] = max(by_buf[buf]["max_dur"], r["duration"])

        print(f"\n--- {core_name} ---")
        for buf in sorted(by_buf, key=lambda b: -by_buf[b]["total_dur"]):
            d = by_buf[buf]
            avg = d["total_dur"] / d["count"] if d["count"] else 0
            print(
                f"  {buf:20s}: count={d['count']:5d}, total_dur={d['total_dur']:8d}, "
                f"avg={avg:.1f}, max={d['max_dur']}"
            )


def print_suggestions(mmads, stores, total_span):
    """Print optimization suggestions based on analysis."""
    if not mmads:
        return

    total_mmad = sum(r["duration"] for r in mmads)
    mmad_util = 100 * total_mmad / total_span if total_span else 0

    print_section("OPTIMIZATION SUGGESTIONS")

    if mmad_util < 30:
        print(
            f"  [!] Cube MMAD utilization is very low ({mmad_util:.1f}%). "
            f"The kernel is sync-stall dominated."
        )

    # Check QK MMAD->TMOV gaps (UB stall)
    ub_stalls = []
    for i in range(len(mmads)):
        if i < len(stores):
            gap = stores[i]["ts_start"] - mmads[i]["ts_end"]
            if gap > 500:
                ub_stalls.append(gap)

    if ub_stalls:
        avg_ub = sum(ub_stalls) / len(ub_stalls)
        print(
            f"  [!] Average UB buffer stall: {avg_ub:.0f} cycles. "
            f"Consider increasing srcVecTNBuffers (ping-pong depth) "
            f"or switching to MODE 2 to allow larger qk_preload."
        )

    print()
    print("  Potential optimizations:")
    print("  1. Increase TILE_S1 to reduce tile count and cross-core sync overhead")
    print("  2. Increase qk_preload (may require MODE 2 for preload > 2)")
    print("  3. Switch to MODE 2 (QK_PV_UB_ONLY) to relax L1 buffer constraints")
    print("  4. Optimize Vec softmax: reduce VCONV, merge VCMAX+VCADD reductions")
    print("  5. Increase srcVecTNBuffers from 2 to 3 (check 256KB UB budget)")


def main():
    parser = argparse.ArgumentParser(
        description="Bottleneck analysis for Flash Attention pipeline simulation logs"
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to timeline.csv produced by pipeline_log_analysis.py",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=500,
        help="Window size (cycles) for Vec compute density analysis (default: 500)",
    )
    args = parser.parse_args()

    rows = load_timeline(args.csv)
    if not rows:
        print("[ERROR] Empty timeline CSV.", file=sys.stderr)
        sys.exit(1)

    total_span = overall_timing(rows)
    utilization_breakdown(rows, total_span)
    cross_core_sync_analysis(rows)
    buffer_breakdown(rows)
    mmads, stores = mmad_analysis(rows)
    per_tile_breakdown(mmads, stores, total_span)
    vec_softmax_ops_breakdown(rows, mmads)
    vec_compute_density(rows, window=args.window)
    print_suggestions(mmads, stores, total_span)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
