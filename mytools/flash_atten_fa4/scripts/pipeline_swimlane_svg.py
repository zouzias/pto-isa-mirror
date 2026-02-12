#!/usr/bin/env python3
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

"""Render core/pipeline swimlane SVG from hardware logs.

The script pairs start/end events by instruction id:
- start log: *.instr_popped_log.dump
- end log:   *.instr_log.dump

Output is an SVG with separate core sections and one lane per pipeline.
"""

import argparse
import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

TS_REGEX = re.compile(r"\[(\d+)\]")
PC_REGEX = re.compile(r"PC:\s*0x([0-9a-fA-F]+)")
PIPELINE_REGEX = re.compile(r"\)\s*([A-Z0-9_]+)")
ID_REGEX = re.compile(r"\(ID:\s*(\d+)\)")
EXCLUDED_PIPELINES = {"PUSHQ", "FLOWCTRL"}
ISSQ_TS_REGEX = re.compile(r"\[info\]\s+(\d+)\s+\[")
ISSQ_PUSH_REGEX = re.compile(r"\[Push Instr\]\s+name=([A-Z0-9_]+)\s+pc=0x([0-9A-Fa-f]+)\s+id=(\d+)")
ISSQ_POP_REGEX = re.compile(r"\[pop Instr\]\s+name=([A-Z0-9_]+)\s+pc=0x([0-9A-Fa-f]+)\s+id=(\d+)")
ISSQ_RETIRE_REGEX = re.compile(r"\[RETIRE INSTR\]\s+instr\.name=([A-Z0-9_]+)\s+instr\.pc=0x([0-9A-Fa-f]+)\s+instr\.id=(\d+)")
ISSQ_EXCLUDED_NAMES = {"WAIT_FLAG", "WAIT_INTRA_BLOCK", "SET_FLAG", "SET_INTRA_BLOCK", "MOV_SPR_XN", "END_LABEL"}


@dataclass
class Event:
    ts: int
    pipeline: str
    instr_id: Optional[int]
    pc: Optional[int]


@dataclass
class Span:
    core: str
    pipeline: str
    ts_start: int
    ts_end: int
    instr_id: Optional[int]
    pc: Optional[int]


def parse_log(path: Path) -> List[Event]:
    events: List[Event] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_ts = TS_REGEX.search(line)
            m_pipe = PIPELINE_REGEX.search(line)
            if not m_ts or not m_pipe:
                continue
            ts = int(m_ts.group(1))
            pipeline = m_pipe.group(1)
            if pipeline in EXCLUDED_PIPELINES:
                continue
            # Merge all RVEC sub-pipelines into one VECTOR lane.
            if pipeline.startswith("RVEC") or pipeline == "VEC":
                pipeline = "VECTOR"
            m_id = ID_REGEX.search(line)
            instr_id = int(m_id.group(1)) if m_id else None
            m_pc = PC_REGEX.search(line)
            pc = int(m_pc.group(1), 16) if m_pc else None
            events.append(Event(ts=ts, pipeline=pipeline, instr_id=instr_id, pc=pc))
    return events


def pair_spans(start_events: List[Event], end_events: List[Event], core: str) -> List[Span]:
    start_by_id: Dict[int, List[Event]] = {}
    end_by_id: Dict[int, List[Event]] = {}
    spans: List[Span] = []

    for ev in start_events:
        if ev.instr_id is not None:
            start_by_id.setdefault(ev.instr_id, []).append(ev)
    for ev in end_events:
        if ev.instr_id is not None:
            end_by_id.setdefault(ev.instr_id, []).append(ev)

    # Pair by id first.
    for instr_id, s_list in start_by_id.items():
        e_list = end_by_id.get(instr_id, [])
        for s_ev, e_ev in zip(s_list, e_list):
            spans.append(
                Span(
                    core=core,
                    pipeline=s_ev.pipeline,
                    ts_start=s_ev.ts,
                    ts_end=max(s_ev.ts, e_ev.ts),
                    instr_id=instr_id,
                    pc=s_ev.pc,
                )
            )

    # Fallback: pair non-id events in-order.
    s_noid = [ev for ev in start_events if ev.instr_id is None]
    e_noid = [ev for ev in end_events if ev.instr_id is None]
    n = min(len(s_noid), len(e_noid))
    for i in range(n):
        s_ev = s_noid[i]
        e_ev = e_noid[i]
        spans.append(
            Span(
                core=core,
                pipeline=s_ev.pipeline,
                ts_start=s_ev.ts,
                ts_end=max(s_ev.ts, e_ev.ts),
                instr_id=None,
                pc=s_ev.pc,
            )
        )

    return spans


def parse_issque_spans(path: Optional[Path], core: str, pipeline: str) -> List[Span]:
    if not path or not path.exists():
        return []

    starts: Dict[int, Tuple[int, int, str]] = {}
    spans: List[Span] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_ts = ISSQ_TS_REGEX.search(line)
            if not m_ts:
                continue
            ts = int(m_ts.group(1))

            m_push = ISSQ_PUSH_REGEX.search(line)
            if m_push:
                name = m_push.group(1)
                if name in ISSQ_EXCLUDED_NAMES:
                    continue
                pc = int(m_push.group(2), 16)
                instr_id = int(m_push.group(3))
                starts[instr_id] = (ts, pc, name)
                continue

            m_pop = ISSQ_POP_REGEX.search(line)
            if not m_pop:
                m_pop = ISSQ_RETIRE_REGEX.search(line)
            if m_pop:
                name = m_pop.group(1)
                if name in ISSQ_EXCLUDED_NAMES:
                    continue
                instr_id = int(m_pop.group(3))
                if instr_id not in starts:
                    continue
                ts_start, pc_start, _ = starts.pop(instr_id)
                spans.append(
                    Span(
                        core=core,
                        pipeline=pipeline,
                        ts_start=ts_start,
                        ts_end=max(ts_start, ts),
                        instr_id=instr_id,
                        pc=pc_start,
                    )
                )

    return spans


def dedupe_spans(spans: List[Span]) -> List[Span]:
    merged: Dict[Tuple[str, str, Optional[int]], Span] = {}
    for sp in spans:
        key = (sp.core, sp.pipeline, sp.instr_id)
        if sp.instr_id is None:
            # Keep non-id spans as independent records.
            key = (sp.core, sp.pipeline, id(sp))
        if key not in merged:
            merged[key] = sp
            continue
        prev = merged[key]
        prev.ts_start = min(prev.ts_start, sp.ts_start)
        prev.ts_end = max(prev.ts_end, sp.ts_end)
    return list(merged.values())


def lane_order(core: str, lanes: List[str]) -> List[str]:
    preferred = {
        "cube": ["MTE1", "CUBE", "FIXP", "MTE2", "SCALAR"],
        "vector": ["SCALAR", "MTE2", "VECTOR", "MTE3"],
    }
    pref = preferred.get(core, ["ALL"])
    rest = [x for x in lanes if x not in pref]
    rest.sort()
    return [x for x in pref if x in lanes] + rest


def color_for_pipeline(pipeline: str) -> str:
    palette = {
        "MTE1": "#2E86DE",
        "MTE2": "#C2185B",
        "MTE3": "#FF7043",
        "CUBE": "#26A69A",
        "FIXP": "#F39C12",
        "SCALAR": "#EF5350",
        "VECTOR": "#F4511E",
    }
    return palette.get(pipeline, "#78909C")


def render_swimlane_svg(
    spans: List[Span],
    output: Path,
    divisor: int = 100,
    slot_width: int = 8,
    lane_height: int = 20,
    lane_gap: int = 8,
    section_gap: int = 26,
    hide_scalar: bool = False,
) -> None:
    if hide_scalar:
        spans = [s for s in spans if s.pipeline != "SCALAR"]

    if not spans:
        output.write_text("", encoding="utf-8")
        return

    div = max(1, divisor)
    spans = sorted(spans, key=lambda x: (x.ts_start, x.core, x.pipeline))

    core_lanes: Dict[str, List[str]] = {}
    for core in ("cube", "vector"):
        lanes = sorted({s.pipeline for s in spans if s.core == core})
        core_lanes[core] = lane_order(core, lanes)

    tmax = max(s.ts_end for s in spans) // div + 1
    left = 180
    right = 40
    top = 28
    width = left + tmax * slot_width + right

    total_lanes = sum(len(core_lanes[c]) for c in ("cube", "vector"))
    height = top + total_lanes * (lane_height + lane_gap) + section_gap + 30

    y_map: Dict[Tuple[str, str], int] = {}
    y = top + 30
    for core in ("cube", "vector"):
        for lane in core_lanes[core]:
            y_map[(core, lane)] = y
            y += lane_height + lane_gap
        y += section_gap

    parts: List[str] = []
    parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append(f'  <rect width="{width}" height="{height}" fill="#ffffff"/>')
    parts.append('  <text x="16" y="22" font-size="15" font-family="Arial" fill="#222" font-weight="bold">Pipeline Swimlane (from logs)</text>')

    # Vertical time grid.
    parts.append('  <g stroke="#efefef" stroke-width="1">')
    step = 10
    for t in range(0, tmax + 1, step):
        x = left + t * slot_width
        parts.append(f'    <line x1="{x}" y1="{top}" x2="{x}" y2="{height - 20}"/>')
    parts.append("  </g>")

    # Section/lane labels.
    for core in ("cube", "vector"):
        if not core_lanes[core]:
            continue
        first_y = y_map[(core, core_lanes[core][0])]
        parts.append(f'  <text x="16" y="{first_y - 8}" font-size="13" font-family="Arial" fill="#333" font-weight="bold">core0.{core}core0</text>')
        for lane in core_lanes[core]:
            ly = y_map[(core, lane)]
            parts.append(f'  <text x="36" y="{ly + 14}" font-size="12" font-family="Arial" fill="#444">{html.escape(lane)}</text>')
            parts.append(f'  <line x1="{left - 8}" y1="{ly + lane_height + 1}" x2="{width - 16}" y2="{ly + lane_height + 1}" stroke="#f6f6f6"/>')

    # Bars on pipeline lanes.
    for sp in spans:
        x = left + (sp.ts_start // div) * slot_width
        w = max(1, (sp.ts_end // div - sp.ts_start // div) * slot_width)
        y0 = y_map[(sp.core, sp.pipeline)]
        color = color_for_pipeline(sp.pipeline)
        parts.append(
            f'  <rect x="{x}" y="{y0}" width="{w}" height="{lane_height}" fill="{color}" rx="3" opacity="0.9"/>'
        )

    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate swimlane SVG from cube/vector logs")
    ap.add_argument("--cube-start", required=True, type=Path, help="cube start log (*.instr_popped_log.dump)")
    ap.add_argument("--cube-end", required=True, type=Path, help="cube end log (*.instr_log.dump)")
    ap.add_argument("--vec-start", required=True, type=Path, help="vector start log (*.instr_popped_log.dump)")
    ap.add_argument("--vec-end", required=True, type=Path, help="vector end log (*.instr_log.dump)")
    ap.add_argument("--vec-mte2-issque", type=Path, default=None, help="Optional vector MTE2 issue queue dump")
    ap.add_argument("--out-svg", type=Path, default=Path("swimlane.svg"))
    ap.add_argument("--svg-divisor", type=int, default=100)
    ap.add_argument("--slot-width", type=int, default=8)
    ap.add_argument("--hide-scalar", action="store_true", help="Hide SCALAR pipeline lanes and bars")
    args = ap.parse_args()

    cube_spans = pair_spans(parse_log(args.cube_start), parse_log(args.cube_end), core="cube")
    vec_spans = pair_spans(parse_log(args.vec_start), parse_log(args.vec_end), core="vector")
    vec_mte2_q_spans = parse_issque_spans(args.vec_mte2_issque, core="vector", pipeline="MTE2")
    spans = dedupe_spans(cube_spans + vec_spans + vec_mte2_q_spans)

    render_swimlane_svg(
        spans=spans,
        output=args.out_svg,
        divisor=args.svg_divisor,
        slot_width=args.slot_width,
        hide_scalar=args.hide_scalar,
    )
    print(f"Wrote {args.out_svg}")


if __name__ == "__main__":
    main()

