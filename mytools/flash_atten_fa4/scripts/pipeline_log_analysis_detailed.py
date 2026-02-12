#!/usr/bin/env python3
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

"""Detailed pipeline timeline extraction with MTE/FIXP subpipe tags.

This script reuses the base pipeline log analysis and optionally tags each
instruction with its issuing subpipe (MTE1/MTE2/MTE3/FIXP/CUBE) from CCU
issque logs. It emits detailed CSV/JSON and an SVG timeline with subpipe
labels. Output filenames are different from the base script to avoid
overwriting.
"""

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pipeline_log_analysis as base

ISSQUE_TS_REGEX = re.compile(r"\[info\]\s+(\d+)\s+")
ISSQUE_NAME_REGEX = re.compile(r"\bname=([A-Z0-9_]+)")
ISSQUE_PC_REGEX = re.compile(r"\bpc=0x([0-9a-fA-F]+)")
ISSQUE_ID_REGEX = re.compile(r"\b(?:id|instr\.id)\s*=\s*(\d+)")

ISSQUE_SKIP_NAMES = {
    "WAIT_FLAG",
    "SET_FLAG",
    "WAIT_INTRA_BLOCK",
    "SET_INTRA_BLOCK",
    "MOV_SPR_XN",
    "FLOWCTRL",
    "BAR",
}

SUBPIPE_PRIORITY = {
    "CUBE": 5,
    "FIXP": 4,
    "MTE3": 3,
    "MTE2": 3,
    "MTE1": 2,
    "VEC_MTE3": 3,
    "VEC_MTE2": 3,
    "VEC": 3,  # Vector compute (RV_* / vec_issque name=VF)
}


@dataclass
class SvgTask:
    name: str
    core: str
    tile: int
    start: int
    end: int
    label_suffix: str = ""


def parse_issque(path: Path, subpipe: str) -> Dict[int, Dict[str, object]]:
    if not path:
        return {}
    if not path.exists():
        return {}
    entries: Dict[int, Dict[str, object]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "name=" not in line or "id=" not in line:
                continue
            ts_match = ISSQUE_TS_REGEX.search(line)
            name_match = ISSQUE_NAME_REGEX.search(line)
            id_match = ISSQUE_ID_REGEX.search(line)
            if not (ts_match and name_match and id_match):
                continue
            name = name_match.group(1)
            if name in ISSQUE_SKIP_NAMES:
                continue
            instr_id = int(id_match.group(1))
            pc_match = ISSQUE_PC_REGEX.search(line)
            entry = {
                "subpipe": subpipe,
                "name": name,
                "ts": int(ts_match.group(1)),
                "pc": int(pc_match.group(1), 16) if pc_match else None,
            }
            if instr_id not in entries:
                entries[instr_id] = entry
            else:
                # Keep higher-priority subpipe if conflicts
                cur = entries[instr_id]
                if SUBPIPE_PRIORITY.get(subpipe, 0) > SUBPIPE_PRIORITY.get(str(cur.get("subpipe", "")), 0):
                    entries[instr_id] = entry
    return entries


def merge_issque_maps(*maps: Dict[int, Dict[str, object]]) -> Dict[int, Dict[str, object]]:
    merged: Dict[int, Dict[str, object]] = {}
    for mp in maps:
        for instr_id, info in mp.items():
            if instr_id not in merged:
                merged[instr_id] = info
            else:
                cur = merged[instr_id]
                cur_pri = SUBPIPE_PRIORITY.get(str(cur.get("subpipe", "")), 0)
                new_pri = SUBPIPE_PRIORITY.get(str(info.get("subpipe", "")), 0)
                if new_pri > cur_pri:
                    merged[instr_id] = info
    return merged


def apply_subpipe(instrs: List[base.Instr], issque_map: Dict[int, Dict[str, object]]) -> None:
    for ins in instrs:
        ins.subpipe = None
        ins.unit_op = None
        if ins.instr_id is None:
            continue
        meta = issque_map.get(ins.instr_id)
        if not meta:
            continue
        ins.subpipe = str(meta.get("subpipe") or "")
        ins.unit_op = str(meta.get("name") or "")


def _parse_stage_name(stage_name: str) -> Tuple[str, int]:
    base_name = "".join(ch for ch in stage_name if ch.isalpha())
    num_part = "".join(ch for ch in stage_name if ch.isdigit())
    idx = int(num_part) if num_part else -1
    return base_name or stage_name, idx


def svg_tasks_from_instrs(instrs: List[base.Instr]) -> List[SvgTask]:
    def core_for(stage: str, fallback: str) -> str:
        if stage in ("compute_qk", "compute_pv"):
            return "cube"
        if stage in ("compute_p", "compute_gu"):
            return "vector"
        return fallback or "vector"

    agg: Dict[Tuple[str, ...], Dict[str, object]] = {}
    load_seq: Dict[Tuple[str, int], int] = {}

    for ins in instrs:
        if not ins.stage_name:
            continue
        base_name, idx = _parse_stage_name(ins.stage_name)
        if idx < 0:
            continue

        stage_label = ins.stage_name

        if ins.op_class == "compute":
            key = ("compute", stage_label)
            name = f"{stage_label}_comp"
        elif ins.op_class in ("load", "store"):
            kind = ins.op_class
            if kind == "load":
                seq = load_seq.setdefault((base_name, idx), 0)
                key = (f"{base_name}_load{seq}", idx, kind)
                name = f"{base_name}_load{seq}"
                load_seq[(base_name, idx)] = seq + 1
            else:
                key = (base_name, idx, kind)
                name = f"{base_name}_{kind}"
        else:
            continue

        label_suffix = ""
        if getattr(ins, "subpipe", None):
            label_suffix = f"@{ins.subpipe}"

        entry = agg.setdefault(
            key,
            {
                "start": ins.ts_start,
                "end": ins.ts_end or ins.ts_start,
                "core": core_for(ins.stage or "", ins.core),
                "name": name,
                "idx": idx,
                "label_suffix": label_suffix,
            },
        )
        entry["start"] = min(entry["start"], ins.ts_start)
        entry["end"] = max(entry["end"], ins.ts_end or ins.ts_start)
        if label_suffix:
            cur = str(entry.get("label_suffix") or "")
            if label_suffix not in cur:
                entry["label_suffix"] = f"{cur},{label_suffix}" if cur else label_suffix

    tasks: List[SvgTask] = []
    for info in agg.values():
        tasks.append(
            SvgTask(
                name=str(info.get("name")),
                core=str(info.get("core") or "vector"),
                tile=int(info.get("idx", -1)),
                start=int(info["start"]),
                end=int(info["end"]),
                label_suffix=str(info.get("label_suffix") or ""),
            )
        )

    return sorted(tasks, key=lambda t: (t.start, t.tile, t.name))


def render_svg(tasks: List[SvgTask], instrs: List[base.Instr], slot_width: int, output: Path, divisor: int = 10) -> None:
    if not tasks:
        return

    div = max(1, divisor)

    tmax_raw = max(t.end for t in tasks)
    tmax = math.ceil(tmax_raw / div)
    width = int(120 + slot_width * tmax + 40)

    colors = {
        "qk": ("#1f77b4", "#1a4f7a"),
        "p": ("#ff9933", "#c66f0a"),
        "pv": ("#2ca25f", "#1f7a47"),
        "gu": ("#9467bd", "#6c3f8b"),
        "cube_load": ("#1f77b4", "#1a4f7a"),
        "cube_comp": ("#0f6bd2", "#0b4b95"),
        "cube_store": ("#1f9ab4", "#146b7b"),
        "vector_load": ("#ff9933", "#c66f0a"),
        "vector_comp": ("#e87412", "#a7520d"),
        "vector_store": ("#ffbf59", "#c8892e"),
    }

    def base_name(name: str) -> str:
        first = name.split("_")[0]
        return first.rstrip("0123456789") or first

    def op_kind(task: SvgTask) -> str:
        suffix = task.name.split("_")[1] if "_" in task.name else "comp"
        if suffix.startswith("load"):
            return "load"
        if suffix.startswith("comp"):
            return "comp"
        return "store"

    def color_for(task: SvgTask) -> Tuple[str, str]:
        op = op_kind(task)
        key = f"{task.core}_{op}"
        if key in colors:
            return colors[key]
        return colors.get(base_name(task.name), ("#888", "#555"))

    mini_h = 12
    mini_gap = 4
    p_load_extra_gap = 2
    p_load_tail_pad = 8
    stage_extra_gap = {"p": 16}

    def assign_lanes(group: List[SvgTask]) -> Tuple[Dict[int, int], int]:
        lane_map: Dict[int, int] = {}
        lane_ends: List[int] = []
        for task in sorted(group, key=lambda t: (t.start, t.end)):
            placed = False
            for idx, end in enumerate(lane_ends):
                if task.start >= end:
                    lane_ends[idx] = task.end
                    lane_map[id(task)] = idx
                    placed = True
                    break
            if not placed:
                lane_map[id(task)] = len(lane_ends)
                lane_ends.append(task.end)
        return lane_map, max(1, len(lane_ends))

    def assign_lanes_by_tile(group: List[SvgTask]) -> Tuple[Dict[int, int], int]:
        lane_map: Dict[int, int] = {}
        tiles = sorted({t.tile for t in group})
        tile_map = {tile: idx for idx, tile in enumerate(tiles)}
        for task in group:
            lane_map[id(task)] = tile_map.get(task.tile, 0)
        return lane_map, max(1, len(tiles))

    def lane_pitch(base: str, op: str) -> int:
        if base == "p" and op == "load":
            return mini_h + mini_gap + p_load_extra_gap
        return mini_h + mini_gap

    def lane_height(lanes: int, base: str, op: str) -> int:
        pitch = lane_pitch(base, op)
        height = pitch * max(1, lanes) - (pitch - mini_h)
        if base == "p" and op == "load":
            height += p_load_tail_pad
        return height

    stage_bases = ("qk", "p", "pv", "gu")
    lane_map: Dict[int, int] = {}
    lane_count: Dict[Tuple[str, str], int] = {}
    for stage_key in stage_bases:
        for op in ("load", "comp", "store"):
            group = [t for t in tasks if base_name(t.name) == stage_key and op_kind(t) == op]
            if stage_key == "p" and op == "load":
                group_map, count = assign_lanes_by_tile(group)
            else:
                group_map, count = assign_lanes(group)
            lane_map.update(group_map)
            lane_count[(stage_key, op)] = count

    band_gap = 16
    op_gap = 8
    rows: Dict[str, int] = {}
    band_bounds: Dict[str, Tuple[int, int]] = {}
    band_heights: Dict[str, int] = {}
    y_cursor = 60
    for stage_key in stage_bases:
        rows[stage_key] = y_cursor
        band_h = (
            lane_height(lane_count.get((stage_key, "load"), 1), stage_key, "load")
            + op_gap
            + lane_height(lane_count.get((stage_key, "comp"), 1), stage_key, "comp")
            + op_gap
            + lane_height(lane_count.get((stage_key, "store"), 1), stage_key, "store")
        )
        band_bounds[stage_key] = (y_cursor, y_cursor + band_h)
        band_heights[stage_key] = band_h
        y_cursor += band_h + band_gap + stage_extra_gap.get(stage_key, 0)
    stage_bottom = y_cursor - band_gap
    cube_timeline_h = max(band_heights.get("qk", 0), band_heights.get("pv", 0), 1)
    vector_timeline_h = max(band_heights.get("p", 0), band_heights.get("gu", 0), 1)
    timeline_gap = 36
    rows["cube"] = stage_bottom + timeline_gap
    rows["vector"] = rows["cube"] + cube_timeline_h + timeline_gap
    summary_gap = 28
    summary_row_h = mini_h + 6
    summary_units_by_core = {
        "cube": ["MTE2", "MTE3", "FIXP", "CUBE"],
        "vector": ["MTE2", "MTE3", "VEC"],
    }
    cube_util_rows = len(summary_units_by_core["cube"])
    vector_util_rows = len(summary_units_by_core["vector"])
    rows["cube_util"] = rows["vector"] + vector_timeline_h + summary_gap
    rows["vector_util"] = rows["cube_util"] + (summary_row_h * cube_util_rows) + summary_gap
    height = rows["vector_util"] + (summary_row_h * vector_util_rows) + 40

    def y_for(task: SvgTask) -> float:
        stage_key = base_name(task.name)
        band_top = rows.get(stage_key, rows.get(task.core, 320))
        op = op_kind(task)
        load_h = lane_height(lane_count.get((stage_key, "load"), 1), stage_key, "load")
        comp_h = lane_height(lane_count.get((stage_key, "comp"), 1), stage_key, "comp")
        load_top = band_top
        comp_top = load_top + load_h + op_gap
        store_top = comp_top + comp_h + op_gap
        lane = lane_map.get(id(task), 0)
        if op == "load":
            return load_top + lane * lane_pitch(stage_key, "load")
        if op == "comp":
            return comp_top + lane * lane_pitch(stage_key, "comp")
        return store_top + lane * lane_pitch(stage_key, "store")

    def rect(task: SvgTask) -> str:
        x = 120 + slot_width * (task.start // div)
        w = slot_width * max(task.end // div - task.start // div, 1)
        y = y_for(task)
        fill, stroke = color_for(task)
        suffix = f"{task.label_suffix}" if task.label_suffix else ""
        label = f"{task.name}({task.tile}){suffix}"
        return (
            f'  <rect x="{x}" y="{y}" width="{w}" height="{mini_h}" fill="{fill}" '
            f'rx="3" stroke="{stroke}" stroke-width="1" />\n'
            f'  <text x="{x + 4}" y="{y + mini_h - 3:.1f}" font-family="Arial" font-size="10" fill="#fff">{label}</text>'
        )

    svg_parts: List[str] = []
    svg_parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg_parts.append(f'  <rect width="{width}" height="{height}" fill="#fff" />')
    svg_parts.append(
        "  <defs>\n"
        "    <marker id=\"arrowhead\" markerWidth=\"10\" markerHeight=\"7\" refX=\"10\" refY=\"3.5\" orient=\"auto\">\n"
        "      <polygon points=\"0 0 10 3.5 0 7\" fill=\"#333\" />\n"
        "    </marker>\n"
        "    <marker id=\"arrowhead-intra\" markerWidth=\"10\" markerHeight=\"7\" refX=\"10\" refY=\"3.5\" orient=\"auto\">\n"
        "      <polygon points=\"0 0 10 3.5 0 7\" fill=\"#1b75d1\" />\n"
        "    </marker>\n"
        "  </defs>"
    )
    svg_parts.append('  <text x="20" y="28" font-family="Arial" font-size="16" font-weight="bold" fill="#222">FA Pipeline Schedule (detailed)</text>')

    svg_parts.append('  <g stroke="#f2f2f2" stroke-width="1">')
    for t in range(0, tmax + 1, 10):
        x = 120 + slot_width * t
        svg_parts.append(f'    <line x1="{x}" y1="40" x2="{x}" y2="{stage_bottom}" />')
    svg_parts.append('  </g>')

    svg_parts.append('  <g stroke="#d9d9d9" stroke-width="1">')
    for stage_key in stage_bases:
        top, bottom = band_bounds[stage_key]
        sep_y = bottom + (band_gap / 2)
        svg_parts.append(f'    <line x1="20" y1="{sep_y:.1f}" x2="{width - 20}" y2="{sep_y:.1f}" />')
    stage_gap = rows["cube"] - stage_bottom
    svg_parts.append(
        f'    <line x1="20" y1="{stage_bottom + stage_gap / 2:.1f}" x2="{width - 20}" y2="{stage_bottom + stage_gap / 2:.1f}" />'
    )
    cube_vec_gap = rows["vector"] - (rows["cube"] + cube_timeline_h)
    svg_parts.append(
        f'    <line x1="20" y1="{rows["cube"] + cube_timeline_h + cube_vec_gap / 2:.1f}" x2="{width - 20}" y2="{rows["cube"] + cube_timeline_h + cube_vec_gap / 2:.1f}" />'
    )
    svg_parts.append('  </g>')

    svg_parts.append(
        f'  <text x="20" y="{rows["qk"] + 20}" font-family="Arial" font-size="13" fill="#222">compute_qk (cube)</text>'
    )
    svg_parts.append(
        f'  <text x="20" y="{rows["p"] + 20}" font-family="Arial" font-size="13" fill="#222">compute_p (vector)</text>'
    )
    svg_parts.append(
        f'  <text x="20" y="{rows["pv"] + 20}" font-family="Arial" font-size="13" fill="#222">compute_pv (cube)</text>'
    )
    svg_parts.append(
        f'  <text x="20" y="{rows["gu"] + 20}" font-family="Arial" font-size="13" fill="#222">compute_gu (vector)</text>'
    )
    svg_parts.append(
        f'  <text x="20" y="{rows["cube"] + 20}" font-family="Arial" font-size="13" fill="#222">cube timeline (qk+pv)</text>'
    )
    svg_parts.append(
        f'  <text x="20" y="{rows["vector"] + 20}" font-family="Arial" font-size="13" fill="#222">vector timeline (p+gu)</text>'
    )
    svg_parts.append(
        f'  <text x="20" y="{rows["cube_util"] - 6}" font-family="Arial" font-size="13" fill="#222">cube util (by subpipe)</text>'
    )
    svg_parts.append(
        f'  <text x="20" y="{rows["vector_util"] - 6}" font-family="Arial" font-size="13" fill="#222">vector util (by subpipe)</text>'
    )

    for task in tasks:
        svg_parts.append(rect(task))

    def rect_at(task: SvgTask, y_row: int) -> str:
        y = y_row + (y_for(task) - rows.get(base_name(task.name), rows.get(task.core, 320)))
        x = 120 + slot_width * (task.start // div)
        w = slot_width * max(task.end // div - task.start // div, 1)
        fill, stroke = color_for(task)
        suffix = f"{task.label_suffix}" if task.label_suffix else ""
        label = f"{task.name}({task.tile}){suffix}"
        return (
            f'  <rect x="{x}" y="{y}" width="{w}" height="{mini_h}" fill="{fill}" '
            f'rx="3" stroke="{stroke}" stroke-width="1" />\n'
            f'  <text x="{x + 4}" y="{y + mini_h - 3:.1f}" font-family="Arial" font-size="10" fill="#fff">{label}</text>'
        )

    for task in sorted(tasks, key=lambda t: (t.start, t.end)):
        if task.core == "cube":
            svg_parts.append(rect_at(task, rows["cube"]))
        else:
            svg_parts.append(rect_at(task, rows["vector"]))

    def norm_subpipe(ins: base.Instr) -> Optional[str]:
        if ins.op_class == "compute":
            return "CUBE" if ins.core == "cube" else "VEC"
        sub = getattr(ins, "subpipe", None)
        if sub:
            text = str(sub)
            if text.startswith("VEC_"):
                return text[4:]
            return text
        pipe = (ins.pipeline or "").upper()
        for tag in ("MTE2", "MTE3", "FIXP", "CUBE"):
            if pipe.startswith(tag):
                return tag
        return None

    def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged

    def util_color(core: str, unit: str) -> Tuple[str, str]:
        if unit in ("MTE2", "MTE3"):
            key = f"{core}_load"
        elif unit == "FIXP":
            key = f"{core}_store"
        else:
            key = f"{core}_comp"
        return colors.get(key, ("#777", "#444"))

    util_rows = {"cube": rows["cube_util"], "vector": rows["vector_util"]}
    def compute_intervals_from_tasks(core: str) -> List[Tuple[int, int]]:
        intervals: List[Tuple[int, int]] = []
        for task in tasks:
            if task.core != core:
                continue
            if "_comp" not in task.name:
                continue
            start = task.start
            end = task.end
            if end < start:
                end = start
            intervals.append((start, end))
        return merge_intervals(intervals)

    for core in ("cube", "vector"):
        base_y = util_rows[core]
        for idx, unit in enumerate(summary_units_by_core[core]):
            if unit in ("CUBE", "VEC"):
                merged = compute_intervals_from_tasks(core)
            else:
                intervals: List[Tuple[int, int]] = []
                for ins in instrs:
                    if ins.core != core:
                        continue
                    sub = norm_subpipe(ins)
                    if sub != unit:
                        continue
                    start = ins.ts_start
                    end = ins.ts_end or ins.ts_start
                    if end < start:
                        end = start
                    intervals.append((start, end))
                merged = merge_intervals(intervals)
            y = base_y + idx * summary_row_h
            label = unit
            svg_parts.append(
                f'  <text x="20" y="{y + mini_h - 2}" font-family="Arial" font-size="10" fill="#333">{label}</text>'
            )
            fill, stroke = util_color(core, unit)
            for start, end in merged:
                x = 120 + slot_width * (start // div)
                w = slot_width * max(end // div - start // div, 1)
                svg_parts.append(
                    f'  <rect x="{x}" y="{y}" width="{w}" height="{mini_h}" fill="{fill}" rx="2" stroke="{stroke}" stroke-width="1" />'
                )
                svg_parts.append(
                    f'  <text x="{x + 4}" y="{y + mini_h - 2:.1f}" font-family="Arial" font-size="9" fill="#fff">{label}</text>'
                )

    def find_pref(name: str, tile: int) -> List[SvgTask]:
        return sorted([tk for tk in tasks if tk.tile == tile and tk.name.startswith(name)], key=lambda t: t.start)

    def find_first(name: str, tile: int) -> Optional[SvgTask]:
        res = find_pref(name, tile)
        return res[0] if res else None

    def adjust_overlap(y1: float, y2: float) -> Tuple[float, float]:
        return y1, y2

    def pick_dep(store_name: str, load_name: str, tile: int) -> Tuple[Optional[SvgTask], Optional[SvgTask]]:
        stores = find_pref(store_name, tile)
        loads = find_pref(load_name, tile)
        if not stores or not loads:
            return None, None
        store = max(stores, key=lambda t: t.end)
        load_candidates = [ld for ld in loads if ld.start >= store.end]
        if not load_candidates:
            return None, None
        return store, load_candidates[0]

    for tile in range(max(t.tile for t in tasks) + 1):
        qk_store, p_load_dep = pick_dep("qk_store", "p_load0", tile)
        if qk_store and p_load_dep:
            x1 = 120 + slot_width * (qk_store.end / div)
            x2 = 120 + slot_width * (p_load_dep.start / div)
            y1 = y_for(qk_store) + mini_h / 2
            y2 = y_for(p_load_dep) + mini_h / 2
            y1, y2 = adjust_overlap(y1, y2)
            svg_parts.append(
                f'  <path d="M{x1:.1f} {y1:.1f} L{x1:.1f} {y2-4:.1f} L{x2:.1f} {y2-4:.1f}" stroke="#333" stroke-width="1.5" fill="none" marker-end="url(#arrowhead)" />'
            )
        p_store, pv_load_dep = pick_dep("p_store", "pv_load1", tile)
        if p_store and pv_load_dep:
            x1 = 120 + slot_width * (p_store.end / div)
            x2 = 120 + slot_width * (pv_load_dep.start / div)
            y1 = y_for(p_store) + mini_h / 2
            y2 = y_for(pv_load_dep) + mini_h / 2
            y1, y2 = adjust_overlap(y1, y2)
            svg_parts.append(
                f'  <path d="M{x1:.1f} {y1} L{x1:.1f} {y2-4} L{x2} {y2-4}" stroke="#333" stroke-width="1.5" fill="none" marker-end="url(#arrowhead)" />'
            )
        pv_store, gu_load = pick_dep("pv_store", "gu_load", tile)
        if pv_store and gu_load:
            x1 = 120 + slot_width * (pv_store.end / div)
            x2 = 120 + slot_width * (gu_load.start / div)
            y1 = y_for(pv_store) + mini_h / 2
            y2 = y_for(gu_load) + mini_h / 2
            y1, y2 = adjust_overlap(y1, y2)
            svg_parts.append(
                f'  <path d="M{x1:.1f} {y1} L{x1:.1f} {y2-4} L{x2} {y2-4}" stroke="#333" stroke-width="1.5" fill="none" marker-end="url(#arrowhead)" />'
            )

        for stage_base in ("qk", "p", "pv", "gu"):
            load = find_first(f"{stage_base}_load", tile)
            comp = find_first(f"{stage_base}_comp", tile)
            store = find_first(f"{stage_base}_store", tile)
            if load and comp and load.start != load.end and comp.start != comp.end and load.end <= comp.start:
                x1 = 120 + slot_width * (load.end / div)
                x2 = 120 + slot_width * (comp.start / div)
                y1 = y_for(load) + mini_h / 2
                y2 = y_for(comp) + mini_h / 2
                svg_parts.append(
                    f'  <path d="M{x1:.1f} {y1} L{x2} {y2}" stroke="#1b75d1" stroke-width="1.2" fill="none" marker-end="url(#arrowhead-intra)" />'
                )
            if comp and store and comp.start != comp.end and store.start != store.end and comp.end <= store.start:
                x1 = 120 + slot_width * (comp.end / div)
                x2 = 120 + slot_width * (store.start / div)
                y1 = y_for(comp) + mini_h / 2
                y2 = y_for(store) + mini_h / 2
                svg_parts.append(
                    f'  <path d="M{x1:.1f} {y1} L{x2} {y2}" stroke="#1b75d1" stroke-width="1.2" fill="none" marker-end="url(#arrowhead-intra)" />'
                )

    svg_parts.append("</svg>")

    with output.open("w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))


def write_csv(instrs: List[base.Instr], path: Path) -> None:
    fieldnames = [
        "core",
        "ts_start",
        "ts_end",
        "pipeline",
        "opcode",
        "op_class",
        "stage",
        "stage_name",
        "buffer",
        "global_buffer",
        "buffer_addr",
        "addresses",
        "pc",
        "line_start",
        "line_end",
        "instr_id",
        "subpipe",
        "unit_op",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for instr in instrs:
            row = {
                "core": instr.core,
                "ts_start": instr.ts_start,
                "ts_end": instr.ts_end,
                "pipeline": instr.pipeline,
                "opcode": instr.opcode,
                "op_class": instr.op_class,
                "stage": instr.stage,
                "stage_name": instr.stage_name,
                "buffer": instr.buffer,
                "global_buffer": instr.global_buffer,
                "buffer_addr": hex(instr.buffer_addr) if instr.buffer_addr is not None else "",
                "addresses": ",".join(hex(a) for a in instr.addresses),
                "pc": hex(instr.pc) if instr.pc is not None else "",
                "line_start": instr.line_start,
                "line_end": instr.line_end,
                "instr_id": instr.instr_id,
                "subpipe": getattr(instr, "subpipe", None),
                "unit_op": getattr(instr, "unit_op", None),
            }
            writer.writerow(row)


def write_json(instrs: List[base.Instr], path: Path) -> None:
    payload = []
    for instr in instrs:
        payload.append(
            {
                "core": instr.core,
                "ts_start": instr.ts_start,
                "ts_end": instr.ts_end,
                "pipeline": instr.pipeline,
                "opcode": instr.opcode,
                "op_class": instr.op_class,
                "stage": instr.stage,
                "stage_name": instr.stage_name,
                "buffer": instr.buffer,
                "global_buffer": instr.global_buffer,
                "buffer_addr": hex(instr.buffer_addr) if instr.buffer_addr is not None else "",
                "addresses": [hex(a) for a in instr.addresses],
                "pc": hex(instr.pc) if instr.pc is not None else "",
                "line_start": instr.line_start,
                "line_end": instr.line_end,
                "instr_id": instr.instr_id,
                "subpipe": getattr(instr, "subpipe", None),
                "unit_op": getattr(instr, "unit_op", None),
            }
        )
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def aggregate_detailed(instrs: List[base.Instr]) -> List[Dict[str, object]]:
    buckets: Dict[Tuple[str, str, str, str], Dict[str, int]] = {}
    for ins in instrs:
        key = (
            ins.core,
            ins.buffer or "unknown",
            ins.op_class,
            getattr(ins, "subpipe", None) or "unknown",
        )
        agg = buckets.setdefault(key, {"start": ins.ts_start, "end": ins.ts_end or ins.ts_start})
        agg["start"] = min(agg["start"], ins.ts_start)
        agg["end"] = max(agg["end"], ins.ts_end or ins.ts_start)
    summary = []
    for (core, buf, op_class, subpipe), val in buckets.items():
        summary.append(
            {
                "core": core,
                "buffer": buf,
                "op_class": op_class,
                "subpipe": subpipe,
                "start": val["start"],
                "end": val["end"],
                "duration": val["end"] - val["start"],
            }
        )
    return sorted(summary, key=lambda x: (x["core"], x["buffer"], x["subpipe"], x["start"]))


def write_aggregate_csv(summary: List[Dict[str, object]], path: Path) -> None:
    fieldnames = ["core", "buffer", "op_class", "subpipe", "start", "end", "duration"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse TFA pipeline logs into detailed timelines")
    ap.add_argument("--device-addrs", required=True, type=Path, help="Path to device_addrs.toml")
    ap.add_argument("--cube-start", required=True, type=Path, help="cube core start log (*.instr_popped_log.dump)")
    ap.add_argument("--cube-end", required=True, type=Path, help="cube core end log (*.instr_log.dump)")
    ap.add_argument("--vec-start", required=True, type=Path, help="vector core start log")
    ap.add_argument("--vec-end", required=True, type=Path, help="vector core end log")
    ap.add_argument("--cube-mte1", type=Path, default=None, help="cube MTE1 issque log")
    ap.add_argument("--cube-mte2", type=Path, default=None, help="cube MTE2 issque log")
    ap.add_argument("--cube-mte3", type=Path, default=None, help="cube MTE3 issque log")
    ap.add_argument("--cube-fixp", type=Path, default=None, help="cube FIXP issque log")
    ap.add_argument("--cube-cube", type=Path, default=None, help="cube CUBE issque log")
    ap.add_argument("--vec-mte2", type=Path, default=None, help="vector MTE2 issque log")
    ap.add_argument("--vec-mte3", type=Path, default=None, help="vector MTE3 issque log")
    ap.add_argument("--vec-vec", type=Path, default=None, help="vector vec_issque log (RV_* compute, name=VF)")
    ap.add_argument("--out-csv", type=Path, default=Path("timeline_detailed.csv"))
    ap.add_argument("--out-json", type=Path, default=Path("timeline_detailed.json"))
    ap.add_argument("--out-agg", type=Path, default=Path("timeline_detailed_agg.csv"))
    ap.add_argument("--out-svg", type=Path, default=None, help="Optional SVG timeline output rendered from parsed log")
    ap.add_argument("--svg-divisor", type=int, default=100, help="Divide timestamps by this factor for SVG scaling")
    args = ap.parse_args()

    buf_map = base.load_device_addrs(args.device_addrs)
    cube_start = base.parse_log(args.cube_start, core="cube")
    cube_end = base.parse_log(args.cube_end, core="cube")
    vec_start = base.parse_log(args.vec_start, core="vector")
    vec_end = base.parse_log(args.vec_end, core="vector")

    cube_instrs = base.to_instrs(base.match_start_end(cube_start, cube_end), buf_map)
    vec_instrs = base.to_instrs(base.match_start_end(vec_start, vec_end), buf_map)
    instrs = sorted(cube_instrs + vec_instrs, key=lambda x: (x.ts_start, x.core))

    base.infer_stage_compute(instrs)
    base.assign_stage_names(instrs)

    issque_map = merge_issque_maps(
        parse_issque(args.cube_mte1, "MTE1") if args.cube_mte1 else {},
        parse_issque(args.cube_mte2, "MTE2") if args.cube_mte2 else {},
        parse_issque(args.cube_mte3, "MTE3") if args.cube_mte3 else {},
        parse_issque(args.cube_fixp, "FIXP") if args.cube_fixp else {},
        parse_issque(args.cube_cube, "CUBE") if args.cube_cube else {},
        parse_issque(args.vec_mte2, "VEC_MTE2") if args.vec_mte2 else {},
        parse_issque(args.vec_mte3, "VEC_MTE3") if args.vec_mte3 else {},
        parse_issque(args.vec_vec, "VEC") if args.vec_vec else {},
    )
    apply_subpipe(instrs, issque_map)

    write_csv(instrs, args.out_csv)
    write_json(instrs, args.out_json)
    write_aggregate_csv(aggregate_detailed(instrs), args.out_agg)

    if args.out_svg:
        tasks = svg_tasks_from_instrs(instrs)
        render_svg(tasks, instrs, slot_width=8, output=args.out_svg, divisor=args.svg_divisor)

    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_agg}")
    if args.out_svg:
        print(f"Wrote {args.out_svg}")


if __name__ == "__main__":
    main()
