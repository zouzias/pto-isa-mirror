#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

"""Generate per-instruction SVG diagrams for PTO ISA docs.

Design goals:
- Use grid-based tiles to visualize the tiled data structure.
- Include a clear, per-instruction conceptual procedure (pseudocode).
- Keep diagrams tidy and consistent across instruction families.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "docs" / "isa" / "manifest.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "figures" / "isa"


CANVAS_W = 1200
CANVAS_H = 720
MARGIN = 24
HEADER_Y = 46
HEADER_DIVIDER_Y = 104

DIAGRAM_TOP = HEADER_DIVIDER_Y + 24
EXPR_Y = DIAGRAM_TOP + 12
SRC_Y = DIAGRAM_TOP + 50
DST_Y = DIAGRAM_TOP + 232

PROC_BOX_Y = 488
PROC_PAD = 16

TILE_ROWS = 5
TILE_COLS = 5
CELL = 22

# One representative element used for callouts.
EX_R = 1
EX_C = 2

ARROW_PAD = 12

COLOR_BY_TEMPLATE = {
    "elementwise": ("#2D5BCE", "#EAF2FF"),
    "scalar": ("#1D8E63", "#E9F7F1"),
    "reduce_expand": ("#C46A1C", "#FFF4E8"),
    "memory": ("#6A47C4", "#F0EDFF"),
    "matmul": ("#1B7F91", "#E8F8FB"),
    "reshape_move": ("#515151", "#F5F5F5"),
    "complex": ("#C53A79", "#FDF0F6"),
    "sync": ("#A37000", "#FFF7D6"),
    "config": ("#4C8A25", "#EEF7E6"),
    "comm": ("#8A4B12", "#FFF3E7"),
}


def _esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def _append_svg_text(
    out: List[str],
    *,
    x: object,
    y: object,
    cls: str,
    text: object,
    text_anchor: Optional[str] = None,
    dominant_baseline: Optional[str] = None,
    fill: Optional[str] = None,
    font_size: Optional[str] = None,
) -> None:
    attrs = [f'x="{x}"', f'y="{y}"', f'class="{cls}"']
    if text_anchor:
        attrs.append(f'text-anchor="{text_anchor}"')
    if dominant_baseline:
        attrs.append(f'dominant-baseline="{dominant_baseline}"')
    if fill:
        attrs.append(f'fill="{fill}"')
    if font_size:
        attrs.append(f'font-size="{font_size}"')
    out.append(f"<text {' '.join(attrs)}>{_esc(text)}</text>")


def _append_rect(
    out: List[str],
    *,
    x: object,
    y: object,
    width: object,
    height: object,
    rx: Optional[object] = None,
    cls: Optional[str] = None,
    fill: Optional[str] = None,
    stroke: Optional[str] = None,
    stroke_width: Optional[object] = None,
) -> None:
    attrs = [f'x="{x}"', f'y="{y}"', f'width="{width}"', f'height="{height}"']
    if rx is not None:
        attrs.append(f'rx="{rx}"')
    if cls:
        attrs.append(f'class="{cls}"')
    if fill:
        attrs.append(f'fill="{fill}"')
    if stroke:
        attrs.append(f'stroke="{_esc(stroke)}"')
    if stroke_width is not None:
        attrs.append(f'stroke-width="{stroke_width}"')
    out.append(f"<rect {' '.join(attrs)}/>")


def load_manifest(path: Path) -> List[Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("instructions", [])
    if not isinstance(entries, list):
        raise ValueError("manifest 'instructions' must be a list")
    return entries


def _tile_cell_text(prefix: str, r: int, c: int) -> str:
    return f"{prefix}{r}{c}"


def _tile_width(cols: int) -> int:
    return cols * CELL


def _tile_height(rows: int) -> int:
    return rows * CELL


def _cell_center(x: int, y: int, r: int, c: int) -> Tuple[int, int]:
    cx = x + c * CELL + CELL // 2
    cy = y + r * CELL + CELL // 2
    return (cx, cy)


def _draw_text_lines(out: List[str], x: int, y: int, lines: Sequence[str], cls: str, line_height: int) -> None:
    out.append(f'<text x="{x}" y="{y}" class="{cls}" xml:space="preserve">')
    first = True
    for ln in lines:
        if first:
            out.append(f'  <tspan x="{x}" dy="0">{_esc(ln)}</tspan>')
            first = False
        else:
            out.append(f'  <tspan x="{x}" dy="{line_height}">{_esc(ln)}</tspan>')
    out.append("</text>")


def _draw_tile_grid(
    out: List[str],
    *,
    x: int,
    y: int,
    label: str,
    prefix: str,
    rows: int = TILE_ROWS,
    cols: int = TILE_COLS,
    highlight_cells: Iterable[Tuple[int, int]] = (),
    highlight_rows: Iterable[int] = (),
    highlight_cols: Iterable[int] = (),
    valid_box: Optional[Tuple[int, int]] = None,
    text_override: Optional[Dict[Tuple[int, int], str]] = None,
    accent: str,
) -> None:
    highlight = set(highlight_cells)
    highlight_r = set(highlight_rows)
    highlight_c = set(highlight_cols)
    text_override = text_override or {}

    w = _tile_width(cols)
    h = _tile_height(rows)

    if valid_box is None:
        # Use a schematic (not full) valid region to make masking visible in most diagrams.
        vr = rows - 1 if rows > 2 else rows
        vc = cols - 1 if cols > 2 else cols
        valid_box = (vr, vc)

    vr, vc = valid_box
    vr = max(0, min(rows, vr))
    vc = max(0, min(cols, vc))

    out.append(f'<text x="{x + w // 2}" y="{y - 10}" class="tileLabel" text-anchor="middle">{_esc(label)}</text>')
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" class="tileBorder" />')

    for r in range(rows):
        for c in range(cols):
            is_hl = (r, c) in highlight or r in highlight_r or c in highlight_c
            rx = x + c * CELL
            ry = y + r * CELL
            masked = (r >= vr) or (c >= vc)
            cls_parts = ["cell"]
            if masked:
                cls_parts.append("cellMasked")
            if is_hl:
                cls_parts.append("cellHL")
            cls = " ".join(cls_parts)
            if is_hl:
                out.append(f'<rect x="{rx}" y="{ry}" width="{CELL}" height="{CELL}" class="{cls}" stroke="{accent}" />')
            else:
                out.append(f'<rect x="{rx}" y="{ry}" width="{CELL}" height="{CELL}" class="{cls}" />')

            text = text_override.get((r, c))
            if text:
                cx, cy = _cell_center(x, y, r, c)
                _append_svg_text(
                    out, x=cx, y=cy + 1, cls="cellText", text=text, text_anchor="middle", dominant_baseline="middle"
                )

    if vr and vc:
        out.append(
            f'<rect x="{x}" y="{y}" width="{vc * CELL}" height="{vr * CELL}" class="validBox" stroke="{accent}" />'
        )

        # Indicate the valid extents (Rv/Cv) without hard-coding numeric sizes.
        out.append(f'<text x="{x + 4}" y="{y + vr * CELL - 4}" class="axisText">{_esc("Rv")}</text>')
        out.append(f'<text x="{x + vc * CELL - 4}" y="{y + 12}" class="axisText" text-anchor="end">{_esc("Cv")}</text>')

    # r/c axes indicator (schematic): row increases downward, col increases to the right.
    _ = prefix  # reserved for future per-cell callouts
    if rows >= 3 and cols >= 3:
        ax = x + 10
        ay = y + 10
        out.append(f'<path d="M {ax} {ay} L {ax + 34} {ay}" class="axisLine" marker-end="url(#axisArrow)" />')
        out.append(f'<path d="M {ax} {ay} L {ax} {ay + 34}" class="axisLine" marker-end="url(#axisArrow)" />')
        out.append(f'<text x="{ax + 38}" y="{ay + 4}" class="axisText">{_esc("c")}</text>')
        out.append(f'<text x="{ax - 2}" y="{ay + 38}" class="axisText" text-anchor="end">{_esc("r")}</text>')


def _draw_scalar_box(out: List[str], *, x: int, y: int, label: str, value: str, accent: str) -> None:
    w = 160
    h = 54
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" class="scalarBox" />')
    out.append(f'<text x="{x + 12}" y="{y + 20}" class="smallLabel">{_esc(label)}</text>')
    out.append(f'<text x="{x + 12}" y="{y + 42}" class="scalarValue" fill="{accent}">{_esc(value)}</text>')


def _draw_mem_row(
    out: List[str],
    *,
    x: int,
    y: int,
    label: str,
    prefix: str,
    cells: int = 12,
    highlight_idx: Optional[int] = None,
    accent: Optional[str] = None,
    text_override: Optional[Dict[int, str]] = None,
) -> None:
    _ = prefix
    text_override = text_override or {}
    cell_w = CELL
    w = cells * cell_w
    h = CELL
    out.append(f'<text x="{x + w // 2}" y="{y - 10}" class="tileLabel" text-anchor="middle">{_esc(label)}</text>')
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" class="tileBorder" />')
    for i in range(cells):
        rx = x + i * cell_w
        is_hl = highlight_idx is not None and i == highlight_idx
        cls = "cell"
        if is_hl:
            cls += " cellHL"
        if is_hl and accent:
            out.append(f'<rect x="{rx}" y="{y}" width="{cell_w}" height="{CELL}" class="{cls}" stroke="{accent}" />')
        else:
            out.append(f'<rect x="{rx}" y="{y}" width="{cell_w}" height="{CELL}" class="{cls}" />')
        text = text_override.get(i)
        if text:
            _append_svg_text(
                out,
                x=rx + cell_w // 2,
                y=y + CELL // 2 + 1,
                cls="cellText",
                text=text,
                text_anchor="middle",
                dominant_baseline="middle",
            )


def _layout_row_lefts(center_x: int, widths: Sequence[int], gap: int) -> List[int]:
    if not widths:
        return []
    total = sum(widths) + gap * (len(widths) - 1)
    start = int(center_x - total / 2)
    xs: List[int] = []
    cur = start
    for w in widths:
        xs.append(cur)
        cur += w + gap
    return xs


def _cell_anchor_top(x: int, y: int, r: int, c: int) -> Tuple[int, int]:
    return (x + c * CELL + CELL // 2, y + r * CELL)


def _cell_anchor_bottom(x: int, y: int, r: int, c: int) -> Tuple[int, int]:
    return (x + c * CELL + CELL // 2, y + (r + 1) * CELL)


def _cell_anchor_left(x: int, y: int, r: int, c: int) -> Tuple[int, int]:
    return (x + c * CELL, y + r * CELL + CELL // 2)


def _cell_anchor_right(x: int, y: int, r: int, c: int) -> Tuple[int, int]:
    return (x + (c + 1) * CELL, y + r * CELL + CELL // 2)


def _tile_port_top(*, x: int, y: int, rows: int, cols: int, c: int) -> Tuple[int, int]:
    _ = rows
    _ = cols
    return (x + c * CELL + CELL // 2, y - ARROW_PAD)


def _tile_port_bottom(*, x: int, y: int, rows: int, cols: int, c: int) -> Tuple[int, int]:
    _ = cols
    return (x + c * CELL + CELL // 2, y + rows * CELL + ARROW_PAD)


def _scalar_port_bottom(*, x: int, y: int, w: int = 160, h: int = 54) -> Tuple[int, int]:
    return (x + w // 2, y + h + ARROW_PAD)


def _scalar_port_top(*, x: int, y: int, w: int = 160, h: int = 54) -> Tuple[int, int]:
    _ = h
    return (x + w // 2, y - ARROW_PAD)


def _draw_op_node(
    out: List[str], *, cx: int, cy: int, instr: str, accent: str
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Draw a circled-square op node and return (left, right, bottom) anchors."""
    max_chars_per_line = 6
    label_lines = [instr[i : i + max_chars_per_line] for i in range(0, len(instr), max_chars_per_line)]
    line_count = max(1, len(label_lines))

    if line_count == 1:
        font_px = 10
    elif line_count == 2:
        font_px = 8
    else:
        font_px = 7

    line_gap = font_px + 1
    text_w = max(len(line) for line in label_lines) * font_px * 0.62
    text_h = line_count * line_gap - 1
    side = int(max(24, text_w + 8, text_h + 8))
    side = min(side, 34)
    r = max(18, side // 2 + 5)

    out.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" class="opCircle" stroke="{_esc(accent)}" />')
    _append_rect(out, x=cx - side // 2, y=cy - side // 2, width=side, height=side, rx=6, cls="opRect", stroke=accent)
    y0 = cy - ((line_count - 1) * line_gap) / 2
    for idx, line in enumerate(label_lines):
        y = y0 + idx * line_gap + 3
        _append_svg_text(
            out, x=cx, y=f"{y:.1f}", cls="opText", text=line, font_size=f"{font_px}px", text_anchor="middle"
        )
    return ((cx - r, cy), (cx + r, cy), (cx, cy + r))


def _draw_binary_flow(
    out: List[str],
    *,
    instr: str,
    left_src: Tuple[int, int],
    right_src: Tuple[int, int],
    dst: Tuple[int, int],
    accent: str,
    op_cx: Optional[int] = None,
    op_cy: Optional[int] = None,
) -> None:
    """Route two sources through a circled-square op node into one destination."""
    # Ensure "left" and "right" are geometrically left/right for clear diagrams.
    l_src, r_src = (left_src, right_src) if left_src[0] <= right_src[0] else (right_src, left_src)
    dx, dy = dst

    if op_cx is None:
        op_cx = int((l_src[0] + r_src[0]) / 2)
    if op_cy is None:
        op_cy = int((max(l_src[1], r_src[1]) + dy) / 2)

    left, right, bottom = _draw_op_node(out, cx=op_cx, cy=op_cy, instr=instr, accent=accent)
    _draw_ortho_arrow(out, x1=l_src[0], y1=l_src[1], x2=left[0], y2=left[1], via_y=left[1], accent=accent)
    _draw_ortho_arrow(out, x1=r_src[0], y1=r_src[1], x2=right[0], y2=right[1], via_y=right[1], accent=accent)
    _draw_ortho_arrow(out, x1=bottom[0], y1=bottom[1], x2=dx, y2=dy, via_y=int((bottom[1] + dy) / 2), accent=accent)


def _mem_anchor_top(x: int, y: int, i: int) -> Tuple[int, int]:
    return (x + i * CELL + CELL // 2, y - ARROW_PAD)


def _mem_anchor_bottom(x: int, y: int, i: int) -> Tuple[int, int]:
    return (x + i * CELL + CELL // 2, y + CELL + ARROW_PAD)


def _mem_anchor_left(x: int, y: int, i: int) -> Tuple[int, int]:
    return (x + i * CELL - ARROW_PAD, y + CELL // 2)


def _mem_anchor_right(x: int, y: int, i: int) -> Tuple[int, int]:
    return (x + (i + 1) * CELL + ARROW_PAD, y + CELL // 2)


def _draw_ortho_arrow(
    out: List[str],
    *,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    accent: str,
    via_y: Optional[int] = None,
    via_x: Optional[int] = None,
) -> None:
    if via_y is None and via_x is None:
        via_y = int((y1 + y2) / 2)
    if via_y is not None and via_x is not None:
        raise ValueError("specify at most one of via_y/via_x")
    if via_y is not None:
        d = f"M {x1} {y1} L {x1} {via_y} L {x2} {via_y} L {x2} {y2}"
    else:
        assert via_x is not None
        d = f"M {x1} {y1} L {via_x} {y1} L {via_x} {y2} L {x2} {y2}"
    out.append(f'<path d="{d}" class="arrow" stroke="{accent}" marker-end="url(#arrow)" />')


def _begin_svg(instr: str, summary: str, template: str, accent: str, bg: str) -> List[str]:
    aria = f"{instr} tile operation diagram"
    out: List[str] = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}" '
        f'viewBox="0 0 {CANVAS_W} {CANVAS_H}" role="img" aria-label="{_esc(aria)}">'
    )
    out.append("<defs>")
    out.append(
        f'  <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" '
        f'orient="auto"><path d="M0,0 L0,12 L12,6 z" fill="{_esc(accent)}"/></marker>'
    )
    out.append(
        '  <marker id="axisArrow" markerWidth="10" markerHeight="10" refX="8" refY="5" '
        'orient="auto"><path d="M0,0 L0,10 L10,5 z" fill="#64748b"/></marker>'
    )
    out.append("</defs>")
    out.append("<style>")
    out.append(
        "\n".join(
            [
                "svg { font-family: Arial, Helvetica, sans-serif; }",
                ".title { font-size: 30px; font-weight: 700; fill: #0f172a; }",
                ".subtitle { font-size: 14px; fill: #334155; }",
                ".meta { font-size: 12px; fill: #64748b; }",
                ".frame { fill: white; }",
                ".panel { fill: " + bg + "; stroke: #e2e8f0; stroke-width: 1.5; rx: 14; }",
                ".tileLabel { font-size: 14px; font-weight: 700; fill: #0f172a; }",
                ".tileBorder { fill: none; stroke: #475569; stroke-width: 1.5; }",
                ".cell { fill: #ffffff; stroke: #94a3b8; stroke-width: 1; }",
                ".cellMasked { fill: #e2e8f0; }",
                ".cellHL { stroke-width: 2; }",
                ".cellText { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "
                "'Liberation Mono', 'Courier New', monospace; font-size: 10px; fill: #0f172a; }",
                ".arrow { stroke-width: 2.5; fill: none; stroke-linejoin: round; stroke-linecap: round; }",
                ".axisLine { stroke: #64748b; stroke-width: 1.5; fill: none; }",
                ".axisText { font-size: 10px; fill: #64748b; font-weight: 700; }",
                ".opCircle { fill: #ffffff; stroke-width: 2; }",
                ".opRect { fill: #ffffff; stroke-width: 2; }",
                ".opText { font-size: 10px; font-weight: 800; fill: #0f172a; }",
                ".procBox { fill: #f8fafc; stroke: #cbd5e1; stroke-width: 1.5; rx: 12; }",
                ".procTitle { font-size: 14px; font-weight: 700; fill: #0f172a; }",
                ".procText { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "
                "'Liberation Mono', 'Courier New', monospace; font-size: 12px; fill: #0f172a; }",
                ".smallLabel { font-size: 12px; fill: #334155; }",
                ".scalarBox { fill: #ffffff; stroke: #cbd5e1; stroke-width: 1.5; }",
                ".scalarValue { font-size: 16px; font-weight: 700; }",
                ".validBox { fill: none; stroke-width: 2; stroke-dasharray: 6 4; }",
            ]
        )
    )
    out.append("</style>")

    out.append(f'<rect x="0" y="0" width="{CANVAS_W}" height="{CANVAS_H}" class="frame" />')
    _append_rect(out, x=MARGIN, y=MARGIN, width=CANVAS_W - 2 * MARGIN, height=CANVAS_H - 2 * MARGIN, cls="panel")
    out.append(f'<text x="{MARGIN + 16}" y="{HEADER_Y}" class="title">{_esc(instr)}</text>')
    out.append(f'<text x="{MARGIN + 16}" y="{HEADER_Y + 26}" class="subtitle">{_esc(summary)}</text>')
    out.append(f'<text x="{MARGIN + 16}" y="{HEADER_Y + 46}" class="meta">Template: {_esc(template)}</text>')
    _append_svg_text(
        out,
        x=CANVAS_W - MARGIN - 16,
        y=HEADER_Y + 46,
        cls="meta",
        text="Legend: outline=example; dashed=valid rows/cols (Rv,Cv); shaded=masked; "
        "r down / c right; ortho arrows=dataflow",
        text_anchor="end",
    )
    out.append(
        f'<line x1="{MARGIN + 12}" y1="{HEADER_DIVIDER_Y}" '
        f'x2="{CANVAS_W - MARGIN - 12}" y2="{HEADER_DIVIDER_Y}" '
        'stroke="#e2e8f0" stroke-width="1.5" />'
    )
    return out


def _end_svg(out: List[str]) -> str:
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _draw_procedure(out: List[str], *, lines: Sequence[str], accent: str) -> None:
    x = MARGIN + 16
    y = PROC_BOX_Y + 14
    w = CANVAS_W - 2 * (MARGIN + 16)
    h = CANVAS_H - PROC_BOX_Y - MARGIN - 14
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" class="procBox" />')
    out.append(f'<text x="{x + PROC_PAD}" y="{y + 26}" class="procTitle">Procedure (conceptual)</text>')
    _draw_text_lines(out, x + PROC_PAD, y + 52, lines, "procText", 16)
    _append_svg_text(
        out,
        x=x + PROC_PAD,
        y=y + h - 16,
        cls="meta",
        text="Note: semantics apply to the valid region unless stated otherwise.",
    )


def _draw_expr(out: List[str], expr: str, accent: str) -> None:
    _append_svg_text(out, x=CANVAS_W // 2, y=EXPR_Y, cls="subtitle", text=expr, text_anchor="middle", fill=accent)


def _elementwise_spec(instr: str) -> Tuple[List[str], str, List[str]]:
    unary = {
        "TABS": "abs(src)",
        "TEXP": "exp(src)",
        "TLOG": "log(src)",
        "TNEG": "-src",
        "TNOT": "~src",
        "TRECIP": "1/src",
        "TRELU": "relu(src)",
        "TRSQRT": "rsqrt(src)",
        "TSQRT": "sqrt(src)",
        "TCVT": "convert(src, roundMode)",
    }
    binary = {
        "TADD": "src0 + src1",
        "TSUB": "src0 - src1",
        "TMUL": "src0 * src1",
        "TDIV": "src0 / src1",
        "TMIN": "min(src0, src1)",
        "TMAX": "max(src0, src1)",
        "TAND": "src0 & src1",
        "TOR": "src0 | src1",
        "TXOR": "src0 ^ src1",
        "TSHL": "src0 << src1",
        "TSHR": "src0 >> src1",
        "TREM": "remainder(src0, src1)",
        "TFMOD": "fmod(src0, src1)",
    }
    ternary = {
        "TMULADDDST": ("src0 * src1 + dst(old)", ["src0", "src1", "dst(old)"]),
        "TFUSEDMULADD": ("src0 * dst(old) + src1", ["src0", "dst(old)", "src1"]),
    }
    if instr in unary:
        expr = f"dst[r,c] = {unary[instr]}"
        proc = ["for r in 0..Rv-1:", "  for c in 0..Cv-1:", f"    {expr}"]
        return (["src"], expr, proc)
    if instr in ternary:
        op_expr, op_inputs = ternary[instr]
        expr = f"dst[r,c] = {op_expr}"
        proc = ["for r in 0..Rv-1:", "  for c in 0..Cv-1:", f"    {expr}"]
        return (op_inputs, expr, proc)
    if instr == "TSEL":
        expr = "dst[r,c] = (mask[r,c] != 0) ? src0[r,c] : src1[r,c]"
        proc = ["for r in 0..Rv-1:", "  for c in 0..Cv-1:", f"    {expr}"]
        return (["mask", "src0", "src1"], expr, proc)
    if instr == "TCMP":
        expr = "dst_mask[r,c] = cmp(src0[r,c], src1[r,c])"
        proc = ["for r in 0..Rv-1:", "  for c in 0..Cv-1:", f"    {expr}"]
        return (["src0", "src1"], expr, proc)
    if instr == "TPRELU":
        expr = "dst[r,c] = (x>0) ? x : slope*x"
        proc = ["for r in 0..Rv-1:", "  for c in 0..Cv-1:", "    x = src0[r,c]", "    slope = src1[r,c]", f"    {expr}"]
        return (["src0", "src1"], expr, proc)
    if instr in binary:
        expr = f"dst[r,c] = {binary[instr]}"
        proc = ["for r in 0..Rv-1:", "  for c in 0..Cv-1:", f"    {expr}"]
        return (["src0", "src1"], expr, proc)

    expr = "dst[r,c] = op(src...)[r,c]"
    proc = ["for r in 0..Rv-1:", "  for c in 0..Cv-1:", f"    {expr}"]
    return (["src0", "src1"], expr, proc)


def _scalar_spec(instr: str) -> Tuple[List[str], str, List[str]]:
    tile_scalar = {
        "TADDS": "src[r,c] + s",
        "TSUBS": "src[r,c] - s",
        "TMULS": "src[r,c] * s",
        "TDIVS": "src[r,c] / s   (or s / src[r,c])",
        "TMAXS": "max(src[r,c], s)",
        "TMINS": "min(src[r,c], s)",
        "TANDS": "src[r,c] & s",
        "TORS": "src[r,c] | s",
        "TXORS": "src[r,c] ^ s",
        "TSHLS": "src[r,c] << s",
        "TSHRS": "src[r,c] >> s",
        "TFMODS": "fmod(src[r,c], s)",
        "TREMS": "remainder(src[r,c], s)",
    }

    if instr == "TEXPANDS":
        expr = "dst[r,c] = s"
        proc = ["for r in 0..Rv-1:", "  for c in 0..Cv-1:", f"    {expr}"]
        return (["src(tile)"], expr, proc)
    if instr == "TCMPS":
        expr = "dst_mask[r,c] = cmp(src[r,c], s)"
        proc = ["for r in 0..Rv-1:", "  for c in 0..Cv-1:", f"    {expr}"]
        return (["src(tile)"], expr, proc)
    if instr == "TSELS":
        expr = "dst = (selectMode) ? src0 : src1"
        proc = ["if selectMode:", "  dst = src0", "else:", "  dst = src1"]
        return (["src0", "src1"], expr, proc)
    if instr == "TLRELU":
        expr = "dst[r,c] = (x>0) ? x : slope*x"
        proc = ["for r in 0..Rv-1:", "  for c in 0..Cv-1:", "    x = src[r,c]", f"    {expr}"]
        return (["src(tile)"], expr, proc)
    if instr in tile_scalar:
        expr = f"dst[r,c] = {tile_scalar[instr]}"
        proc = ["for r in 0..Rv-1:", "  for c in 0..Cv-1:", f"    {expr}"]
        return (["src(tile)"], expr, proc)

    expr = "dst[r,c] = op(src[r,c], s)"
    proc = ["for r in 0..Rv-1:", "  for c in 0..Cv-1:", f"    {expr}"]
    return (["src(tile)"], expr, proc)


def _reduce_expand_kind(instr: str) -> Tuple[str, str, str]:
    # Returns (mode, axis, op)
    # mode: reduce|expand|expand_op
    if instr in {"TROWSUM", "TROWMAX", "TROWMIN"}:
        return ("reduce", "row", instr.replace("TROW", "").lower())
    if instr in {"TCOLSUM", "TCOLMAX", "TCOLMIN"}:
        return ("reduce", "col", instr.replace("TCOL", "").lower())
    if instr in {"TROWEXPAND", "TCOLEXPAND"}:
        return ("expand", "row" if instr.startswith("TROW") else "col", "broadcast")
    if instr.startswith("TROWEXPAND"):
        return ("expand_op", "row", instr.replace("TROWEXPAND", "").lower() or "broadcast")
    if instr.startswith("TCOLEXPAND"):
        return ("expand_op", "col", instr.replace("TCOLEXPAND", "").lower() or "broadcast")
    return ("reduce_expand", "row", "op")


def _render_elementwise(instr: str, summary: str, accent: str, bg: str) -> str:
    if instr == "TDEQUANT":
        return _render_tdequant(instr, summary, accent, bg)

    inputs, expr, proc = _elementwise_spec(instr)
    out = _begin_svg(instr, summary, "elementwise", accent, bg)

    _draw_expr(out, expr, accent)

    tile_w = _tile_width(TILE_COLS)
    tile_h = _tile_height(TILE_ROWS)
    gap = 80
    y_src = SRC_Y
    y_dst = DST_Y

    if instr == "TSEL":
        prefixes = ["m", "a", "b"]
    else:
        prefixes = ["a", "b", "c"][: len(inputs)]

    xs = _layout_row_lefts(CANVAS_W // 2, [tile_w] * len(inputs), gap)
    for x, label, pfx in zip(xs, inputs, prefixes, strict=False):
        _draw_tile_grid(out, x=x, y=y_src, label=label, prefix=pfx, highlight_cells=[(EX_R, EX_C)], accent=accent)

    out_label = "dst(mask)" if instr == "TCMP" else "dst"
    out_prefix = "m" if instr == "TCMP" else "d"
    x_dst = (CANVAS_W - tile_w) // 2
    _draw_tile_grid(
        out, x=x_dst, y=y_dst, label=out_label, prefix=out_prefix, highlight_cells=[(EX_R, EX_C)], accent=accent
    )

    dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)

    # For binary ops, route through an explicit op node (circled square mnemonic).
    if len(inputs) == 2 and len(xs) == 2:
        s0x, s0y = _tile_port_bottom(x=xs[0], y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        s1x, s1y = _tile_port_bottom(x=xs[1], y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        _draw_binary_flow(
            out,
            instr=instr,
            left_src=(s0x, s0y),
            right_src=(s1x, s1y),
            dst=(dx, dy),
            accent=accent,
            op_cx=CANVAS_W // 2,
        )
    else:
        via_base = int((y_src + tile_h + y_dst) / 2)
        n = max(1, len(xs))
        for i, x in enumerate(xs):
            sx, sy = _tile_port_bottom(x=x, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
            via_y = via_base + int((i - (n - 1) / 2) * 14)
            _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=via_y, accent=accent)

    _draw_procedure(out, lines=proc, accent=accent)
    return _end_svg(out)


def _render_taxpy(instr: str, summary: str, accent: str, bg: str) -> str:
    out = _begin_svg(instr, summary, "scalar", accent, bg)
    expr = "dst[r,c] = dst(old)[r,c] + src[r,c] * scalar"
    _draw_expr(out, expr, accent)

    tile_w = _tile_width(TILE_COLS)
    y_src = SRC_Y
    y_dst = DST_Y
    xs = _layout_row_lefts(CANVAS_W // 2, [tile_w, tile_w, 160], 70)
    x_dst_old, x_src, x_scalar = xs[0], xs[1], xs[2]

    _draw_tile_grid(
        out, x=x_dst_old, y=y_src, label="dst(old)", prefix="d", highlight_cells=[(EX_R, EX_C)], accent=accent
    )
    _draw_tile_grid(out, x=x_src, y=y_src, label="src", prefix="a", highlight_cells=[(EX_R, EX_C)], accent=accent)
    _draw_scalar_box(out, x=x_scalar, y=y_src + 8, label="scalar", value="s", accent=accent)

    x_dst = (CANVAS_W - tile_w) // 2
    _draw_tile_grid(out, x=x_dst, y=y_dst, label="dst(out)", prefix="d", highlight_cells=[(EX_R, EX_C)], accent=accent)

    dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
    sources = [
        _tile_port_bottom(x=x_dst_old, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C),
        _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C),
        _scalar_port_bottom(x=x_scalar, y=y_src + 8),
    ]
    via_base = int((y_src + _tile_height(TILE_ROWS) + y_dst) / 2)
    for idx, (sx, sy) in enumerate(sources):
        _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=via_base + (idx - 1) * 14, accent=accent)

    proc = [
        "for r in 0..Rv-1:",
        "  for c in 0..Cv-1:",
        "    dst[r,c] = dst[r,c] + src[r,c] * scalar",
        "dst is read and written in-place.",
    ]
    _draw_procedure(out, lines=proc, accent=accent)
    return _end_svg(out)


def _render_tdequant(instr: str, summary: str, accent: str, bg: str) -> str:
    out = _begin_svg(instr, summary, "elementwise", accent, bg)
    expr = "dst[r,c] = (src[r,c] - offset[r,pc]) * scale[r,pc]"
    _draw_expr(out, expr, accent)

    tile_w = _tile_width(TILE_COLS)
    scale_cols = 3
    scale_w = _tile_width(scale_cols)
    y_src = SRC_Y
    y_para = y_src + 22
    y_dst = DST_Y
    xs = _layout_row_lefts(CANVAS_W // 2, [tile_w, scale_w, scale_w], 80)
    x_src, x_scale, x_offset = xs[0], xs[1], xs[2]

    _draw_tdequant_inputs(out, accent, x_src, x_scale, x_offset, y_src, y_para, scale_cols)

    x_dst = (CANVAS_W - tile_w) // 2
    _draw_tile_grid(
        out, x=x_dst, y=y_dst, label="dst (FP32)", prefix="d", highlight_cells=[(EX_R, EX_C)], accent=accent
    )

    dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
    sources = [
        _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C),
        _tile_port_bottom(x=x_scale, y=y_para, rows=TILE_ROWS, cols=scale_cols, c=min(EX_C, scale_cols - 1)),
        _tile_port_bottom(x=x_offset, y=y_para, rows=TILE_ROWS, cols=scale_cols, c=min(EX_C, scale_cols - 1)),
    ]
    via_base = int((y_src + _tile_height(TILE_ROWS) + y_dst) / 2)
    for idx, (sx, sy) in enumerate(sources):
        _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=via_base + (idx - 1) * 14, accent=accent)

    proc = [
        "paraCols = max(1, scale.validCols)",
        "for r in 0..Rv-1:",
        "  for c in 0..Cv-1:",
        "    pc = min(c, paraCols - 1)",
        "    dst[r,c] = (src[r,c] - offset[r,pc]) * scale[r,pc]",
    ]
    _draw_procedure(out, lines=proc, accent=accent)
    return _end_svg(out)


def _draw_tdequant_inputs(
    out: List[str], accent: str, x_src: int, x_scale: int, x_offset: int, y_src: int, y_para: int, scale_cols: int
) -> None:
    _draw_tile_grid(
        out, x=x_src, y=y_src, label="src (S8/S16)", prefix="q", highlight_cells=[(EX_R, EX_C)], accent=accent
    )
    for x, label, prefix in ((x_scale, "scale", "s"), (x_offset, "offset", "o")):
        _draw_tile_grid(
            out,
            x=x,
            y=y_para,
            label=label,
            prefix=prefix,
            rows=TILE_ROWS,
            cols=scale_cols,
            highlight_cells=[(EX_R, min(EX_C, scale_cols - 1))],
            accent=accent,
        )


def _render_scalar(instr: str, summary: str, accent: str, bg: str) -> str:
    if instr == "TAXPY":
        return _render_taxpy(instr, summary, accent, bg)

    _inputs, expr, proc = _scalar_spec(instr)
    out = _begin_svg(instr, summary, "scalar", accent, bg)
    _draw_expr(out, expr, accent)

    tile_w = _tile_width(TILE_COLS)
    tile_h = _tile_height(TILE_ROWS)
    gap = 80
    y_src = SRC_Y
    y_dst = DST_Y

    src_labels = ["src0", "src1"] if instr == "TSELS" else ["src"]
    src_prefixes = ["a", "b"] if len(src_labels) == 2 else ["a"]

    xs = _layout_row_lefts(CANVAS_W // 2, [tile_w] * len(src_labels), gap)
    for x, label, pfx in zip(xs, src_labels, src_prefixes, strict=False):
        _draw_tile_grid(out, x=x, y=y_src, label=label, prefix=pfx, highlight_cells=[(EX_R, EX_C)], accent=accent)

    scalar_x, scalar_y = _draw_scalar_operand(out, instr, y_src, accent)
    x_dst = _draw_scalar_output(out, instr, tile_w, y_dst, accent)
    dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
    via_base = int((y_src + tile_h + y_dst) / 2)
    sources = _scalar_sources(xs, scalar_x, scalar_y, y_src)
    _draw_scalar_arrows(out, instr, sources, (dx, dy), via_base, accent)
    _draw_procedure(out, lines=proc, accent=accent)
    return _end_svg(out)


def _draw_scalar_operand(out: List[str], instr: str, y_src: int, accent: str) -> Tuple[int, int]:
    scalar_label = "selectMode" if instr == "TSELS" else "scalar"
    scalar_value = "mode" if instr == "TSELS" else "s"
    scalar_x = CANVAS_W - MARGIN - 16 - 160
    scalar_y = y_src + 8
    _draw_scalar_box(out, x=scalar_x, y=scalar_y, label=scalar_label, value=scalar_value, accent=accent)
    return (scalar_x, scalar_y)


def _draw_scalar_output(out: List[str], instr: str, tile_w: int, y_dst: int, accent: str) -> int:
    out_label = "dst(mask)" if instr == "TCMPS" else "dst"
    out_prefix = "m" if instr == "TCMPS" else "d"
    x_dst = (CANVAS_W - tile_w) // 2
    _draw_tile_grid(
        out, x=x_dst, y=y_dst, label=out_label, prefix=out_prefix, highlight_cells=[(EX_R, EX_C)], accent=accent
    )
    return x_dst


def _scalar_sources(xs: Sequence[int], scalar_x: int, scalar_y: int, y_src: int) -> List[Tuple[int, int]]:
    sources: List[Tuple[int, int]] = []
    for x in xs:
        sources.append(_tile_port_bottom(x=x, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C))
    sources.append(_scalar_port_bottom(x=scalar_x, y=scalar_y))
    return sources


def _draw_scalar_arrows(
    out: List[str], instr: str, sources: Sequence[Tuple[int, int]], dst: Tuple[int, int], via_base: int, accent: str
) -> None:
    if len(sources) == 2:
        _draw_binary_flow(
            out, instr=instr, left_src=sources[0], right_src=sources[1], dst=dst, accent=accent, op_cx=CANVAS_W // 2
        )
    else:
        n = len(sources)
        for i, (sx, sy) in enumerate(sources):
            via_y = via_base + int((i - (n - 1) / 2) * 14)
            _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dst[0], y2=dst[1], via_y=via_y, accent=accent)


def _render_reduce_expand(instr: str, summary: str, accent: str, bg: str) -> str:
    mode, axis, op = _reduce_expand_kind(instr)
    out = _begin_svg(instr, summary, "reduce_expand", accent, bg)

    tile_w = _tile_width(TILE_COLS)
    tile_h = _tile_height(TILE_ROWS)
    y_src = SRC_Y
    y_dst = DST_Y

    if mode == "reduce":
        if axis == "row":
            expr = f"dst[r,0] = {op}_c src[r,c]"
            proc = ["for r in 0..Rv-1:", f"  dst[r,0] = {op} over c=0..Cv-1 of src[r,c]"]
            x_src = (CANVAS_W - tile_w) // 2
            x_dst = (CANVAS_W - _tile_width(1)) // 2
            _draw_tile_grid(out, x=x_src, y=y_src, label="src", prefix="a", highlight_rows=[EX_R], accent=accent)
            _draw_tile_grid(
                out,
                x=x_dst,
                y=y_dst,
                label="dst (column vector)",
                prefix="d",
                rows=TILE_ROWS,
                cols=1,
                highlight_cells=[(EX_R, 0)],
                accent=accent,
            )
            sx, sy = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
            dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=1, c=0)
        else:
            expr = f"dst[0,c] = {op}_r src[r,c]"
            proc = ["for c in 0..Cv-1:", f"  dst[0,c] = {op} over r=0..Rv-1 of src[r,c]"]
            x_src = (CANVAS_W - tile_w) // 2
            dst_rows, dst_cols = 1, TILE_COLS
            dst_w, dst_h = _tile_width(dst_cols), _tile_height(dst_rows)
            x_dst = (CANVAS_W - dst_w) // 2
            y_dst2 = y_dst + (tile_h - dst_h) // 2
            _draw_tile_grid(out, x=x_src, y=y_src, label="src", prefix="a", highlight_cols=[EX_C], accent=accent)
            _draw_tile_grid(
                out,
                x=x_dst,
                y=y_dst2,
                label="dst (row vector)",
                prefix="d",
                rows=dst_rows,
                cols=dst_cols,
                highlight_cells=[(0, EX_C)],
                accent=accent,
            )
            sx, sy = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
            dx, dy = _tile_port_top(x=x_dst, y=y_dst2, rows=dst_rows, cols=dst_cols, c=EX_C)

        _draw_expr(out, expr, accent)
        via_y = int((sy + dy) / 2)
        _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=via_y, accent=accent)
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if mode == "expand":
        if axis == "row":
            expr = "dst[r,c] = src[r,0]"
            proc = ["for r in 0..Rv-1:", "  v = src[r,0]", "  for c in 0..Cv-1:", "    dst[r,c] = v"]
            x_src = (CANVAS_W - tile_w) // 2
            x_dst = (CANVAS_W - tile_w) // 2
            _draw_tile_grid(out, x=x_src, y=y_src, label="src", prefix="a", highlight_cells=[(EX_R, 0)], accent=accent)
            _draw_tile_grid(
                out,
                x=x_dst,
                y=y_dst,
                label="dst",
                prefix="d",
                highlight_rows=[EX_R],
                highlight_cells=[(EX_R, EX_C)],
                accent=accent,
            )
            sx, sy = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=0)
            dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        else:
            expr = "dst[r,c] = src[0,c]"
            proc = ["for c in 0..Cv-1:", "  v = src[0,c]", "  for r in 0..Rv-1:", "    dst[r,c] = v"]
            x_src = (CANVAS_W - tile_w) // 2
            x_dst = (CANVAS_W - tile_w) // 2
            _draw_tile_grid(out, x=x_src, y=y_src, label="src", prefix="a", highlight_cells=[(0, EX_C)], accent=accent)
            _draw_tile_grid(
                out,
                x=x_dst,
                y=y_dst,
                label="dst",
                prefix="d",
                highlight_cols=[EX_C],
                highlight_cells=[(EX_R, EX_C)],
                accent=accent,
            )
            sx, sy = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
            dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)

        _draw_expr(out, expr, accent)
        via_y = int((sy + dy) / 2)
        _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=via_y, accent=accent)
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    # expand_op
    if axis == "row":
        s_rows, s_cols = TILE_ROWS, 1
        s_h = _tile_height(s_rows)
        y_s = y_src
        s_cell = (EX_R, 0)
        scalar_ref = "s = src1[r,0]"
    else:
        s_rows, s_cols = 1, TILE_COLS
        s_h = _tile_height(s_rows)
        y_s = y_src + (tile_h - s_h) // 2
        s_cell = (0, EX_C)
        scalar_ref = "s = src1[0,c]"

    if op == "expdif":
        expr = "dst[r,c] = exp(src0[r,c] - s)"
    elif op == "add":
        expr = "dst[r,c] = src0[r,c] + s"
    elif op == "sub":
        expr = "dst[r,c] = src0[r,c] - s"
    elif op == "mul":
        expr = "dst[r,c] = src0[r,c] * s"
    elif op == "div":
        expr = "dst[r,c] = src0[r,c] / s"
    elif op == "max":
        expr = "dst[r,c] = max(src0[r,c], s)"
    elif op == "min":
        expr = "dst[r,c] = min(src0[r,c], s)"
    else:
        expr = "dst[r,c] = op(src0[r,c], s)"

    proc = ["for r in 0..Rv-1:", "  for c in 0..Cv-1:", f"    {scalar_ref}", f"    {expr}"]

    widths = [tile_w, _tile_width(s_cols)]
    xs = _layout_row_lefts(CANVAS_W // 2, widths, 80)
    x_src0, x_src1 = xs[0], xs[1]
    x_dst = (CANVAS_W - tile_w) // 2

    _draw_tile_grid(out, x=x_src0, y=y_src, label="src0", prefix="a", highlight_cells=[(EX_R, EX_C)], accent=accent)
    _draw_tile_grid(
        out,
        x=x_src1,
        y=y_s,
        label="src1 (per-row)" if axis == "row" else "src1 (per-col)",
        prefix="s",
        rows=s_rows,
        cols=s_cols,
        highlight_cells=[s_cell],
        accent=accent,
    )
    _draw_tile_grid(out, x=x_dst, y=y_dst, label="dst", prefix="d", highlight_cells=[(EX_R, EX_C)], accent=accent)

    _draw_expr(out, expr, accent)

    dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
    a_x, a_y = _tile_port_bottom(x=x_src0, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
    s_x, s_y = _tile_port_bottom(x=x_src1, y=y_s, rows=s_rows, cols=s_cols, c=s_cell[1])
    _draw_binary_flow(
        out, instr=instr, left_src=(a_x, a_y), right_src=(s_x, s_y), dst=(dx, dy), accent=accent, op_cx=CANVAS_W // 2
    )

    _draw_procedure(out, lines=proc, accent=accent)
    return _end_svg(out)


def _render_memory(instr: str, summary: str, accent: str, bg: str) -> str:
    out = _begin_svg(instr, summary, "memory", accent, bg)
    tile_w = _tile_width(TILE_COLS)
    tile_h = _tile_height(TILE_ROWS)
    y_src = SRC_Y
    y_dst = DST_Y

    if instr == "TPREFETCH_ASYNC":
        expr = "AsyncEvent = SDMA_CMO_PREFETCH(GM base, totalBytes)"
        proc = [
            "if src is null, not flat-contiguous, or byte count is zero:",
            "  return empty SDMA AsyncEvent",
            "initialize or reuse SDMA session from PrefetchAsyncContext",
            "submit SDMA CMO prefetch for src.data(), totalBytes",
            "return AsyncEvent for optional wait/test by the caller",
        ]
        _draw_expr(out, expr, accent)

        mem_w = 12 * CELL
        cache_w = 300
        cache_h = 86
        xs = _layout_row_lefts(CANVAS_W // 2, [mem_w, 160, cache_w], 100)
        x_mem, x_ctx, x_cache = xs[0], xs[1], xs[2]
        y_mem = y_src + 46
        y_ctx = y_src + 28
        y_cache = y_src + 20

        _draw_mem_row(out, x=x_mem, y=y_mem, label="GlobalTensor / GM", prefix="g", highlight_idx=6, accent=accent)
        _draw_scalar_box(out, x=x_ctx, y=y_ctx, label="context", value="SDMA", accent=accent)
        _append_rect(
            out,
            x=x_cache,
            y=y_cache,
            width=cache_w,
            height=cache_h,
            rx=14,
            fill="#ffffff",
            stroke=accent,
            stroke_width=2,
        )
        out.append(f'<text x="{x_cache + 18}" y="{y_cache + 34}" class="tileLabel">L2 cache</text>')
        _draw_text_lines(
            out, x_cache + 18, y_cache + 56, ["cache lines warmed", "data remains in GM"], "smallLabel", 18
        )

        event_x = (CANVAS_W - 240) // 2
        event_y = y_dst + 20
        _append_rect(
            out, x=event_x, y=event_y, width=240, height=70, rx=14, fill="#ffffff", stroke=accent, stroke_width=2
        )
        _append_svg_text(out, x=event_x + 120, y=event_y + 32, cls="tileLabel", text="AsyncEvent", text_anchor="middle")
        _append_svg_text(
            out, x=event_x + 120, y=event_y + 54, cls="smallLabel", text="DmaEngine::SDMA", text_anchor="middle"
        )

        gm_x, gm_y = _mem_anchor_right(x_mem, y_mem, 6)
        ctx_x, ctx_y = _scalar_port_bottom(x=x_ctx, y=y_ctx)
        _draw_ortho_arrow(out, x1=gm_x, y1=gm_y, x2=x_cache, y2=y_cache + cache_h // 2, via_x=x_ctx - 28, accent=accent)
        _draw_ortho_arrow(
            out, x1=ctx_x, y1=ctx_y, x2=event_x + 120, y2=event_y, via_y=int((ctx_y + event_y) / 2), accent=accent
        )
        _draw_ortho_arrow(
            out,
            x1=x_cache + cache_w // 2,
            y1=y_cache + cache_h,
            x2=event_x + 120,
            y2=event_y,
            via_y=int((y_cache + cache_h + event_y) / 2),
            accent=accent,
        )
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr in {"TLOAD", "TPREFETCH"}:
        expr = "dst[r,c] = GM[...]"
        proc = ["for r,c in valid(dst):", "  dst[r,c] = GM[base + (row0+r)*stride + (col0+c)]"]
        _draw_expr(out, expr, accent)
        mem_w = 12 * CELL
        x_mem = (CANVAS_W - mem_w) // 2
        y_mem = y_src + (tile_h - CELL) // 2
        _draw_mem_row(out, x=x_mem, y=y_mem, label="GlobalTensor / GM", prefix="g", highlight_idx=6, accent=accent)

        x_tile = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(
            out, x=x_tile, y=y_dst, label="dst tile", prefix="d", highlight_cells=[(EX_R, EX_C)], accent=accent
        )

        sx, sy = _mem_anchor_bottom(x_mem, y_mem, 6)
        dx, dy = _tile_port_top(x=x_tile, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        via_y = int((sy + dy) / 2)
        _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=via_y, accent=accent)
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr == "TSTORE":
        expr = "GM[...] = src[r,c]"
        proc = ["for r,c in valid(src):", "  GM[base + (row0+r)*stride + (col0+c)] = src[r,c]"]
        _draw_expr(out, expr, accent)
        x_src = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(
            out, x=x_src, y=y_src, label="src tile", prefix="a", highlight_cells=[(EX_R, EX_C)], accent=accent
        )

        mem_w = 12 * CELL
        x_mem = (CANVAS_W - mem_w) // 2
        y_mem = y_dst + (tile_h - CELL) // 2
        _draw_mem_row(out, x=x_mem, y=y_mem, label="GlobalTensor / GM", prefix="g", highlight_idx=6, accent=accent)

        sx, sy = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        dx, dy = _mem_anchor_top(x_mem, y_mem, 6)
        via_y = int((sy + dy) / 2)
        _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=via_y, accent=accent)
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr == "TSTORE_FP":
        expr = "GM[...] = quantize(src, fp)"
        proc = ["for r,c in valid(src):", "  q = quantize(src[r,c], fp[r,c])", "  GM[...] = q"]
        _draw_expr(out, expr, accent)
        xs = _layout_row_lefts(CANVAS_W // 2, [tile_w, tile_w], 80)
        x_src, x_fp = xs[0], xs[1]
        _draw_tile_grid(
            out, x=x_src, y=y_src, label="src (acc)", prefix="a", highlight_cells=[(EX_R, EX_C)], accent=accent
        )
        _draw_tile_grid(
            out, x=x_fp, y=y_src, label="fp/scale", prefix="s", highlight_cells=[(EX_R, EX_C)], accent=accent
        )

        mem_w = 12 * CELL
        x_mem = (CANVAS_W - mem_w) // 2
        y_mem = y_dst + (tile_h - CELL) // 2
        _draw_mem_row(out, x=x_mem, y=y_mem, label="GlobalTensor / GM", prefix="g", highlight_idx=6, accent=accent)

        dx, dy = _mem_anchor_top(x_mem, y_mem, 6)
        a_x, a_y = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        f_x, f_y = _tile_port_bottom(x=x_fp, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        _draw_binary_flow(
            out,
            instr=instr,
            left_src=(a_x, a_y),
            right_src=(f_x, f_y),
            dst=(dx, dy),
            accent=accent,
            op_cx=CANVAS_W // 2,
        )
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr == "MGATHER":
        expr = "dst[r,c] = mem[indexes[r,c]]"
        proc = ["for r,c in valid(dst):", "  idx = indexes[r,c]", "  dst[r,c] = mem[idx]"]
        _draw_expr(out, expr, accent)
        mem_w = 12 * CELL
        widths = [mem_w, tile_w]
        xs = _layout_row_lefts(CANVAS_W // 2, widths, 80)
        x_mem, x_idx = xs[0], xs[1]
        y_mem = y_src + (tile_h - CELL) // 2
        _draw_mem_row(out, x=x_mem, y=y_mem, label="GM / mem", prefix="g", highlight_idx=10, accent=accent)
        _draw_tile_grid(
            out, x=x_idx, y=y_src, label="idx tile", prefix="i", highlight_cells=[(EX_R, EX_C)], accent=accent
        )

        x_dst = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(
            out, x=x_dst, y=y_dst, label="dst tile", prefix="d", highlight_cells=[(EX_R, EX_C)], accent=accent
        )

        dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        m_x, m_y = _mem_anchor_bottom(x_mem, y_mem, 10)
        i_x, i_y = _tile_port_bottom(x=x_idx, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        _draw_binary_flow(
            out,
            instr=instr,
            left_src=(m_x, m_y),
            right_src=(i_x, i_y),
            dst=(dx, dy),
            accent=accent,
            op_cx=CANVAS_W // 2,
        )
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr == "MSCATTER":
        expr = "mem[indexes[r,c]] = src[r,c]"
        proc = ["for r,c in valid(src):", "  idx = indexes[r,c]", "  mem[idx] = src[r,c]"]
        _draw_expr(out, expr, accent)
        xs = _layout_row_lefts(CANVAS_W // 2, [tile_w, tile_w], 80)
        x_src, x_idx = xs[0], xs[1]
        _draw_tile_grid(
            out, x=x_src, y=y_src, label="src tile", prefix="a", highlight_cells=[(EX_R, EX_C)], accent=accent
        )
        _draw_tile_grid(
            out, x=x_idx, y=y_src, label="idx tile", prefix="i", highlight_cells=[(EX_R, EX_C)], accent=accent
        )

        mem_w = 12 * CELL
        x_mem = (CANVAS_W - mem_w) // 2
        y_mem = y_dst + (tile_h - CELL) // 2
        _draw_mem_row(out, x=x_mem, y=y_mem, label="GM / mem", prefix="g", highlight_idx=10, accent=accent)

        dx, dy = _mem_anchor_top(x_mem, y_mem, 10)
        s_x, s_y = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        i_x, i_y = _tile_port_bottom(x=x_idx, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        _draw_binary_flow(
            out,
            instr=instr,
            left_src=(s_x, s_y),
            right_src=(i_x, i_y),
            dst=(dx, dy),
            accent=accent,
            op_cx=CANVAS_W // 2,
        )
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    _draw_procedure(out, lines=["(implementation-defined)"], accent=accent)
    return _end_svg(out)


def _render_matmul(instr: str, summary: str, accent: str, bg: str) -> str:
    if instr == "TMATMUL_MX_HIF4":
        return _render_matmul_mx_hif4(instr, summary, accent, bg)

    out = _begin_svg(instr, summary, "matmul", accent, bg)

    tile_w = _tile_width(TILE_COLS)
    y_src = SRC_Y
    y_dst = DST_Y

    label_a = "A" if instr.startswith("TMATMUL") else "A (vec)"
    label_b = "B" if instr.startswith("TMATMUL") else "B (vec)"

    xs = _layout_row_lefts(CANVAS_W // 2, [tile_w, tile_w], 120)
    x_a, x_b = xs[0], xs[1]
    x_c = (CANVAS_W - tile_w) // 2

    _draw_tile_grid(out, x=x_a, y=y_src, label=label_a, prefix="a", highlight_cells=[(EX_R, 1)], accent=accent)
    _draw_tile_grid(out, x=x_b, y=y_src, label=label_b, prefix="b", highlight_cells=[(1, EX_C)], accent=accent)
    _draw_tile_grid(out, x=x_c, y=y_dst, label="C / dst", prefix="c", highlight_cells=[(EX_R, EX_C)], accent=accent)

    if instr.startswith("TMATMUL"):
        expr = "C[i,j] = sum_k A[i,k] * B[k,j]"
    else:
        expr = "C[0,j] = sum_k A[0,k] * B[k,j]"

    _draw_expr(out, expr, accent)

    dx, dy = _tile_port_top(x=x_c, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
    a_x, a_y = _tile_port_bottom(x=x_a, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=1)
    b_x, b_y = _tile_port_bottom(x=x_b, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
    _draw_binary_flow(
        out, instr=instr, left_src=(a_x, a_y), right_src=(b_x, b_y), dst=(dx, dy), accent=accent, op_cx=CANVAS_W // 2
    )

    proc = [
        "for i,j in output valid region:",
        "  acc = 0",
        "  for k in 0..K-1:",
        "    acc += A[i,k] * B[k,j]",
        "  C[i,j] = acc   (+ bias/acc, if applicable)",
    ]
    _draw_procedure(out, lines=proc, accent=accent)
    return _end_svg(out)


def _render_matmul_mx_hif4(instr: str, summary: str, accent: str, bg: str) -> str:
    out = _begin_svg(instr, summary, "matmul", accent, bg)
    expr = "L0C[i,j] += dequant_hif4(A, scaleA)[i,k] * dequant_hif4(B, scaleB)[k,j]"
    _draw_expr(out, expr, accent)

    tile_w = _tile_width(TILE_COLS)
    scale_w = _tile_width(3)
    y_data = SRC_Y
    y_scale = SRC_Y + 126
    y_dst = DST_Y + 18
    xs = _layout_row_lefts(CANVAS_W // 2, [tile_w, scale_w, tile_w, scale_w], 48)
    x_a, x_sa, x_b, x_sb = xs[0], xs[1], xs[2], xs[3]

    _draw_hif4_matmul_inputs(out, accent, x_a, x_sa, x_b, x_sb, y_data, y_scale)
    x_c = (CANVAS_W - tile_w) // 2
    _draw_tile_grid(
        out, x=x_c, y=y_dst, label="L0C / dst (FP32)", prefix="c", highlight_cells=[(EX_R, EX_C)], accent=accent
    )
    _draw_hif4_matmul_arrows(out, accent, (x_a, x_sa, x_b, x_sb), x_c, y_data, y_scale, y_dst)
    _draw_procedure(out, lines=_hif4_matmul_proc(), accent=accent)
    return _end_svg(out)


def _draw_hif4_matmul_inputs(
    out: List[str], accent: str, x_a: int, x_sa: int, x_b: int, x_sb: int, y_data: int, y_scale: int
) -> None:
    _draw_tile_grid(
        out, x=x_a, y=y_data, label="A data (HiF4 PK4)", prefix="a", highlight_cells=[(EX_R, 1)], accent=accent
    )
    _draw_hif4_scale_patch(out, x_sa, y_scale, "A scale", "sa", accent)
    _draw_tile_grid(
        out, x=x_b, y=y_data, label="B data (HiF4 PK4)", prefix="b", highlight_cells=[(1, EX_C)], accent=accent
    )
    _draw_hif4_scale_patch(out, x_sb, y_scale, "B scale", "sb", accent)


def _draw_hif4_scale_patch(out: List[str], x: int, y: int, label: str, prefix: str, accent: str) -> None:
    _draw_tile_grid(
        out,
        x=x,
        y=y,
        label=label,
        prefix=prefix,
        rows=2,
        cols=3,
        highlight_cells=[(0, 1)],
        text_override={(0, 0): "Ea", (0, 1): "Eb", (1, 1): "Ec"},
        accent=accent,
    )


def _draw_hif4_matmul_arrows(
    out: List[str], accent: str, input_xs: Tuple[int, int, int, int], x_c: int, y_data: int, y_scale: int, y_dst: int
) -> None:
    x_a, x_sa, x_b, x_sb = input_xs
    dx, dy = _tile_port_top(x=x_c, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
    sources = [
        _tile_port_bottom(x=x_a, y=y_data, rows=TILE_ROWS, cols=TILE_COLS, c=1),
        _tile_port_bottom(x=x_sa, y=y_scale, rows=2, cols=3, c=1),
        _tile_port_bottom(x=x_b, y=y_data, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C),
        _tile_port_bottom(x=x_sb, y=y_scale, rows=2, cols=3, c=1),
    ]
    via_base = int((y_scale + _tile_height(2) + y_dst) / 2)
    for idx, (sx, sy) in enumerate(sources):
        _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=via_base + (idx - 2) * 10, accent=accent)


def _hif4_matmul_proc() -> List[str]:
    return [
        "TEXTRACT places HiF4 data in L0A/L0B and scale patches in L0AMX/L0BMX.",
        "Each 64B scale patch carries Ea/Eb and Ec metadata for 64-value groups.",
        "mad_mx(hifloat4x2_t) applies the three-level scale inside Cube.",
        "Accumulate into L0C as FP32; TSTORE/FIXPIPE may cast the result to BF16.",
    ]


def _render_reshape_move(instr: str, summary: str, accent: str, bg: str) -> str:
    out = _begin_svg(instr, summary, "reshape_move", accent, bg)

    tile_w = _tile_width(TILE_COLS)
    y_src = SRC_Y
    y_dst = DST_Y

    if instr.startswith("TEXTRACT"):
        expr = "dst = slice(src, offset)"
        proc = ["for r,c in valid(dst):", "  dst[r,c] = src[r + row_off, c + col_off]"]
        _draw_expr(out, expr, accent)
        x_src = (CANVAS_W - tile_w) // 2
        x_dst = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(
            out, x=x_src, y=y_src, label="src", prefix="a", valid_box=(3, 3), highlight_cells=[(0, 0)], accent=accent
        )
        _draw_tile_grid(
            out, x=x_dst, y=y_dst, label="dst (window)", prefix="d", highlight_cells=[(0, 0)], accent=accent
        )
        sx, sy = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=0)
        dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=0)
        via_y = int((sy + dy) / 2)
        _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=via_y, accent=accent)
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr.startswith("TINSERT"):
        expr = "dst[off + (r,c)] = src[r,c]"
        proc = ["for r,c in valid(src):", "  dst[r + row_off, c + col_off] = src[r,c]"]
        _draw_expr(out, expr, accent)
        xs = _layout_row_lefts(CANVAS_W // 2, [tile_w, tile_w], 120)
        x_dst_in, x_src_win = xs[0], xs[1]
        x_dst_out = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(
            out, x=x_dst_in, y=y_src, label="dst (in)", prefix="d", highlight_cells=[(EX_R, EX_C)], accent=accent
        )
        _draw_tile_grid(
            out,
            x=x_src_win,
            y=y_src,
            label="src (window)",
            prefix="a",
            valid_box=(2, 2),
            highlight_cells=[(0, 0)],
            accent=accent,
        )
        _draw_tile_grid(
            out, x=x_dst_out, y=y_dst, label="dst (out)", prefix="d", highlight_cells=[(EX_R, EX_C)], accent=accent
        )
        dx, dy = _tile_port_top(x=x_dst_out, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        d_x, d_y = _tile_port_bottom(x=x_dst_in, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        s_x, s_y = _tile_port_bottom(x=x_src_win, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=0)
        _draw_binary_flow(
            out,
            instr=instr,
            left_src=(d_x, d_y),
            right_src=(s_x, s_y),
            dst=(dx, dy),
            accent=accent,
            op_cx=CANVAS_W // 2,
        )
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr.startswith("TFILLPAD"):
        expr = "dst = pad(src, pad_value)"
        proc = [
            "for r,c in full tile domain:",
            "  if (r,c) in valid(src):",
            "    dst[r,c] = src[r,c]",
            "  else:",
            "    dst[r,c] = pad_value",
        ]
        _draw_expr(out, expr, accent)
        x_src = (CANVAS_W - tile_w) // 2
        x_dst = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(
            out,
            x=x_src,
            y=y_src,
            label="src (valid)",
            prefix="a",
            valid_box=(3, 3),
            highlight_cells=[(EX_R, EX_C)],
            accent=accent,
        )
        _draw_tile_grid(
            out,
            x=x_dst,
            y=y_dst,
            label="dst (padded)",
            prefix="d",
            valid_box=(3, 3),
            highlight_cells=[(3, 3)],
            accent=accent,
        )
        sx, sy = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        via_y = int((sy + dy) / 2)
        _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=via_y, accent=accent)
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr == "TTRANS":
        expr = "dst[r,c] = src[c,r]"
        proc = ["for r,c in valid(dst):", "  dst[r,c] = src[c,r]"]
        _draw_expr(out, expr, accent)
        x_src = (CANVAS_W - tile_w) // 2
        x_dst = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(out, x=x_src, y=y_src, label="src", prefix="a", highlight_cells=[(EX_R, EX_C)], accent=accent)
        _draw_tile_grid(out, x=x_dst, y=y_dst, label="dst", prefix="d", highlight_cells=[(EX_C, EX_R)], accent=accent)
        sx, sy = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_R)
        via_y = int((sy + dy) / 2)
        _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=via_y, accent=accent)
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    # Default: movement/reshape
    expr = "dst = move/reshape(src)"
    proc = ["for r,c in valid(dst):", "  dst[r,c] = transform(src[r,c])   (layout/location dependent)"]
    _draw_expr(out, expr, accent)
    x_src = (CANVAS_W - tile_w) // 2
    x_dst = (CANVAS_W - tile_w) // 2
    _draw_tile_grid(out, x=x_src, y=y_src, label="src", prefix="a", highlight_cells=[(EX_R, EX_C)], accent=accent)
    _draw_tile_grid(out, x=x_dst, y=y_dst, label="dst", prefix="d", highlight_cells=[(EX_R, EX_C)], accent=accent)
    sx, sy = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
    dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
    via_y = int((sy + dy) / 2)
    _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=via_y, accent=accent)
    _draw_procedure(out, lines=proc, accent=accent)
    return _end_svg(out)


def _render_quant_dn(instr: str, summary: str, accent: str, bg: str) -> str:
    out = _begin_svg(instr, summary, "complex", accent, bg)
    expr = "TQUANT<0>: dst RowMajor + exp/max/scaling DN; TMOV<0> converts exp DN -> ZZ"
    _draw_expr(out, expr, accent)
    tile_w = _tile_width(TILE_COLS)
    y_src = SRC_Y
    y_dst = DST_Y
    x_src = (CANVAS_W - tile_w) // 2
    _draw_tile_grid(out, x=x_src, y=y_src, label="src (FP MxN)", prefix="a", highlight_cols=[EX_C], accent=accent)

    exp_rows = 3
    xs = _layout_row_lefts(CANVAS_W // 2, [tile_w, tile_w, tile_w], 70)
    x_dst, x_exp, x_zz = xs[0], xs[1], xs[2]
    _draw_tile_grid(out, x=x_dst, y=y_dst, label="dst (FP8/FP4 RM)", prefix="q", highlight_cols=[EX_C], accent=accent)
    _draw_quant_dn_exp_tiles(out, accent, x_exp, x_zz, y_dst + 22, exp_rows)
    _draw_quant_dn_arrows(out, accent, x_src, x_dst, x_exp, x_zz, y_src, y_dst, exp_rows)
    _draw_procedure(out, lines=_quant_dn_proc(), accent=accent)
    return _end_svg(out)


def _draw_quant_dn_exp_tiles(out: List[str], accent: str, x_exp: int, x_zz: int, y_exp: int, exp_rows: int) -> None:
    exp_tiles = [
        (x_exp, "exp/max/scale (DN)", "e", {(0, 0): "Mhat", (0, 1): "xN"}),
        (x_zz, "exp (ZZ via TMOV<0>)", "z", {(0, 0): "cb", (1, 1): "p/q"}),
    ]
    for x, label, prefix, text_override in exp_tiles:
        _draw_tile_grid(
            out,
            x=x,
            y=y_exp,
            label=label,
            prefix=prefix,
            rows=exp_rows,
            cols=TILE_COLS,
            highlight_cells=[(1, EX_C)],
            text_override=text_override,
            accent=accent,
        )


def _draw_quant_dn_arrows(
    out: List[str], accent: str, x_src: int, x_dst: int, x_exp: int, x_zz: int, y_src: int, y_dst: int, exp_rows: int
) -> None:
    sx, sy = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
    d_x, d_y = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
    e_x, e_y = _tile_port_top(x=x_exp, y=y_dst + 22, rows=exp_rows, cols=TILE_COLS, c=EX_C)
    z_x, z_y = _tile_port_top(x=x_zz, y=y_dst + 22, rows=exp_rows, cols=TILE_COLS, c=EX_C)
    via = int((sy + d_y) / 2)
    _draw_ortho_arrow(out, x1=sx, y1=sy, x2=d_x, y2=d_y, via_y=via - 10, accent=accent)
    _draw_ortho_arrow(out, x1=sx, y1=sy, x2=e_x, y2=e_y, via_y=via + 10, accent=accent)
    _draw_ortho_arrow(out, x1=e_x, y1=e_y, x2=z_x, y2=z_y, via_y=e_y - 24, accent=accent)


def _quant_dn_proc() -> List[str]:
    return [
        "TQUANT<0,...> groups source rows along axis 0 in 32-row groups.",
        "dst keeps the source MxN RowMajor data layout after FP8/FP4 quantization.",
        "exp, max, and scaling use DN shape Mhat x N, not the Cube ZZ layout.",
        "Use TMOV<0>(expZZ, expDN, tmp) only when the exponent feeds MMAD_MX.",
    ]


def _render_quant_hif4(instr: str, summary: str, accent: str, bg: str) -> str:
    out = _begin_svg(instr, summary, "complex", accent, bg)
    expr = "dst(fp4 packed) + Ea/Eb/Ec metadata = HiF4Quant(src BF16)"
    _draw_expr(out, expr, accent)
    tile_w = _tile_width(TILE_COLS)
    y_src = SRC_Y
    y_dst = DST_Y
    xs = _layout_row_lefts(CANVAS_W // 2, [tile_w, tile_w, tile_w], 70)
    x_src, x_dst, x_meta = xs[0], xs[1], xs[2]
    _draw_tile_grid(
        out, x=x_src, y=y_src, label="src (BF16)", prefix="b", highlight_cells=[(EX_R, EX_C)], accent=accent
    )
    _draw_tile_grid(
        out,
        x=x_dst,
        y=y_dst,
        label="dst FP4 packed",
        prefix="h",
        rows=TILE_ROWS,
        cols=TILE_COLS,
        highlight_cells=[(EX_R, EX_C)],
        accent=accent,
    )
    _draw_tile_grid(
        out,
        x=x_meta,
        y=y_dst + 22,
        label="scale metadata",
        prefix="m",
        rows=3,
        cols=TILE_COLS,
        highlight_cells=[(0, 0), (1, EX_C), (2, EX_C)],
        text_override={(0, 0): "Ea", (1, 0): "Eb", (2, 0): "Ec"},
        accent=accent,
    )
    dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
    mx, my = _tile_port_top(x=x_meta, y=y_dst + 22, rows=3, cols=TILE_COLS, c=EX_C)
    s0 = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
    via = int((s0[1] + dy) / 2)
    _draw_ortho_arrow(out, x1=s0[0], y1=s0[1], x2=dx, y2=dy, via_y=via - 10, accent=accent)
    _draw_ortho_arrow(out, x1=s0[0], y1=s0[1], x2=mx, y2=my, via_y=via + 10, accent=accent)
    proc = [
        "for each 64-element HiF4 block:",
        "  Ma/Mb/Mc = max(abs(src)) over 64/8/4 element groups.",
        "  derive Ea(e6m2), Eb bits, Ec bits, and per-4 reciprocal scale.",
        "  q = round_to_e1m2(src * scale); pack FP4 codes into dst bytes.",
    ]
    _draw_procedure(out, lines=proc, accent=accent)
    return _end_svg(out)


def _render_histogram(instr: str, summary: str, accent: str, bg: str) -> str:
    out = _begin_svg(instr, summary, "complex", accent, bg)
    expr = "dst[row,bin] = cumulative_count(selected_byte(src[row,*]) <= bin)"
    _draw_expr(out, expr, accent)
    tile_w = _tile_width(TILE_COLS)
    y_src = SRC_Y
    y_dst = DST_Y
    xs = _layout_row_lefts(CANVAS_W // 2, [tile_w, _tile_width(1)], 120)
    x_src, x_idx = xs[0], xs[1]
    _draw_tile_grid(out, x=x_src, y=y_src, label="src", prefix="a", highlight_rows=[EX_R], accent=accent)
    _draw_tile_grid(
        out,
        x=x_idx,
        y=y_src,
        label="idx/filter",
        prefix="i",
        rows=TILE_ROWS,
        cols=1,
        highlight_cells=[(0, 0), (1, 0), (2, 0)],
        accent=accent,
    )
    x_dst = (CANVAS_W - tile_w) // 2
    override = {(EX_R, 0): "bin0", (EX_R, 1): "...", (EX_R, 4): "bin255"}
    _draw_tile_grid(
        out,
        x=x_dst,
        y=y_dst,
        label="dst cumulative bins",
        prefix="h",
        highlight_rows=[EX_R],
        text_override=override,
        accent=accent,
    )
    dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
    s0 = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
    s1 = _tile_port_bottom(x=x_idx, y=y_src, rows=TILE_ROWS, cols=1, c=0)
    _draw_binary_flow(out, instr=instr, left_src=s0, right_src=s1, dst=(dx, dy), accent=accent, op_cx=CANVAS_W // 2)
    proc = [
        "for each source row:",
        "  filter higher bytes using idx rows when required",
        "  count selected byte values into 256 bins",
        "  write prefix sums to dst[row,0..255]",
    ]
    _draw_procedure(out, lines=proc, accent=accent)
    return _end_svg(out)


def _render_complex(instr: str, summary: str, accent: str, bg: str) -> str:
    if instr == "TQUANT_DN":
        return _render_quant_dn(instr, summary, accent, bg)
    if instr == "TQUANT_HIF4":
        return _render_quant_hif4(instr, summary, accent, bg)
    if instr == "THISTOGRAM":
        return _render_histogram(instr, summary, accent, bg)

    out = _begin_svg(instr, summary, "complex", accent, bg)

    tile_w = _tile_width(TILE_COLS)
    tile_h = _tile_height(TILE_ROWS)
    y_src = SRC_Y
    y_dst = DST_Y

    if instr == "TCI":
        expr = "dst[r,c] = base + r*stride + c"
        proc = ["for r in 0..Rv-1:", "  for c in 0..Cv-1:", f"    {expr}"]
        _draw_expr(out, expr, accent)
        scalar_gap = 40
        scalar_y = y_src + 8
        x_base = (CANVAS_W - (160 * 2 + scalar_gap)) // 2
        x_stride = x_base + 160 + scalar_gap
        _draw_scalar_box(out, x=x_base, y=scalar_y, label="base", value="base", accent=accent)
        _draw_scalar_box(out, x=x_stride, y=scalar_y, label="stride", value="stride", accent=accent)

        override: Dict[Tuple[int, int], str] = {(EX_R, EX_C): "base+..."}
        x_dst = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(
            out,
            x=x_dst,
            y=y_dst,
            label="dst",
            prefix="d",
            text_override=override,
            highlight_cells=[(EX_R, EX_C)],
            accent=accent,
        )
        dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        sources = [_scalar_port_bottom(x=x_base, y=scalar_y), _scalar_port_bottom(x=x_stride, y=scalar_y)]
        _draw_binary_flow(
            out,
            instr=instr,
            left_src=sources[0],
            right_src=sources[1],
            dst=(dx, dy),
            accent=accent,
            op_cx=CANVAS_W // 2,
        )
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr == "TTRI":
        expr = "mask[r,c] = (r >= c) ? 1 : 0"
        proc = ["for r in 0..Rv-1:", "  for c in 0..Cv-1:", f"    {expr}"]
        _draw_expr(out, expr, accent)
        scalar_x = (CANVAS_W - 160) // 2
        scalar_y = y_src + 8
        _draw_scalar_box(out, x=scalar_x, y=scalar_y, label="pattern", value="r>=c", accent=accent)

        override = {(r, c): ("1" if r >= c else "0") for r in range(TILE_ROWS) for c in range(TILE_COLS)}
        x_dst = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(
            out,
            x=x_dst,
            y=y_dst,
            label="mask tile",
            prefix="m",
            text_override=override,
            highlight_cells=[(EX_R, EX_C)],
            accent=accent,
        )
        dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        sx, sy = _scalar_port_bottom(x=scalar_x, y=scalar_y)
        _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=int((sy + dy) / 2), accent=accent)
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr in {"TGATHER", "TGATHERB"}:
        expr = "dst[r,c] = src0[ indices[r,c] ]"
        proc = ["for r,c in valid(dst):", "  k = indices[r,c]", "  dst[r,c] = src0[k]"]
        _draw_expr(out, expr, accent)
        xs = _layout_row_lefts(CANVAS_W // 2, [tile_w, tile_w], 120)
        x_src0, x_idx = xs[0], xs[1]
        idx_text = {(EX_R, EX_C): "k"}
        _draw_tile_grid(out, x=x_src0, y=y_src, label="src0", prefix="a", highlight_cells=[(EX_R, EX_C)], accent=accent)
        _draw_tile_grid(
            out,
            x=x_idx,
            y=y_src,
            label="indices",
            prefix="i",
            highlight_cells=[(EX_R, EX_C)],
            text_override=idx_text,
            accent=accent,
        )

        x_dst = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(out, x=x_dst, y=y_dst, label="dst", prefix="d", highlight_cells=[(EX_R, EX_C)], accent=accent)

        dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        a_x, a_y = _tile_port_bottom(x=x_src0, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        i_x, i_y = _tile_port_bottom(x=x_idx, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        _draw_binary_flow(
            out,
            instr=instr,
            left_src=(a_x, a_y),
            right_src=(i_x, i_y),
            dst=(dx, dy),
            accent=accent,
            op_cx=CANVAS_W // 2,
        )
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr == "TSCATTER":
        expr = "dst[ idx[r,c], c ] = src[r,c]"
        proc = ["for r,c in valid(src):", "  rr = idx[r,c]", "  dst[rr, c] = src[r,c]"]
        _draw_expr(out, expr, accent)
        xs = _layout_row_lefts(CANVAS_W // 2, [tile_w, tile_w], 120)
        x_src, x_idx = xs[0], xs[1]
        idx_text = {(EX_R, EX_C): "rr"}
        _draw_tile_grid(out, x=x_src, y=y_src, label="src", prefix="a", highlight_cells=[(EX_R, EX_C)], accent=accent)
        _draw_tile_grid(
            out,
            x=x_idx,
            y=y_src,
            label="row idx",
            prefix="i",
            highlight_cells=[(EX_R, EX_C)],
            text_override=idx_text,
            accent=accent,
        )

        x_dst = (CANVAS_W - tile_w) // 2
        dst_r = min(TILE_ROWS - 2, EX_R + 1) if TILE_ROWS > 2 else EX_R
        _draw_tile_grid(out, x=x_dst, y=y_dst, label="dst", prefix="d", highlight_cells=[(dst_r, EX_C)], accent=accent)

        dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        s_x, s_y = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        i_x, i_y = _tile_port_bottom(x=x_idx, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        _draw_binary_flow(
            out,
            instr=instr,
            left_src=(s_x, s_y),
            right_src=(i_x, i_y),
            dst=(dx, dy),
            accent=accent,
            op_cx=CANVAS_W // 2,
        )
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr == "TSORT32":
        expr = "dst[i,k] = src[i, pi_i(k)] ; idx = pi"
        proc = ["for each row i:", "  (dst_row, idx_row) = sort_with_indices(src_row)"]
        _draw_expr(out, expr, accent)
        x_src = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(
            out, x=x_src, y=y_src, label="src (block)", prefix="a", highlight_cells=[(EX_R, EX_C)], accent=accent
        )

        xs = _layout_row_lefts(CANVAS_W // 2, [tile_w, tile_w], 120)
        x_dst, x_idx = xs[0], xs[1]
        _draw_tile_grid(
            out, x=x_dst, y=y_dst, label="dst (sorted)", prefix="d", highlight_cells=[(EX_R, EX_C)], accent=accent
        )
        _draw_tile_grid(
            out, x=x_idx, y=y_dst, label="idx (perm)", prefix="p", highlight_cells=[(EX_R, EX_C)], accent=accent
        )

        s_x, s_y = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        d1_x, d1_y = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        d2_x, d2_y = _tile_port_top(x=x_idx, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        via_base = int((y_src + tile_h + y_dst) / 2)
        _draw_ortho_arrow(out, x1=s_x, y1=s_y, x2=d1_x, y2=d1_y, via_y=via_base - 10, accent=accent)
        _draw_ortho_arrow(out, x1=s_x, y1=s_y, x2=d2_x, y2=d2_y, via_y=via_base + 10, accent=accent)
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr == "TMRGSORT":
        expr = "dst = merge(src0, src1, ...)"
        proc = ["dst = merge(sorted_lists...)", "(ordering/format are implementation-defined)"]
        _draw_expr(out, expr, accent)
        xs = _layout_row_lefts(CANVAS_W // 2, [tile_w, tile_w], 120)
        x0, x1 = xs[0], xs[1]
        _draw_tile_grid(out, x=x0, y=y_src, label="src0 (sorted)", prefix="a", accent=accent)
        _draw_tile_grid(out, x=x1, y=y_src, label="src1 (sorted)", prefix="b", accent=accent)
        x_dst = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(
            out, x=x_dst, y=y_dst, label="dst (merged)", prefix="d", highlight_cells=[(EX_R, EX_C)], accent=accent
        )
        dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        a_x, a_y = _tile_port_bottom(x=x0, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        b_x, b_y = _tile_port_bottom(x=x1, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        _draw_binary_flow(
            out,
            instr=instr,
            left_src=(a_x, a_y),
            right_src=(b_x, b_y),
            dst=(dx, dy),
            accent=accent,
            op_cx=CANVAS_W // 2,
        )
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr == "TQUANT":
        expr = "dst, exp, max, scaling = quantize(src, mode)"
        proc = [
            "max = max(abs(src))",
            "scaling = compute_scaling(max, mode)",
            "for r,c in valid(src):",
            "  (dst[r,c], exp[r,c]) = quantize(src[r,c], scaling, mode)",
        ]
        _draw_expr(out, expr, accent)
        x_src = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(
            out, x=x_src, y=y_src, label="src (fp32)", prefix="a", highlight_cells=[(EX_R, EX_C)], accent=accent
        )

        # Outputs: exp tile, dst tile, scaling/max (vector).
        scale_rows, scale_cols = 1, TILE_COLS
        scale_h = _tile_height(scale_rows)
        y_scale = y_dst + (tile_h - scale_h) // 2
        widths = [tile_w, tile_w, tile_w]
        xs = _layout_row_lefts(CANVAS_W // 2, widths, 80)
        x_exp, x_dst, x_scale = xs[0], xs[1], xs[2]
        _draw_tile_grid(
            out, x=x_exp, y=y_dst, label="exp tile", prefix="e", highlight_cells=[(EX_R, EX_C)], accent=accent
        )
        _draw_tile_grid(
            out, x=x_dst, y=y_dst, label="dst (quant)", prefix="q", highlight_cells=[(EX_R, EX_C)], accent=accent
        )
        _draw_tile_grid(
            out, x=x_scale, y=y_scale, label="max/scale", prefix="s", rows=scale_rows, cols=scale_cols, accent=accent
        )

        s_x, s_y = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        d_x, d_y = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        e_x, e_y = _tile_port_top(x=x_exp, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        m_x, m_y = _tile_port_top(x=x_scale, y=y_scale, rows=scale_rows, cols=scale_cols, c=EX_C)
        via_base = int((y_src + tile_h + y_dst) / 2)
        _draw_ortho_arrow(out, x1=s_x, y1=s_y, x2=d_x, y2=d_y, via_y=via_base - 12, accent=accent)
        _draw_ortho_arrow(out, x1=s_x, y1=s_y, x2=e_x, y2=e_y, via_y=via_base + 0, accent=accent)
        _draw_ortho_arrow(out, x1=s_x, y1=s_y, x2=m_x, y2=m_y, via_y=via_base + 12, accent=accent)
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr.startswith("TPART"):
        op = instr.replace("TPART", "").lower()
        if op == "add":
            body = "src0 + src1"
        elif op == "mul":
            body = "src0 * src1"
        elif op == "max":
            body = "max(src0, src1)"
        elif op == "min":
            body = "min(src0, src1)"
        else:
            body = "op(src0, src1)"
        expr = f"dst[r,c] = partial({body})   (validity-dependent)"
        proc = [
            "for r,c in valid(dst):",
            "  if defined(src0[r,c]) and defined(src1[r,c]):",
            f"    dst[r,c] = {body}",
            "  elif only one is defined:",
            "    dst[r,c] = that defined value",
            "  else:",
            "    dst[r,c] = implementation-defined",
        ]
        _draw_expr(out, expr, accent)
        xs = _layout_row_lefts(CANVAS_W // 2, [tile_w, tile_w], 120)
        x0, x1 = xs[0], xs[1]
        _draw_tile_grid(
            out,
            x=x0,
            y=y_src,
            label="src0 (valid A)",
            prefix="a",
            valid_box=(3, 3),
            highlight_cells=[(EX_R, EX_C)],
            accent=accent,
        )
        _draw_tile_grid(
            out,
            x=x1,
            y=y_src,
            label="src1 (valid B)",
            prefix="b",
            valid_box=(2, 4),
            highlight_cells=[(EX_R, EX_C)],
            accent=accent,
        )
        x_dst = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(
            out,
            x=x_dst,
            y=y_dst,
            label="dst (valid D)",
            prefix="d",
            valid_box=(3, 3),
            highlight_cells=[(EX_R, EX_C)],
            accent=accent,
        )
        dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        a_x, a_y = _tile_port_bottom(x=x0, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        b_x, b_y = _tile_port_bottom(x=x1, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        _draw_binary_flow(
            out,
            instr=instr,
            left_src=(a_x, a_y),
            right_src=(b_x, b_y),
            dst=(dx, dy),
            accent=accent,
            op_cx=CANVAS_W // 2,
        )
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr == "TPRINT":
        expr = "print(src)   (implementation-defined formatting)"
        proc = [expr]
        _draw_expr(out, expr, accent)
        x_src = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(
            out, x=x_src, y=y_src, label="src tile", prefix="a", highlight_cells=[(EX_R, EX_C)], accent=accent
        )
        box_x = (CANVAS_W - 160) // 2
        box_y = y_dst + 20
        _draw_scalar_box(out, x=box_x, y=box_y, label="side effect", value="print/log", accent=accent)
        sx, sy = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        dx, dy = _scalar_port_top(x=box_x, y=box_y)
        _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=int((sy + dy) / 2), accent=accent)
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    _draw_procedure(out, lines=["(implementation-defined)"], accent=accent)
    return _end_svg(out)


def _render_sync(instr: str, summary: str, accent: str, bg: str) -> str:
    out = _begin_svg(instr, summary, "sync", accent, bg)
    expr = "synchronization establishes ordering: producer -> consumer"
    _draw_expr(out, expr, accent)

    box_w = 520
    box_h = 92
    box_x = (CANVAS_W - box_w) // 2
    y_prod = SRC_Y + 6
    y_cons = DST_Y + 6

    def stage_box(y: int, title: str, detail: str) -> None:
        _append_rect(out, x=box_x, y=y, width=box_w, height=box_h, rx=14, fill="#ffffff", stroke=accent, stroke_width=2)
        out.append(
            f'<text x="{box_x + box_w // 2}" y="{y + 38}" text-anchor="middle" class="tileLabel">{_esc(title)}</text>'
        )
        out.append(
            f'<text x="{box_x + box_w // 2}" y="{y + 64}" text-anchor="middle" class="smallLabel">{_esc(detail)}</text>'
        )

    stage_box(y_prod, "Producer stage", "ops tagged as producer_class")
    stage_box(y_cons, "Consumer stage", "ops tagged as consumer_class after synchronization")

    src_x, src_y = (box_x + box_w // 2, y_prod + box_h)
    dst_x, dst_y = (box_x + box_w // 2, y_cons)
    via_x = box_x + box_w + 56
    _draw_ortho_arrow(out, x1=src_x, y1=src_y, x2=dst_x, y2=dst_y, via_x=via_x, accent=accent)

    proc = [
        "synchronize(producer_class, consumer_class)",
        "1) Let P be all earlier ops issued with class=producer_class.",
        "2) Wait until P are complete or until their events are satisfied.",
        "3) For all later ops with class=consumer_class: observe results of P.",
        "Ordering: P happens-before consumer_class ops after synchronization.",
    ]
    _draw_procedure(out, lines=proc, accent=accent)
    return _end_svg(out)


def _render_config(instr: str, summary: str, accent: str, bg: str) -> str:
    if instr == "SET_QUANT_VECTOR":
        return _render_set_quant_vector(instr, summary, accent, bg)

    out = _begin_svg(instr, summary, "config", accent, bg)
    tile_w = _tile_width(TILE_COLS)
    y_src = SRC_Y
    y_dst = DST_Y

    def draw_state_box(*, x: int, y: int, title: str, lines: Sequence[str]) -> Tuple[int, int]:
        w = 520
        h = 92
        _append_rect(out, x=x, y=y, width=w, height=h, rx=14, fill="#ffffff", stroke=accent, stroke_width=2)
        out.append(f'<text x="{x + 18}" y="{y + 34}" class="tileLabel">{_esc(title)}</text>')
        # Two lines is enough for these config diagrams; keep text tidy.
        _draw_text_lines(out, x + 18, y + 56, lines[:2], "smallLabel", 18)
        return (x + w // 2, y)

    if instr == "TASSIGN":
        expr = "tile.bind(address)   (implementation-defined mapping)"
        proc = [
            "TASSIGN(tile, address, ...waitEvents)",
            "1) address := immediate/scalar (implementation-defined base).",
            "2) Bind tile handle to an on-chip address range starting at address.",
            "3) Subsequent memory ops on this tile use the bound mapping.",
        ]

        _draw_expr(out, expr, accent)

        widths = [tile_w, 160]
        xs = _layout_row_lefts(CANVAS_W // 2, widths, 220)
        x_tile = xs[0]
        x_addr = xs[1]
        scalar_y = y_src + 8

        _draw_tile_grid(
            out,
            x=x_tile,
            y=y_src,
            label="tile handle (unbound)",
            prefix="t",
            highlight_cells=[(EX_R, EX_C)],
            accent=accent,
        )
        _draw_scalar_box(out, x=x_addr, y=scalar_y, label="address", value="0x....", accent=accent)

        x_dst = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(
            out,
            x=x_dst,
            y=y_dst,
            label="tile handle (bound)",
            prefix="t",
            highlight_cells=[(EX_R, EX_C)],
            accent=accent,
        )

        dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        sx, sy = _tile_port_bottom(x=x_tile, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        ax, ay = _scalar_port_bottom(x=x_addr, y=scalar_y)
        _draw_binary_flow(
            out, instr=instr, left_src=(sx, sy), right_src=(ax, ay), dst=(dx, dy), accent=accent, op_cx=CANVAS_W // 2
        )
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr == "SETFMATRIX":
        expr = "set FMATRIX state (used by later ops)"
        proc = [
            "SETFMATRIX(value, ...waitEvents)",
            "1) Update FMATRIX register/state.",
            "2) Ordering: update takes effect before dependent ops.",
            "3) Affects subsequent IMG2COL / layout-sensitive operations.",
        ]
        scalar_label = "FMATRIX"
        scalar_value = "set"
        state_lines = ["FMATRIX state updated", "consulted by later ops"]
    elif instr == "SET_QUANT_SCALAR":
        expr = "QUANT_SCALAR_REG = bitcast(preQuantScalar) with int8 sign bit"
        proc = [
            "SET_QUANT_SCALAR<OutType>(preQuantScalar, ...waitEvents)",
            "1) bitcast float scalar into low 32 bits of quantConfig.",
            "2) For 8-bit OutType, write the signed/unsigned flag bit.",
            "3) Copy quantConfig to QUANT_SCALAR_REG for later TPUSH.",
        ]
        scalar_label = "preQuant"
        scalar_value = "float"
        state_lines = ["QUANT_SCALAR_REG updated", "consumed by later TPUSH"]
    else:
        mode = "HF32" if instr == "TSETHF32MODE" else "TF32" if instr == "TSETTF32MODE" else "mode"
        expr = f"set transform mode ({mode})"
        proc = [
            f"{instr}(enable/mode, ...waitEvents)",
            "1) Update backend transform/rounding mode (implementation-defined).",
            "2) Ordering: update takes effect before dependent ops.",
            "3) Affects subsequent GEMV/MATMUL or conversion paths (if applicable).",
        ]
        scalar_label = "mode"
        scalar_value = "enable/mode"
        state_lines = ["backend mode state updated", "used by later ops"]

    _draw_expr(out, expr, accent)

    scalar_x = CANVAS_W - MARGIN - 16 - 160
    scalar_y = y_src + 8
    _draw_scalar_box(out, x=scalar_x, y=scalar_y, label=scalar_label, value=scalar_value, accent=accent)

    state_x = (CANVAS_W - 520) // 2
    state_y = y_dst + 6
    dx, dy = draw_state_box(x=state_x, y=state_y, title="Execution state", lines=state_lines)
    sx, sy = (scalar_x + 160 // 2, scalar_y + 54)
    via_y = int((sy + dy) / 2)
    _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=via_y, accent=accent)
    _draw_procedure(out, lines=proc, accent=accent)
    return _end_svg(out)


def _render_set_quant_vector(instr: str, summary: str, accent: str, bg: str) -> str:
    out = _begin_svg(instr, summary, "config", accent, bg)
    expr = "QUANT_VECTOR_REG = address(fp Scaling tile)"
    _draw_expr(out, expr, accent)

    tile_w = _tile_width(TILE_COLS)
    y_src = SRC_Y
    y_dst = DST_Y
    x_tile = (CANVAS_W - tile_w) // 2
    _draw_tile_grid(
        out,
        x=x_tile,
        y=y_src,
        label="fpTile (Scaling)",
        prefix="s",
        rows=1,
        cols=TILE_COLS,
        highlight_cells=[(0, EX_C)],
        accent=accent,
    )

    state_x = (CANVAS_W - 520) // 2
    state_y = y_dst + 6
    _append_rect(out, x=state_x, y=state_y, width=520, height=92, rx=14, fill="#ffffff", stroke=accent, stroke_width=2)
    out.append(f'<text x="{state_x + 18}" y="{state_y + 34}" class="tileLabel">Execution state</text>')
    _draw_text_lines(
        out, state_x + 18, state_y + 56, ["QUANT_VECTOR_REG updated", "points to Scaling tile"], "smallLabel", 18
    )

    sx, sy = _tile_port_bottom(x=x_tile, y=y_src, rows=1, cols=TILE_COLS, c=EX_C)
    dx, dy = (state_x + 260, state_y)
    _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=int((sy + dy) / 2), accent=accent)

    proc = [
        "SET_QUANT_VECTOR(fpTile, ...waitEvents)",
        "1) Require fpTile.Loc == TileType::Scaling.",
        "2) Reinterpret fpTile.data() as the scaling tile address.",
        "3) Copy address to QUANT_VECTOR_REG for later TPUSH.",
    ]
    _draw_procedure(out, lines=proc, accent=accent)
    return _end_svg(out)


def _render_comm(instr: str, summary: str, accent: str, bg: str) -> str:
    out = _begin_svg(instr, summary, "comm", accent, bg)
    tile_w = _tile_width(TILE_COLS)
    y_src = SRC_Y
    y_dst = DST_Y
    layout = _comm_layout(y_src)
    expr, proc = _comm_spec(instr)
    _draw_comm_main(out, instr, accent, layout, tile_w, y_src)
    _draw_expr(out, expr, accent)
    _draw_comm_lifecycle(out, instr, accent, y_dst)
    _draw_procedure(out, lines=proc, accent=accent)
    return _end_svg(out)


def _comm_layout(y_src: int) -> Dict[str, int]:
    return {"pipe_x": (CANVAS_W - 260) // 2, "pipe_y": y_src + 20, "pipe_w": 260, "pipe_h": 92}


def _comm_spec(instr: str) -> Tuple[str, List[str]]:
    specs = {
        "TALLOC": (
            "gmTensor.data = GM_SLOT_BUFFER + slotOffset (+ split offset)",
            [
                "if producer allocateStatus is set and the FIFO slot needs space:",
                "  pipe.prod.allocate<Split>() waits for free space",
                "entryBase = GM_SLOT_BUFFER + (tileIndex % SLOT_NUM) * SLOT_SIZE",
                "entryBase += split offset for V2C/Both vector-side views",
                "tileIndex++; TASSIGN_IMPL(gmTensor, entryBase)",
            ],
        ),
        "TPUSH": (
            "producer tile -> FIFO slot; record token when required",
            [
                "if producer allocateStatus is set: allocate slot",
                "copy tile into GM slot, V2C buffer, or C2V shared slot",
                "for split mode, use subblock-dependent row/col offset",
                "if producer recordStatus is set: record<Split>() publishes token",
            ],
        ),
        "TPOP": (
            "consumer waits token -> load FIFO slot into tile/GlobalTensor",
            [
                "if consumer waitStatus is set: wait<Split>()",
                "slotIndex = cons.tileId % SLOT_NUM",
                "read GM slot, C2V shared slot, or V2C buffer",
                "materialize consumer Tile or GlobalTensor view",
            ],
        ),
        "TFREE": (
            "consumer free -> release FIFO slot/token",
            [
                "if consumer freeStatus is set:",
                "  pipe.cons.free<Split>() releases the consumed entry",
                "TileData TPOP flows may be no-op on targets that do not need release",
            ],
        ),
    }
    return specs.get(instr, ("communication operation", ["(diagram template not implemented)"]))


def _draw_pipe_state(out: List[str], layout: Dict[str, int], title: str, detail: str, accent: str) -> None:
    pipe_x, pipe_y = layout["pipe_x"], layout["pipe_y"]
    pipe_w, pipe_h = layout["pipe_w"], layout["pipe_h"]
    _append_rect(
        out, x=pipe_x, y=pipe_y, width=pipe_w, height=pipe_h, rx=14, fill="#ffffff", stroke=accent, stroke_width=2
    )
    _append_svg_text(out, x=pipe_x + pipe_w // 2, y=pipe_y + 34, cls="tileLabel", text=title, text_anchor="middle")
    _append_svg_text(out, x=pipe_x + pipe_w // 2, y=pipe_y + 58, cls="smallLabel", text=detail, text_anchor="middle")


def _draw_comm_main(out: List[str], instr: str, accent: str, layout: Dict[str, int], tile_w: int, y_src: int) -> None:
    if instr == "TALLOC":
        _draw_comm_token(out, layout, "TPipe producer", "reserve slot and entryBase", "slot", "id/base", y_src, accent)
    elif instr == "TPUSH":
        _draw_comm_push(out, layout, tile_w, y_src, accent)
    elif instr == "TPOP":
        _draw_comm_pop(out, layout, tile_w, y_src, accent)
    elif instr == "TFREE":
        _draw_comm_token(out, layout, "TPipe consumer", "release consumed slot", "free", "token", y_src, accent)
    else:
        _draw_pipe_state(out, layout, "Communication", "implementation-defined", accent)


def _draw_comm_token(
    out: List[str], layout: Dict[str, int], title: str, detail: str, label: str, value: str, y_src: int, accent: str
) -> None:
    pipe_x, pipe_y = layout["pipe_x"], layout["pipe_y"]
    pipe_w, pipe_h = layout["pipe_w"], layout["pipe_h"]
    token_x = CANVAS_W - MARGIN - 176
    _draw_pipe_state(out, layout, title, detail, accent)
    _draw_scalar_box(out, x=token_x, y=y_src + 38, label=label, value=value, accent=accent)
    _draw_ortho_arrow(
        out,
        x1=pipe_x + pipe_w,
        y1=pipe_y + pipe_h // 2,
        x2=token_x,
        y2=y_src + 65,
        via_x=pipe_x + pipe_w + 46,
        accent=accent,
    )


def _draw_comm_push(out: List[str], layout: Dict[str, int], tile_w: int, y_src: int, accent: str) -> None:
    _ = tile_w
    x_tile = MARGIN + 90
    _draw_tile_grid(
        out, x=x_tile, y=y_src, label="producer tile", prefix="p", highlight_cells=[(EX_R, EX_C)], accent=accent
    )
    _draw_pipe_state(out, layout, "TPipe FIFO slot", "GM / V2C / C2V storage", accent)
    sx, sy = _tile_port_bottom(x=x_tile, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
    _draw_ortho_arrow(
        out, x1=sx, y1=sy, x2=layout["pipe_x"], y2=layout["pipe_y"] + 46, via_y=layout["pipe_y"] + 128, accent=accent
    )


def _draw_comm_pop(out: List[str], layout: Dict[str, int], tile_w: int, y_src: int, accent: str) -> None:
    _draw_pipe_state(out, layout, "TPipe FIFO slot", "published producer data", accent)
    x_tile = CANVAS_W - MARGIN - 90 - tile_w
    _draw_tile_grid(
        out, x=x_tile, y=y_src, label="consumer tile", prefix="c", highlight_cells=[(EX_R, EX_C)], accent=accent
    )
    dx, dy = _tile_port_top(x=x_tile, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
    _draw_ortho_arrow(
        out,
        x1=layout["pipe_x"] + layout["pipe_w"],
        y1=layout["pipe_y"] + 46,
        x2=dx,
        y2=dy,
        via_x=layout["pipe_x"] + layout["pipe_w"] + 46,
        accent=accent,
    )


def _draw_comm_lifecycle(out: List[str], instr: str, accent: str, y_dst: int) -> None:
    states = ["allocate", "push", "pop", "free"]
    state_w = 150
    state_gap = 34
    start_x = (CANVAS_W - (state_w * len(states) + state_gap * (len(states) - 1))) // 2
    state_y = y_dst + 34
    prev_right = None
    for idx, state in enumerate(states):
        x = start_x + idx * (state_w + state_gap)
        cls_stroke = accent if state.upper() == instr[1:] or (instr == "TALLOC" and state == "allocate") else "#cbd5e1"
        _append_rect(
            out, x=x, y=state_y, width=state_w, height=58, rx=12, fill="#ffffff", stroke=cls_stroke, stroke_width=2
        )
        _append_svg_text(out, x=x + state_w // 2, y=state_y + 35, cls="tileLabel", text=state, text_anchor="middle")
        if prev_right is not None:
            _draw_ortho_arrow(
                out, x1=prev_right, y1=state_y + 29, x2=x, y2=state_y + 29, via_y=state_y + 29, accent=accent
            )
        prev_right = x + state_w


def render_svg(entry: Dict[str, object]) -> str:
    instr = str(entry.get("instruction", "UNKNOWN")).strip()
    template = str(entry.get("diagram_template", "elementwise")).strip()
    summary = str(entry.get("summary_en", "")).strip()

    accent, bg = COLOR_BY_TEMPLATE.get(template, COLOR_BY_TEMPLATE["elementwise"])

    if template == "elementwise":
        return _render_elementwise(instr, summary, accent, bg)
    if template == "scalar":
        return _render_scalar(instr, summary, accent, bg)
    if template == "reduce_expand":
        return _render_reduce_expand(instr, summary, accent, bg)
    if template == "memory":
        return _render_memory(instr, summary, accent, bg)
    if template == "matmul":
        return _render_matmul(instr, summary, accent, bg)
    if template == "reshape_move":
        return _render_reshape_move(instr, summary, accent, bg)
    if template == "complex":
        return _render_complex(instr, summary, accent, bg)
    if template == "sync":
        return _render_sync(instr, summary, accent, bg)
    if template == "config":
        return _render_config(instr, summary, accent, bg)
    if template == "comm":
        return _render_comm(instr, summary, accent, bg)

    # Fallback
    out = _begin_svg(instr, summary, template, accent, bg)
    _draw_procedure(out, lines=["(diagram template not implemented)"], accent=accent)
    return _end_svg(out)


def check_svgs(entries: List[Dict[str, object]], output_dir: Path) -> List[str]:
    errors: List[str] = []
    for e in entries:
        instr = str(e.get("instruction", "")).strip()
        if not instr:
            continue
        svg = output_dir / f"{instr}.svg"
        if not svg.exists():
            errors.append(f"missing svg: {svg}")
            continue
        data = svg.read_text(encoding="utf-8", errors="ignore")
        if "<svg" not in data:
            errors.append(f"invalid svg: {svg}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate per-instruction SVG diagrams")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    entries = load_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.check:
        errors = check_svgs(entries, args.output_dir)
        if errors:
            for err in errors:
                print(f"ERROR: {err}")
            return 1
        print("OK: all instruction SVG files are present.")
        return 0

    for e in entries:
        instr = str(e.get("instruction", "")).strip()
        if not instr:
            continue
        out_path = args.output_dir / f"{instr}.svg"
        out_path.write_text(render_svg(e), encoding="utf-8")

    print(f"Generated {len(entries)} SVG files in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
