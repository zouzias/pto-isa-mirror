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

"""Render state, configuration, and communication ISA SVG diagrams."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from gen_isa_svg_types import CommRenderContext, CommTokenSpec, ConfigRenderContext

from gen_isa_svgs import (
    CANVAS_W,
    DST_Y,
    EX_C,
    EX_R,
    MARGIN,
    SRC_Y,
    TILE_COLS,
    TILE_ROWS,
    _append_rect,
    _append_svg_text,
    _begin_svg,
    _draw_binary_flow,
    _draw_expr,
    _draw_ortho_arrow,
    _draw_procedure,
    _draw_scalar_box,
    _draw_text_lines,
    _draw_tile_grid,
    _end_svg,
    _esc,
    _layout_row_lefts,
    _scalar_port_bottom,
    _tile_height,
    _tile_port_bottom,
    _tile_port_top,
    _tile_width,
)


def render_sync(instr: str, summary: str, accent: str, bg: str) -> str:
    out = _begin_svg(instr, summary, "sync", accent, bg)
    expr = "synchronization establishes ordering: producer -> consumer"
    _draw_expr(out, expr, accent)

    box_w = 520
    box_h = 92
    box_x = (CANVAS_W - box_w) // 2
    y_prod = SRC_Y + 6
    y_cons = DST_Y + 6

    def stage_box(y: int, title: str, detail: str) -> None:
        _append_rect(
            out,
            x=box_x,
            y=y,
            width=box_w,
            height=box_h,
            rx=14,
            fill="#ffffff",
            stroke=accent,
            stroke_width=2,
        )
        _append_svg_text(
            out,
            x=box_x + box_w // 2,
            y=y + 38,
            cls="tileLabel",
            text=title,
            text_anchor="middle",
        )
        _append_svg_text(
            out,
            x=box_x + box_w // 2,
            y=y + 64,
            cls="smallLabel",
            text=detail,
            text_anchor="middle",
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


def render_config(instr: str, summary: str, accent: str, bg: str) -> str:
    if instr == "SET_QUANT_VECTOR":
        return _render_set_quant_vector(instr, summary, accent, bg)

    out = _begin_svg(instr, summary, "config", accent, bg)
    tile_w = _tile_width(TILE_COLS)
    y_src = SRC_Y
    y_dst = DST_Y
    context = ConfigRenderContext(tile_w, y_src, y_dst)

    def draw_state_box(*, x: int, y: int, title: str, lines: Sequence[str]) -> Tuple[int, int]:
        w = 520
        h = 92
        _append_rect(out, x=x, y=y, width=w, height=h, rx=14, fill="#ffffff", stroke=accent, stroke_width=2)
        _append_svg_text(out, x=x + 18, y=y + 34, cls="tileLabel", text=title)
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
        return _render_tassign_config(out, accent, context, expr, proc)

    expr, proc, scalar_label, scalar_value, state_lines = _config_state_spec(instr)
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


def _render_tassign_config(
    out: List[str], accent: str, context: ConfigRenderContext, expr: str, proc: List[str]
) -> str:
    _draw_expr(out, expr, accent)
    widths = [context.tile_w, 160]
    xs = _layout_row_lefts(CANVAS_W // 2, widths, 220)
    x_tile = xs[0]
    x_addr = xs[1]
    scalar_y = context.y_src + 8
    _draw_tile_grid(
        out,
        x=x_tile,
        y=context.y_src,
        label="tile handle (unbound)",
        prefix="t",
        highlight_cells=[(EX_R, EX_C)],
        accent=accent,
    )
    _draw_scalar_box(out, x=x_addr, y=scalar_y, label="address", value="0x....", accent=accent)
    x_dst = (CANVAS_W - context.tile_w) // 2
    _draw_tile_grid(
        out,
        x=x_dst,
        y=context.y_dst,
        label="tile handle (bound)",
        prefix="t",
        highlight_cells=[(EX_R, EX_C)],
        accent=accent,
    )
    dx, dy = _tile_port_top(x=x_dst, y=context.y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
    sx, sy = _tile_port_bottom(x=x_tile, y=context.y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
    ax, ay = _scalar_port_bottom(x=x_addr, y=scalar_y)
    _draw_binary_flow(
        out, instr="TASSIGN", left_src=(sx, sy), right_src=(ax, ay), dst=(dx, dy), accent=accent, op_cx=CANVAS_W // 2
    )
    _draw_procedure(out, lines=proc, accent=accent)
    return _end_svg(out)


def _config_state_spec(instr: str) -> Tuple[str, List[str], str, str, List[str]]:
    if instr == "SETFMATRIX":
        return (
            "set FMATRIX state (used by later ops)",
            [
                "SETFMATRIX(value, ...waitEvents)",
                "1) Update FMATRIX register/state.",
                "2) Ordering: update takes effect before dependent ops.",
                "3) Affects subsequent IMG2COL / layout-sensitive operations.",
            ],
            "FMATRIX",
            "set",
            ["FMATRIX state updated", "consulted by later ops"],
        )
    if instr == "SET_QUANT_SCALAR":
        return (
            "QUANT_SCALAR_REG = bitcast(preQuantScalar) with int8 sign bit",
            [
                "SET_QUANT_SCALAR<OutType>(preQuantScalar, ...waitEvents)",
                "1) bitcast float scalar into low 32 bits of quantConfig.",
                "2) For 8-bit OutType, write the signed/unsigned flag bit.",
                "3) Copy quantConfig to QUANT_SCALAR_REG for later TPUSH.",
            ],
            "preQuant",
            "float",
            ["QUANT_SCALAR_REG updated", "consumed by later TPUSH"],
        )
    mode = "HF32" if instr == "TSETHF32MODE" else "TF32" if instr == "TSETTF32MODE" else "mode"
    return (
        f"set transform mode ({mode})",
        [
            f"{instr}(enable/mode, ...waitEvents)",
            "1) Update backend transform/rounding mode (implementation-defined).",
            "2) Ordering: update takes effect before dependent ops.",
            "3) Affects subsequent GEMV/MATMUL or conversion paths (if applicable).",
        ],
        "mode",
        "enable/mode",
        ["backend mode state updated", "used by later ops"],
    )


def _draw_set_quant_vector_state(out: List[str], accent: str, state_x: int, state_y: int) -> None:
    _append_rect(
        out,
        x=state_x,
        y=state_y,
        width=520,
        height=92,
        rx=14,
        fill="#ffffff",
        stroke=accent,
        stroke_width=2,
    )
    _append_svg_text(out, x=state_x + 18, y=state_y + 34, cls="tileLabel", text="Execution state")
    _draw_text_lines(
        out,
        state_x + 18,
        state_y + 56,
        ["QUANT_VECTOR_REG updated", "points to Scaling tile"],
        "smallLabel",
        18,
    )


def _set_quant_vector_proc() -> List[str]:
    return [
        "SET_QUANT_VECTOR(fpTile, ...waitEvents)",
        "1) Require fpTile.Loc == TileType::Scaling.",
        "2) Reinterpret fpTile.data() as the scaling tile address.",
        "3) Copy address to QUANT_VECTOR_REG for later TPUSH.",
    ]


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
    _draw_set_quant_vector_state(out, accent, state_x, state_y)

    sx, sy = _tile_port_bottom(x=x_tile, y=y_src, rows=1, cols=TILE_COLS, c=EX_C)
    dx, dy = (state_x + 260, state_y)
    _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=int((sy + dy) / 2), accent=accent)

    proc = _set_quant_vector_proc()
    _draw_procedure(out, lines=proc, accent=accent)
    return _end_svg(out)


def render_comm(instr: str, summary: str, accent: str, bg: str) -> str:
    out = _begin_svg(instr, summary, "comm", accent, bg)
    tile_w = _tile_width(TILE_COLS)
    y_src = SRC_Y
    y_dst = DST_Y
    layout = _comm_layout(y_src)
    expr, proc = _comm_spec(instr)
    context = CommRenderContext(instr, layout, tile_w, y_src)
    _draw_comm_main(out, accent, context)
    _draw_expr(out, expr, accent)
    _draw_comm_lifecycle(out, instr, accent, y_dst)
    _draw_procedure(out, lines=proc, accent=accent)
    return _end_svg(out)


def _comm_layout(y_src: int) -> Dict[str, int]:
    return {
        "pipe_x": (CANVAS_W - 260) // 2,
        "pipe_y": y_src + 20,
        "pipe_w": 260,
        "pipe_h": 92,
    }


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
        out,
        x=pipe_x,
        y=pipe_y,
        width=pipe_w,
        height=pipe_h,
        rx=14,
        fill="#ffffff",
        stroke=accent,
        stroke_width=2,
    )
    _append_svg_text(out, x=pipe_x + pipe_w // 2, y=pipe_y + 34, cls="tileLabel", text=title, text_anchor="middle")
    _append_svg_text(out, x=pipe_x + pipe_w // 2, y=pipe_y + 58, cls="smallLabel", text=detail, text_anchor="middle")


def _draw_comm_main(out: List[str], accent: str, context: CommRenderContext) -> None:
    layout = context.layout
    if context.instr == "TALLOC":
        spec = CommTokenSpec("TPipe producer", "reserve slot and entryBase", "slot", "id/base")
        _draw_comm_token(out, accent, layout, context.y_src, spec)
    elif context.instr == "TPUSH":
        _draw_comm_push(out, layout, context.tile_w, context.y_src, accent)
    elif context.instr == "TPOP":
        _draw_comm_pop(out, layout, context.tile_w, context.y_src, accent)
    elif context.instr == "TFREE":
        spec = CommTokenSpec("TPipe consumer", "release consumed slot", "free", "token")
        _draw_comm_token(out, accent, layout, context.y_src, spec)
    else:
        _draw_pipe_state(out, layout, "Communication", "implementation-defined", accent)


def _draw_comm_token(
    out: List[str],
    accent: str,
    layout: Dict[str, int],
    y_src: int,
    spec: CommTokenSpec,
) -> None:
    pipe_x, pipe_y = layout["pipe_x"], layout["pipe_y"]
    pipe_w, pipe_h = layout["pipe_w"], layout["pipe_h"]
    token_x = CANVAS_W - MARGIN - 176
    _draw_pipe_state(out, layout, spec.title, spec.detail, accent)
    _draw_scalar_box(out, x=token_x, y=y_src + 38, label=spec.label, value=spec.value, accent=accent)
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
        out,
        x=x_tile,
        y=y_src,
        label="producer tile",
        prefix="p",
        highlight_cells=[(EX_R, EX_C)],
        accent=accent,
    )
    _draw_pipe_state(out, layout, "TPipe FIFO slot", "GM / V2C / C2V storage", accent)
    sx, sy = _tile_port_bottom(x=x_tile, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
    _draw_ortho_arrow(
        out,
        x1=sx,
        y1=sy,
        x2=layout["pipe_x"],
        y2=layout["pipe_y"] + 46,
        via_y=layout["pipe_y"] + 128,
        accent=accent,
    )


def _draw_comm_pop(out: List[str], layout: Dict[str, int], tile_w: int, y_src: int, accent: str) -> None:
    _draw_pipe_state(out, layout, "TPipe FIFO slot", "published producer data", accent)
    x_tile = CANVAS_W - MARGIN - 90 - tile_w
    _draw_tile_grid(
        out,
        x=x_tile,
        y=y_src,
        label="consumer tile",
        prefix="c",
        highlight_cells=[(EX_R, EX_C)],
        accent=accent,
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
            out,
            x=x,
            y=state_y,
            width=state_w,
            height=58,
            rx=12,
            fill="#ffffff",
            stroke=cls_stroke,
            stroke_width=2,
        )
        _append_svg_text(out, x=x + state_w // 2, y=state_y + 35, cls="tileLabel", text=state, text_anchor="middle")
        if prev_right is not None:
            _draw_ortho_arrow(
                out,
                x1=prev_right,
                y1=state_y + 29,
                x2=x,
                y2=state_y + 29,
                via_y=state_y + 29,
                accent=accent,
            )
        prev_right = x + state_w
