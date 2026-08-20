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

"""Shared small data structures for ISA SVG rendering."""

from __future__ import annotations

from typing import Dict, NamedTuple, Sequence, Tuple


class TDequantLayout(NamedTuple):
    x_src: int
    x_scale: int
    x_offset: int
    y_src: int
    y_para: int
    scale_cols: int


class ScalarArrowFlow(NamedTuple):
    instr: str
    sources: Sequence[Tuple[int, int]]
    dst: Tuple[int, int]
    via_base: int


class Hif4MatmulLayout(NamedTuple):
    x_a: int
    x_sa: int
    x_b: int
    x_sb: int
    x_c: int
    y_data: int
    y_scale: int
    y_dst: int


class TilePatchSpec(NamedTuple):
    x: int
    y: int
    label: str
    prefix: str


class QuantDnLayout(NamedTuple):
    x_src: int
    x_dst: int
    x_exp: int
    x_zz: int
    y_src: int
    y_dst: int
    exp_rows: int


class QuantHif4Layout(NamedTuple):
    x_src: int
    x_dst: int
    x_meta: int
    y_src: int
    y_dst: int


class CommRenderContext(NamedTuple):
    instr: str
    layout: Dict[str, int]
    tile_w: int
    y_src: int


class CommTokenSpec(NamedTuple):
    title: str
    detail: str
    label: str
    value: str


class ConfigRenderContext(NamedTuple):
    tile_w: int
    y_src: int
    y_dst: int
