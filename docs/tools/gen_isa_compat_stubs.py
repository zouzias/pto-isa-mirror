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

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ISA_DIR = REPO_ROOT / "docs" / "isa"
MANIFEST_PATH = ISA_DIR / "manifest.yaml"

CATEGORY_TO_DIR = {
    "Synchronization": "tile/ops/sync-and-config",
    "Manual / Resource Binding": "tile/ops/sync-and-config",
    "Elementwise (Tile-Tile)": "tile/ops/elementwise-tile-tile",
    "Tile-Scalar / Tile-Immediate": "tile/ops/tile-scalar-and-immediate",
    "Axis Reduce / Expand": "tile/ops/reduce-and-expand",
    "Memory (GM <-> Tile)": "tile/ops/memory-and-data-movement",
    "Matrix Multiply": "tile/ops/matrix-and-matrix-vector",
    "Data Movement / Layout": "tile/ops/layout-and-rearrangement",
    "Complex": "tile/ops/irregular-and-complex",
}

SPECIAL_SLUGS = {
    "SET_IMG2COL_RPT": "set-img2col-rpt",
    "SET_IMG2COL_PADDING": "set-img2col-padding",
    "TGEMV_MX": "tgemv-mx",
    "TMATMUL_MX": "tmatmul-mx",
    "TMATMUL_ACC": "tmatmul-acc",
    "TMATMUL_BIAS": "tmatmul-bias",
    "TGEMV_ACC": "tgemv-acc",
    "TGEMV_BIAS": "tgemv-bias",
    "TFILLPAD_INPLACE": "tfillpad-inplace",
    "TFILLPAD_EXPAND": "tfillpad-expand",
    "TEXTRACT_FP": "textract",
    "TINSERT_FP": "tinsert",
    "TMOV_FP": "tmov",
    "TSUBVIEW": "subview",
    "TGET_SCALE_ADDR": "get-scale-addr",
}

DOC_ONLY_KEEP = {
    "TALLOC",
    "TPUSH",
    "TPOP",
    "TFREE",
    "TPARTARGMAX",
    "TPARTARGMIN",
    "TCONCAT",
    "SETFMATRIX",
    "SET_IMG2COL_RPT",
    "SET_IMG2COL_PADDING",
    "SETHF32MODE",
    "SETTF32MODE",
}


def load_manifest() -> list[dict[str, object]]:
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return list(data.get("instructions", []))


def instr_to_slug(instr: str) -> str:
    if instr in SPECIAL_SLUGS:
        return SPECIAL_SLUGS[instr]
    return instr.lower()


def redirect_target(instr: str, category: str, zh: bool) -> str | None:
    if instr in DOC_ONLY_KEEP:
        return None
    directory = CATEGORY_TO_DIR.get(category)
    if not directory:
        return None
    suffix = "_zh" if zh else ""
    return f"{directory}/{instr_to_slug(instr)}{suffix}.md"


def render_stub(instr: str, target: str, zh: bool) -> str:
    title = instr
    if zh:
        return (
            f"# {title}\n\n"
            f"> 此页面已迁移。请参阅 [{title}]({target})。\n"
        )
    return (
        f"# {title}\n\n"
        f"> This page has moved. See [{title}]({target}).\n"
    )


def main() -> int:
    for entry in load_manifest():
        instr = str(entry["instruction"])
        category = str(entry["category"])
        for zh in (False, True):
            target = redirect_target(instr, category, zh)
            if not target:
                continue
            name = f"{instr}{'_zh' if zh else ''}.md"
            path = ISA_DIR / name
            path.write_text(render_stub(instr, target, zh), encoding="utf-8")
    print("Generated ISA compatibility stubs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
