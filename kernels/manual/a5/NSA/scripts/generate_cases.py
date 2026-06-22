#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

"""
Generate NSA case configuration.

Case entry format:
  HEAD_SIZE,S0,S1,CUBE_S0[,TILE_S1[,BLOCK_SIZE[,NUM_SELECTED[,SELECT_BLOCK_SIZE[,WINDOW[,COMPRESS_DIM]]]]]]

Derived branch sequence lengths (must each be divisible by TILE_S1):
  S1_CMP  = S1 / BLOCK_SIZE
  S1_SLC  = NUM_SELECTED * SELECT_BLOCK_SIZE
  S1_WIN  = WINDOW
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

TILE_S1_DEFAULT = 128
QK_PRELOAD_DEFAULT = 4
BLOCK_SIZE_DEFAULT = 4
NUM_SELECTED_DEFAULT = 32
SELECT_BLOCK_SIZE_DEFAULT = 4
WINDOW_DEFAULT = 128
COMPRESS_DIM_DEFAULT = 256

DEFAULT_CASES = [
    (128, 128, 512, 128, TILE_S1_DEFAULT, BLOCK_SIZE_DEFAULT, NUM_SELECTED_DEFAULT,
     SELECT_BLOCK_SIZE_DEFAULT, WINDOW_DEFAULT, COMPRESS_DIM_DEFAULT, False),
]


def _derived(case: Dict[str, int]) -> Dict[str, int]:
    s1 = case["s1"]
    block_size = case["block_size"]
    if s1 % block_size != 0:
        raise ValueError("S1 must be divisible by BLOCK_SIZE")
    num_blocks = s1 // block_size
    s1_cmp = num_blocks
    s1_slc = case["num_selected"] * case["select_block_size"]
    s1_win = case["window_size"]
    tile_s1 = case["tile_s1"]
    for name, val in [("S1_CMP", s1_cmp), ("S1_SLC", s1_slc), ("S1_WIN", s1_win)]:
        if val % tile_s1 != 0:
            raise ValueError(f"{name}={val} must be divisible by TILE_S1={tile_s1}")
    case["num_blocks"] = num_blocks
    case["s1_cmp"] = s1_cmp
    case["s1_slc"] = s1_slc
    case["s1_win"] = s1_win
    return case


def _parse_case_entry(raw: str, qk_preload: int, causal_mask: bool) -> Dict[str, int]:
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    if len(parts) < 4:
        raise ValueError(
            "Expected at least HEAD,S0,S1,CUBE_S0 "
            "[,TILE_S1[,BLOCK_SIZE[,NUM_SELECTED[,SELECT_BLOCK_SIZE[,WINDOW[,COMPRESS_DIM]]]]]]"
        )
    head, s0, s1, cube_s0 = map(int, parts[:4])
    tile_s1 = int(parts[4]) if len(parts) >= 5 else TILE_S1_DEFAULT
    block_size = int(parts[5]) if len(parts) >= 6 else BLOCK_SIZE_DEFAULT
    num_selected = int(parts[6]) if len(parts) >= 7 else NUM_SELECTED_DEFAULT
    select_block_size = int(parts[7]) if len(parts) >= 8 else SELECT_BLOCK_SIZE_DEFAULT
    window_size = int(parts[8]) if len(parts) >= 9 else WINDOW_DEFAULT
    compress_dim = int(parts[9]) if len(parts) >= 10 else COMPRESS_DIM_DEFAULT
    case = {
        "head_size": head,
        "s0": s0,
        "s1": s1,
        "cube_s0": cube_s0,
        "cube_s1": 128,
        "tile_s1": tile_s1,
        "block_size": block_size,
        "num_selected": num_selected,
        "select_block_size": select_block_size,
        "window_size": window_size,
        "compress_dim": compress_dim,
        "qk_preload": qk_preload,
        "causal_mask": int(causal_mask),
    }
    return _derived(case)


def _default_cases(qk_preload: int) -> List[Dict[str, int]]:
    cases = []
    for row in DEFAULT_CASES:
        head, s0, s1, cube_s0, tile_s1, block_size, num_selected, select_block_size, window_size, compress_dim, causal_mask = row
        cases.append(_derived({
            "head_size": head,
            "s0": s0,
            "s1": s1,
            "cube_s0": cube_s0,
            "cube_s1": 128,
            "tile_s1": tile_s1,
            "block_size": block_size,
            "num_selected": num_selected,
            "select_block_size": select_block_size,
            "window_size": window_size,
            "compress_dim": compress_dim,
            "qk_preload": qk_preload,
            "causal_mask": int(causal_mask),
        }))
    return cases


def _normalize_case(case: Dict[str, int]) -> Dict[str, int]:
    if case["qk_preload"] < 1:
        raise ValueError("qk_preload must be >= 1")
    if case["cube_s0"] > case["s0"] or case["s0"] % case["cube_s0"] != 0:
        case["cube_s0"] = case["s0"]
    if case["s1"] % case["cube_s1"] != 0:
        raise ValueError("S1 must be divisible by CUBE_S1 (128)")
    if case["tile_s1"] % case["cube_s1"] != 0:
        raise ValueError("TILE_S1 must be divisible by CUBE_S1")
    return _derived(case)


def _case_name(case: Dict[str, int]) -> str:
    return (
        f"case_float_H_{case['head_size']}_S0_{case['s0']}_S1_{case['s1']}"
        f"_B{case['block_size']}_N{case['num_selected']}_W{case['window_size']}"
    )


def _render_macro(cases: List[Dict[str, int]]) -> str:
    lines = ["#define TNSA_FOR_EACH_CASE(MACRO) \\"]
    for idx, case in enumerate(cases):
        causal = "true" if bool(case["causal_mask"]) else "false"
        suffix = " \\" if idx + 1 != len(cases) else ""
        line = (
            f"    MACRO({case['s0']}, {case['head_size']}, {case['s1']}, {case['s1_cmp']}, "
            f"{case['s1_slc']}, {case['s1_win']}, {case['block_size']}, {case['num_selected']}, "
            f"{case['window_size']}, {case['cube_s0']}, {case['cube_s1']}, {case['tile_s1']}, "
            f"{case['qk_preload']}, {causal}){suffix}"
        )
        lines.append(line)
    return "\n".join(lines)


def _render_branch_macro(cases: List[Dict[str, int]]) -> str:
    seen = set()
    branches: List[tuple] = []
    for case in cases:
        causal = "true" if bool(case["causal_mask"]) else "false"
        for s1 in (case["s1_cmp"], case["s1_slc"], case["s1_win"]):
            key = (
                case["s0"],
                case["head_size"],
                s1,
                case["cube_s0"],
                case["cube_s1"],
                case["tile_s1"],
                case["qk_preload"],
                causal,
            )
            if key not in seen:
                seen.add(key)
                branches.append(key)
    lines = ["#define TNSA_FOR_EACH_BRANCH(MACRO) \\"]
    for idx, branch in enumerate(branches):
        suffix = " \\" if idx + 1 != len(branches) else ""
        s0, head, s1, cube_s0, cube_s1, tile_s1, qk_preload, causal = branch
        lines.append(
            f"    MACRO({s0}, {head}, {s1}, {cube_s0}, {cube_s1}, {tile_s1}, {qk_preload}, {causal}){suffix}"
        )
    return "\n".join(lines)


def _render_header(cases: List[Dict[str, int]]) -> str:
    macro_block = _render_macro(cases)
    branch_macro_block = _render_branch_macro(cases)
    entries = []
    for case in cases:
        entries.append(
            "    {" + ", ".join([
                str(case["s0"]),
                str(case["head_size"]),
                str(case["s1"]),
                str(case["s1_cmp"]),
                str(case["s1_slc"]),
                str(case["s1_win"]),
                str(case["block_size"]),
                str(case["num_selected"]),
                str(case["select_block_size"]),
                str(case["window_size"]),
                str(case["compress_dim"]),
                str(case["cube_s0"]),
                str(case["cube_s1"]),
                str(case["tile_s1"]),
                str(case["qk_preload"]),
                str("true" if bool(case["causal_mask"]) else "false"),
                f'"{_case_name(case)}"',
            ]) + "}"
        )
    return f"""#pragma once
// Auto-generated by scripts/generate_cases.py. Do not edit manually.
// clang-format off
#include <cstddef>

{macro_block}

{branch_macro_block}

#define TFA_FOR_EACH_CASE(MACRO) TNSA_FOR_EACH_BRANCH(MACRO)

struct GeneratedTnsaCase {{
    int s0;
    int head_size;
    int s1;
    int s1_cmp;
    int s1_slc;
    int s1_win;
    int block_size;
    int num_selected;
    int select_block_size;
    int window_size;
    int compress_dim;
    int cube_s0;
    int cube_s1;
    int tile_s1;
    int qk_preload;
    bool causal_mask;
    const char *name;
}};

static constexpr GeneratedTnsaCase kGeneratedTnsaCases[] = {{
{",".join(entries)}
}};
static constexpr std::size_t kGeneratedTnsaCasesCount = sizeof(kGeneratedTnsaCases) / sizeof(kGeneratedTnsaCases[0]);
// clang-format on
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NSA case header/JSON")
    parser.add_argument("--cases", action="append", default=None)
    parser.add_argument("--qk-preload", type=int, default=QK_PRELOAD_DEFAULT)
    parser.add_argument(
        "--output-header",
        default=str(Path(__file__).resolve().parent.parent / "build" / "generated_cases.h"),
    )
    parser.add_argument(
        "--output-json",
        default=str(Path(__file__).resolve().parent.parent / "build" / "generated_cases.json"),
    )
    parser.add_argument("--causal-mask", default=False)
    args = parser.parse_args()

    if args.cases:
        cases = [_normalize_case(_parse_case_entry(e, args.qk_preload, args.causal_mask)) for e in args.cases]
    else:
        cases = [_normalize_case(c) for c in _default_cases(args.qk_preload)]

    header_path = Path(args.output_header)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(_render_header(cases))

    json_path = Path(args.output_json)
    json_payload = [{"name": _case_name(c), **c} for c in cases]
    json_path.write_text(json.dumps(json_payload, indent=2))

    print(f"[INFO] Wrote {header_path}")
    print(f"[INFO] Wrote {json_path}")
    for case in json_payload:
        print(
            f"  - {case['name']} (H={case['head_size']}, S0={case['s0']}, S1={case['s1']}, "
            f"S1_CMP={case['s1_cmp']}, S1_SLC={case['s1_slc']}, S1_WIN={case['s1_win']})"
        )


if __name__ == "__main__":
    main()
