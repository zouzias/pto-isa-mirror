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
Generate GQA (Grouped Query Attention) case configuration and emit a shared header/JSON.

Each --cases entry format: NUM_Q_HEADS,NUM_KV_HEADS,HEAD_SIZE,S0,S1,CUBE_S0[,TILE_S1]
CUBE_S1 is fixed at 128; TILE_S1 defaults to 128 if omitted.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

TILE_S1_DEFAULT = 128
QK_PRELOAD_DEFAULT = 4

DEFAULT_CASES = [
    (8, 2, 128, 128, 512, 128, TILE_S1_DEFAULT, False),
    (8, 2, 128, 128, 1024, 128, TILE_S1_DEFAULT, False),
    (4, 2, 128, 128, 512, 128, TILE_S1_DEFAULT, False),
    (4, 1, 128, 128, 512, 128, TILE_S1_DEFAULT, False),
]


def _parse_case_entry(raw: str, qk_preload: int, causal_mask: bool) -> Dict[str, int]:
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    if len(parts) not in (6, 7):
        raise ValueError(
            "Expected 6 or 7 comma-separated values "
            "(NUM_Q_HEADS,NUM_KV_HEADS,HEAD_SIZE,S0,S1,CUBE_S0[,TILE_S1]), "
            f"got '{raw}'"
        )
    num_q_heads, num_kv_heads, head_size, s0, s1, cube_s0 = map(int, parts[:6])
    tile_s1 = int(parts[6]) if len(parts) == 7 else TILE_S1_DEFAULT
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("NUM_Q_HEADS must be divisible by NUM_KV_HEADS")
    return {
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_size": head_size,
        "s0": s0,
        "s1": s1,
        "cube_s0": cube_s0,
        "cube_s1": 128,
        "tile_s1": tile_s1,
        "qk_preload": qk_preload,
        "causal_mask": int(causal_mask),
    }


def _default_cases(qk_preload: int) -> List[Dict[str, int]]:
    return [
        {
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_size": head_size,
            "s0": s0,
            "s1": s1,
            "cube_s0": cube_s0,
            "cube_s1": 128,
            "tile_s1": tile_s1,
            "qk_preload": qk_preload,
            "causal_mask": int(causal_mask),
        }
        for (num_q_heads, num_kv_heads, head_size, s0, s1, cube_s0, tile_s1, causal_mask) in DEFAULT_CASES
    ]


def _case_name(case: Dict[str, int]) -> str:
    return (
        f"case_float_GQA_Q{case['num_q_heads']}_K{case['num_kv_heads']}"
        f"_H{case['head_size']}_S0_{case['s0']}_S1_{case['s1']}"
    )


def _normalize_case(case: Dict[str, int]) -> Dict[str, int]:
    if case["qk_preload"] < 1:
        raise ValueError("qk_preload must be >= 1")

    if case["num_q_heads"] % case["num_kv_heads"] != 0:
        raise ValueError("NUM_Q_HEADS must be divisible by NUM_KV_HEADS")

    if case["cube_s0"] > case["s0"] or case["s0"] % case["cube_s0"] != 0:
        case["cube_s0"] = case["s0"]

    if case["cube_s1"] != 128:
        case["cube_s1"] = 128
    if case["s1"] % case["cube_s1"] != 0:
        raise ValueError("S1 must be divisible by CUBE_S1 (128)")

    if case["tile_s1"] % case["cube_s1"] != 0:
        raise ValueError("TILE_S1 must be divisible by CUBE_S1 (128)")
    if case["s1"] % case["tile_s1"] != 0:
        raise ValueError("S1 must be divisible by TILE_S1")

    return case


def _render_macro(cases: List[Dict[str, int]]) -> str:
    lines = ["#define TGQA_FOR_EACH_CASE(MACRO) \\"]
    for idx, case in enumerate(cases):
        causal_mask = str("true" if bool(case["causal_mask"]) else "false")
        suffix = " \\" if idx + 1 != len(cases) else ""
        line = (
            f"    MACRO({case['num_q_heads']}, {case['num_kv_heads']}, "
            f"{case['head_size']}, {case['s0']}, {case['s1']}, "
            f"{case['cube_s0']}, {case['cube_s1']}, {case['tile_s1']}, "
            f"{case['qk_preload']}, {causal_mask}){suffix}"
        )
        lines.append(line)
    return "\n".join(lines)


def _render_header(cases: List[Dict[str, int]]) -> str:
    macro_block = _render_macro(cases)
    array_entries = []
    for case in cases:
        array_entries.append(
            "    {" + ", ".join(
                [
                    str(case["num_q_heads"]),
                    str(case["num_kv_heads"]),
                    str(case["head_size"]),
                    str(case["s0"]),
                    str(case["s1"]),
                    str(case["cube_s0"]),
                    str(case["cube_s1"]),
                    str(case["tile_s1"]),
                    str(case["qk_preload"]),
                    str("true" if bool(case["causal_mask"]) else "false"),
                    f'"{_case_name(case)}"',
                ]
            ) + "}"
        )
    array_block = ",\n".join(array_entries)

    return f"""#pragma once
// Auto-generated by scripts/generate_cases.py. Do not edit manually.
// clang-format off
#include <cstddef>

{macro_block}

struct GeneratedGqaCase {{
    int num_q_heads;
    int num_kv_heads;
    int head_size;
    int s0;
    int s1;
    int cube_s0;
    int cube_s1;
    int tile_s1;
    int qk_preload;
    bool causal_mask;
    const char *name;
}};

static constexpr GeneratedGqaCase kGeneratedGqaCases[] = {{
{array_block}
}};
static constexpr std::size_t kGeneratedGqaCasesCount = sizeof(kGeneratedGqaCases) / sizeof(kGeneratedGqaCases[0]);
// clang-format on
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GQA case header/JSON")
    parser.add_argument(
        "--cases",
        action="append",
        default=None,
        help="Case entry NUM_Q_HEADS,NUM_KV_HEADS,HEAD_SIZE,S0,S1,CUBE_S0[,TILE_S1] (repeatable)",
    )
    parser.add_argument(
        "--qk-preload",
        type=int,
        default=QK_PRELOAD_DEFAULT,
        help="qkPreloadNum applied to all generated cases",
    )
    parser.add_argument(
        "--output-header",
        default=str((Path(__file__).resolve().parent.parent / "build" / "generated_cases.h")),
        help="Output header path",
    )
    parser.add_argument(
        "--output-json",
        default=str((Path(__file__).resolve().parent.parent / "build" / "generated_cases.json")),
        help="Output JSON path",
    )
    parser.add_argument(
        "--causal-mask",
        default=False,
        help="Enable causal mask",
    )
    parser.add_argument(
        "--num-q-heads",
        type=int,
        default=None,
        help="Override NUM_Q_HEADS for default cases",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=None,
        help="Override NUM_KV_HEADS for default cases",
    )
    args = parser.parse_args()

    if args.cases:
        cases = [_normalize_case(_parse_case_entry(entry, args.qk_preload, args.causal_mask)) for entry in args.cases]
    else:
        cases = [_normalize_case(case) for case in _default_cases(args.qk_preload)]
        if args.num_q_heads is not None and args.num_kv_heads is not None:
            for case in cases:
                case["num_q_heads"] = args.num_q_heads
                case["num_kv_heads"] = args.num_kv_heads

    header_text = _render_header(cases)
    header_path = Path(args.output_header)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(header_text)

    json_payload = [{"name": _case_name(case), **case} for case in cases]
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(json_payload, indent=2))

    print(f"[INFO] Wrote {header_path}")
    print(f"[INFO] Wrote {json_path}")
    print("[INFO] Cases generated:")
    for case in json_payload:
        print(
            f"  - {case['name']} (Q={case['num_q_heads']}, K={case['num_kv_heads']}, "
            f"H={case['head_size']}, S0={case['s0']}, S1={case['s1']}, "
            f"CUBE_S0={case['cube_s0']}, CUBE_S1={case['cube_s1']}, "
            f"TILE_S1={case['tile_s1']}, QK_PRELOAD={case['qk_preload']}, "
            f"CAUSAL_MASK={case['causal_mask']})"
        )


if __name__ == "__main__":
    main()
