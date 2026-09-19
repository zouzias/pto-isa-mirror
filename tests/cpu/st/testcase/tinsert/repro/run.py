#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Validate TINSERT CPU reproductions against the current checkout."""

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


def run_case(cxx: str, source: Path, repo: Path, output: Path, case: tuple[str, str, list[str], str]) -> dict:
    name, filename, defines, expected = case
    binary = output / name
    command = [
        cxx,
        "-std=c++23",
        "-O1",
        "-D__CPU_SIM",
        "-pthread",
        "-I",
        str(repo / "include"),
        *[f"-D{value}" for value in defines],
        str(source / filename),
        "-o",
        str(binary),
    ]
    build = subprocess.run(command, capture_output=True, text=True, timeout=120)
    log = "$ " + shlex.join(command) + "\n" + build.stdout + build.stderr
    run_rc = None
    if build.returncode == 0:
        run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=120, cwd=output)
        run_rc = run.returncode
        log += "$ " + str(binary) + "\n" + run.stdout + run.stderr
    matched = (
        (build.returncode != 0 and expected in build.stderr) if expected else (build.returncode == 0 and run_rc == 0)
    )
    result = {
        "name": name,
        "build_rc": build.returncode,
        "run_rc": run_rc,
        "expected": expected or "PASS",
        "matched": matched,
    }
    (output / f"{name}.log").write_text(log)
    return result


def main() -> int:
    source = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expect-original-failures",
        action="store_true",
        help="Check the original compile errors against an unfixed checkout",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    repo = source.parents[5]
    jobs = [(f"split{n}", "split.cpp", [f"SPLIT_COUNT={n}"], "no matching function") for n in (2, 4)]
    for name, a, b, m in (
        ("half_control", "half", "half", 16),
        ("bf16_control", "bfloat16_t", "bfloat16_t", 16),
        ("float_control", "float", "float", 16),
        ("e4m3", "float8_e4m3_t", "float8_e4m3_t", 16),
        ("e5m2", "float8_e5m2_t", "float8_e5m2_t", 16),
        ("e4m3_e5m2", "float8_e4m3_t", "float8_e5m2_t", 16),
        ("e5m2_e4m3", "float8_e5m2_t", "float8_e4m3_t", 16),
        ("hif8", "hifloat8_t", "hifloat8_t", 16),
        ("e4m3_m2032", "float8_e4m3_t", "float8_e4m3_t", 2032),
    ):
        jobs.append(
            (
                f"acc_{name}",
                "acc2mat.cpp",
                [f"A_TYPE={a}", f"B_TYPE={b}", f"M_DIM={m}"],
                "" if name.endswith("control") else "Not supported data type",
            )
        )
    jobs = [
        (name, filename, defines, expected if args.expect_original_failures else "")
        for name, filename, defines, expected in jobs
    ]
    jobs.append(
        (
            "split_unsupported_uint16",
            "split.cpp",
            ["SPLIT_TYPE=uint16_t"],
            "no matching function" if args.expect_original_failures else "TINSERT SPLIT: Unsupported data type",
        )
    )
    results = []
    for name, filename, defines, expected in jobs:
        result = run_case(args.cxx, source, repo, output, (name, filename, defines, expected))
        results.append(result)
        print(json.dumps(result), flush=True)
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0 if all(row["matched"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
