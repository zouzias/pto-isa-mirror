# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

"""Check A2/A3 TEXTRACT TileType and layout dispatch with Bisheng, without device execution."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


UNSUPPORTED_PATH = "TEXTRACT A2A3: Unsupported source and destination TileType combination."
UNSUPPORTED_LAYOUT = "TEXTRACT A2A3 Vec->Vec: Only ND-to-ND and NZ-to-NZ are supported."


def tile(location, dtype="half", nz=False, rows=32, valid_rows=32):
    if location == "Acc":
        dtype = "float"
    row_major = location in ("Left", "Right") or (location == "Vec" and not nz)
    layout = "RowMajor" if row_major else "ColMajor"
    fractal = "NoneBox" if location == "Vec" and not nz else "RowMajor"
    if location == "Right":
        fractal = "ColMajor"
    return f"Tile<TileType::{location}, {dtype}, {rows}, 32, BLayout::{layout}, {valid_rows}, 32, SLayout::{fractal}>"


def make_source(src, dst, dual=False):
    extra = "DstTile dst1; TASSIGN(dst1, 4096);" if dual else ""
    call = "TEXTRACT(dst, dst1, src, 0, 0, 0, 0)" if dual else "TEXTRACT(dst, src, 0, 0)"
    return f"""#include <pto/pto-inst.hpp>
using namespace pto;
__global__ AICORE void checkExtract() {{
    using SrcTile = {src};
    using DstTile = {dst};
    SrcTile src;
    DstTile dst;
    TASSIGN(src, 0);
    TASSIGN(dst, 8192);
    {extra}
    {call};
}}
"""


def cases():
    supported = {("Vec", "Vec"), ("Mat", "Left"), ("Mat", "Right"), ("Acc", "Mat")}
    for src in ("Vec", "Mat", "Left", "Right", "Acc"):
        for dst in ("Vec", "Mat", "Left", "Right", "Acc"):
            core = "vec" if src == "Vec" and dst in ("Vec", "Mat") else "cube"
            diagnostic = "" if (src, dst) in supported else UNSUPPORTED_PATH
            yield f"{src}_to_{dst}", core, diagnostic, make_source(tile(src), tile(dst))
    # Cover the Mat/Acc dtype set with both ND and NZ UB inputs.
    for dtype in ("half", "bfloat16_t", "float", "int8_t"):
        for nz in (False, True):
            if dtype == "half" and not nz:
                continue  # Covered by the TileType matrix.
            name = f"Vec_to_Mat_{dtype}_{'nz' if nz else 'nd'}"
            yield name, "vec", UNSUPPORTED_PATH, make_source(tile("Vec", dtype, nz), tile("Mat", dtype))
    yield "Vec_to_Vec_nz", "vec", "", make_source(tile("Vec", nz=True), tile("Vec", nz=True))
    yield "Mat_to_Left_small_m", "cube", "", make_source(tile("Mat"), tile("Left", rows=16, valid_rows=8))
    yield "Vec_nd_to_two_nz", "vec", "", make_source(tile("Vec"), tile("Vec", nz=True), dual=True)

    layouts = [(outer, inner) for outer in ("RowMajor", "ColMajor") for inner in ("NoneBox", "RowMajor", "ColMajor")]
    for src_outer, src_inner in layouts:
        for dst_outer, dst_inner in layouts:
            if (src_outer, src_inner) == (dst_outer, dst_inner) and (src_outer, src_inner) in (
                ("RowMajor", "NoneBox"),
                ("ColMajor", "RowMajor"),
            ):
                continue  # ND-to-ND and NZ-to-NZ already have positive controls.
            src = f"Tile<TileType::Vec, half, 32, 32, BLayout::{src_outer}, 32, 32, SLayout::{src_inner}>"
            dst = f"Tile<TileType::Vec, half, 32, 32, BLayout::{dst_outer}, 32, 32, SLayout::{dst_inner}>"
            name = f"Vec_{src_outer}_{src_inner}_to_{dst_outer}_{dst_inner}"
            yield name, "vec", UNSUPPORTED_LAYOUT, make_source(src, dst)


def check_case(compiler, repo, cann, directory, case):
    name, core, diagnostic, code = case
    source = directory / (name + ".cpp")
    source.write_text(code)
    command = [
        compiler,
        "-x",
        "cce",
        "--cce-aicore-arch=dav-c220-" + core,
        "--cce-aicore-only",
        "-std=c++17",
        "-O2",
        "-I",
        str(repo / "include"),
        "-isystem",
        str(cann / "include"),
        "-c",
        str(source),
        "-o",
        str(directory / (name + ".o")),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if not diagnostic:
        if result.returncode != 0:
            raise RuntimeError(f"{name}: supported path failed to compile\n{result.stderr}")
    elif result.returncode == 0 or diagnostic not in result.stderr:
        raise RuntimeError(f"{name}: expected compilation failure containing {diagnostic!r}\n{result.stderr}")
    print(f"PASS {name} ({'rejected' if diagnostic else 'supported'})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="bisheng", help="CANN Bisheng compiler")
    args = parser.parse_args()
    compiler = shutil.which(args.compiler)
    if compiler is None or not os.environ.get("ASCEND_HOME_PATH"):
        parser.error("source the CANN set_env.sh before running this check")
    repo = Path(__file__).resolve().parents[2]
    all_cases = list(cases())
    with tempfile.TemporaryDirectory(prefix="pto-a2a3-textract-") as temporary:
        for case in all_cases:
            check_case(compiler, repo, Path(os.environ["ASCEND_HOME_PATH"]), Path(temporary), case)
    print(f"Passed {len(all_cases)} A2/A3 TEXTRACT compiler checks (no device execution).")


if __name__ == "__main__":
    main()
