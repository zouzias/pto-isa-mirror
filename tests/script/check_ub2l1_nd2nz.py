# --------------------------------------------------------------------------------
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

"""Check A5 UB ND-to-L1 NZ diagnostics and debug assertion lowering without an NPU."""

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


ALIGNMENT = "UB-to-L1 ND-to-NZ columns and column offsets must be 32-byte aligned."
SOURCE_WINDOW = "TEXTRACT ND-to-NZ source window exceeds valid shape."
DESTINATION_WINDOW = "UB-to-L1 ND-to-NZ destination window exceeds storage."


def make_source(operation, **overrides):
    params = dict(
        src_type="half",
        dst_type="half",
        src_rows=16,
        src_cols=64,
        dst_rows=16,
        dst_cols=64,
        src_valid_rows=7,
        src_valid_cols=32,
        dst_valid_rows=7,
        dst_valid_cols=32,
        row=0,
        col=0,
        compact="Null",
        fractal=512,
    )
    params.update(overrides)
    for tile in ("src", "dst"):
        rows, cols = params[tile + "_valid_rows"], params[tile + "_valid_cols"]
        params[tile + "_shape"] = f"{rows}, {cols}" if rows and cols else "DYNAMIC, DYNAMIC"
        params[tile + "_init"] = "" if rows and cols else f"({rows}, {cols})"
    call = "TMOV(dst, src)" if operation == "TMOV" else f"{operation}(dst, src, {params['row']}, {params['col']})"
    return f"""#include <pto/pto-inst.hpp>
using namespace pto;
__global__ AICORE void checkNdToNz() {{
    Tile<TileType::Vec, {params["src_type"]}, {params["src_rows"]}, {params["src_cols"]},
         BLayout::RowMajor, {params["src_shape"]}> src{params["src_init"]};
    Tile<TileType::Mat, {params["dst_type"]}, {params["dst_rows"]}, {params["dst_cols"]},
         BLayout::ColMajor, {params["dst_shape"]}, SLayout::RowMajor, {params["fractal"]},
         PadValue::Null, CompactMode::{params["compact"]}> dst{params["dst_init"]};
    TASSIGN(src, 0);
    TASSIGN(dst, 0);
    {call};
}}
"""


def cases():
    for operation in ("TMOV", "TEXTRACT", "TINSERT"):
        yield operation + "_valid", operation, "copy", "", {}
        yield (
            operation + "_65536_columns",
            operation,
            "copy",
            "",
            dict(
                src_type="float4_e2m1x2_t",
                dst_type="float4_e2m1x2_t",
                src_rows=1,
                src_cols=65536,
                dst_cols=65536,
                src_valid_rows=1,
                dst_valid_rows=1,
                src_valid_cols=65536,
                dst_valid_cols=65536,
            ),
        )
        yield operation + "_empty_rows", operation, "empty", "", dict(src_valid_rows=0, dst_valid_rows=0)
        yield operation + "_empty_cols", operation, "empty", "", dict(src_valid_cols=0, dst_valid_cols=0)
        yield operation + "_unaligned_width", operation, "trap", ALIGNMENT, dict(src_valid_cols=31, dst_valid_cols=31)
        yield (
            operation + "_compact",
            operation,
            "compile_error",
            "requires a non-compact NZ512 Mat destination",
            dict(compact="Normal"),
        )
    yield "TMOV_shape_mismatch", "TMOV", "trap", "TMOV ND-to-NZ requires matching valid shapes.", dict(dst_valid_rows=6)
    for dimension, offset in (("row", 10), ("col", 48)):
        yield "TEXTRACT_valid_" + dimension, "TEXTRACT", "trap", SOURCE_WINDOW, {dimension: offset}
        yield "TINSERT_storage_" + dimension, "TINSERT", "trap", DESTINATION_WINDOW, {dimension: offset}
    for operation in ("TEXTRACT", "TINSERT"):
        yield (
            operation + "_unaligned_offset",
            operation,
            "trap",
            ALIGNMENT,
            dict(col=1, src_valid_cols=64, dst_valid_cols=32) if operation == "TEXTRACT" else dict(col=1),
        )
    for operation, prefix in (("TMOV", "TMov"), ("TEXTRACT", "TExtract")):
        yield (
            operation + "_int32",
            operation,
            "compile_error",
            prefix + ": Unsupported data type!",
            dict(src_type="int32_t", dst_type="int32_t"),
        )
    yield (
        "TINSERT_uint8",
        "TINSERT",
        "compile_error",
        "TINSERT : Unsupported data type.",
        dict(src_type="uint8_t", dst_type="uint8_t"),
    )
    yield (
        "TMOV_type_mismatch",
        "TMOV",
        "compile_error",
        "Destination and Source tile data types must be the same",
        dict(dst_type="float"),
    )


def check_case(compiler, objdump, repo, cann, directory, case):
    name, operation, expected, diagnostic, params = case
    source = directory / (name + ".cpp")
    obj = directory / (name + ".o")
    source.write_text(make_source(operation, **params))
    # Mark CANN headers as system headers so --cce-enable-print cannot shadow this checkout's PTO headers.
    command = [
        compiler,
        "-x",
        "cce",
        "--cce-aicore-arch=dav-c310-vec",
        "--cce-aicore-only",
        "--cce-enable-print",
        "-DREGISTER_BASE",
        "-D_DEBUG",
        "-std=c++17",
        "-O2",
        "-I",
        str(repo / "include"),
        "-isystem",
        str(cann / "include"),
        "-c",
        str(source),
        "-o",
        str(obj),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if expected == "compile_error":
        if result.returncode == 0 or diagnostic not in result.stderr:
            raise RuntimeError(f"{name}: expected compilation failure containing {diagnostic!r}\n{result.stderr}")
    else:
        if result.returncode != 0:
            raise RuntimeError(f"{name}: compilation failed\n{result.stderr}")
        assembly = subprocess.run(
            [objdump, "-d", str(obj)], capture_output=True, text=True, check=True, timeout=30
        ).stdout
        if "<not available>" in assembly or "<unknown>" in assembly:
            raise RuntimeError(
                f"{name}: CANN llvm-objdump cannot decode the A5 instructions; "
                "use a CANN toolchain with A5 disassembly support"
            )
        has_trap = re.search(r"\bTRAP\b", assembly) is not None
        has_copy = re.search(r"\bMOV_UB_TO_L1\b", assembly) is not None
        # Some Bisheng versions retain unreachable copies after a debug trap.
        if has_trap != (expected == "trap") or (expected != "trap" and has_copy != (expected == "copy")):
            raise RuntimeError(f"{name}: expected {expected}, got trap={has_trap}, copy={has_copy}")
        if diagnostic and diagnostic.encode() not in obj.read_bytes():
            raise RuntimeError(f"{name}: missing assertion diagnostic {diagnostic!r}")
    print(f"PASS {name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="bisheng", help="CANN Bisheng compiler")
    args = parser.parse_args()
    compiler = shutil.which(args.compiler)
    if compiler is None or not os.environ.get("ASCEND_HOME_PATH"):
        parser.error("source the CANN set_env.sh before running this check")
    objdump = Path(compiler).resolve().with_name("llvm-objdump")
    if not objdump.is_file():
        parser.error(f"missing CANN disassembler: {objdump}")
    repo = Path(__file__).resolve().parents[2]
    all_cases = list(cases())
    with tempfile.TemporaryDirectory(prefix="pto-nd2nz-") as temporary:
        for case in all_cases:
            check_case(compiler, str(objdump), repo, Path(os.environ["ASCEND_HOME_PATH"]), Path(temporary), case)
    print(f"Passed {len(all_cases)} A5 ND-to-NZ compiler contract checks (no device execution).")


if __name__ == "__main__":
    main()
