#!/usr/bin/env python3
"""
Convert elements-per-cycle (epc) to bytes-per-cycle by multiplying epc by
an inferred dtype size in bytes.
"""

import argparse
import csv
import os
import re
from pathlib import Path


DTYPE_BYTES = {
    "half": 2,
    "float": 4,
    "double": 8,
    "bfloat16": 2,
    "bf16": 2,
    "fp16": 2,
    "fp32": 4,
    "fp64": 8,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "s8": 1,
    "u8": 1,
    "s16": 2,
    "u16": 2,
    "s32": 4,
    "u32": 4,
    "s64": 8,
    "u64": 8,
    "float16": 2,
    "float32": 4,
    "float64": 8,
}


def normalize_dtype_name(raw: str) -> str:
    if raw is None:
        return ""
    t = raw.strip().lower()
    t = t.replace(" ", "")
    # Strip leading dimension hints like 1d_, 2d_, 3d_.
    t = re.sub(r"^\d+d_", "", t)
    return t


def dtype_bytes(dtype_name: str):
    t = normalize_dtype_name(dtype_name)
    if not t:
        return None

    if t in DTYPE_BYTES:
        return DTYPE_BYTES[t]

    # Handle composite names like "uint16_uint8" or "half_uint8".
    parts = [p for p in t.split("_") if p]
    if len(parts) > 1 and all(p in DTYPE_BYTES for p in parts):
        return sum(DTYPE_BYTES[p] for p in parts)

    # Handle int/uint with bit width, e.g. int16, uint32.
    m = re.fullmatch(r"(u?int|s)(\d+)", t)
    if m:
        bits = int(m.group(2))
        if bits % 8 == 0:
            return bits // 8

    return None


def pick_dtype_bytes(row, mode: str):
    if mode == "src":
        return dtype_bytes(row.get("src_dtype", ""))
    if mode == "dst":
        return dtype_bytes(row.get("dst_dtype", ""))
    if mode == "both":
        src = dtype_bytes(row.get("src_dtype", ""))
        dst = dtype_bytes(row.get("dst_dtype", ""))
        if src is None or dst is None:
            return None
        return src + dst
    return None


def process_csv(path: Path, out_path: Path, dtype_mode: str, fail_on_unknown: bool):
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if "epc" not in fieldnames:
            raise ValueError(f"Missing 'epc' column in {path}")

        dtype_col = "dtype_bytes"
        bpc_col = "bytes_per_cycle"
        if dtype_col not in fieldnames:
            fieldnames.append(dtype_col)
        if bpc_col not in fieldnames:
            fieldnames.append(bpc_col)

        with out_path.open("w", newline="") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                size = pick_dtype_bytes(row, dtype_mode)
                row[dtype_col] = "" if size is None else str(size)

                epc_raw = (row.get("epc") or "").strip()
                if epc_raw and size is not None:
                    try:
                        epc_val = float(epc_raw)
                        row[bpc_col] = f"{epc_val * size:.6f}"
                    except ValueError:
                        row[bpc_col] = ""
                else:
                    row[bpc_col] = ""

                if fail_on_unknown and size is None:
                    raise ValueError(
                        f"Unknown dtype for row: src={row.get('src_dtype')} dst={row.get('dst_dtype')}"
                    )

                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Convert epc (elements/cycle) to bytes/cycle using dtype size.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="CSV files to process",
    )
    parser.add_argument(
        "--dtype-field",
        choices=["src", "dst", "both"],
        default="dst",
        help="Which dtype to use for sizing (default: dst)",
    )
    parser.add_argument(
        "--suffix",
        default="_bpc",
        help="Suffix for output files (default: _bpc)",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Optional output directory; defaults to input file directory",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input CSV files",
    )
    parser.add_argument(
        "--fail-on-unknown",
        action="store_true",
        help="Fail if a dtype size cannot be inferred",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    for input_path in args.inputs:
        in_path = Path(input_path)
        if not in_path.exists():
            raise FileNotFoundError(in_path)

        if args.in_place:
            out_path = in_path
        else:
            base = in_path.stem + args.suffix + in_path.suffix
            out_path = (out_dir or in_path.parent) / base

        process_csv(in_path, out_path, args.dtype_field, args.fail_on_unknown)


if __name__ == "__main__":
    main()
