#!/usr/bin/env python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------
"""Generate the A2/A3 CCE cycle profile header from its CSV source."""

from __future__ import annotations

import argparse
import csv
import difflib
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "cce_cycle_profiles.csv"
OUT_PATH = SCRIPT_DIR / "cce_cycle_profiles_generated.hpp"

EXPECTED_HEADER = [
    "symbol",
    "profile_kind",
    "fp16_startup",
    "fp16_slope",
    "fp32_startup",
    "fp32_slope",
    "integer_startup",
    "integer_slope",
    "startup",
    "slope",
    "fill",
    "decay",
    "valid_repeat_max",
    "serialize_every_call",
]

PROFILE_TYPES = {
    "linear": "LinearCycleProfile",
    "integer_linear": "IntegerLinearCycleProfile",
    "exp_fill": "ExpFillCycleProfile",
}

PROFILE_FIELDS = {
    "linear": ("fp16_startup", "fp16_slope", "fp32_startup", "fp32_slope"),
    "integer_linear": ("fp16_startup", "fp16_slope", "fp32_startup", "fp32_slope", "integer_startup", "integer_slope"),
    "exp_fill": ("startup", "slope", "fill", "decay", "valid_repeat_max", "serialize_every_call"),
}

SYMBOL_PATTERN = re.compile(r"^k[A-Z][A-Za-z0-9]*Profile$")


def _parse_decimal(text: str, field: str, symbol: str) -> str:
    value_text = text.strip()
    if not value_text:
        raise ValueError(f"{symbol}: missing required field '{field}'")
    try:
        value = Decimal(value_text)
    except InvalidOperation as exc:
        raise ValueError(f"{symbol}: invalid decimal '{value_text}' in '{field}'") from exc
    if not value.is_finite():
        raise ValueError(f"{symbol}: non-finite value '{value_text}' in '{field}'")
    if "." not in value_text and "e" not in value_text.lower():
        value_text += ".0"
    return value_text


def _parse_non_negative_int(text: str, field: str, symbol: str) -> str:
    value_text = text.strip()
    try:
        value = int(value_text)
    except ValueError as exc:
        raise ValueError(f"{symbol}: invalid integer '{value_text}' in '{field}'") from exc
    if value < 0:
        raise ValueError(f"{symbol}: '{field}' must be non-negative")
    return str(value)


def _parse_bool(text: str, field: str, symbol: str) -> str:
    value_text = text.strip().lower()
    if value_text not in {"true", "false"}:
        raise ValueError(f"{symbol}: '{field}' must be true or false")
    return value_text


def _validate_header(fieldnames: list[str] | None) -> None:
    if fieldnames != EXPECTED_HEADER:
        raise ValueError(f"Invalid header in {CSV_PATH}: {fieldnames}, expected {EXPECTED_HEADER}")


def _parse_row(row: dict[str, str], seen_symbols: set[str]) -> tuple[str, str, list[str]]:
    if set(row) != set(EXPECTED_HEADER) or any(value is None for value in row.values()):
        raise ValueError(f"Malformed CSV row: {row}")
    symbol = row["symbol"].strip()
    profile_kind = row["profile_kind"].strip().lower()
    if not SYMBOL_PATTERN.fullmatch(symbol):
        raise ValueError(f"Invalid C++ profile symbol '{symbol}'")
    if symbol in seen_symbols:
        raise ValueError(f"Duplicate C++ profile symbol '{symbol}'")
    if profile_kind not in PROFILE_TYPES:
        raise ValueError(f"{symbol}: unsupported profile_kind '{profile_kind}'")

    required_fields = set(PROFILE_FIELDS[profile_kind])
    parameter_fields = set(EXPECTED_HEADER[2:])
    unexpected_fields = sorted(field for field in parameter_fields - required_fields if row[field].strip())
    if unexpected_fields:
        raise ValueError(f"{symbol}: fields not used by {profile_kind}: {unexpected_fields}")

    values: list[str] = []
    for field in PROFILE_FIELDS[profile_kind]:
        if field == "valid_repeat_max":
            values.append(_parse_non_negative_int(row[field], field, symbol))
        elif field == "serialize_every_call":
            values.append(_parse_bool(row[field], field, symbol))
        else:
            values.append(_parse_decimal(row[field], field, symbol))

    if profile_kind == "exp_fill" and Decimal(row["decay"].strip()) <= 0:
        raise ValueError(f"{symbol}: decay must be positive")

    seen_symbols.add(symbol)
    return symbol, profile_kind, values


def _load_rows() -> list[tuple[str, str, list[str]]]:
    rows: list[tuple[str, str, list[str]]] = []
    seen_symbols: set[str] = set()
    with CSV_PATH.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        _validate_header(reader.fieldnames)
        for row in reader:
            rows.append(_parse_row(row, seen_symbols))
    if not rows:
        raise ValueError(f"No CCE cycle profiles found in {CSV_PATH}")
    return rows


def _render_header(rows: list[tuple[str, str, list[str]]]) -> str:
    lines = [
        "/**",
        "Copyright (c) 2026 Huawei Technologies Co., Ltd.",
        "This program is free software, you can redistribute it and/or modify it under the terms and conditions of",
        'CANN Open Software License Agreement Version 2.0 (the "License").',
        "Please refer to the License for details. You may not use this file except in compliance with the License.",
        'THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,',
        "INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.",
        "See LICENSE in the root of the software repository for the full text of the License.",
        "*/",
        "",
        "// Auto-generated by gen_cce_cycle_profiles_header.py from cce_cycle_profiles.csv.",
        "// Do not edit this file manually.",
        "",
        "#pragma once",
        "",
        "#include <pto/costmodel/a2a3/cce_costmodel/cce_costmodel_core.hpp>",
        "",
        "namespace cce_cycle_profiles {",
        "",
        "// clang-format off",
        "// NOLINTBEGIN(readability-magic-numbers)",
    ]
    for symbol, profile_kind, values in rows:
        profile_type = PROFILE_TYPES[profile_kind]
        lines.append(f"inline constexpr cce_costmodel_detail::{profile_type} {symbol}{{{', '.join(values)}}};")
    lines.extend(
        ["// NOLINTEND(readability-magic-numbers)", "// clang-format on", "", "} // namespace cce_cycle_profiles", ""]
    )
    return "\n".join(lines)


def _check_output(expected: str) -> None:
    actual = OUT_PATH.read_text(encoding="utf-8") if OUT_PATH.exists() else ""
    if actual == expected:
        return
    diff = "".join(
        difflib.unified_diff(
            actual.splitlines(keepends=True),
            expected.splitlines(keepends=True),
            fromfile=str(OUT_PATH),
            tofile=f"{OUT_PATH} (generated)",
        )
    )
    raise RuntimeError(f"{OUT_PATH} is stale; run {Path(__file__).name}\n{diff}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the committed header is stale")
    args = parser.parse_args()

    content = _render_header(_load_rows())
    if args.check:
        _check_output(content)
    else:
        OUT_PATH.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
