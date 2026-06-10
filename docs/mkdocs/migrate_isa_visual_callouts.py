#!/usr/bin/env python3
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------
"""Migrate PTO ISA Markdown sections to MkDocs/PyMdown visual callouts.

The rewrite is intentionally conservative and idempotent. It preserves each
section heading, wraps known legality/profile sections in callout containers,
and only promotes blockquote notes or explicit diagnostic lines when the source
already marks their intent.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ISA_ROOT = REPO_ROOT / "docs" / "isa"

SECTION_CALLOUTS = {
    "Constraints": "!!! warning",
    "约束": "!!! warning",
    "Target-Profile Restrictions": "??? info",
    "目标 Profile 限制": "??? info",
    "Exceptions": "!!! danger",
    "异常与非法情形": "!!! danger",
    "Cases That Are Not Allowed": "!!! danger",
    "不允许的情形": "!!! danger",
}

BLOCKQUOTE_KINDS = (
    ("warning", ("warning", "warnings", "limitation", "limitations", "警告", "限制")),
    ("danger", ("illegal", "invalid", "not allowed", "非法", "不允许")),
    ("note", ("note", "notes", "注意", "说明")),
)


@dataclass
class Stats:
    files_changed: int = 0
    sections: int = 0
    target_tabs: int = 0
    blockquotes: int = 0
    diagnostics: int = 0


def strip_blank_edges(lines: list[str]) -> list[str]:
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def indent_for_admonition(lines: list[str]) -> list[str]:
    return [f"    {line}" if line else "" for line in lines]


def is_already_wrapped(lines: list[str], expected_marker: str) -> bool:
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        return stripped.startswith(expected_marker) or stripped.startswith("!!! ") or stripped.startswith("??? ")
    return False


def canonical_target_label(heading: str) -> str | None:
    normalized = heading.lower().replace("-", "").replace("_", "").replace("/", "").replace(" ", "")
    if normalized.startswith("cpusim"):
        return "CPU_SIM"
    if normalized.startswith("a2a3") or normalized.startswith("a2/a3"):
        return "A2/A3"
    if normalized.startswith("a5"):
        return "A5"
    return None


def convert_target_h3_to_tabs(body: list[str]) -> tuple[list[str], bool]:
    h3_indices = [index for index, line in enumerate(body) if line.startswith("### ")]
    if len(h3_indices) < 2:
        return body, False

    labels = [canonical_target_label(body[index][4:].strip()) for index in h3_indices]
    if any(label is None for label in labels) or len(set(labels)) != len(labels):
        return body, False

    output = body[: h3_indices[0]]
    for pos, start in enumerate(h3_indices):
        end = h3_indices[pos + 1] if pos + 1 < len(h3_indices) else len(body)
        content = strip_blank_edges(body[start + 1 : end])
        output.append(f'=== "{labels[pos]}"')
        output.extend(indent_for_admonition(content or ["No additional restriction is documented for this target."]))
        output.append("")
    return strip_blank_edges(output), True


def wrap_sections(lines: list[str], stats: Stats) -> list[str]:
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        match = re.match(r"^## (.+?)\s*$", line)
        if not match or match.group(1) not in SECTION_CALLOUTS:
            output.append(line)
            index += 1
            continue

        heading = match.group(1)
        marker = SECTION_CALLOUTS[heading]
        end = index + 1
        while end < len(lines) and not re.match(r"^## [^#]", lines[end]):
            end += 1

        body = lines[index + 1 : end]
        if is_already_wrapped(body, marker):
            output.extend(lines[index:end])
            index = end
            continue

        trimmed = strip_blank_edges(body)
        if heading in {"Target-Profile Restrictions", "目标 Profile 限制"}:
            trimmed, converted_tabs = convert_target_h3_to_tabs(trimmed)
            if converted_tabs:
                stats.target_tabs += 1

        if not trimmed:
            output.extend(lines[index:end])
            index = end
            continue

        output.extend([line, "", f'{marker} "{heading}"'])
        output.extend(indent_for_admonition(trimmed))
        output.append("")
        stats.sections += 1
        index = end
    return output


def blockquote_kind(label: str) -> str | None:
    lowered = label.lower()
    for kind, needles in BLOCKQUOTE_KINDS:
        if any(needle in lowered for needle in needles):
            return kind
    return None


def convert_blockquote(lines: list[str], index: int) -> tuple[list[str] | None, int]:
    if not lines[index].startswith(">"):
        return None, index

    end = index
    block: list[str] = []
    while end < len(lines) and (lines[end].startswith(">") or not lines[end].strip()):
        if lines[end].startswith(">"):
            block.append(lines[end][1:].lstrip())
        else:
            block.append("")
        end += 1

    body = strip_blank_edges(block)
    if not body:
        return None, index

    match = re.match(r"^\*\*(?P<label>[^*]+?)\*\*:?\s*(?P<rest>.*)$", body[0])
    if not match:
        return None, index

    label = match.group("label").strip()
    kind = blockquote_kind(label)
    if kind is None:
        return None, index

    rest = match.group("rest").strip()
    content = ([rest] if rest else []) + body[1:]
    content = strip_blank_edges(content) or [label]
    return [f'!!! {kind} "{label}"', *indent_for_admonition(content), ""], end


def convert_inline_diagnostic(line: str) -> list[str] | None:
    match = re.match(r"^\*\*(Diagnostic|诊断)\*\*:?\s*(.+)$", line)
    if not match:
        return None
    title = match.group(1)
    message = match.group(2).strip()
    return [f'!!! failure "{title}"', f"    {message}", ""]


def convert_top_level_callouts(lines: list[str], stats: Stats) -> list[str]:
    output: list[str] = []
    index = 0
    in_fence = False
    while index < len(lines):
        line = lines[index]
        if line.startswith("```") or line.startswith("~~~"):
            in_fence = not in_fence
            output.append(line)
            index += 1
            continue

        if not in_fence:
            converted_quote, next_index = convert_blockquote(lines, index)
            if converted_quote is not None:
                output.extend(converted_quote)
                stats.blockquotes += 1
                index = next_index
                continue

            converted_diagnostic = convert_inline_diagnostic(line)
            if converted_diagnostic is not None:
                output.extend(converted_diagnostic)
                stats.diagnostics += 1
                index += 1
                continue

        output.append(line)
        index += 1
    return output


def migrate_file(path: Path, stats: Stats, check: bool) -> bool:
    original_bytes = path.read_bytes()
    newline = "\r\n" if b"\r\n" in original_bytes else "\n"
    original = original_bytes.decode("utf-8")
    had_trailing_newline = original.endswith("\n")
    lines = original.splitlines()

    local_stats = Stats()
    lines = wrap_sections(lines, local_stats)
    lines = convert_top_level_callouts(lines, local_stats)

    migrated = newline.join(lines)
    if had_trailing_newline:
        migrated += newline

    if migrated == original:
        return False

    stats.sections += local_stats.sections
    stats.target_tabs += local_stats.target_tabs
    stats.blockquotes += local_stats.blockquotes
    stats.diagnostics += local_stats.diagnostics
    stats.files_changed += 1

    if not check:
        path.write_text(migrated, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report pending rewrites without modifying files")
    args = parser.parse_args()

    stats = Stats()
    changed = []
    for path in sorted(ISA_ROOT.rglob("*.md")):
        if migrate_file(path, stats, args.check):
            changed.append(path.relative_to(REPO_ROOT))

    print(f"files_changed: {stats.files_changed}")
    print(f"sections_wrapped: {stats.sections}")
    print(f"target_tab_groups: {stats.target_tabs}")
    print(f"blockquotes_converted: {stats.blockquotes}")
    print(f"diagnostics_converted: {stats.diagnostics}")
    if args.check and changed:
        for path in changed:
            print(path)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
