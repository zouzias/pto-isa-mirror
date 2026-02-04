#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


try:
    from deep_translator import GoogleTranslator
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: deep-translator.\n"
        "Recommended:\n"
        "  python3 -m venv .venv-isa-zh\n"
        "  source .venv-isa-zh/bin/activate\n"
        "  python -m pip install deep-translator==1.11.4\n"
        f"\nImport error: {e}"
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
ISA_EN_DIR = REPO_ROOT / "docs" / "isa"
ISA_ZH_DIR = REPO_ROOT / "docs" / "isa_zh"
ISA_ZH_FIG_DIR = ISA_ZH_DIR / "figures"
CACHE_PATH = REPO_ROOT / "output" / "isa_zh_translate_cache.json"


HEADING_MAP = {
    "## Introduction": "## 简介",
    "## Math Interpretation": "## 数学解释",
    "## Assembly Syntax": "## 汇编语法",
    "## C++ Intrinsic": "## C++ Intrinsic（内建接口）",
    "## Constraints": "## 约束",
    "## Examples": "## 示例",
    "### Auto": "### 自动（Auto）",
    "### Manual": "### 手动（Manual）",
}


POST_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    # Prefer "实现定义" over "实施定义".
    (re.compile(r"实施定义"), "实现定义"),
    (re.compile(r"实施检查"), "实现检查"),
    (re.compile(r"实施说明"), "实现说明"),
    (re.compile(r"已实施"), "已实现"),
    (re.compile(r"针对特定目标实施"), "针对特定目标实现"),
    (re.compile(r"强制实施"), "强制执行"),
    # Prefer "逐元素" over "按元素".
    (re.compile(r"按元素"), "逐元素"),
    # Keep the technical term "Tile" un-translated.
    (re.compile(r"(图块|磁贴|瓷砖|瓦片|平铺)"), "Tile"),
    (re.compile(r"平铺布局"), "Tile 布局"),
    (re.compile(r"平铺位置"), "Tile 位置"),
    # Improve layout phrasing.
    (re.compile(r"行主\b"), "行主序"),
    (re.compile(r"列主\b"), "列主序"),
    (re.compile(r"以行为主"), "行主序"),
    (re.compile(r"以列为主"), "列主序"),
    # Use a consistent term for Intrinsics.
    (re.compile(r"C\\+\\+\\s*内在函数"), "C++ Intrinsic（内建接口）"),
    (re.compile(r"内在函数"), "Intrinsic（内建接口）"),
    # Fix common translation artifacts around compiler lowering / scratch tiles.
    (re.compile(r"降低可能会导致内部划伤\s*Tile"), "Lowering 过程中可能会引入内部 scratch Tile"),
    (re.compile(r"\bLowering 过程中"), "Lowering（降级）过程中"),
    (re.compile(r"；\s+C\+\+"), "；C++"),
    # Preferred Chinese terminology in ISA/programming context.
    (re.compile(r"掩模"), "掩码"),
    (re.compile(r"行优先"), "行主序"),
    (re.compile(r"列优先"), "列主序"),
    (re.compile(r"选择/减少"), "选择/归约"),
    (re.compile(r"掩码图案"), "掩码模式"),
    (re.compile(r"掩码模式收集"), "掩码模式 gather"),
    (re.compile(r"掩码模式聚集"), "掩码模式 gather"),
    (re.compile(r"掩码图案收集"), "掩码模式 gather"),
    (re.compile(r"掩码图案聚集"), "掩码模式 gather"),
    (re.compile(r"\bcol-major\b"), "列主序"),
    (re.compile(r"\brow-major\b"), "行主序"),
    (re.compile(r"TileType::Vec\s+负载"), "TileType::Vec 加载"),
    (re.compile(r"TileType::Mat\s+负载"), "TileType::Mat 加载"),
    (re.compile(r"`TileType::Vec`\s+负载"), "`TileType::Vec` 加载"),
    (re.compile(r"`TileType::Mat`\s+负载"), "`TileType::Mat` 加载"),
    (re.compile(r"DN 带有 列主序"), "DN 列主序"),
    (re.compile(r"对于scaleA"), "对于 scaleA"),
    (re.compile(r"对于scaleB"), "对于 scaleB"),
    (re.compile(r"gather是"), "gather 是"),
    (re.compile(r"shape/valid"), "形状/有效区域"),
    (re.compile(r"编写打包谓词掩码"), "输出打包的谓词掩码"),
    (re.compile(r"^让`", flags=re.M), "设 `"),
    (re.compile(r"^让\s", flags=re.M), "设 "),
    (re.compile(r"^让[:：]", flags=re.M), "设："),
    (re.compile(r"附件："), "累加器："),
    (re.compile(r"矢量"), "向量"),
]


PROTECT_PATTERNS: list[re.Pattern[str]] = [
    # Fenced code blocks
    re.compile(r"```[\s\S]*?```"),
    # Display math blocks
    re.compile(r"\$\$[\s\S]*?\$\$"),
    # Inline code
    re.compile(r"`[^`\n]*`"),
    # Protect headings that are just ISA opcodes (avoid transliteration like '# TADD' -> '# 塔德').
    re.compile(r"^#\s+[A-Z0-9_]+\s*$", flags=re.M),
]

_PROTECT_BLOCK_PATTERNS: list[re.Pattern[str]] = [
    # Fenced code blocks
    re.compile(r"```[\s\S]*?```"),
    # Display math blocks
    re.compile(r"\$\$[\s\S]*?\$\$"),
]


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_cache(cache_path: Path) -> dict[str, str]:
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        # Corrupted cache -> ignore.
        return {}


def _save_cache(cache_path: Path, cache: dict[str, str]) -> None:
    _ensure_parent_dir(cache_path)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _protect(text: str, extra_patterns: list[re.Pattern[str]] | None = None) -> tuple[str, list[str]]:
    placeholders: list[str] = []

    def _sub(m: re.Match[str]) -> str:
        placeholders.append(m.group(0))
        return f"<<<PTOISA_ZH_PH_{len(placeholders) - 1}>>>"

    out = text
    for pat in [*PROTECT_PATTERNS, *(extra_patterns or [])]:
        out = pat.sub(_sub, out)
    return out, placeholders


def _restore(text: str, placeholders: list[str]) -> str:
    out = text
    for i, block in enumerate(placeholders):
        out = out.replace(f"<<<PTOISA_ZH_PH_{i}>>>", block)
    return out


def _protect_custom(text: str, patterns: list[re.Pattern[str]]) -> tuple[str, list[str]]:
    placeholders: list[str] = []

    def _sub(m: re.Match[str]) -> str:
        placeholders.append(m.group(0))
        return f"<<<PTOISA_ZH_PH_{len(placeholders) - 1}>>>"

    out = text
    for pat in patterns:
        out = pat.sub(_sub, out)
    return out, placeholders


def _fix_zh_markdown_spacing(md: str) -> str:
    protected, placeholders = _protect_custom(md, _PROTECT_BLOCK_PATTERNS)
    out = protected
    # Add spaces between CJK text and inline code for readability.
    out = re.sub(r"([\u4e00-\u9fff])`", r"\1 `", out)
    out = re.sub(r"`([\u4e00-\u9fff])", r"` \1", out)
    return _restore(out, placeholders)


def _chunk_text(text: str, max_len: int) -> Iterable[str]:
    # Split on double newlines to keep Markdown structure stable.
    parts = re.split(r"(\n\n+)", text)
    buf = ""
    for part in parts:
        if not part:
            continue
        if buf and len(buf) + len(part) > max_len:
            yield buf
            buf = ""
        buf += part
    if buf:
        yield buf


def _translate_en_to_zh(
    markdown: str, translator: GoogleTranslator, cache: dict[str, str], *, protect_headings: bool = False
) -> str:
    extra_patterns: list[re.Pattern[str]] = []
    if protect_headings:
        extra_patterns.append(re.compile(r"^#{1,6} .*$", flags=re.M))

    protected, placeholders = _protect(markdown, extra_patterns=extra_patterns)

    translated_chunks: list[str] = []
    for chunk in _chunk_text(protected, max_len=3500):
        key = chunk
        if key in cache:
            translated_chunks.append(cache[key])
            continue

        last_err: Exception | None = None
        for attempt in range(5):
            try:
                # Gentle pacing to reduce throttling.
                if attempt == 0:
                    time.sleep(0.05)
                else:
                    time.sleep(0.4 * attempt)
                zh = translator.translate(chunk)
                cache[key] = zh
                translated_chunks.append(zh)
                break
            except Exception as e:  # pragma: no cover
                last_err = e
                continue
        else:  # pragma: no cover
            raise RuntimeError(f"Translation failed after retries: {last_err}") from last_err

    translated = "".join(translated_chunks)
    translated = _restore(translated, placeholders)

    for pat, repl in POST_REPLACEMENTS:
        translated = pat.sub(repl, translated)

    # Add spaces around key terms when adjacent to CJK chars.
    translated = re.sub(r"([\u4e00-\u9fff])Tile", r"\1 Tile", translated)
    translated = re.sub(r"Tile([\u4e00-\u9fff])", r"Tile \1", translated)

    translated = _fix_zh_markdown_spacing(translated)
    return translated


def _apply_heading_map(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    for line in lines:
        out.append(HEADING_MAP.get(line.strip(), line))
    return "\n".join(out) + ("\n" if md.endswith("\n") else "")


def _diagram_section(op: str) -> str:
    return f"## 计算流程图\n\n![{op} 计算流程图](figures/{op}.svg)\n\n"


def _insert_diagram(md: str, op: str) -> str:
    # Insert after the Introduction section, before the next '##' heading.
    lines = md.splitlines(keepends=True)
    intro_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "## 简介":
            intro_idx = i
            break

    if intro_idx is None:
        # Fallback: insert after title.
        for i, line in enumerate(lines):
            if line.startswith("# "):
                insert_at = i + 1
                return "".join(lines[:insert_at]) + "\n" + _diagram_section(op) + "".join(lines[insert_at:])
        return _diagram_section(op) + md

    # Find next section heading after introduction.
    insert_at = len(lines)
    for j in range(intro_idx + 1, len(lines)):
        if lines[j].startswith("## ") and lines[j].strip() != "## 简介":
            insert_at = j
            break

    return "".join(lines[:insert_at]) + _diagram_section(op) + "".join(lines[insert_at:])


@dataclass(frozen=True)
class SvgBox:
    label: str
    kind: str = "tile"
    rows: int = 4
    cols: int = 4
    highlight_cells: tuple[tuple[int, int], ...] = ()
    highlight_rows: tuple[int, ...] = ()
    highlight_cols: tuple[int, ...] = ()


@dataclass(frozen=True)
class SvgDiagram:
    title: str
    inputs: list[SvgBox]
    outputs: list[SvgBox]
    op_label: str
    formula: str | None = None


def _xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


_SAMPLE_CELL = (1, 1)


def _svg_tile(
    label: str,
    *,
    highlight_cell: tuple[int, int] | None = None,
    highlight_rows: tuple[int, ...] = (),
    highlight_cols: tuple[int, ...] = (),
) -> SvgBox:
    return SvgBox(
        label=label,
        kind="tile",
        rows=4,
        cols=4,
        highlight_cells=((highlight_cell,) if highlight_cell is not None else ()),
        highlight_rows=highlight_rows,
        highlight_cols=highlight_cols,
    )


def _svg_row_vec(label: str, *, highlight_col: int | None = None) -> SvgBox:
    return SvgBox(
        label=label,
        kind="row_vec",
        rows=1,
        cols=4,
        highlight_cells=(((0, highlight_col),) if highlight_col is not None else ()),
    )


def _svg_col_vec(label: str, *, highlight_row: int | None = None) -> SvgBox:
    return SvgBox(
        label=label,
        kind="col_vec",
        rows=4,
        cols=1,
        highlight_cells=(((highlight_row, 0),) if highlight_row is not None else ()),
    )


def _svg_scalar(label: str) -> SvgBox:
    return SvgBox(label=label, kind="scalar", rows=1, cols=1)


def _svg_token(label: str) -> SvgBox:
    return SvgBox(label=label, kind="token", rows=1, cols=1)


def _svg_gm(label: str, *, highlight: bool = True) -> SvgBox:
    highlight_cells: tuple[tuple[int, int], ...] = ()
    if highlight:
        # Highlight a 4x4 window (like a Tile view) inside a larger GM grid.
        highlight_cells = tuple((r, c) for r in range(1, 5) for c in range(1, 5))
    return SvgBox(label=label, kind="gm", rows=6, cols=6, highlight_cells=highlight_cells)


def _infer_diagram(op: str) -> SvgDiagram:
    # Memory ops
    if op == "TLOAD":
        return SvgDiagram(
            title=op,
            inputs=[_svg_gm("src (GlobalTensor / GM)")],
            outputs=[_svg_tile("dst (Tile)", highlight_cell=_SAMPLE_CELL)],
            op_label=op,
            formula="dst[i,j] = src[r0+i, c0+j]",
        )
    if op == "TSTORE":
        return SvgDiagram(
            title=op,
            inputs=[_svg_tile("src (Tile)", highlight_cell=_SAMPLE_CELL)],
            outputs=[_svg_gm("dst (GlobalTensor / GM)")],
            op_label=op,
            formula="dst[r0+i, c0+j] = src[i,j]",
        )
    if op == "TSTORE_FP":
        return SvgDiagram(
            title=op,
            inputs=[
                _svg_tile("acc (TileAcc)", highlight_cell=_SAMPLE_CELL),
                _svg_col_vec("fp (scale Tile)", highlight_row=_SAMPLE_CELL[0]),
            ],
            outputs=[_svg_gm("dst (GlobalTensor / GM)")],
            op_label=op,
            formula="store(acc, scale=fp)",
        )
    if op == "MGATHER":
        return SvgDiagram(
            title=op,
            inputs=[_svg_gm("src (GM)"), _svg_tile("idx (Tile)", highlight_cell=_SAMPLE_CELL)],
            outputs=[_svg_tile("dst (Tile)", highlight_cell=_SAMPLE_CELL)],
            op_label=op,
            formula="dst[i,j] = src[idx[i,j]]",
        )
    if op == "MSCATTER":
        return SvgDiagram(
            title=op,
            inputs=[_svg_tile("src (Tile)", highlight_cell=_SAMPLE_CELL), _svg_tile("idx (Tile)", highlight_cell=_SAMPLE_CELL)],
            outputs=[_svg_gm("dst (GM)")],
            op_label=op,
            formula="dst[idx[i,j]] = src[i,j]",
        )
    if op == "TPREFETCH":
        return SvgDiagram(
            title=op,
            inputs=[_svg_gm("src (GM)", highlight=False)],
            outputs=[_svg_token("cache / buffer")],
            op_label=op,
            formula="prefetch hint",
        )

    # GEMM/GEMV
    if op == "TMATMUL":
        return SvgDiagram(
            title=op,
            inputs=[_svg_tile("A (TileLeft)", highlight_rows=(1,)), _svg_tile("B (TileRight)", highlight_cols=(2,))],
            outputs=[_svg_tile("C (TileAcc)", highlight_cell=(1, 2))],
            op_label="Σ",
            formula="C[i,j] = Σ_k A[i,k]·B[k,j]",
        )
    if op == "TMATMUL_ACC":
        return SvgDiagram(
            title=op,
            inputs=[
                _svg_tile("C0 (acc in)", highlight_cell=(1, 2)),
                _svg_tile("A", highlight_rows=(1,)),
                _svg_tile("B", highlight_cols=(2,)),
            ],
            outputs=[_svg_tile("C1 (acc out)", highlight_cell=(1, 2))],
            op_label="Σ+",
            formula="C1 = C0 + A×B",
        )
    if op == "TMATMUL_BIAS":
        return SvgDiagram(
            title=op,
            inputs=[
                _svg_tile("A", highlight_rows=(1,)),
                _svg_tile("B", highlight_cols=(2,)),
                _svg_row_vec("Bias", highlight_col=2),
            ],
            outputs=[_svg_tile("C (TileAcc)", highlight_cell=(1, 2))],
            op_label="Σ+",
            formula="C = A×B + Bias",
        )
    if op == "TMATMUL_MX":
        return SvgDiagram(
            title=op,
            inputs=[
                _svg_tile("A", highlight_rows=(1,)),
                _svg_tile("scaleA", highlight_cell=_SAMPLE_CELL),
                _svg_tile("B", highlight_cols=(2,)),
                _svg_tile("scaleB", highlight_cell=_SAMPLE_CELL),
            ],
            outputs=[_svg_tile("C (TileAcc)", highlight_cell=(1, 2))],
            op_label="Σ",
            formula="C = A×B (mx/quantized)",
        )
    if op == "TGEMV":
        return SvgDiagram(
            title=op,
            inputs=[_svg_tile("A (matrix Tile)", highlight_rows=(1,)), _svg_col_vec("x (vector Tile)", highlight_row=2)],
            outputs=[_svg_col_vec("y (acc/out Tile)", highlight_row=1)],
            op_label="Σ",
            formula="y = A·x",
        )
    if op == "TGEMV_ACC":
        return SvgDiagram(
            title=op,
            inputs=[
                _svg_col_vec("y0 (acc in)", highlight_row=1),
                _svg_tile("A", highlight_rows=(1,)),
                _svg_col_vec("x", highlight_row=2),
            ],
            outputs=[_svg_col_vec("y1 (acc out)", highlight_row=1)],
            op_label="Σ+",
            formula="y1 = y0 + A·x",
        )
    if op == "TGEMV_BIAS":
        return SvgDiagram(
            title=op,
            inputs=[
                _svg_tile("A", highlight_rows=(1,)),
                _svg_col_vec("x", highlight_row=2),
                _svg_col_vec("Bias", highlight_row=1),
            ],
            outputs=[_svg_col_vec("y (acc/out)", highlight_row=1)],
            op_label="Σ+",
            formula="y = A·x + Bias",
        )

    # Row/col reduction
    if op in {"TROWSUM", "TROWMAX", "TROWMIN"}:
        op_label = {"TROWSUM": "Σ", "TROWMAX": "max", "TROWMIN": "min"}[op]
        formula = {
            "TROWSUM": "dst[i] = Σ_j src[i,j]",
            "TROWMAX": "dst[i] = max_j src[i,j]",
            "TROWMIN": "dst[i] = min_j src[i,j]",
        }[op]
        return SvgDiagram(
            title=op,
            inputs=[_svg_tile("src (Tile)", highlight_rows=(1,)), _svg_tile("tmp (scratch)")],
            outputs=[_svg_col_vec("dst (row reduce out)", highlight_row=1)],
            op_label=op_label,
            formula=formula,
        )
    if op in {"TCOLSUM", "TCOLMAX", "TCOLMIN"}:
        op_label = {"TCOLSUM": "Σ", "TCOLMAX": "max", "TCOLMIN": "min"}[op]
        formula = {
            "TCOLSUM": "dst[j] = Σ_i src[i,j]",
            "TCOLMAX": "dst[j] = max_i src[i,j]",
            "TCOLMIN": "dst[j] = min_i src[i,j]",
        }[op]
        return SvgDiagram(
            title=op,
            inputs=[_svg_tile("src (Tile)", highlight_cols=(2,)), _svg_tile("tmp (scratch)")],
            outputs=[_svg_row_vec("dst (col reduce out)", highlight_col=2)],
            op_label=op_label,
            formula=formula,
        )

    # Broadcast expand families
    if op == "TROWEXPAND":
        return SvgDiagram(
            title=op,
            inputs=[_svg_tile("src (Tile)", highlight_cols=(0,))],
            outputs=[_svg_tile("dst (Tile)", highlight_rows=(1,))],
            op_label="bcast",
            formula="dst[i,j] = src[i,0]",
        )
    if op == "TCOLEXPAND":
        return SvgDiagram(
            title=op,
            inputs=[_svg_tile("src (Tile)", highlight_rows=(0,))],
            outputs=[_svg_tile("dst (Tile)", highlight_cols=(2,))],
            op_label="bcast",
            formula="dst[i,j] = src[0,j]",
        )
    if op.startswith("TROWEXPAND") and op != "TROWEXPAND":
        suffix = op.removeprefix("TROWEXPAND")
        sym = {
            "ADD": "+",
            "SUB": "-",
            "MUL": "×",
            "DIV": "÷",
            "MAX": "max",
            "MIN": "min",
            "EXPDIF": "exp(x - s)",
        }.get(suffix, "op")
        return SvgDiagram(
            title=op,
            inputs=[
                _svg_tile("src0 (Tile)", highlight_cell=(1, 2)),
                _svg_col_vec("src1 (row scalars)", highlight_row=1),
            ],
            outputs=[_svg_tile("dst (Tile)", highlight_cell=(1, 2))],
            op_label=sym,
            formula=f"dst[i,j] = src0[i,j] {sym} src1[i]",
        )
    if op.startswith("TCOLEXPAND") and op != "TCOLEXPAND":
        suffix = op.removeprefix("TCOLEXPAND")
        sym = {
            "ADD": "+",
            "SUB": "-",
            "MUL": "×",
            "DIV": "÷",
            "MAX": "max",
            "MIN": "min",
            "EXPDIF": "exp(x - s)",
        }.get(suffix, "op")
        return SvgDiagram(
            title=op,
            inputs=[
                _svg_tile("src0 (Tile)", highlight_cell=(1, 2)),
                _svg_row_vec("src1 (col scalars)", highlight_col=2),
            ],
            outputs=[_svg_tile("dst (Tile)", highlight_cell=(1, 2))],
            op_label=sym,
            formula=f"dst[i,j] = src0[i,j] {sym} src1[j]",
        )

    # Padding
    if op == "TFILLPAD":
        return SvgDiagram(
            title=op,
            inputs=[_svg_tile("src (Tile)", highlight_cell=_SAMPLE_CELL)],
            outputs=[_svg_tile("dst (Tile)", highlight_cell=_SAMPLE_CELL)],
            op_label="pad",
            formula="copy + pad(dst.PadVal)",
        )
    if op in {"TFILLPAD_INPLACE", "TFILLPAD_EXPAND"}:
        return SvgDiagram(
            title=op,
            inputs=[_svg_tile("src/dst (Tile)", highlight_cell=_SAMPLE_CELL)],
            outputs=[_svg_tile("dst (Tile)", highlight_cell=_SAMPLE_CELL)],
            op_label="pad",
            formula="pad variant (impl-defined)",
        )

    # Data movement/layout
    if op in {"TMOV", "TMOV_FP", "TTRANS", "TRESHAPE", "TEXTRACT", "TEXTRACT_FP", "TINSERT", "TINSERT_FP"}:
        return SvgDiagram(
            title=op,
            inputs=[_svg_tile("src (Tile)", highlight_cell=_SAMPLE_CELL)],
            outputs=[_svg_tile("dst (Tile)", highlight_cell=_SAMPLE_CELL)],
            op_label="move",
            formula="layout/data move",
        )
    if op == "TASSIGN":
        return SvgDiagram(
            title=op,
            inputs=[_svg_scalar("addr (SR/imm)")],
            outputs=[_svg_tile("tile (bound)")],
            op_label="bind",
            formula="tile ↦ address binding",
        )
    if op in {"TIMG2COL", "TSETFMATRIX"}:
        return SvgDiagram(
            title=op,
            inputs=[_svg_token("src/config")],
            outputs=[_svg_token("dst/config")],
            op_label="cfg",
            formula="im2col / config",
        )

    # Complex / misc
    if op == "TCI":
        return SvgDiagram(
            title=op,
            inputs=[_svg_scalar("start/step (imm)")],
            outputs=[_svg_tile("dst (Tile)", highlight_cell=_SAMPLE_CELL)],
            op_label="iota",
            formula="dst = iota()",
        )
    if op in {"TGATHER", "TGATHERB"}:
        return SvgDiagram(
            title=op,
            inputs=[_svg_tile("src (Tile)", highlight_cell=_SAMPLE_CELL), _svg_tile("idx/mask", highlight_cell=_SAMPLE_CELL)],
            outputs=[_svg_tile("dst (Tile)", highlight_cell=_SAMPLE_CELL)],
            op_label="gather",
            formula="dst = gather(src, idx)",
        )
    if op == "TSCATTER":
        return SvgDiagram(
            title=op,
            inputs=[_svg_tile("src (Tile)", highlight_cell=_SAMPLE_CELL), _svg_tile("idx (Tile)", highlight_cell=_SAMPLE_CELL)],
            outputs=[_svg_tile("dst (Tile)", highlight_cell=_SAMPLE_CELL)],
            op_label="scatter",
            formula="dst[idx[i,j], j] = src[i,j]",
        )
    if op == "TSORT32":
        return SvgDiagram(
            title=op,
            inputs=[_svg_tile("src (Tile)", highlight_cell=_SAMPLE_CELL)],
            outputs=[_svg_tile("dst (sorted Tile)", highlight_cell=_SAMPLE_CELL), _svg_tile("idx (perm Tile)", highlight_cell=_SAMPLE_CELL)],
            op_label="sort",
            formula="sort32 + permutation",
        )
    if op == "TMRGSORT":
        return SvgDiagram(
            title=op,
            inputs=[_svg_row_vec("src0..srcN (sorted lists)", highlight_col=2)],
            outputs=[_svg_row_vec("dst (merged)", highlight_col=2), _svg_token("executed (counts)")],
            op_label="merge",
            formula="merge(src*)",
        )
    if op in {"TPARTADD", "TPARTMAX", "TPARTMIN"}:
        return SvgDiagram(
            title=op,
            inputs=[_svg_tile("src0 (Tile)", highlight_cell=_SAMPLE_CELL), _svg_tile("src1 (Tile)", highlight_cell=_SAMPLE_CELL)],
            outputs=[_svg_tile("dst (Tile)", highlight_cell=_SAMPLE_CELL)],
            op_label=op.removeprefix("TPART").lower(),
            formula="partial elementwise op",
        )
    if op == "TTRI":
        return SvgDiagram(
            title=op,
            inputs=[_svg_scalar("mode/shape")],
            outputs=[_svg_tile("dst (mask Tile)", highlight_rows=(0,), highlight_cols=(0,))],
            op_label="tri",
            formula="triangular mask",
        )
    if op == "TQUANT":
        return SvgDiagram(
            title=op,
            inputs=[_svg_tile("src (FP32 Tile)", highlight_cell=_SAMPLE_CELL)],
            outputs=[
                _svg_tile("dst (e.g. FP8)", highlight_cell=_SAMPLE_CELL),
                _svg_col_vec("exp", highlight_row=_SAMPLE_CELL[0]),
                _svg_scalar("max"),
                _svg_col_vec("scaling", highlight_row=_SAMPLE_CELL[0]),
            ],
            op_label="quant",
            formula="quantize(mode)",
        )
    if op == "TPRINT":
        return SvgDiagram(
            title=op,
            inputs=[_svg_tile("src (Tile)", highlight_cell=_SAMPLE_CELL)],
            outputs=[_svg_token("debug output")],
            op_label="print",
            formula="print (impl-defined)",
        )
    if op == "TSYNC":
        return SvgDiagram(
            title=op,
            inputs=[_svg_token("events")],
            outputs=[_svg_token("barrier")],
            op_label="sync",
            formula="wait(events...)",
        )

    # Elementwise defaults
    unary_ops = {
        "TABS": "abs(x)",
        "TEXP": "exp(x)",
        "TLOG": "log(x)",
        "TSQRT": "sqrt(x)",
        "TRSQRT": "rsqrt(x)",
        "TRECIP": "1/x",
        "TNEG": "-x",
        "TNOT": "~x",
        "TRELU": "relu(x)",
        "TEXPANDS": "scalar",
        "TCVT": "cast(x)",
    }
    if op in unary_ops:
        return SvgDiagram(
            title=op,
            inputs=[_svg_tile("src (Tile)", highlight_cell=_SAMPLE_CELL)],
            outputs=[_svg_tile("dst (Tile)", highlight_cell=_SAMPLE_CELL)],
            op_label=unary_ops[op],
            formula=f"dst[i,j] = {unary_ops[op]}",
        )

    if op == "TLRELU":
        return SvgDiagram(
            title=op,
            inputs=[_svg_tile("src (Tile)", highlight_cell=_SAMPLE_CELL), _svg_scalar("slope (scalar)")],
            outputs=[_svg_tile("dst (Tile)", highlight_cell=_SAMPLE_CELL)],
            op_label="lrelu",
            formula="dst[i,j] = (x>0)?x:x·slope",
        )
    if op == "TPRELU":
        return SvgDiagram(
            title=op,
            inputs=[
                _svg_tile("src0 (Tile)", highlight_cell=_SAMPLE_CELL),
                _svg_tile("src1 (slope Tile)", highlight_cell=_SAMPLE_CELL),
            ],
            outputs=[_svg_tile("dst (Tile)", highlight_cell=_SAMPLE_CELL)],
            op_label="prelu",
            formula="dst[i,j] = (x>0)?x:x·slope",
        )
    if op == "TSEL":
        return SvgDiagram(
            title=op,
            inputs=[
                _svg_tile("mask (Tile)", highlight_cell=_SAMPLE_CELL),
                _svg_tile("src0 (Tile)", highlight_cell=_SAMPLE_CELL),
                _svg_tile("src1 (Tile)", highlight_cell=_SAMPLE_CELL),
            ],
            outputs=[_svg_tile("dst (Tile)", highlight_cell=_SAMPLE_CELL)],
            op_label="sel",
            formula="dst = mask ? src0 : src1",
        )
    if op == "TSELS":
        return SvgDiagram(
            title=op,
            inputs=[
                _svg_tile("src0 (Tile)", highlight_cell=_SAMPLE_CELL),
                _svg_tile("src1 (Tile)", highlight_cell=_SAMPLE_CELL),
                _svg_scalar("selectMode (scalar)"),
            ],
            outputs=[_svg_tile("dst (Tile)", highlight_cell=_SAMPLE_CELL)],
            op_label="sel",
            formula="dst = (selectMode==1) ? src0 : src1",
        )
    if op == "TCMP":
        return SvgDiagram(
            title=op,
            inputs=[
                _svg_tile("src0 (Tile)", highlight_cell=_SAMPLE_CELL),
                _svg_tile("src1 (Tile)", highlight_cell=_SAMPLE_CELL),
                _svg_scalar("cmpMode"),
            ],
            outputs=[_svg_tile("dst (mask Tile)", highlight_cell=_SAMPLE_CELL)],
            op_label="cmp",
            formula="dst = (src0 cmp src1)",
        )

    # Scalar tile ops (tile + scalar)
    scalar_ops = {
        "TADDS": "+",
        "TSUBS": "-",
        "TMULS": "×",
        "TDIVS": "÷",
        "TREMS": "rem",
        "TMAXS": "max",
        "TMINS": "min",
        "TANDS": "&",
        "TORS": "|",
        "TXORS": "^",
        "TCMPS": "cmp",
        "TSHLS": "<<",
        "TSHRS": ">>",
        "TREM": "rem",
    }
    if op in scalar_ops:
        return SvgDiagram(
            title=op,
            inputs=[_svg_tile("src (Tile)", highlight_cell=_SAMPLE_CELL), _svg_scalar("scalar (imm)")],
            outputs=[_svg_tile("dst (Tile)", highlight_cell=_SAMPLE_CELL)],
            op_label=scalar_ops[op],
            formula=f"dst[i,j] = src[i,j] {scalar_ops[op]} scalar",
        )

    if op == "TADDSC":
        return SvgDiagram(
            title=op,
            inputs=[
                _svg_tile("src0 (Tile)", highlight_cell=_SAMPLE_CELL),
                _svg_scalar("scalar"),
                _svg_tile("src1 (Tile)", highlight_cell=_SAMPLE_CELL),
            ],
            outputs=[_svg_tile("dst (Tile)", highlight_cell=_SAMPLE_CELL)],
            op_label="+",
            formula="dst = src0 + scalar + src1",
        )
    if op == "TSUBSC":
        return SvgDiagram(
            title=op,
            inputs=[
                _svg_tile("src0 (Tile)", highlight_cell=_SAMPLE_CELL),
                _svg_scalar("scalar"),
                _svg_tile("src1 (Tile)", highlight_cell=_SAMPLE_CELL),
            ],
            outputs=[_svg_tile("dst (Tile)", highlight_cell=_SAMPLE_CELL)],
            op_label="±",
            formula="dst = src0 - scalar + src1",
        )
    if op == "TADDC":
        return SvgDiagram(
            title=op,
            inputs=[
                _svg_tile("src0 (Tile)", highlight_cell=_SAMPLE_CELL),
                _svg_tile("src1 (Tile)", highlight_cell=_SAMPLE_CELL),
                _svg_tile("src2 (Tile)", highlight_cell=_SAMPLE_CELL),
            ],
            outputs=[_svg_tile("dst (Tile)", highlight_cell=_SAMPLE_CELL)],
            op_label="+",
            formula="dst = src0 + src1 + src2",
        )
    if op == "TSUBC":
        return SvgDiagram(
            title=op,
            inputs=[
                _svg_tile("src0 (Tile)", highlight_cell=_SAMPLE_CELL),
                _svg_tile("src1 (Tile)", highlight_cell=_SAMPLE_CELL),
                _svg_tile("src2 (Tile)", highlight_cell=_SAMPLE_CELL),
            ],
            outputs=[_svg_tile("dst (Tile)", highlight_cell=_SAMPLE_CELL)],
            op_label="±",
            formula="dst = src0 - src1 + src2",
        )

    binary_ops = {
        "TADD": "+",
        "TSUB": "-",
        "TMUL": "×",
        "TDIV": "÷",
        "TAND": "&",
        "TOR": "|",
        "TXOR": "^",
        "TMIN": "min",
        "TMAX": "max",
        "TSHL": "<<",
        "TSHR": ">>",
        "TREM": "rem",
    }
    if op in binary_ops:
        sym = binary_ops[op]
        return SvgDiagram(
            title=op,
            inputs=[_svg_tile("src0 (Tile)", highlight_cell=_SAMPLE_CELL), _svg_tile("src1 (Tile)", highlight_cell=_SAMPLE_CELL)],
            outputs=[_svg_tile("dst (Tile)", highlight_cell=_SAMPLE_CELL)],
            op_label=sym,
            formula=f"dst[i,j] = src0[i,j] {sym} src1[i,j]",
        )

    # Fallback: generic one-in/one-out.
    return SvgDiagram(
        title=op,
        inputs=[_svg_tile("src (Tile)", highlight_cell=_SAMPLE_CELL)],
        outputs=[_svg_tile("dst (Tile)", highlight_cell=_SAMPLE_CELL)],
        op_label=op,
        formula=None,
    )


def _render_svg(diagram: SvgDiagram) -> str:
    cell = 18
    label_h = 20
    gap_y = 26
    margin_x = 56
    margin_y = 18
    title_h = 54
    formula_h = 32 if diagram.formula else 0
    op_r = 30
    gap_x = 110

    def _box_size(box: SvgBox) -> tuple[int, int]:
        if box.kind == "token":
            return (180, 48)
        w = max(1, box.cols) * cell
        h = max(1, box.rows) * cell
        if box.kind == "scalar":
            w = max(w, 54)
            h = max(h, 54)
        return (w, h)

    def _stack_height(boxes: list[SvgBox]) -> int:
        if not boxes:
            return 0
        heights = [((0 if b.kind == "token" else label_h) + _box_size(b)[1]) for b in boxes]
        return sum(heights) + gap_y * (len(boxes) - 1)

    in_max_w = max((_box_size(b)[0] for b in diagram.inputs), default=0)
    out_max_w = max((_box_size(b)[0] for b in diagram.outputs), default=0)

    width = margin_x + in_max_w + gap_x + (op_r * 2) + gap_x + out_max_w + margin_x
    width = max(width, 920)

    in_total_h = _stack_height(diagram.inputs)
    out_total_h = _stack_height(diagram.outputs)
    main_h = max(in_total_h, out_total_h, op_r * 2 + 40)
    height = title_h + main_h + formula_h + margin_y

    cx = width / 2
    op_cy = title_h + main_h / 2
    op_cx = cx

    def _arrow(x1: float, y1: float, x2: float, y2: float) -> str:
        return (
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            'stroke="#374151" stroke-width="1.6" marker-end="url(#arrow)"/>'
        )

    def _draw_grid_box(
        box: SvgBox,
        *,
        x: float,
        y: float,
        fill: str,
        stroke: str,
    ) -> tuple[str, tuple[float, float]]:
        w, h = _box_size(box)
        parts: list[str] = []
        label = _xml_escape(box.label)

        grid_y = y if box.kind == "token" else y + label_h

        if box.kind == "token":
            parts.append(
                f'<rect x="{x:.1f}" y="{grid_y:.1f}" rx="12" ry="12" width="{w:.1f}" height="{h:.1f}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1.2" stroke-dasharray="4 4"/>'
            )
            parts.append(
                f'<text x="{x + w / 2:.1f}" y="{grid_y + h / 2 + 5:.1f}" text-anchor="middle" '
                'font-family="Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial" '
                'font-size="14" fill="#111827">'
                f"{label}</text>"
            )
            return "\n".join(parts), (x + w / 2, grid_y + h / 2)

        parts.append(
            f'<text x="{x + w / 2:.1f}" y="{y + 15:.1f}" text-anchor="middle" '
            'font-family="Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial" '
            'font-size="13" fill="#111827">'
            f"{label}</text>"
        )

        parts.append(
            f'<rect x="{x:.1f}" y="{grid_y:.1f}" rx="12" ry="12" width="{w:.1f}" height="{h:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
        )

        # Highlights (draw first; grid lines later so they are visible on top).
        hl_color = "#FDE68A"
        hl_opacity = "0.75"

        for r in box.highlight_rows:
            if 0 <= r < box.rows:
                parts.append(
                    f'<rect x="{x:.1f}" y="{grid_y + r * cell:.1f}" width="{w:.1f}" height="{cell:.1f}" '
                    f'fill="{hl_color}" fill-opacity="{hl_opacity}" rx="10" ry="10"/>'
                )
        for c in box.highlight_cols:
            if 0 <= c < box.cols:
                parts.append(
                    f'<rect x="{x + c * cell:.1f}" y="{grid_y:.1f}" width="{cell:.1f}" height="{h:.1f}" '
                    f'fill="{hl_color}" fill-opacity="{hl_opacity}" rx="10" ry="10"/>'
                )
        for r, c in box.highlight_cells:
            if 0 <= r < box.rows and 0 <= c < box.cols:
                parts.append(
                    f'<rect x="{x + c * cell:.1f}" y="{grid_y + r * cell:.1f}" '
                    f'width="{cell:.1f}" height="{cell:.1f}" fill="{hl_color}" fill-opacity="{hl_opacity}" rx="8" ry="8"/>'
                )

        # Grid lines
        line_stroke = "#D1D5DB"
        for r in range(1, box.rows):
            y1 = grid_y + r * cell
            parts.append(
                f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x + w:.1f}" y2="{y1:.1f}" '
                f'stroke="{line_stroke}" stroke-width="1"/>'
            )
        for c in range(1, box.cols):
            x1 = x + c * cell
            parts.append(
                f'<line x1="{x1:.1f}" y1="{grid_y:.1f}" x2="{x1:.1f}" y2="{grid_y + h:.1f}" '
                f'stroke="{line_stroke}" stroke-width="1"/>'
            )

        # Anchor point:
        # - highlighted cell(s): use cell center; for multiple cells use the bounding-box center
        # - highlighted row/col: use row/col center
        # - else: grid center
        if box.highlight_cells:
            if len(box.highlight_cells) == 1:
                r, c = box.highlight_cells[0]
                anchor = (x + (c + 0.5) * cell, grid_y + (r + 0.5) * cell)
            else:
                rs = [rc[0] for rc in box.highlight_cells]
                cs = [rc[1] for rc in box.highlight_cells]
                r0, r1 = min(rs), max(rs)
                c0, c1 = min(cs), max(cs)
                anchor = (x + ((c0 + c1 + 1) / 2) * cell, grid_y + ((r0 + r1 + 1) / 2) * cell)
        elif box.highlight_rows:
            r = box.highlight_rows[0]
            anchor = (x + w / 2, grid_y + (r + 0.5) * cell)
        elif box.highlight_cols:
            c = box.highlight_cols[0]
            anchor = (x + (c + 0.5) * cell, grid_y + h / 2)
        else:
            anchor = (x + w / 2, grid_y + h / 2)

        return "\n".join(parts), anchor

    def _stack_positions(boxes: list[SvgBox], start_y: float) -> list[float]:
        ys: list[float] = []
        y = start_y
        for b in boxes:
            ys.append(y)
            y += (0 if b.kind == "token" else label_h) + _box_size(b)[1] + gap_y
        return ys

    in_start_y = title_h + (main_h - in_total_h) / 2 if diagram.inputs else title_h + main_h / 2
    out_start_y = title_h + (main_h - out_total_h) / 2 if diagram.outputs else title_h + main_h / 2

    in_x = margin_x
    out_right = width - margin_x

    svg_parts: list[str] = []
    svg_parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}">'
    )
    svg_parts.append(
        "<defs>"
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">'
        '<path d="M0,0 L0,6 L9,3 z" fill="#374151" />'
        "</marker>"
        "</defs>"
    )
    svg_parts.append(
        f'<text x="{cx:.1f}" y="28" text-anchor="middle" '
        'font-family="Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial" '
        'font-size="20" font-weight="700" fill="#111827">'
        f"{_xml_escape(diagram.title)}</text>"
    )
    svg_parts.append(
        f'<text x="{cx:.1f}" y="48" text-anchor="middle" '
        'font-family="Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial" '
        'font-size="12" fill="#6B7280">'
        "Tile（2D SIMD）元素网格示意</text>"
    )

    # Operation node (center)
    svg_parts.append(
        f'<circle cx="{op_cx:.1f}" cy="{op_cy:.1f}" r="{op_r}" '
        'fill="#F3F4F6" stroke="#9CA3AF" stroke-width="1.2"/>'
    )
    op_label = _xml_escape(diagram.op_label)
    op_font = 14 if len(diagram.op_label) <= 6 else 12
    svg_parts.append(
        f'<text x="{op_cx:.1f}" y="{op_cy + 5:.1f}" text-anchor="middle" '
        'font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" '
        f'font-size="{op_font}" fill="#111827">{op_label}</text>'
    )

    # Boxes + anchors
    in_ys = _stack_positions(diagram.inputs, in_start_y)
    out_ys = _stack_positions(diagram.outputs, out_start_y)
    in_anchors: list[tuple[float, float]] = []
    out_anchors: list[tuple[float, float]] = []

    for b, y in zip(diagram.inputs, in_ys, strict=False):
        fill = "#EFF6FF" if b.kind != "gm" else "#FFF7ED"
        stroke = "#60A5FA" if b.kind != "gm" else "#FB923C"
        box_svg, anchor = _draw_grid_box(b, x=in_x, y=y, fill=fill, stroke=stroke)
        svg_parts.append(box_svg)
        in_anchors.append(anchor)

    for b, y in zip(diagram.outputs, out_ys, strict=False):
        w, _ = _box_size(b)
        x = out_right - w
        fill = "#ECFDF5" if b.kind != "gm" else "#FFF7ED"
        stroke = "#34D399" if b.kind != "gm" else "#FB923C"
        box_svg, anchor = _draw_grid_box(b, x=x, y=y, fill=fill, stroke=stroke)
        svg_parts.append(box_svg)
        out_anchors.append(anchor)

    # Arrows
    for x1, y1 in in_anchors:
        svg_parts.append(_arrow(x1, y1, op_cx - op_r, op_cy))
    for x2, y2 in out_anchors:
        svg_parts.append(_arrow(op_cx + op_r, op_cy, x2, y2))

    if diagram.formula:
        svg_parts.append(
            f'<text x="{cx:.1f}" y="{height - margin_y:.1f}" text-anchor="middle" '
            'font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" '
            'font-size="13" fill="#111827">'
            f"{_xml_escape(diagram.formula)}</text>"
        )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts) + "\n"


def _intro_cn(op: str) -> str:
    # Memory
    if op == "TLOAD":
        return "从 GlobalTensor（GM）加载到片上 Tile；传输范围由 `dst` 的有效区域（`GetValidRow/Col()`）决定。"
    if op == "TSTORE":
        return "将片上 Tile 按有效区域写回 GlobalTensor（GM）。"
    if op == "TSTORE_FP":
        return "将 accumulator Tile 写回 GM，并使用 `fp`（scale）Tile 作为向量量化/缩放参数（实现定义）。"
    if op == "MGATHER":
        return "使用逐元素索引从 GM 进行 gather-load，结果写入 `dst` Tile。"
    if op == "MSCATTER":
        return "使用逐元素索引将 `src` Tile scatter-store 写回 GM（越界/冲突写入行为实现定义）。"
    if op == "TPREFETCH":
        return "预取指令：将 GM 数据预取到 Tile 侧缓存/缓冲（hint，行为实现定义），用于降低后续访问延迟。"

    # GEMM / GEMV
    if op == "TMATMUL":
        return "矩阵乘（GEMM）：计算 `C = A × B` 并写入 accumulator/output Tile。"
    if op == "TMATMUL_ACC":
        return "融合累加的矩阵乘：计算 `C1 = C0 + A × B`（`C0` 为 accumulator 输入）。"
    if op == "TMATMUL_BIAS":
        return "带 bias 的矩阵乘：计算 `C = A × B + Bias`（bias 的广播语义实现定义）。"
    if op == "TMATMUL_MX":
        return "带额外缩放/量化参数的矩阵乘（mx 变体，支持范围由目标后端实现定义）。"
    if op == "TGEMV":
        return "矩阵-向量乘（GEMV）：计算 `y = A · x` 并写入 accumulator/output Tile。"
    if op == "TGEMV_ACC":
        return "融合累加的 GEMV：计算 `y1 = y0 + A · x`（`y0` 为 accumulator 输入）。"
    if op == "TGEMV_BIAS":
        return "带 bias 的 GEMV：计算 `y = A · x + Bias`（bias 语义实现定义）。"

    # Reductions
    if op == "TROWSUM":
        return "按行归约（求和）：沿列方向对每一行求和，输出为每行一个标量（C++ Intrinsic 需要额外的 `tmp` scratch Tile）。"
    if op == "TROWMAX":
        return "按行归约（最大值）：沿列方向对每一行取最大值，输出为每行一个标量（C++ Intrinsic 需要额外的 `tmp` scratch Tile）。"
    if op == "TROWMIN":
        return "按行归约（最小值）：沿列方向对每一行取最小值，输出为每行一个标量（C++ Intrinsic 需要额外的 `tmp` scratch Tile）。"
    if op == "TCOLSUM":
        return "按列归约（求和）：沿行方向对每一列求和，输出为每列一个标量（C++ Intrinsic 需要额外的 `tmp` scratch Tile）。"
    if op == "TCOLMAX":
        return "按列归约（最大值）：沿行方向对每一列取最大值，输出为每列一个标量（C++ Intrinsic 需要额外的 `tmp` scratch Tile）。"
    if op == "TCOLMIN":
        return "按列归约（最小值）：沿行方向对每一列取最小值，输出为每列一个标量（C++ Intrinsic 需要额外的 `tmp` scratch Tile）。"

    # Broadcast / expand
    if op == "TROWEXPAND":
        return "按行广播：将每行的 `src[i,0]` 广播到该行所有列，写入 `dst`。"
    if op == "TCOLEXPAND":
        return "按列广播：将每列的 `src[0,j]` 广播到该列所有行，写入 `dst`。"
    if op.startswith("TROWEXPAND") and op != "TROWEXPAND":
        return "按行广播运算：对 `src0` 的每一行使用 `src1` 提供的行标量（每行一个值）进行广播，并对整行逐元素计算后写入 `dst`。"
    if op.startswith("TCOLEXPAND") and op != "TCOLEXPAND":
        return "按列广播运算：对 `src0` 的每一列使用 `src1` 提供的列标量（每列一个值）进行广播，并对整列逐元素计算后写入 `dst`。"

    # Padding
    if op == "TFILLPAD":
        return "按有效区域将 `src` 拷贝到 `dst`，其余位置用编译期 `PadVal` 指定的 padding 值填充。"
    if op == "TFILLPAD_INPLACE":
        return "`TFILLPAD` 的 in-place 变体（实现定义）。"
    if op == "TFILLPAD_EXPAND":
        return "`TFILLPAD` 的 expand 变体：允许 `dst` 大于 `src`（实现定义）。"

    # Data movement / layout
    if op == "TMOV":
        return "Tile 间搬运/拷贝（可带实现定义的转换模式）。"
    if op == "TMOV_FP":
        return "从 accumulator Tile 搬运/转换到目标 Tile，并使用 `fp`（scale）Tile 作为向量量化/缩放参数（实现定义）。"
    if op == "TTRANS":
        return "Tile 转置（lowering 过程中可能引入内部 scratch Tile）。"
    if op == "TEXTRACT":
        return "从源 Tile 中按 `(indexRow, indexCol)` 提取 sub-tile 并写入 `dst`。"
    if op == "TEXTRACT_FP":
        return "带 `fp`（scale）Tile 的 `TEXTRACT` 变体（向量量化/缩放参数，行为实现定义）。"
    if op == "TINSERT":
        return "将源 sub-tile 按 `(indexRow, indexCol)` 插入到目标 Tile。"
    if op == "TINSERT_FP":
        return "带 `fp`（scale）Tile 的 `TINSERT` 变体（向量量化/缩放参数，行为实现定义）。"
    if op == "TRESHAPE":
        return "在不改变底层字节序列的前提下，将 Tile 重新解释为另一种 Tile 类型/形状。"
    if op == "TASSIGN":
        return "将 Tile 对象绑定到实现定义的片上地址（手动 placement / 地址绑定）。"
    if op == "TIMG2COL":
        return "将输入特征图 Tile 变换为 im2col 矩阵 Tile（卷积类 workload），参数由 `Img2colTileConfig` 与 `(posM, posK)` 指定。"
    if op == "TSETFMATRIX":
        return "配置 `TIMG2COL` 使用的 FMATRIX 参数（实现定义）。"

    # Complex / misc
    if op == "TCI":
        return "在目标 Tile 中生成连续整数序列（iota），用于构造索引/掩码等。"
    if op == "TGATHER":
        return "按索引 Tile 或编译期 `MaskPattern` 从源 Tile 中 gather 选取元素并写入 `dst`（越界行为实现定义）。"
    if op == "TGATHERB":
        return "按字节偏移从源 Tile 中 gather 选取元素并写入 `dst`（具体偏移解释实现定义）。"
    if op == "TSCATTER":
        return "使用逐元素行索引将 `src` 的元素写入 `dst[idx[i,j], j]`（冲突写入行为实现定义）。"
    if op == "TSORT32":
        return "对固定大小的 32 元素块排序，并输出排序后的值与索引映射。"
    if op == "TMRGSORT":
        return "归并排序：将多个已排序列表合并为一个有序列表（元素格式/布局与 executed 语义实现定义）。"
    if op in {"TPARTADD", "TPARTMAX", "TPARTMIN"}:
        return "部分逐元素运算：用于有效区域不一致场景（覆盖范围与边界行为实现定义）。"
    if op == "TTRI":
        return "生成上/下三角掩码 Tile（由编译期参数控制，常用于 attention / band mask）。"
    if op == "TQUANT":
        return "量化：将 FP32 Tile 量化为更低精度格式（如 FP8），并输出 exp/scaling/max 等辅助 Tile（实现定义）。"
    if op == "TPRINT":
        return "调试指令：打印/导出 Tile 内容（实现定义）。"
    if op == "TSYNC":
        return (
            "PTO 执行同步：\n\n"
            "- `TSYNC(events...)`：等待一组显式事件令牌。\n"
            "- `TSYNC<Op>()`：对单个向量 op 类插入 pipeline barrier。\n\n"
            "许多 Intrinsic 会在发出指令前内部调用 `TSYNC(events...)`。"
        )

    # Elementwise (Tile SIMD)
    bin_op = {
        "TADD": "加法",
        "TSUB": "减法",
        "TMUL": "乘法",
        "TDIV": "除法",
        "TREM": "取余",
        "TSHL": "左移",
        "TSHR": "右移",
        "TAND": "按位与",
        "TOR": "按位或",
        "TXOR": "按位异或",
        "TMIN": "取最小值",
        "TMAX": "取最大值",
    }
    if op in bin_op:
        return f"在 `dst` 的有效区域内，对 `src0` 与 `src1` 执行逐元素{bin_op[op]}，结果写入 `dst`（Tile 以 2D SIMD 方式并行计算）。"

    unary_op = {
        "TABS": "绝对值",
        "TEXP": "指数",
        "TLOG": "自然对数",
        "TSQRT": "平方根",
        "TRSQRT": "倒平方根",
        "TRECIP": "倒数",
        "TNEG": "取负",
        "TNOT": "按位取反",
        "TRELU": "ReLU",
        "TCVT": "类型转换（带舍入模式）",
    }
    if op in unary_op:
        return f"在 `dst` 的有效区域内，对 `src` 执行逐元素{unary_op[op]}，结果写入 `dst`（Tile 以 2D SIMD 方式并行计算）。"

    scalar_op = {
        "TADDS": "加",
        "TSUBS": "减",
        "TMULS": "乘",
        "TDIVS": "除",
        "TREMS": "取余",
        "TMAXS": "取最大值",
        "TMINS": "取最小值",
        "TANDS": "按位与",
        "TORS": "按位或",
        "TXORS": "按位异或",
        "TCMPS": "比较",
        "TSHLS": "左移",
        "TSHRS": "右移",
    }
    if op in scalar_op:
        return f"在 `dst` 的有效区域内，对 `src` 的每个元素与标量 `scalar` 执行逐元素{scalar_op[op]}，结果写入 `dst`。"

    if op == "TADDSC":
        return "在 `dst` 的有效区域内逐元素计算 `src0 + scalar + src1` 并写入 `dst`（融合标量与双输入）。"
    if op == "TSUBSC":
        return "在 `dst` 的有效区域内逐元素计算 `src0 - scalar + src1` 并写入 `dst`（融合标量与双输入）。"
    if op == "TADDC":
        return "在 `dst` 的有效区域内逐元素计算 `src0 + src1 + src2` 并写入 `dst`（三输入融合加法）。"
    if op == "TSUBC":
        return "在 `dst` 的有效区域内逐元素计算 `src0 - src1 + src2` 并写入 `dst`（三输入融合运算）。"
    if op == "TSEL":
        return "使用掩码 Tile 在 `src0` 与 `src1` 间逐元素选择并写入 `dst`。"
    if op == "TSELS":
        return "使用标量 `selectMode` 在 `src0` 与 `src1` 间做全局选择（每个元素一致），并写入 `dst`。"
    if op == "TCMP":
        return "按 `cmpMode` 对 `src0/src1` 做逐元素比较，并将结果以打包谓词掩码写入 `dst`（编码/布局实现定义）。"
    if op == "TEXPANDS":
        return "将标量广播填充到 `dst` Tile（构造常量 Tile / 广播常数）。"
    if op == "TLRELU":
        return "Leaky ReLU：使用标量 `slope` 对负半轴逐元素缩放。"
    if op == "TPRELU":
        return "PReLU：使用 `slope` Tile 对负半轴逐元素缩放。"

    return "（本指令页面已按 PTO ISA 术语与编程语境进行中文化整理。）"


def _rewrite_intro_section(md: str, op: str) -> str:
    pat = re.compile(r"(?ms)^## 简介\s*\n\n?(.*?)(\n(?=^## )|\Z)")
    m = pat.search(md)
    if not m:
        return md

    intro = _intro_cn(op).rstrip() + "\n"
    replacement = f"## 简介\n\n{intro}\n"
    return md[: m.start()] + replacement + md[m.end() :]


def _is_instruction_doc(filename: str, content: str) -> bool:
    if filename in {"README.md", "README_zh.md", "conventions.md"}:
        return False
    first_line = content.splitlines()[0].strip() if content else ""
    return bool(re.fullmatch(r"#\s+[A-Z0-9_]+", first_line))


def _gen_one(md_path: Path, translator: GoogleTranslator, cache: dict[str, str]) -> None:
    src = md_path.read_text(encoding="utf-8")
    filename = md_path.name

    # README in zh is already provided in-tree; use it as the index page for isa_zh.
    if filename == "README_zh.md":
        out_path = ISA_ZH_DIR / "README.md"
        _ensure_parent_dir(out_path)
        out_path.write_text(src, encoding="utf-8")
        return

    if filename == "README.md":
        # Skip the English index; we write a Chinese index from README_zh.md.
        return

    should_diagram = _is_instruction_doc(filename, src)
    op = md_path.stem

    md = _apply_heading_map(src)
    if should_diagram:
        md = _insert_diagram(md, op)

    zh = _translate_en_to_zh(md, translator=translator, cache=cache, protect_headings=should_diagram)
    if should_diagram:
        zh = _rewrite_intro_section(zh, op)

    out_md_path = ISA_ZH_DIR / filename
    _ensure_parent_dir(out_md_path)
    out_md_path.write_text(zh, encoding="utf-8")

    if should_diagram:
        diagram = _infer_diagram(op)
        svg = _render_svg(diagram)
        out_svg_path = ISA_ZH_FIG_DIR / f"{op}.svg"
        _ensure_parent_dir(out_svg_path)
        out_svg_path.write_text(svg, encoding="utf-8")


def main(argv: list[str]) -> int:
    global ISA_EN_DIR, ISA_ZH_DIR, ISA_ZH_FIG_DIR

    ap = argparse.ArgumentParser(description="Generate Chinese ISA docs + per-op SVG diagrams.")
    ap.add_argument("--input", type=Path, default=ISA_EN_DIR, help="Input directory (default: docs/isa)")
    ap.add_argument("--output", type=Path, default=ISA_ZH_DIR, help="Output directory (default: docs/isa_zh)")
    ap.add_argument("--only", type=str, default=None, help="Generate only a single opcode (e.g. TADD)")
    args = ap.parse_args(argv)
    ISA_EN_DIR = args.input
    ISA_ZH_DIR = args.output
    ISA_ZH_FIG_DIR = ISA_ZH_DIR / "figures"

    if not ISA_EN_DIR.exists():
        raise SystemExit(f"Input dir not found: {ISA_EN_DIR}")

    ISA_ZH_DIR.mkdir(parents=True, exist_ok=True)
    ISA_ZH_FIG_DIR.mkdir(parents=True, exist_ok=True)

    cache = _load_cache(CACHE_PATH)
    translator = GoogleTranslator(source="en", target="zh-CN")

    md_files = sorted(p for p in ISA_EN_DIR.glob("*.md"))
    if args.only:
        md_files = [p for p in md_files if p.stem == args.only or p.name == f"{args.only}.md"]
        if not md_files:
            raise SystemExit(f"No matching ISA markdown for --only={args.only}")

    for p in md_files:
        _gen_one(p, translator=translator, cache=cache)

    _save_cache(CACHE_PATH, cache)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
